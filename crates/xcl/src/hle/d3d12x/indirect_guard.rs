//! The group counts of an ExecuteIndirect of dispatches, looked at before the GPU runs them.
//!
//! The counts come from memory the GPU (or an older copy) wrote. Where they are not what the
//! title meant, they can be anything: a float's bits are a billion groups, and a dispatch of a
//! billion groups holds the GPU for seconds or until Windows resets the driver (a retail title's
//! culling pass on an RTX 5090: 1 in about 2,000 of its ExecuteIndirects ran for 1.4 to 13.7 s,
//! the others for 30 µs). So a program of XCL's own reads the counts first and hands them to the
//! ExecuteIndirect through a buffer of XCL's own: 0 groups where they are out of range. It also
//! writes them where the CPU reads them: the log names the first ones out of range, and the
//! profile (`XCL_GPU_PROFILE=1`) shows how many groups an ExecuteIndirect ran.
//!
//! `XCL_INDIRECT_GUARD=0` turns it off; `XCL_INDIRECT_MAX_GROUPS` is the most groups one
//! ExecuteIndirect may run (4 Mi: a 4K target in 8x8 groups has 130 thousand).

use std::collections::HashSet;
use std::ffi::c_void;
use std::mem::{transmute_copy, ManuallyDrop};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

use super::real;

/// ExecuteIndirects whose counts are kept (a ring): more than the queues hold at once
const SLOTS: u32 = 1 << 15;
/// bytes of a slot in `out` and in `seen`
const SLOT: u32 = 16;
/// `seen`: how many counts were out of range, the first of them, then the slots
const BAD_AT: u32 = 32;
const BAD_SIZE: u32 = 48;
const BAD_KEPT: u32 = 64;
const SEEN_AT: u32 = BAD_AT + BAD_KEPT * BAD_SIZE;

fn on() -> bool {
    static ON: LazyLock<bool> = LazyLock::new(|| std::env::var("XCL_INDIRECT_GUARD").map_or(true, |v| v.trim() != "0"));
    *ON
}

fn max_groups() -> u32 {
    static MAX: LazyLock<u32> = LazyLock::new(|| std::env::var("XCL_INDIRECT_MAX_GROUPS").ok().and_then(|v| v.trim().parse().ok()).filter(|&n| n > 0).unwrap_or(1 << 22));
    *MAX
}

/// command signatures of one dispatch and nothing else (its counts are the first 12 bytes)
static PLAIN: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

/// A command signature was made (at each creation: one at a freed one's address does not keep
/// what that was).
pub(super) fn note_signature(sig: usize, one_dispatch: bool) {
    let mut g = PLAIN.lock().unwrap();
    let g = g.get_or_insert_with(Default::default);
    if one_dispatch {
        g.insert(sig);
    } else {
        g.remove(&sig);
    }
}

fn program() -> String {
    format!(
        r#"
cbuffer C : register(b0) {{ uint max_groups, slot, id_lo, id_hi, at_lo, at_hi; }};
RWByteAddressBuffer args : register(u0);
RWByteAddressBuffer outp : register(u1);
RWByteAddressBuffer seen : register(u2);
// the same memory as the title's buffer shows it
RWByteAddressBuffer shown : register(u3);
[numthreads(1, 1, 1)]
void main() {{
    uint3 a = args.Load3(0u);
    uint3 b = shown.Load3(0u);
    // (as floats: the product of three counts does not fit 32 bits)
    bool ok = a.x <= max_groups && a.y <= 65535u && a.z <= 65535u && (float)a.x * (float)a.y * (float)a.z <= (float)max_groups;
    outp.Store3(slot * {SLOT}u, ok ? a : uint3(0u, 0u, 0u));
    seen.Store4({SEEN_AT}u + slot * {SLOT}u, uint4(a, 1u));
    bool differs = any(a != b);
    if (differs) {{
        uint m;
        seen.InterlockedAdd(4u, 1u, m);
    }}
    if (!ok) {{
        uint n;
        seen.InterlockedAdd(0u, 1u, n);
        if (n < {BAD_KEPT}u) {{
            seen.Store4({BAD_AT}u + n * {BAD_SIZE}u, uint4(a, slot));
            seen.Store4({BAD_AT}u + n * {BAD_SIZE}u + 16u, uint4(id_lo, id_hi, at_lo, at_hi));
            seen.Store4({BAD_AT}u + n * {BAD_SIZE}u + 32u, uint4(b, differs ? 1u : 0u));
        }}
    }}
}}
"#
    )
}

struct Guard {
    dev: usize,
    rs: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    /// the counts the ExecuteIndirects read: COMMON between its uses
    out: ID3D12Resource,
    out_va: u64,
    /// what the GPU found, on the CPU
    _seen: ID3D12Resource,
    seen_va: u64,
    seen_at: usize,
    next: AtomicU32,
}

// SAFETY: the mapping lives as long as `_seen`; D3D12 objects are free-threaded.
unsafe impl Send for Guard {}
unsafe impl Sync for Guard {}

impl Guard {
    fn word(&self, byte: u32) -> &AtomicU32 {
        unsafe { &*((self.seen_at + byte as usize) as *const AtomicU32) }
    }
}

static GUARD: Mutex<Option<Arc<Guard>>> = Mutex::new(None);

/// The device's guard, made on first use; None: not possible (logged once)
unsafe fn guard_of(list: &ID3D12GraphicsCommandList) -> Option<Arc<Guard>> {
    static FAILED: AtomicU32 = AtomicU32::new(0);
    if FAILED.load(Ordering::Relaxed) != 0 {
        return None;
    }
    let mut dev: Option<ID3D12Device> = None;
    list.GetDevice(&mut dev).ok()?;
    let dev = dev?;
    let mut g = GUARD.lock().unwrap();
    if let Some(g) = g.as_ref().filter(|g| g.dev == dev.as_raw() as usize) {
        return Some(g.clone());
    }
    match make(&dev) {
        Ok(n) => {
            let n = Arc::new(n);
            *g = Some(n.clone());
            Some(n)
        }
        Err(e) => {
            FAILED.store(1, Ordering::Relaxed);
            crate::xlog_warn!("D3D12.X: ExecuteIndirect: the counts are not checked ({e})");
            None
        }
    }
}

unsafe fn make(dev: &ID3D12Device) -> Result<Guard, String> {
    let cs = real::compile(&program(), "cs_5_0");
    if cs.is_empty() {
        return Err("its program does not compile".into());
    }
    let root = |ty, anonymous| D3D12_ROOT_PARAMETER { ParameterType: ty, Anonymous: anonymous, ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL };
    let uav = |k| root(D3D12_ROOT_PARAMETER_TYPE_UAV, D3D12_ROOT_PARAMETER_0 { Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: k, RegisterSpace: 0 } });
    let params = [
        root(D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS, D3D12_ROOT_PARAMETER_0 { Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: 0, RegisterSpace: 0, Num32BitValues: 6 } }),
        uav(0),
        uav(1),
        uav(2),
        uav(3),
    ];
    let desc = D3D12_ROOT_SIGNATURE_DESC { NumParameters: params.len() as u32, pParameters: params.as_ptr(), ..Default::default() };
    let mut blob: Option<ID3DBlob> = None;
    D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, None).map_err(|e| format!("root signature: {e}"))?;
    let blob = blob.ok_or("root signature")?;
    let rs: ID3D12RootSignature = dev.CreateRootSignature(0, std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize())).map_err(|e| format!("CreateRootSignature: {e}"))?;
    let code = D3D12_SHADER_BYTECODE { pShaderBytecode: cs.as_ptr() as *const c_void, BytecodeLength: cs.len() };
    let pso: ID3D12PipelineState = dev
        .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC { pRootSignature: transmute_copy(&rs), CS: code, ..Default::default() })
        .map_err(|e| format!("CreateComputePipelineState: {e}"))?;
    let buffer = |heap: D3D12_HEAP_PROPERTIES, width: u64| -> Result<ID3D12Resource, String> {
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: width,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut r: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut r).map_err(|e| format!("a buffer of {width} bytes: {e}"))?;
        r.ok_or_else(|| "a buffer".into())
    };
    let out = buffer(D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, SLOTS as u64 * SLOT as u64)?;
    let cpu = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_CUSTOM, CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, MemoryPoolPreference: D3D12_MEMORY_POOL_L0, ..Default::default() };
    let bytes = SEEN_AT as u64 + SLOTS as u64 * SLOT as u64;
    let seen = buffer(cpu, bytes)?;
    let mut p = std::ptr::null_mut();
    seen.Map(0, None, Some(&mut p)).map_err(|e| format!("Map: {e}"))?;
    std::ptr::write_bytes(p.cast::<u8>(), 0, bytes as usize);
    Ok(Guard {
        dev: dev.as_raw() as usize,
        out_va: out.GetGPUVirtualAddress(),
        seen_va: seen.GetGPUVirtualAddress(),
        seen_at: p as usize,
        rs,
        pso,
        out,
        _seen: seen,
        next: AtomicU32::new(0),
    })
}

/// An ExecuteIndirect whose counts XCL's program reads: what it runs instead of the title's.
pub(super) struct Checked {
    guard: Arc<Guard>,
    pub(super) slot: u32,
}

/// Record the check of an ExecuteIndirect's counts (`args` + `args_off`, the PC buffer and the
/// offset in it) on `list`; None: not one this checks, it runs as it is. Binds a compute root
/// signature and pipeline: the caller binds the dispatch's after it.
pub(super) unsafe fn check(list: *mut c_void, pso: usize, sig: usize, max: u32, args: usize, args_off: u64, count_buf: usize) -> Option<Checked> {
    if !on() || max != 1 || count_buf != 0 || !PLAIN.lock().unwrap().as_ref().is_some_and(|s| s.contains(&sig)) {
        return None;
    }
    if !super::list_x::tracked_list_type(list).is_some_and(|t| t == D3D12_COMMAND_LIST_TYPE_DIRECT || t == D3D12_COMMAND_LIST_TYPE_COMPUTE) {
        return None;
    }
    let l = ID3D12GraphicsCommandList::from_raw_borrowed(&list)?;
    let res = args as *mut c_void;
    let va = ID3D12Resource::from_raw_borrowed(&res)?.GetGPUVirtualAddress();
    if va == 0 || (va + args_off) % 4 != 0 {
        return None;
    }
    let shown = va + args_off;
    // The memory window is what the title's shaders write: the counts are read there. (The
    // title's buffer is a resource of its own over the same video memory.)
    let delta = real::buffer_delta(args);
    let va = real::console_buffer(args).and_then(|(p, _)| real::uma_va_of(p + args_off.checked_sub(delta)? as usize)).filter(|v| v % 4 == 0).unwrap_or(shown);
    let g = guard_of(l)?;
    let slot = g.next.fetch_add(1, Ordering::Relaxed) % SLOTS;
    g.word(SEEN_AT + slot * SLOT + 12).store(0, Ordering::Release);
    let id = real::pso_shaders(pso).map_or(0, |(a, _)| a);
    // the counts were written as UAVs: they are read after those writes
    let written = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 { UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: ManuallyDrop::new(None) }) },
    };
    l.ResourceBarrier(&[written]);
    l.SetComputeRootSignature(&g.rs);
    l.SetPipelineState(&g.pso);
    let constants = [max_groups(), slot, id as u32, (id >> 32) as u32, va as u32, (va >> 32) as u32];
    l.SetComputeRoot32BitConstants(0, constants.len() as u32, constants.as_ptr() as *const c_void, 0);
    l.SetComputeRootUnorderedAccessView(1, va);
    l.SetComputeRootUnorderedAccessView(2, g.out_va);
    l.SetComputeRootUnorderedAccessView(3, g.seen_va);
    l.SetComputeRootUnorderedAccessView(4, shown);
    l.Dispatch(1, 1, 1);
    report(&g);
    Some(Checked { guard: g, slot })
}

impl Checked {
    /// `op(buffer, offset)`: the ExecuteIndirect with the checked counts.
    pub(super) unsafe fn run(&self, list: *mut c_void, op: impl FnOnce(usize, u64)) {
        let out = self.guard.out.as_raw() as usize;
        // (the driver reads the counts after the program's writes with this transition only,
        // `bind::fresh_indirect_arguments`)
        super::list_x::transition(list, out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
        op(out, self.slot as u64 * SLOT as u64);
        super::list_x::transition(list, out, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COMMON);
    }
}

/// The counts slot `slot`'s ExecuteIndirect was asked to run, once the GPU has read them.
pub(super) fn counts(slot: u32) -> Option<[u32; 3]> {
    let g = GUARD.lock().unwrap().clone()?;
    let at = SEEN_AT + slot * SLOT;
    (g.word(at + 12).load(Ordering::Acquire) != 0).then(|| std::array::from_fn(|k| g.word(at + 4 * k as u32).load(Ordering::Relaxed)))
}

/// Counts out of range so far.
fn bad_so_far(g: &Guard) -> u32 {
    g.word(0).load(Ordering::Acquire)
}

/// Log the counts found out of range since the last time (and how often the window and the
/// title's buffer held different ones).
fn report(g: &Guard) {
    static DIFFERED: AtomicU64 = AtomicU64::new(0);
    let d = g.word(4).load(Ordering::Acquire) as u64;
    if d > DIFFERED.load(Ordering::Relaxed) && (d < 4 || d.is_power_of_two()) && DIFFERED.swap(d, Ordering::Relaxed) < d {
        crate::xlog_warn!("D3D12.X: ExecuteIndirect: {d} time(s) the title's buffer did not show the counts the memory window holds");
    }
    static REPORTED: AtomicU64 = AtomicU64::new(0);
    let n = bad_so_far(g) as u64;
    let last = REPORTED.load(Ordering::Relaxed);
    if n <= last || REPORTED.compare_exchange(last, n, Ordering::Relaxed, Ordering::Relaxed).is_err() {
        return;
    }
    for k in last..n.min(BAD_KEPT as u64) {
        let at = BAD_AT + k as u32 * BAD_SIZE;
        let w: [u32; 12] = std::array::from_fn(|i| g.word(at + 4 * i as u32).load(Ordering::Relaxed));
        let id = w[4] as u64 | (w[5] as u64) << 32;
        let va = w[6] as u64 | (w[7] as u64) << 32;
        crate::xlog_warn!(
            "D3D12.X: ExecuteIndirect of CS {id:016x}: {:#x} x {:#x} x {:#x} groups at GPU {va:#x}{} – out of range, not run ({n} so far)",
            w[0],
            w[1],
            w[2],
            if w[11] != 0 { format!(" (the title's buffer shows {:#x} x {:#x} x {:#x})", w[8], w[9], w[10]) } else { String::new() }
        );
    }
    if last < BAD_KEPT as u64 || n.ilog2() == last.max(1).ilog2() {
        return;
    }
    crate::xlog_warn!("D3D12.X: ExecuteIndirect: {n} group counts out of range so far, not run");
}

#[cfg(test)]
mod tests {
    use super::super::list_x;
    use super::*;

    /// Garbage counts run 0 groups and are kept for the log; counts in range run as they are.
    #[test]
    fn an_execute_indirect_with_garbage_counts_runs_no_groups() {
        use crate::gpu::{fence, recomp::runner::skip_test};
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;

        // the counts the memory holds, and the groups that may run
        let cases: [([u32; 3], u32); 6] = [
            ([7, 1, 1], 7),
            ([3, 4, 5], 60),
            ([0, 1, 1], 0),
            // the bits of 1.0f, a count below 2^32 that is none, and a product too large
            ([0x3F80_0000, 1, 1], 0),
            ([1, 0xFFFF_FFFF, 1], 0),
            ([60_000, 60_000, 2], 0),
        ];
        let src = r#"
RWByteAddressBuffer c : register(u0);
[numthreads(1, 1, 1)]
void main() { c.InterlockedAdd(0, 1u); }
"#;
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                skip_test("no GPU");
                return;
            }
            let dev = dev.unwrap();
            let cs = real::compile(src, "cs_5_0");
            assert!(!cs.is_empty());
            let param = D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                Anonymous: D3D12_ROOT_PARAMETER_0 { Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 0, RegisterSpace: 0 } },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            };
            let desc = D3D12_ROOT_SIGNATURE_DESC { NumParameters: 1, pParameters: &param, ..Default::default() };
            let mut blob: Option<ID3DBlob> = None;
            D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, None).unwrap();
            let blob = blob.unwrap();
            let rs: ID3D12RootSignature = dev.CreateRootSignature(0, std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize())).unwrap();
            let code = D3D12_SHADER_BYTECODE { pShaderBytecode: cs.as_ptr() as _, BytecodeLength: cs.len() };
            let pso: ID3D12PipelineState = dev.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC { pRootSignature: transmute_copy(&rs), CS: code, ..Default::default() }).unwrap();
            let arg = D3D12_INDIRECT_ARGUMENT_DESC { Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH, ..Default::default() };
            let mut sig: Option<ID3D12CommandSignature> = None;
            dev.CreateCommandSignature(&D3D12_COMMAND_SIGNATURE_DESC { ByteStride: 12, NumArgumentDescs: 1, pArgumentDescs: &arg, NodeMask: 0 }, None, &mut sig).unwrap();
            let sig = sig.unwrap();
            note_signature(sig.as_raw() as usize, true);
            let ty = D3D12_COMMAND_LIST_TYPE_COMPUTE;
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: ty, ..Default::default() }).unwrap();
            let done: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let buffer = |width: u64| {
                let cpu = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_CUSTOM, CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, MemoryPoolPreference: D3D12_MEMORY_POOL_L0, ..Default::default() };
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: width,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    ..Default::default()
                };
                let mut r: Option<ID3D12Resource> = None;
                dev.CreateCommittedResource(&cpu, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut r).unwrap();
                let r = r.unwrap();
                let mut p = std::ptr::null_mut();
                r.Map(0, None, Some(&mut p)).unwrap();
                (r, p as *mut u32)
            };
            let ((args, args_at), (groups, groups_at)) = (buffer(256), buffer(256));
            for (k, (want, runs)) in cases.iter().enumerate() {
                // (at an offset: the title's arguments are anywhere in its buffer)
                let off = 4 * k as u64;
                for (i, w) in want.iter().enumerate() {
                    *args_at.add(k + i) = *w;
                }
                *groups_at = 0;
                let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).unwrap();
                let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, ty, &alloc, &pso).unwrap();
                let raw = list.as_raw();
                real::adopt(raw, real::Kind::CommandList);
                list_x::on_create(raw, alloc.as_raw() as usize, 0, ty);
                type Ei = unsafe extern "system" fn(*mut c_void, usize, u32, usize, u64, usize, u64);
                type Close = unsafe extern "system" fn(*mut c_void) -> windows::core::HRESULT;
                let ei: Ei = std::mem::transmute(real::orig_fn(raw, 59));
                let close: Close = std::mem::transmute(real::orig_fn(raw, 9));
                let before = GUARD.lock().unwrap().as_ref().map_or(0, |g| bad_so_far(g));
                let checked = check(raw, 0, sig.as_raw() as usize, 1, args.as_raw() as usize, off, 0).expect("checked");
                list.SetComputeRootSignature(&rs);
                list.SetPipelineState(&pso);
                list.SetComputeRootUnorderedAccessView(0, groups.GetGPUVirtualAddress());
                checked.run(raw, |b, o| ei(raw, sig.as_raw() as usize, 1, b, o, 0, 0));
                close(raw).unwrap();
                queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
                let v = done.GetCompletedValue() + 1;
                queue.Signal(&done, v).unwrap();
                fence::wait(&done, v, &dev, None, 10_000).unwrap();
                assert_eq!(std::ptr::read_volatile(groups_at), *runs, "{want:x?}: groups run");
                assert_eq!(counts(checked.slot), Some(*want), "{want:x?}: what the CPU sees");
                let bad = bad_so_far(&checked.guard) - before;
                assert_eq!(bad, (*runs == 0 && want[0] != 0) as u32, "{want:x?}: counted as out of range");
            }
            // a signature with more than a dispatch, a count buffer or several commands: not checked
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, ty, &alloc, &pso).unwrap();
            let raw = list.as_raw();
            real::adopt(raw, real::Kind::CommandList);
            list_x::on_create(raw, alloc.as_raw() as usize, 0, ty);
            let s = sig.as_raw() as usize;
            let a = args.as_raw() as usize;
            assert!(check(raw, 0, s, 2, a, 0, 0).is_none());
            assert!(check(raw, 0, s, 1, a, 0, a).is_none());
            note_signature(s, false);
            assert!(check(raw, 0, s, 1, a, 0, 0).is_none());
        }
    }
}
