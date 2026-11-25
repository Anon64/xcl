//! Depth buffers that the console also lets shaders write (ALLOW_DEPTH_STENCIL |
//! ALLOW_UNORDERED_ACCESS). The PC refuses the pair: `device_x::pc_resource_desc` keeps the depth
//! buffer and drops ALLOW_UNORDERED_ACCESS, so a UAV of such a buffer is a null view and what a
//! compute program stores through it is lost.
//!
//! This module keeps what XCL knows of these buffers: each object by its address, the objects at
//! one title address and shape together in a `Cell` (on the PC they share texels, `vram`), and the
//! DSVs that name them. Behind `XCL_DEPTH_UAV_TWIN=1`; with the switch off nothing is registered
//! and every hook stops at one atomic load.
//!
//! (After the depth+UAV twins of xcl-dev/xcl#1989 by raylitebull: registering at creation, DSVs and
//! forgetting at the last release; and #1063 by IvyNotFound: the PC state a buffer was created in.)

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use super::sharded::Sharded;

/// The switch's value when `XCL_DEPTH_UAV_TWIN` is not set.
const DEFAULT_ON: bool = false;

/// Whether depth buffers with unordered access are kept (`XCL_DEPTH_UAV_TWIN`).
pub(super) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| on(std::env::var("XCL_DEPTH_UAV_TWIN").ok().as_deref(), DEFAULT_ON))
}

/// "0" off, "1" on, anything else (or nothing) `default`.
fn on(v: Option<&str>, default: bool) -> bool {
    match v.map(str::trim) {
        Some("0") => false,
        Some("1") => true,
        _ => default,
    }
}

/// What makes objects one buffer: on the PC, objects at one title address share texels.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct Key {
    /// `vram::canonical_va` of the title address, or the object's address when it has none
    at: usize,
    /// `at` is a title address
    placed: bool,
    width: u32,
    height: u32,
    array: u16,
    mips: u16,
    /// the PC resource format: 19, 39, 44 or 53
    format: u32,
}

struct Cell {
    key: Key,
    inner: Mutex<Inner>,
}

#[derive(Default)]
struct Inner {
    /// registered objects of this buffer
    members: u32,
}

#[derive(Clone)]
#[allow(dead_code)] // (the state is read once twins are made)
struct Object {
    cell: Arc<Cell>,
    /// the PC state the object was created in
    state: u32,
}

#[derive(Clone)]
#[allow(dead_code)] // (read once DSV binds are followed)
struct Dsv {
    res: usize,
    /// D3D12_DSV_FLAGS of the view: 1 read-only depth, 2 read-only stencil
    flags: u32,
    cell: Weak<Cell>,
}

/// Every Cell by its key: taken at creation and release only.
static CELLS: Mutex<Option<HashMap<Key, Arc<Cell>>>> = Mutex::new(None);
/// Registered objects by their address.
static OBJECTS: Sharded<Object> = Sharded::new();
/// DSV handles that name a registered object.
static DSVS: Sharded<Dsv> = Sharded::new();
/// The gates: entries in OBJECTS and DSVS.
static REGISTERED: AtomicUsize = AtomicUsize::new(0);
static DSV_ENTRIES: AtomicUsize = AtomicUsize::new(0);

fn u32_at(x: &[u8; 0x38], o: usize) -> u32 {
    u32::from_le_bytes(x[o..o + 4].try_into().unwrap())
}

/// The PC object `res` was made for the console description `console_desc` in PC state
/// `pc_state` (the state of the call that succeeded), at the title address `va` (None: memory of
/// its own). Kept when it is a single-sample 2D depth buffer with unordered access.
pub(super) fn note_created(res: *mut c_void, console_desc: &[u8; 0x38], va: Option<usize>, pc_state: u32) {
    if !enabled() || res.is_null() {
        return;
    }
    register(res as usize, console_desc, va.map(super::vram::canonical_va), pc_state);
}

/// `note_created` without the switch, `at` already canonical. True: kept.
fn register(res: usize, x: &[u8; 0x38], at: Option<usize>, pc_state: u32) -> bool {
    let format = u32_at(&super::device_x::pc_resource_desc(x), 0x20);
    if u32_at(x, 0) != D3D12_RESOURCE_DIMENSION_TEXTURE2D.0 as u32 || u32_at(x, 0x30) & 6 != 6 || u32_at(x, 0x24) > 1 || !matches!(format, 19 | 39 | 44 | 53) {
        return false;
    }
    let key = Key {
        at: at.unwrap_or(res),
        placed: at.is_some(),
        width: u64::from_le_bytes(x[0x10..0x18].try_into().unwrap()) as u32,
        height: u32_at(x, 0x18),
        array: u16::from_le_bytes(x[0x1C..0x1E].try_into().unwrap()),
        mips: u16::from_le_bytes(x[0x1E..0x20].try_into().unwrap()),
        format,
    };
    let mut g = CELLS.lock().unwrap();
    let cells = g.get_or_insert_with(HashMap::new);
    let cell = cells.entry(key).or_insert_with(|| Arc::new(Cell { key, inner: Mutex::new(Inner::default()) })).clone();
    cell.inner.lock().unwrap().members += 1;
    match OBJECTS.insert(res, Object { cell, state: pc_state }) {
        None => {
            REGISTERED.fetch_add(1, Ordering::Relaxed);
        }
        // (an object XCL did not see released: its membership goes back)
        Some(old) => leave(cells, &old.cell),
    }
    true
}

/// One object of `cell` is gone; the last one takes the Cell with it.
fn leave(cells: &mut HashMap<Key, Arc<Cell>>, cell: &Arc<Cell>) {
    let mut inner = cell.inner.lock().unwrap();
    inner.members = inner.members.saturating_sub(1);
    if inner.members == 0 && cells.get(&cell.key).is_some_and(|c| Arc::ptr_eq(c, cell)) {
        cells.remove(&cell.key);
    }
}

/// The last reference of `res` goes (`real::res_release`).
pub(super) fn forget(res: usize) {
    if REGISTERED.load(Ordering::Relaxed) == 0 {
        return;
    }
    let Some(o) = OBJECTS.remove(res) else { return };
    REGISTERED.fetch_sub(1, Ordering::Relaxed);
    if let Some(cells) = CELLS.lock().unwrap().as_mut() {
        leave(cells, &o.cell);
    }
}

/// CreateDepthStencilView of `res` at `handle` with D3D12_DSV_FLAGS `flags`.
pub(super) fn note_dsv(handle: usize, res: usize, flags: u32) {
    if REGISTERED.load(Ordering::Relaxed) == 0 && DSV_ENTRIES.load(Ordering::Relaxed) == 0 {
        return;
    }
    match OBJECTS.get(res) {
        Some(o) => {
            if DSVS.insert(handle, Dsv { res, flags, cell: Arc::downgrade(&o.cell) }).is_none() {
                DSV_ENTRIES.fetch_add(1, Ordering::Relaxed);
            }
        }
        None => forget_dsv(handle),
    }
}

fn forget_dsv(handle: usize) {
    if DSV_ENTRIES.load(Ordering::Relaxed) != 0 && DSVS.remove(handle).is_some() {
        DSV_ENTRIES.fetch_sub(1, Ordering::Relaxed);
    }
}

/// The DSV handle increment of `dev`, read once.
unsafe fn dsv_increment(dev: *mut c_void) -> u64 {
    static INC: AtomicU32 = AtomicU32::new(0);
    let mut inc = INC.load(Ordering::Relaxed);
    if inc == 0 {
        let Some(d) = ID3D12Device::from_raw_borrowed(&dev) else { return 0 };
        inc = d.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
        INC.store(inc, Ordering::Relaxed);
    }
    inc as u64
}

/// CopyDescriptorsSimple of `n` DSVs from `src` to `dst`: each destination names what its source
/// names, or nothing.
pub(super) unsafe fn dsvs_copied(dev: *mut c_void, n: u32, dst: u64, src: u64) {
    if DSV_ENTRIES.load(Ordering::Relaxed) == 0 {
        return;
    }
    copy_dsvs(n, dst, src, dsv_increment(dev));
}

/// CopyDescriptors of DSVs: destination and source ranges (start, length; no lengths: 1 each)
/// walked in step.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn dsv_ranges_copied(dev: *mut c_void, nd: u32, dsts: *const u64, dsizes: *const u32, ns: u32, srcs: *const u64, ssizes: *const u32) {
    if DSV_ENTRIES.load(Ordering::Relaxed) == 0 || dsts.is_null() || srcs.is_null() {
        return;
    }
    let ranges = |starts: *const u64, sizes: *const u32, n: u32| -> Vec<(u64, u32)> {
        (0..n as usize).map(|r| (*starts.add(r), if sizes.is_null() { 1 } else { *sizes.add(r) })).collect()
    };
    let inc = dsv_increment(dev);
    for (dst, src, n) in pieces(&ranges(dsts, dsizes, nd), &ranges(srcs, ssizes, ns), inc) {
        copy_dsvs(n, dst, src, inc);
    }
}

/// Destination and source ranges cut where either one ends: (dst, src, n) runs.
fn pieces(dsts: &[(u64, u32)], srcs: &[(u64, u32)], inc: u64) -> Vec<(u64, u64, u32)> {
    let (mut out, mut d, mut s, mut dk, mut sk) = (Vec::new(), 0, 0, 0u32, 0u32);
    while d < dsts.len() && s < srcs.len() {
        let n = (dsts[d].1 - dk).min(srcs[s].1 - sk);
        if n > 0 {
            out.push((dsts[d].0 + dk as u64 * inc, srcs[s].0 + sk as u64 * inc, n));
        }
        dk += n;
        sk += n;
        if dk == dsts[d].1 {
            (d, dk) = (d + 1, 0);
        }
        if sk == srcs[s].1 {
            (s, sk) = (s + 1, 0);
        }
    }
    out
}

fn copy_dsvs(n: u32, dst: u64, src: u64, inc: u64) {
    for k in 0..n as u64 {
        let (to, from) = ((dst + k * inc) as usize, (src + k * inc) as usize);
        match DSVS.get(from) {
            Some(v) => {
                if DSVS.insert(to, v).is_none() {
                    DSV_ENTRIES.fetch_add(1, Ordering::Relaxed);
                }
            }
            None => forget_dsv(to),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A console resource description: 2D, `w`x`h`, one slice and mip, `samples`, `flags`.
    fn desc(dim: u32, w: u64, h: u32, format: u32, samples: u32, flags: u32) -> [u8; 0x38] {
        let mut x = [0u8; 0x38];
        x[0..4].copy_from_slice(&dim.to_le_bytes());
        x[0x10..0x18].copy_from_slice(&w.to_le_bytes());
        x[0x18..0x1C].copy_from_slice(&h.to_le_bytes());
        x[0x1C..0x1E].copy_from_slice(&1u16.to_le_bytes());
        x[0x1E..0x20].copy_from_slice(&1u16.to_le_bytes());
        x[0x20..0x24].copy_from_slice(&format.to_le_bytes());
        x[0x24..0x28].copy_from_slice(&samples.to_le_bytes());
        x[0x30..0x34].copy_from_slice(&flags.to_le_bytes());
        x
    }

    fn cell_of(res: usize) -> Option<Arc<Cell>> {
        OBJECTS.get(res).map(|o| o.cell)
    }

    #[test]
    fn the_switch_reads_0_and_1() {
        assert!(!on(Some("0"), true));
        assert!(on(Some("1"), false));
        assert!(on(Some(" 1\n"), false), "trimmed");
        assert!(on(Some("yes"), true) && !on(Some("yes"), false), "anything else: the default");
        assert!(on(None, true) && !on(None, false));
    }

    #[test]
    fn only_depth_buffers_that_allow_unordered_access_are_kept() {
        let base = 0x7A00_0000_0000usize;
        // console D24S8 (45), D16S8 (118), D32 (40), D16 (55), D32S8 (20) with flags 6
        for (k, f) in [45u32, 118, 40, 55, 20].into_iter().enumerate() {
            let res = base + k * 0x100;
            assert!(register(res, &desc(3, 64, 32, f, 1, 6), Some(res), 0x10), "format {f}");
            assert_eq!(cell_of(res).unwrap().key.format, [44, 44, 39, 53, 19][k]);
            forget(res);
            assert!(cell_of(res).is_none());
        }
        let refused = [
            ("depth only", desc(3, 64, 32, 45, 1, 2)),
            ("unordered access only", desc(3, 64, 32, 41, 1, 4)),
            ("4 samples", desc(3, 64, 32, 45, 4, 6)),
            ("a 3D texture", desc(4, 64, 32, 45, 1, 6)),
            ("a colour format", desc(3, 64, 32, 28, 1, 6)),
        ];
        for (k, (what, x)) in refused.iter().enumerate() {
            let res = base + 0x1_0000 + k * 0x100;
            assert!(!register(res, x, None, 0x10), "{what}");
            assert!(cell_of(res).is_none(), "{what}");
        }
    }

    #[test]
    fn depth_buffers_at_one_title_address_share_a_cell_until_the_last_is_released() {
        let (a, b, c, at) = (0x7B00_0000_1000usize, 0x7B00_0000_2000usize, 0x7B00_0000_3000usize, 0x2_0000_0000usize);
        let x = desc(3, 2560, 1440, 45, 1, 6);
        assert!(register(a, &x, Some(at), 0x10));
        assert!(register(b, &x, Some(at), 0));
        let (ca, cb) = (cell_of(a).unwrap(), cell_of(b).unwrap());
        assert!(Arc::ptr_eq(&ca, &cb), "one address, one shape: one Cell");
        assert_eq!(ca.inner.lock().unwrap().members, 2);
        assert_eq!((OBJECTS.get(a).unwrap().state, OBJECTS.get(b).unwrap().state), (0x10, 0), "each object keeps its own state");
        // another shape at the address, and memory of its own: Cells of their own
        assert!(register(c, &desc(3, 1280, 720, 45, 1, 6), Some(at), 0x10));
        assert!(!Arc::ptr_eq(&ca, &cell_of(c).unwrap()));
        let own = 0x7B00_0000_4000usize;
        assert!(register(own, &x, None, 0x10));
        let co = cell_of(own).unwrap();
        assert!(!co.key.placed && co.key.at == own && !Arc::ptr_eq(&ca, &co));

        let in_cells = |k: &Key| CELLS.lock().unwrap().as_ref().is_some_and(|m| m.contains_key(k));
        forget(a);
        assert!(in_cells(&ca.key) && ca.inner.lock().unwrap().members == 1, "b still holds it");
        forget(b);
        assert!(!in_cells(&ca.key), "the last release takes the Cell");
        // a second registration of one address without a release gives the first one's membership back
        assert!(register(own, &x, None, 0x10));
        assert_eq!(cell_of(own).unwrap().inner.lock().unwrap().members, 1);
        forget(own);
        forget(c);
        assert!(!in_cells(&co.key) && cell_of(c).is_none());
    }

    #[test]
    fn a_copied_dsv_names_what_its_source_names() {
        let (res, inc) = (0x7C00_0000_1000usize, 0x20u64);
        let (heap, other) = (0x5_0000_0000u64, 0x6_0000_0000u64);
        assert!(register(res, &desc(3, 64, 64, 45, 1, 6), None, 0x10));
        note_dsv(heap as usize, res, 2);
        assert!(DSVS.get(heap as usize).is_some_and(|d| d.res == res && d.flags == 2));
        // a DSV of a resource that is not kept: not known (and an older entry there goes)
        note_dsv(heap as usize + 0x20, 0x7C00_0000_9000, 0);
        assert!(DSVS.get(heap as usize + 0x20).is_none());

        // two descriptors from `heap`: the first names `res`, the second nothing
        note_dsv(other as usize + 0x20, res, 0);
        copy_dsvs(2, other, heap, inc);
        assert!(DSVS.get(other as usize).is_some_and(|d| d.res == res && d.flags == 2 && d.cell.upgrade().is_some()));
        assert!(DSVS.get(other as usize + 0x20).is_none(), "a copy from an unknown handle forgets the destination");

        // ranges are cut where either side ends
        assert_eq!(pieces(&[(0x100, 3)], &[(0x500, 1), (0x900, 2)], inc), vec![(0x100, 0x500, 1), (0x120, 0x900, 2)]);
        assert_eq!(pieces(&[(0x100, 1), (0x300, 2)], &[(0x500, 2)], inc), vec![(0x100, 0x500, 1), (0x300, 0x520, 1)]);
        assert_eq!(pieces(&[(0x100, 0), (0x300, 1)], &[(0x500, 1)], inc), vec![(0x300, 0x500, 1)]);

        forget(res);
        assert!(DSVS.get(other as usize).is_some_and(|d| d.cell.upgrade().is_none()), "a DSV of a buffer that went names no Cell");
        forget_dsv(heap as usize);
        forget_dsv(other as usize);
    }
}
