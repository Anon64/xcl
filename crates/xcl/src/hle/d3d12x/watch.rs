//! What XCL does when the graphics driver stops answering, and what it was doing when that began.
//!
//! A retail title's world ended twice in a driver error whose cause the log did not hold: the
//! process was killed from outside seconds after the driver's first event, before the device was
//! reported removed, and the second time it then hung half-terminated in the driver for five
//! minutes with the desktop behind it. Three things, all cheap enough to be always on:
//!
//! - a ring of the last draws and dispatches (pipeline, kind, size, frame), written at each one
//!   (an index and two stores) and read only for a report: which programs the GPU had been given
//!   when it stopped;
//! - a thread (`xcl-gpu-watch`) that looks for a removed device ten times a second (a title
//!   whose threads all wait submits nothing and would never notice), and times the calls into
//!   the driver that return at once on a working system: an ExecuteCommandLists or a Present that
//!   has not returned after `XCL_GPU_STALL_MS` (2 000) is reported with the ring, and the title's
//!   other submits wait until it returns – work piled on a driver that has stopped does not help
//!   it recover. A call that took a quarter of a second and returned is logged too (video
//!   memory being paged shows there first);
//! - `XCL_STOP_FILE=<path>`: when that file appears the title ends itself in order
//!   (`real::end_title`: its threads suspended, its fences signalled, the queues drained) – a
//!   process killed from outside with GPU work in flight can hang in the driver. A file that
//!   begins with `fault` (the driver reported an error) first gives the driver its time-out to
//!   say whether the device is removed, with nothing more submitted meanwhile, so that the
//!   reason and DRED's report are written.
//!
//! `XCL_GPU_WATCH=0`: no thread (the ring is still written).

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------------------
// the ring
// ---------------------------------------------------------------------------------------

/// draws and dispatches kept (about two frames of a retail title's world)
const RING: usize = 1 << 14;

static AT: AtomicUsize = AtomicUsize::new(0);
static RING_PSO: [AtomicUsize; RING] = [const { AtomicUsize::new(0) }; RING];
/// frame (28 bits) << 36 | work (32 bits) << 4 | kind
static RING_INFO: [AtomicU64; RING] = [const { AtomicU64::new(0) }; RING];

/// kinds of work in the ring (`gpu_profile::Kind` + 1, then the two-pass mesh draws)
pub(super) const MESH: u8 = 6;
pub(super) const INDIRECT_MESH: u8 = 7;

fn kind_name(kind: u8) -> &'static str {
    match kind {
        1 => "Dispatch",
        2 => "ExecuteIndirect (dispatches)",
        3 => "draw",
        4 => "ExecuteIndirect (draws)",
        5 => "ExecuteIndirect with a PC pipeline",
        MESH => "DispatchMesh in two passes",
        INDIRECT_MESH => "ExecuteIndirect (DispatchMesh) in two passes",
        _ => "?",
    }
}

fn dispatches(kind: u8) -> bool {
    kind == 1 || kind == 2
}

/// the product of a size's parts that are not 0 (a dispatch's groups, a draw's vertices x
/// instances, an ExecuteIndirect's largest count), at most `u32::MAX`
fn work_of(size: [u32; 3]) -> u32 {
    size.iter().filter(|&&d| d != 0).fold(1u64, |a, &d| a.saturating_mul(d as u64)).min(u32::MAX as u64) as u32
}

fn pack(kind: u8, work: u32, frame: u64) -> u64 {
    (frame & 0x0FFF_FFFF) << 36 | (work as u64) << 4 | (kind & 15) as u64
}

fn unpack(info: u64) -> (u8, u32, u32) {
    ((info & 15) as u8, (info >> 4) as u32, (info >> 36) as u32)
}

/// A draw or dispatch with pipeline `pso` (the title's handle) goes onto a list.
pub(super) fn note(pso: usize, kind: u8, size: [u32; 3]) {
    let i = AT.fetch_add(1, Ordering::Relaxed) & (RING - 1);
    RING_PSO[i].store(pso, Ordering::Relaxed);
    RING_INFO[i].store(pack(kind, work_of(size), super::present::activity().0), Ordering::Relaxed);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Entry {
    pso: usize,
    kind: u8,
    work: u32,
    frame: u32,
}

/// the ring, oldest first (entries never written left out)
fn snapshot() -> Vec<Entry> {
    let at = AT.load(Ordering::Relaxed);
    let n = at.min(RING);
    (at - n..at)
        .map(|k| {
            let i = k & (RING - 1);
            let (kind, work, frame) = unpack(RING_INFO[i].load(Ordering::Relaxed));
            Entry { pso: RING_PSO[i].load(Ordering::Relaxed), kind, work, frame }
        })
        .filter(|e| e.kind != 0)
        .collect()
}

/// The lines of a report of `entries` (oldest first): the dispatching pipelines by the groups
/// they were given, the drawing ones by their calls, and the last ones in order. `name`: what is
/// known of a pipeline.
fn summarise(entries: &[Entry], name: impl Fn(usize) -> String) -> Vec<String> {
    if entries.is_empty() {
        return vec!["in flight: no draw or dispatch recorded yet".into()];
    }
    // (pipeline, kind) → calls, work in all, largest, last frame
    let mut by: std::collections::HashMap<(usize, u8), (u64, u64, u32, u32)> = std::collections::HashMap::new();
    for e in entries {
        let s = by.entry((e.pso, e.kind)).or_default();
        s.0 += 1;
        s.1 += e.work as u64;
        s.2 = s.2.max(e.work);
        s.3 = e.frame;
    }
    let (first, last) = (entries[0].frame, entries[entries.len() - 1].frame);
    let mut lines = vec![format!("in flight: the last {} draws and dispatches (frames {first} to {last}), {} pipelines", entries.len(), by.len())];
    let mut all: Vec<((usize, u8), (u64, u64, u32, u32))> = by.into_iter().collect();
    // dispatches by the groups given (a program that runs long runs on every one of them),
    // then draws by calls; ties by address, for a stable order
    all.sort_by(|a, b| {
        dispatches(b.0 .1)
            .cmp(&dispatches(a.0 .1))
            .then(if dispatches(a.0 .1) { b.1 .1.cmp(&a.1 .1) } else { b.1 .0.cmp(&a.1 .0) })
            .then(a.0.cmp(&b.0))
    });
    let (mut shown_dispatch, mut shown_draw) = (0, 0);
    for ((pso, kind), (calls, work, largest, frame)) in &all {
        let shown = if dispatches(*kind) { &mut shown_dispatch } else { &mut shown_draw };
        *shown += 1;
        if *shown > if dispatches(*kind) { 24 } else { 12 } {
            continue;
        }
        lines.push(format!("in flight:   {} {pso:#x} ({}): {calls} calls, {work} in all, largest {largest}, last in frame {frame}", kind_name(*kind), name(*pso)));
    }
    let tail: Vec<String> = entries[entries.len().saturating_sub(24)..].iter().map(|e| format!("{} {:#x} {}", kind_name(e.kind), e.pso, e.work)).collect();
    lines.push(format!("in flight: the last {} in order: {}", tail.len(), tail.join("; ")));
    lines
}

/// What is known of the pipeline behind the title's handle `pso`: its console shaders.
fn pipeline_name(pso: usize) -> String {
    match super::real::pso_shaders(super::pso_async::resolve(pso)) {
        Some((cs, 0)) => format!("cs {cs:016x}"),
        Some((vs, ps)) => format!("vs {vs:016x} ps {ps:016x}"),
        None => "no console shader known".into(),
    }
}

/// The ring into the log (`why`: what happened), at most once a second.
pub(super) fn report(why: &str) {
    static LAST: AtomicU64 = AtomicU64::new(0);
    let now = now_ms();
    let last = LAST.load(Ordering::Relaxed);
    if last != 0 && now.saturating_sub(last) < 1000 {
        return;
    }
    LAST.store(now, Ordering::Relaxed);
    crate::xlog_error!("in flight ({why}):");
    for line in summarise(&snapshot(), pipeline_name) {
        crate::xlog_error!("{line}");
    }
}

// ---------------------------------------------------------------------------------------
// calls into the driver
// ---------------------------------------------------------------------------------------

/// threads whose calls are timed (a title submits from a handful)
const SLOTS: usize = 64;

/// when the thread of slot k entered the driver (`now_ms`; 0: it is not in it)
static SINCE: [AtomicU64; SLOTS] = [const { AtomicU64::new(0) }; SLOTS];
static WHAT: [AtomicU32; SLOTS] = [const { AtomicU32::new(0) }; SLOTS];
static TID: [AtomicU32; SLOTS] = [const { AtomicU32::new(0) }; SLOTS];
static NEXT_SLOT: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// this thread's slot (`usize::MAX`: none yet; `SLOTS`: none left)
    static SLOT: Cell<usize> = const { Cell::new(usize::MAX) };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Call {
    Execute = 1,
    Present = 2,
}

fn call_name(what: u32) -> &'static str {
    match what {
        1 => "ExecuteCommandLists",
        2 => "Present",
        _ => "a call",
    }
}

/// milliseconds since the first use, never 0
fn now_ms() -> u64 {
    static T0: OnceLock<Instant> = OnceLock::new();
    T0.get_or_init(Instant::now).elapsed().as_millis() as u64 + 1
}

/// a call that returned after this long is logged
const SLOW_MS: u64 = 250;

/// The thread is inside a driver call until this is dropped.
pub(super) struct InDriver {
    slot: usize,
    since: u64,
}

/// The calling thread enters `call` (one the driver returns from at once when it works).
pub(super) fn in_driver(call: Call) -> InDriver {
    let slot = SLOT.with(|s| {
        if s.get() == usize::MAX {
            s.set(NEXT_SLOT.fetch_add(1, Ordering::Relaxed).min(SLOTS));
            if s.get() < SLOTS {
                TID[s.get()].store(unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() }, Ordering::Relaxed);
            }
        }
        s.get()
    });
    // (a call inside a call of the same thread is the outer one's time)
    if slot >= SLOTS || SINCE[slot].load(Ordering::Relaxed) != 0 {
        return InDriver { slot: SLOTS, since: 0 };
    }
    let since = now_ms();
    WHAT[slot].store(call as u32, Ordering::Relaxed);
    SINCE[slot].store(since, Ordering::Release);
    InDriver { slot, since }
}

impl Drop for InDriver {
    fn drop(&mut self) {
        if self.slot >= SLOTS {
            return;
        }
        SINCE[self.slot].store(0, Ordering::Release);
        let took = now_ms().saturating_sub(self.since);
        if took >= SLOW_MS {
            static N: AtomicU64 = AtomicU64::new(0);
            let n = N.fetch_add(1, Ordering::Relaxed) + 1;
            if n <= 16 || n.is_power_of_two() {
                crate::xlog_warn!("D3D12: {} took {took} ms in the driver ({n} such calls so far)", call_name(WHAT[self.slot].load(Ordering::Relaxed)));
            }
        }
    }
}

/// The call that has been in the driver longest: (milliseconds, which, thread).
fn longest_call(now: u64) -> Option<(u64, u32, u32)> {
    (0..SLOTS)
        .filter_map(|k| {
            let since = SINCE[k].load(Ordering::Acquire);
            (since != 0).then(|| (now.saturating_sub(since), WHAT[k].load(Ordering::Relaxed), TID[k].load(Ordering::Relaxed)))
        })
        .max()
}

/// a driver call has not returned for the stall time: the title's submits wait
static STALLED: AtomicBool = AtomicBool::new(false);

/// The title submits: not while a call into the driver stands still (or a stop was asked for).
pub(super) fn wait_while_stalled() {
    if !STALLED.load(Ordering::Relaxed) {
        return;
    }
    static N: AtomicU64 = AtomicU64::new(0);
    if N.fetch_add(1, Ordering::Relaxed) < 8 {
        crate::xlog_warn!("D3D12: a submit waits until the graphics driver answers again");
    }
    while STALLED.load(Ordering::Relaxed) {
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn stall_ms() -> u64 {
    static MS: OnceLock<u64> = OnceLock::new();
    *MS.get_or_init(|| std::env::var("XCL_GPU_STALL_MS").ok().and_then(|v| v.trim().parse().ok()).filter(|&ms| ms >= 100).unwrap_or(2000))
}

/// What the watch does about the longest call (`longest` ms, None: no thread is in the driver),
/// stalled or not before: (stalled now, whether that is news).
fn stall_step(longest: Option<u64>, limit: u64, stalled: bool) -> (bool, bool) {
    let now = longest.is_some_and(|ms| ms >= limit);
    (now, now != stalled)
}

/// What a stop file asks for: `fault…` – the driver reported an error; anything else – end now.
fn asks_for_fault(content: &str) -> bool {
    content.trim_start().to_ascii_lowercase().starts_with("fault")
}

/// The watching thread, once the device is made.
pub(super) fn start() {
    static STARTED: AtomicBool = AtomicBool::new(false);
    if STARTED.swap(true, Ordering::AcqRel) || std::env::var("XCL_GPU_WATCH").is_ok_and(|v| v.trim() == "0") {
        return;
    }
    let stop_file = std::env::var_os("XCL_STOP_FILE").map(std::path::PathBuf::from);
    let _ = std::thread::Builder::new().name("xcl-gpu-watch".into()).spawn(move || {
        let limit = stall_ms();
        let (mut stalled, mut said) = (false, 0u64);
        loop {
            std::thread::sleep(Duration::from_millis(100));
            super::real::stop_if_device_removed();
            let now = now_ms();
            let longest = longest_call(now);
            let (is, news) = stall_step(longest.map(|l| l.0), limit, stalled);
            if news && is {
                let (ms, what, tid) = longest.unwrap_or_default();
                STALLED.store(true, Ordering::Release);
                crate::xlog_error!("D3D12: {} on thread {tid} has not returned for {ms} ms – the graphics driver has stopped answering; the title's other submits wait until it does", call_name(what));
                report("a call into the driver stands still");
                crate::log::flush();
                said = now;
            } else if news {
                STALLED.store(false, Ordering::Release);
                crate::xlog_warn!("D3D12: the graphics driver answers again after {} ms – the title goes on", now.saturating_sub(said) + limit);
            } else if is && now.saturating_sub(said) >= 5000 {
                let (ms, what, tid) = longest.unwrap_or_default();
                crate::xlog_error!("D3D12: {} on thread {tid} still has not returned ({ms} ms)", call_name(what));
                crate::log::flush();
                said = now;
            }
            stalled = is;
            if let Some(content) = stop_file.as_deref().and_then(|p| std::fs::read_to_string(p).ok()) {
                stop_asked(&content);
            }
        }
    });
}

/// The stop file is there: the title ends itself.
fn stop_asked(content: &str) -> ! {
    let fault = asks_for_fault(content);
    crate::xlog_warn!("XCL_STOP_FILE: a stop was asked for from outside ({}) – the title ends itself", if fault { "the graphics driver reported an error" } else { "end" });
    report("a stop was asked for");
    crate::log::flush();
    if fault {
        // nothing more goes to the driver while it decides (its time-out, then the recovery);
        // a removed device is reported – reason, DRED – and ends the process there
        STALLED.store(true, Ordering::Release);
        for _ in 0..120 {
            super::real::stop_if_device_removed();
            std::thread::sleep(Duration::from_millis(100));
        }
        crate::xlog_warn!("XCL_STOP_FILE: the device was not reported removed within 12 s");
    }
    crate::report::write_final(Some("stopped from outside (XCL_STOP_FILE)".into()));
    crate::log::flush();
    super::real::end_title(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_entry_packs_and_unpacks() {
        for (kind, work, frame) in [(1u8, 0u32, 0u64), (7, u32::MAX, 0x0FFF_FFFF), (3, 1234, 5)] {
            assert_eq!(unpack(pack(kind, work, frame)), (kind, work, frame as u32));
        }
        // a frame count past 28 bits wraps, and does not reach the work
        assert_eq!(unpack(pack(2, 7, 0x1_0000_0003)), (2, 7, 3));
        assert_eq!(work_of([4, 5, 6]), 120);
        assert_eq!(work_of([4, 0, 0]), 4, "an ExecuteIndirect's count");
        assert_eq!(work_of([0, 0, 0]), 1);
        assert_eq!(work_of([u32::MAX, u32::MAX, 2]), u32::MAX, "saturates");
    }

    #[test]
    fn the_report_ranks_dispatches_by_groups_and_draws_by_calls() {
        let e = |pso, kind, work, frame| Entry { pso, kind, work, frame };
        let entries = [
            e(0x10, 3, 6, 1),      // draw
            e(0x20, 1, 100, 1),    // small dispatch
            e(0x30, 1, 4_000_000, 2), // a full-screen dispatch
            e(0x10, 3, 6, 2),
            e(0x20, 1, 100, 2),
            e(0x40, MESH, 12, 2),
        ];
        let lines = summarise(&entries, |pso| format!("p{pso:x}"));
        assert_eq!(lines[0], "in flight: the last 6 draws and dispatches (frames 1 to 2), 4 pipelines");
        assert!(lines[1].contains("Dispatch 0x30 (p30): 1 calls, 4000000 in all, largest 4000000, last in frame 2"), "{}", lines[1]);
        assert!(lines[2].contains("Dispatch 0x20 (p20): 2 calls, 200 in all, largest 100"), "{}", lines[2]);
        assert!(lines[3].contains("draw 0x10 (p10): 2 calls"), "{}", lines[3]);
        assert!(lines[4].contains("DispatchMesh in two passes 0x40 (p40): 1 calls"), "{}", lines[4]);
        assert!(lines[5].starts_with("in flight: the last 6 in order: draw 0x10 6; Dispatch 0x20 100; "), "{}", lines[5]);
        assert_eq!(summarise(&[], |_| String::new()), vec!["in flight: no draw or dispatch recorded yet".to_string()]);
    }

    #[test]
    fn the_report_keeps_to_a_few_pipelines_of_each_kind() {
        let entries: Vec<Entry> = (0..100).map(|k| Entry { pso: 0x1000 + k, kind: if k % 2 == 0 { 1 } else { 3 }, work: k as u32, frame: 1 }).collect();
        let lines = summarise(&entries, |_| String::new());
        // the head line, 24 dispatching pipelines, 12 drawing ones, the tail
        assert_eq!(lines.len(), 1 + 24 + 12 + 1);
        assert!(lines[1].contains("Dispatch 0x1062"), "the largest first: {}", lines[1]);
    }

    #[test]
    fn a_stall_is_news_once_and_its_end_is_news_once() {
        assert_eq!(stall_step(None, 2000, false), (false, false));
        assert_eq!(stall_step(Some(1999), 2000, false), (false, false));
        assert_eq!(stall_step(Some(2000), 2000, false), (true, true));
        assert_eq!(stall_step(Some(9000), 2000, true), (true, false));
        assert_eq!(stall_step(Some(3), 2000, true), (false, true), "that call returned, another began");
        assert_eq!(stall_step(None, 2000, true), (false, true));
    }

    #[test]
    fn a_stop_file_says_whether_the_driver_reported_an_error() {
        assert!(asks_for_fault("fault run139 2026-11-25T05:36:37\r\n"));
        assert!(asks_for_fault("  FAULT"));
        assert!(!asks_for_fault("end"));
        assert!(!asks_for_fault(""));
    }

    #[test]
    fn a_thread_in_the_driver_is_seen_until_it_returns() {
        // (its own thread: the slot is the thread's)
        std::thread::spawn(|| {
            let tid = unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() };
            let seen = |now: u64| (0..SLOTS).any(|k| TID[k].load(Ordering::Relaxed) == tid && SINCE[k].load(Ordering::Relaxed) != 0 && now >= SINCE[k].load(Ordering::Relaxed));
            {
                let _call = in_driver(Call::Execute);
                assert!(seen(now_ms()));
                // a call inside it is the outer one's time and does not end it
                drop(in_driver(Call::Present));
                assert!(seen(now_ms()));
                assert!(longest_call(now_ms() + 5000).is_some_and(|(ms, _, _)| ms >= 5000));
            }
            assert!(!seen(now_ms()));
        })
        .join()
        .unwrap();
    }

    #[test]
    fn the_ring_holds_what_was_noted_last() {
        // (other tests of the crate may note work too: only what this one put in is looked for)
        note(0xABCD_0001, MESH, [7, 0, 0]);
        note(0xABCD_0002, 1, [3, 5, 2]);
        let snap = snapshot();
        let a = snap.iter().rposition(|e| e.pso == 0xABCD_0001).expect("noted");
        let b = snap.iter().rposition(|e| e.pso == 0xABCD_0002).expect("noted");
        assert!(a < b, "oldest first");
        assert_eq!((snap[a].kind, snap[a].work), (MESH, 7));
        assert_eq!((snap[b].kind, snap[b].work), (1, 30));
    }
}
