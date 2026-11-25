//! The title's GPU view of its own memory, as the GPU is told about it.
//!
//! A retail title reserves two ranges for the same physical pages: one the CPU reads and writes,
//! and a GPU view the CPU cannot touch (`kernelx::mirrors`). It gives the GPU only addresses in
//! the GPU view: buffer views made by address, root views, vertex and index buffers, the
//! operands of memory operations. With the 16-window layout that view lies in view windows
//! (`Uma::is_view`), whose 64 KiB tiles get video memory, filled from the CPU's range, when an
//! address in them is first used (`sparse::map_mirror`).
//!
//! Every path that takes such an address from the title calls `want` with what it knows of the
//! range. The counters say which paths a title uses and how much of the view each asks for.

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};

/// The paths a GPU-view address reaches the GPU by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Path {
    /// a buffer view made by address (device #70 / #71)
    ViewAt = 0,
    /// a constant buffer view
    Cbv,
    /// a view of a buffer resource placed in the view
    BufferView,
    /// a buffer resource placed in the view
    PlacedBuffer,
    /// a root CBV / SRV / UAV (its size is not known)
    RootView,
    /// a vertex or index buffer
    VertexIndex,
    /// a memory operation of the list (copy, fill, write, timestamps, predication)
    MemoryOp,
    /// a wait on a value in memory
    Wait,
    /// a file read into the view
    FileRead,
    /// a Map of a resource in the view
    MapRead,
    /// a GPU copy or query resolve into a buffer in the view
    CopyInto,
    /// a root view of a two-pass mesh stage
    MeshStage,
    /// an acceleration-structure view (asked for from its root on: its size is not known)
    Bvh,
}

const PATHS: usize = 13;
const NAMES: [&str; PATHS] = [
    "views by address",
    "constant buffer views",
    "buffer views",
    "placed buffers",
    "root views",
    "vertex / index buffers",
    "memory operations",
    "waits",
    "file reads",
    "maps",
    "copies into it",
    "mesh stage views",
    "acceleration-structure views",
];

#[allow(clippy::declare_interior_mutable_const)]
const ZERO: AtomicU64 = AtomicU64::new(0);
static CALLS: [AtomicU64; PATHS] = [ZERO; PATHS];
static BYTES: [AtomicU64; PATHS] = [ZERO; PATHS];

/// Ranges a thread remembers having asked for (`asked_lately`).
const LATELY: usize = 8;

thread_local! {
    /// The ranges this thread asked for last, the generation of the GPU views they were asked
    /// in (`kernelx::gpu_view_map_gen`) and where the next one goes. A title binds the same
    /// vertex buffers and root views for draw after draw, on the thread that records the list;
    /// asking again walks every tile of the range (`sparse::map_mirror`: a shared set).
    static ASKED: Cell<([(u64, u64); LATELY], u64, usize)> = const { Cell::new(([(0, 0); LATELY], 0, 0)) };
}

/// Whether this thread asked for exactly `[va, va + len)` among its last `LATELY` ranges while
/// the GPU views were as they are now (`generation`); it is remembered if not. A view that went
/// or came since (its tiles are forgotten with it) empties the memory.
fn asked_lately(va: u64, len: u64, generation: u64) -> bool {
    ASKED.with(|a| {
        let (mut ranges, asked_in, mut next) = a.get();
        if asked_in != generation {
            (ranges, next) = ([(0, 0); LATELY], 0);
        } else if ranges.contains(&(va, len)) {
            return true;
        }
        ranges[next] = (va, len);
        a.set((ranges, generation, (next + 1) % LATELY));
        false
    })
}

/// Whether `va` lies in a view window (false without the 16-window layout).
pub fn is_view(va: u64) -> bool {
    crate::gpu::uma::get().is_some_and(|u| u.is_view(va as usize))
}

/// The title gave `[va, va + len)` to the GPU on `path`: when it lies in a view window its
/// tiles are asked for (mapped and filled from the CPU's range at the next submit). True when
/// it was a view address.
pub fn want(path: Path, va: u64, len: u64) -> bool {
    if va == 0 || len == 0 || !is_view(va) {
        return false;
    }
    CALLS[path as usize].fetch_add(1, Ordering::Relaxed);
    if !asked_lately(va, len, crate::hle::kernelx::gpu_view_map_gen()) {
        BYTES[path as usize].fetch_add(len, Ordering::Relaxed);
        super::sparse::map_mirror(va as usize, len as usize);
    }
    true
}

/// The CPU address behind `va` (itself unless `va` lies in a GPU view of the title's memory,
/// in a view window or not: `sparse::cpu_of` asks `kernelx::mirrors`, a lock). Callers on a hot
/// path ask `is_view` first.
pub fn cpu(va: u64) -> u64 {
    super::sparse::cpu_of(va as usize) as u64
}

/// (calls, bytes asked for) of each path so far. The bytes of a range a thread asks for again
/// and again count once (`asked_lately`).
pub fn totals() -> [(u64, u64); PATHS] {
    std::array::from_fn(|i| (CALLS[i].load(Ordering::Relaxed), BYTES[i].load(Ordering::Relaxed)))
}

/// What was asked for between two `totals`: the paths with calls, for the periodic log line.
pub fn line(now: &[(u64, u64); PATHS], before: &[(u64, u64); PATHS]) -> Option<String> {
    let parts: Vec<String> = (0..PATHS)
        .filter(|&i| now[i].0 > before[i].0)
        .map(|i| format!("{} {} ({} KB new)", now[i].0 - before[i].0, NAMES[i], (now[i].1 - before[i].1) >> 10))
        .collect();
    (!parts.is_empty()).then(|| parts.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The line names only the paths used since the totals before, with their new bytes.
    #[test]
    fn the_line_names_the_paths_used_since() {
        let mut before = [(0u64, 0u64); PATHS];
        let mut now = before;
        assert_eq!(line(&now, &before), None, "nothing asked for: no line");
        now[Path::ViewAt as usize] = (3, 3 << 20);
        now[Path::RootView as usize] = (1, 65536);
        assert_eq!(line(&now, &before).as_deref(), Some("3 views by address (3072 KB new), 1 root views (64 KB new)"));
        before = now;
        now[Path::RootView as usize] = (2, 65536);
        assert_eq!(line(&now, &before).as_deref(), Some("1 root views (0 KB new)"), "a range asked for again adds a call, no bytes");
        assert_eq!(NAMES[Path::Bvh as usize], "acceleration-structure views");
    }

    /// An address outside a view window (every address without the 16-window layout, and null)
    /// asks for nothing and is not counted.
    #[test]
    fn an_address_outside_the_views_asks_for_nothing() {
        let before = totals();
        assert!(!want(Path::Cbv, 0, 256));
        assert!(!want(Path::Cbv, 0x1000, 0));
        assert_eq!(totals(), before);
    }

    /// A thread that asks for the same range again (a vertex buffer bound for every draw) does
    /// not walk its tiles again; another range, a ninth range that pushes the first out, and
    /// GPU views that changed since make it ask again.
    #[test]
    fn a_range_asked_for_lately_is_not_asked_for_again() {
        std::thread::spawn(|| {
            assert!(!asked_lately(0x14_0000_1000, 0x2000, 7), "the first time");
            assert!(asked_lately(0x14_0000_1000, 0x2000, 7), "again: remembered");
            assert!(!asked_lately(0x14_0000_1000, 0x4000, 7), "another length is another range");
            for k in 0..LATELY as u64 - 1 {
                assert!(!asked_lately(0x15_0000_0000 + (k << 16), 0x1000, 7));
            }
            assert!(!asked_lately(0x14_0000_1000, 0x2000, 7), "pushed out by {LATELY} later ranges");
            assert!(asked_lately(0x14_0000_1000, 0x2000, 7));
            assert!(!asked_lately(0x14_0000_1000, 0x2000, 8), "the views changed: asked again");
            assert!(asked_lately(0x14_0000_1000, 0x2000, 8));
        })
        .join()
        .unwrap();
    }
}
