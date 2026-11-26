//! The live placements of the title's textures on video memory shared by address (`vram`): each
//! PC texture, the title pages it has (`vram::owned_pages`: tile i on the page at `start + i *
//! 64 KiB`), its kind and when it came. Two textures on the same pages share video memory as on
//! the console – what one is written with, the other shows.
//!
//! `XCL_PLACE_CHECK=1` logs placements over the pages of a live texture (the first 32 in full)
//! and counts them every 2 s, by how their pages lie and by kind.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DESC;

/// What the GPU does with a placed texture.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Kind {
    /// a render target, depth buffer or UAV texture (`vram::gpu_owned`)
    Target,
    /// sampled: kept in step with the title's memory (`title_textures`)
    Sampled,
}

impl Kind {
    fn word(self) -> &'static str {
        match self {
            Kind::Target => "target",
            Kind::Sampled => "sampled",
        }
    }
}

/// Where a placement's video memory is: the title's pages of a device, or an arena (`vram`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Backing {
    Pages(usize),
    Arena(u64),
}

/// The PC description of a placed texture, for the log.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(super) struct Desc {
    pub width: u64,
    pub height: u32,
    pub depth: u16,
    pub mips: u16,
    pub format: i32,
    pub flags: i32,
}

impl Desc {
    pub(super) fn of(d: &D3D12_RESOURCE_DESC) -> Desc {
        Desc { width: d.Width, height: d.Height, depth: d.DepthOrArraySize, mips: d.MipLevels, format: d.Format.0, flags: d.Flags.0 }
    }
}

impl std::fmt::Display for Desc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x{}x{} format {} flags {:#x} {} mips", self.width, self.height, self.depth, self.format, self.flags, self.mips)
    }
}

#[derive(Clone, Debug)]
struct Placement {
    /// the title address it was placed at (`vram::canonical_va`)
    va: usize,
    /// its title pages `[start, end)` (none of its own: `start == end`)
    start: usize,
    end: usize,
    backing: Backing,
    kind: Kind,
    seq: u64,
    at: Instant,
    desc: Desc,
    /// the title released it: it goes on the PC once the GPU is past it (`late_release`)
    released: bool,
}

/// How a newer placement's pages lie to an older one's.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Overlap {
    /// the same pages: one memory under two descriptions
    Same,
    /// all its pages are the older one's
    Inside,
    /// all the older one's pages are its
    Around,
    /// some of each one's pages are the other's
    Partial,
}

impl Overlap {
    fn words(self) -> &'static str {
        match self {
            Overlap::Same => "on the same pages as",
            Overlap::Inside => "inside",
            Overlap::Around => "around",
            Overlap::Partial => "partly over",
        }
    }
}

/// How placement `new`'s pages lie to `old`'s (None: they share no video memory). Placements at
/// one address share their tiles past their own pages too (`vram::Pages::extra_pages`).
fn overlap(new: &Placement, old: &Placement) -> Option<Overlap> {
    if new.backing != old.backing {
        return None;
    }
    let shared = new.start < old.end && old.start < new.end;
    if !shared && new.va != old.va {
        return None;
    }
    Some(if (new.start, new.end) == (old.start, old.end) {
        Overlap::Same
    } else if old.start <= new.start && new.end <= old.end {
        Overlap::Inside
    } else if new.start <= old.start && old.end <= new.end {
        Overlap::Around
    } else {
        Overlap::Partial
    })
}

/// A placement over the pages of an older live one.
#[derive(Clone, Debug)]
pub(super) struct Case {
    pub old: usize,
    pub overlap: Overlap,
    /// (the newer one's, the older one's)
    pub kinds: (Kind, Kind),
    /// placements between them
    pub ago: u64,
    pub ms: u64,
    /// the older one was released by the title and is not gone on the PC yet
    pub released: bool,
    pub old_desc: Desc,
    /// the older one's (address, first page, end of its pages)
    pub old_range: (usize, usize, usize),
    pub old_seq: u64,
}

/// the placements are found by the 2 MiB blocks of the title's addresses they cover
const BLOCK_SHIFT: usize = 21;

#[derive(Default)]
struct Registry {
    live: HashMap<usize, Placement>,
    /// block → the placements over it
    blocks: BTreeMap<usize, Vec<usize>>,
    seq: u64,
}

impl Registry {
    /// The blocks of `p`'s pages and of its address.
    fn blocks_of(p: &Placement) -> std::ops::RangeInclusive<usize> {
        let last = if p.end > p.start { p.end - 1 } else { p.start };
        (p.va.min(p.start) >> BLOCK_SHIFT)..=(last.max(p.va) >> BLOCK_SHIFT)
    }

    /// The live placements but `p` itself that share video memory with `p`.
    fn over(&self, res: usize, p: &Placement) -> Vec<(usize, Overlap)> {
        let mut out: Vec<(usize, Overlap)> = Vec::new();
        for b in Self::blocks_of(p) {
            for &o in self.blocks.get(&b).into_iter().flatten() {
                if o == res || out.iter().any(|&(k, _)| k == o) {
                    continue;
                }
                if let Some(ov) = self.live.get(&o).and_then(|old| overlap(p, old)) {
                    out.push((o, ov));
                }
            }
        }
        out
    }

    /// Texture `res` placed; with `look`, the older live ones it shares video memory with (oldest
    /// first). Returns its sequence number too.
    #[allow(clippy::too_many_arguments)]
    fn place(&mut self, res: usize, va: usize, (start, end): (usize, usize), backing: Backing, kind: Kind, desc: Desc, now: Instant, look: bool) -> (u64, Vec<Case>) {
        // (an object's address again: the one before went without a word)
        self.remove(res);
        self.seq += 1;
        let p = Placement { va, start, end: end.max(start), backing, kind, seq: self.seq, at: now, desc, released: false };
        let mut cases: Vec<Case> = if look {
            self.over(res, &p)
                .into_iter()
                .map(|(o, overlap)| {
                    let old = &self.live[&o];
                    Case {
                        old: o,
                        overlap,
                        kinds: (kind, old.kind),
                        ago: p.seq - old.seq,
                        ms: now.saturating_duration_since(old.at).as_millis() as u64,
                        released: old.released,
                        old_desc: old.desc,
                        old_range: (old.va, old.start, old.end),
                        old_seq: old.seq,
                    }
                })
                .collect()
        } else {
            Vec::new()
        };
        cases.sort_by_key(|c| c.old_seq);
        for b in Self::blocks_of(&p) {
            self.blocks.entry(b).or_default().push(res);
        }
        self.live.insert(res, p);
        (self.seq, cases)
    }

    /// The title released `res`; it stays until it is gone on the PC (`remove`).
    fn release(&mut self, res: usize) -> bool {
        self.live.get_mut(&res).map(|p| p.released = true).is_some()
    }

    /// `res` is gone on the PC.
    fn remove(&mut self, res: usize) -> bool {
        let Some(p) = self.live.remove(&res) else { return false };
        for b in Self::blocks_of(&p) {
            if let Some(v) = self.blocks.get_mut(&b) {
                v.retain(|&r| r != res);
                if v.is_empty() {
                    self.blocks.remove(&b);
                }
            }
        }
        true
    }
}

static REGISTRY: Mutex<Option<Registry>> = Mutex::new(None);

/// `XCL_PLACE_CHECK=1`: placements over the pages of live ones logged and counted.
pub(super) fn check() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("XCL_PLACE_CHECK").is_ok_and(|v| v.trim() == "1");
        if on {
            crate::xlog_info!("D3D12.X: XCL_PLACE_CHECK=1 - placements over the pages of live textures are logged (the first 32) and counted every 2 s");
        }
        on
    })
}

/// The counts of the 2 s line: [made, aliases at the same range, over released ones not gone
/// yet, partial (tt, st, ts, ss), one inside the other (tt, st, ts, ss), aliases of a sampled
/// texture over a target] – tt: a target over a target, st: a sampled texture over a target, ...
static COUNTS: [AtomicU64; 12] = [const { AtomicU64::new(0) }; 12];
static LAST_LINE: Mutex<Option<Instant>> = Mutex::new(None);

/// The kinds' place in `COUNTS`' groups of four.
fn kinds_index((new, old): (Kind, Kind)) -> usize {
    (new == Kind::Sampled) as usize + 2 * (old == Kind::Sampled) as usize
}

/// The 2 s line of `counts` (`COUNTS`).
fn counts_line(c: &[u64; 12]) -> String {
    let p = c[3] + c[4] + c[5] + c[6];
    let i = c[7] + c[8] + c[9] + c[10];
    format!(
        "D3D12.X: placements (2 s): {} made, {} aliases at the same range ({} sampled over a target), {p} partial overlaps (tt/st/ts/ss {}/{}/{}/{}), {i} one inside the other (tt/st/ts/ss {}/{}/{}/{}), {} over ones the title released (not gone yet)",
        c[0], c[1], c[11], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10], c[2]
    )
}

/// The full line of a case (the first 32).
fn case_line(res: usize, seq: u64, kind: Kind, desc: &Desc, (va, start, end): (usize, usize, usize), c: &Case) -> String {
    format!(
        "D3D12.X: placement #{seq} ({} {desc}, object {res:#x}) at {va:#x}, pages {start:#x}..{end:#x}, is {} #{} ({} {}, object {:#x}) at {:#x}, pages {:#x}..{:#x}, placed {} placements / {} ms before{}",
        kind.word(),
        c.overlap.words(),
        c.old_seq,
        c.kinds.1.word(),
        c.old_desc,
        c.old,
        c.old_range.0,
        c.old_range.1,
        c.old_range.2,
        c.ago,
        c.ms,
        if c.released { ", released by the title (not gone yet)" } else { "" }
    )
}

/// The cases of a placement into `COUNTS`, the first 32 logged; the 2 s line when it is due.
fn note(res: usize, seq: u64, kind: Kind, desc: &Desc, range: (usize, usize, usize), cases: &[Case]) {
    static LOGGED: AtomicU64 = AtomicU64::new(0);
    COUNTS[0].fetch_add(1, Ordering::Relaxed);
    for c in cases {
        let k = match c.overlap {
            Overlap::Same => 1,
            Overlap::Partial => 3 + kinds_index(c.kinds),
            Overlap::Inside | Overlap::Around => 7 + kinds_index(c.kinds),
        };
        COUNTS[k].fetch_add(1, Ordering::Relaxed);
        if c.overlap == Overlap::Same && c.kinds == (Kind::Sampled, Kind::Target) {
            COUNTS[11].fetch_add(1, Ordering::Relaxed);
        }
        if c.released {
            COUNTS[2].fetch_add(1, Ordering::Relaxed);
        }
        // (an alias in full only of another kind: a sampled one is uploaded over what the GPU wrote)
        if (c.overlap != Overlap::Same || c.kinds.0 != c.kinds.1) && LOGGED.fetch_add(1, Ordering::Relaxed) < 32 {
            crate::xlog_info!("{}", case_line(res, seq, kind, desc, range, c));
        }
    }
    let now = Instant::now();
    let due = {
        let mut last = LAST_LINE.lock().unwrap();
        let due = last.is_some_and(|t| now.duration_since(t).as_secs() >= 2);
        if due || last.is_none() {
            *last = Some(now);
        }
        due
    };
    if due {
        let c: [u64; 12] = std::array::from_fn(|k| COUNTS[k].swap(0, Ordering::Relaxed));
        crate::xlog_info!("{}", counts_line(&c));
    }
}

/// Texture `res` placed at title address `va` on `pages` (`[start, end)`: its own title pages).
pub(super) fn placed(res: usize, va: usize, pages: (usize, usize), backing: Backing, kind: Kind, desc: &D3D12_RESOURCE_DESC) {
    let look = check();
    let desc = Desc::of(desc);
    let (seq, cases) = REGISTRY.lock().unwrap().get_or_insert_with(Registry::default).place(res, va, pages, backing, kind, desc, Instant::now(), look);
    if look {
        note(res, seq, kind, &desc, (va, pages.0, pages.1.max(pages.0)), &cases);
    }
}

/// The title released placed texture `res` (`vram::forget`).
pub(super) fn released(res: usize) {
    if let Some(r) = REGISTRY.lock().unwrap().as_mut() {
        r.release(res);
    }
}

/// Resource `res` is gone on the PC (or goes now).
pub(super) fn gone(res: usize) {
    if let Some(r) = REGISTRY.lock().unwrap().as_mut() {
        r.remove(res);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const K64: usize = 0x1_0000;
    const MIB: usize = 1 << 20;

    fn place(r: &mut Registry, res: usize, va: usize, pages: (usize, usize), kind: Kind) -> (u64, Vec<Case>) {
        r.place(res, va, pages, Backing::Pages(1), kind, Desc::default(), Instant::now(), true)
    }

    fn overlaps(cases: &[Case]) -> Vec<(usize, Overlap)> {
        cases.iter().map(|c| (c.old, c.overlap)).collect()
    }

    /// Placements are classified by how their pages lie to the live ones'.
    #[test]
    fn a_placement_is_classified_by_its_pages() {
        let mut r = Registry::default();
        let x = 0x40_0000_0000usize;
        // a 14.4 MiB target, then: the same pages, an R8 target inside it, one partly over its end,
        // one around all of it, one past it
        assert!(place(&mut r, 1, x, (x, x + 0xe7_0000), Kind::Target).1.is_empty());
        assert_eq!(overlaps(&place(&mut r, 2, x, (x, x + 0xe7_0000), Kind::Target).1), [(1, Overlap::Same)]);
        let (_, c) = place(&mut r, 3, x + 0xba_0000, (x + 0xba_0000, x + 0xe4_0000), Kind::Target);
        assert_eq!(overlaps(&c), [(1, Overlap::Inside), (2, Overlap::Inside)]);
        let (_, c) = place(&mut r, 4, x + 0xe0_0000, (x + 0xe0_0000, x + 0x120_0000), Kind::Sampled);
        assert_eq!(overlaps(&c), [(1, Overlap::Partial), (2, Overlap::Partial), (3, Overlap::Partial)]);
        assert_eq!(c[0].kinds, (Kind::Sampled, Kind::Target));
        let (_, c) = place(&mut r, 5, x - K64, (x - K64, x + 0x200_0000), Kind::Sampled);
        assert_eq!(overlaps(&c), [(1, Overlap::Around), (2, Overlap::Around), (3, Overlap::Around), (4, Overlap::Around)]);
        assert!(place(&mut r, 6, x + 0x200_0000, (x + 0x200_0000, x + 0x210_0000), Kind::Sampled).1.is_empty());
    }

    /// Ranges that only touch share nothing; nor do placements on other video memory.
    #[test]
    fn neighbours_and_other_memory_share_nothing() {
        let mut r = Registry::default();
        let x = 0x10_0000_0000usize;
        place(&mut r, 1, x, (x, x + 4 * MIB), Kind::Target);
        assert!(place(&mut r, 2, x + 4 * MIB, (x + 4 * MIB, x + 8 * MIB), Kind::Target).1.is_empty());
        assert!(place(&mut r, 3, x - MIB, (x - MIB, x), Kind::Sampled).1.is_empty());
        // (another device's pages, an arena)
        assert!(r.place(4, x, (x, x + 4 * MIB), Backing::Pages(2), Kind::Target, Desc::default(), Instant::now(), true).1.is_empty());
        assert!(r.place(5, x, (x, x + 4 * MIB), Backing::Arena(7), Kind::Target, Desc::default(), Instant::now(), true).1.is_empty());
        let (_, c) = r.place(6, x + MIB, (x + MIB, x + 2 * MIB), Backing::Arena(7), Kind::Target, Desc::default(), Instant::now(), true);
        assert_eq!(overlaps(&c), [(5, Overlap::Inside)]);
    }

    /// A texture at a 4 KiB-aligned address with no page of its own (`vram::owned_pages`) shares
    /// tiles only with one at the same address.
    #[test]
    fn textures_without_pages_of_their_own_share_by_address() {
        let mut r = Registry::default();
        let x = 0x20_0000_0000usize + 0x1000;
        let first = x.next_multiple_of(K64);
        place(&mut r, 1, x, (first, first), Kind::Sampled);
        assert!(place(&mut r, 2, x + 0x2000, (first, first), Kind::Sampled).1.is_empty(), "another address: other tiles");
        assert_eq!(overlaps(&place(&mut r, 3, x, (first, first), Kind::Sampled).1), [(1, Overlap::Same)]);
        // (one with pages from the next 64 KiB on)
        assert_eq!(overlaps(&place(&mut r, 4, x, (first, first + K64), Kind::Target).1), [(1, Overlap::Around), (3, Overlap::Around)]);
    }

    /// A placement over 2 MiB blocks is found from each of them, and once.
    #[test]
    fn a_large_placement_is_found_once_from_every_block() {
        let mut r = Registry::default();
        let x = 0x30_0000_0000usize;
        place(&mut r, 1, x, (x, x + 64 * MIB), Kind::Target);
        for at in [x, x + 31 * MIB, x + 64 * MIB - K64] {
            let (_, c) = place(&mut r, 100 + at, at, (at, at + K64), Kind::Sampled);
            assert_eq!(overlaps(&c), [(1, Overlap::Inside)], "at {at:#x}");
            r.remove(100 + at);
        }
    }

    /// Sequence numbers and ages count placements; a released one is found until it is gone, and
    /// an object's address placed again is a new placement.
    #[test]
    fn released_ones_stay_until_gone_and_sequence_numbers_count_placements() {
        let mut r = Registry::default();
        let x = 0x50_0000_0000usize;
        let (s1, _) = place(&mut r, 1, x, (x, x + 2 * MIB), Kind::Target);
        place(&mut r, 2, x + 8 * MIB, (x + 8 * MIB, x + 9 * MIB), Kind::Target);
        assert!(r.release(1));
        let (s3, c) = place(&mut r, 3, x + MIB, (x + MIB, x + 3 * MIB), Kind::Sampled);
        assert_eq!((s3 - s1, c.len(), c[0].ago, c[0].old_seq, c[0].released), (2, 1, 2, s1, true));
        assert!(r.remove(1) && !r.remove(1));
        assert!(place(&mut r, 4, x, (x, x + MIB), Kind::Target).1.is_empty(), "1 is gone, 3 starts where 4 ends");
        // (the address of object 4 again, elsewhere: the old placement is not found)
        let (_, c) = place(&mut r, 4, x + 8 * MIB, (x + 8 * MIB, x + 9 * MIB), Kind::Target);
        assert_eq!(overlaps(&c), [(2, Overlap::Same)]);
        assert!(place(&mut r, 5, x, (x, x + K64), Kind::Target).1.is_empty(), "object 4 was moved, object 3 starts past it");
        assert!(r.blocks.values().flatten().all(|res| r.live.contains_key(res)));
    }

    /// The 2 s line's counts by kind: the newer one's first.
    #[test]
    fn the_counts_line_names_the_kinds_newer_first() {
        assert_eq!(kinds_index((Kind::Target, Kind::Target)), 0);
        assert_eq!(kinds_index((Kind::Sampled, Kind::Target)), 1);
        assert_eq!(kinds_index((Kind::Target, Kind::Sampled)), 2);
        assert_eq!(kinds_index((Kind::Sampled, Kind::Sampled)), 3);
        let line = counts_line(&[9, 2, 1, 1, 2, 0, 0, 0, 0, 3, 0, 1]);
        assert!(line.starts_with("D3D12.X: placements (2 s): 9 made, 2 aliases at the same range (1 sampled over a target), 3 partial overlaps (tt/st/ts/ss 1/2/0/0), 3 one inside the other (tt/st/ts/ss 0/0/3/0)"), "{line}");
    }
}
