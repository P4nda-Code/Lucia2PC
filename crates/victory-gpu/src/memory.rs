//! Title memory by address, for translated shaders that read and write
//! through raw 40-bit addresses (`victory_shader::emit::SLOT_MEMORY`: the
//! console's ray-tracing structure builder and the programs that walk its
//! structures; the structures themselves lie in plain title memory, in no
//! buffer object).
//!
//! The copies are the ones every other access uses ([`crate::shadow`]: a
//! title buffer's copy, or the 4 MiB window of memory outside buffers), so
//! a root view and a raw address reach the same host bytes. This module
//! keeps what a shader needs to find them:
//!
//! - the **directory**: per 64 KiB page of title address space the number
//!   of the copy that holds the page (0: none) and the page's byte offset
//!   in it; entries are published before dispatch and retired when their
//!   backing title buffer is released;
//! - per view heap a run of slots with a raw unordered access view of each
//!   numbered copy (slot 0 of the run: a null view, for pages without a
//!   copy);
//! - the **faults**: a monotonic atomic bitmap of pages shaders found empty,
//!   read here once a frame so the host can retry making their copies. A
//!   pointer a shader takes out of memory cannot be known before it runs;
//!   addresses a dispatch receives as root views are made ready before it.
//!
//! **A page that is not there when a shader touches it is a lost access**:
//! the read gives 0 and the store is dropped; the fault bitmap lets the host
//! retry mapping on later frames. For the structure builder that is a dropped
//! stage, so before a dispatch the host makes ready everything it can see
//! the dispatch name: the whole copy around each root view, around each
//! pair of root constants that is a title address and around each root
//! constant that is one in 256-byte units (the builder passes structures
//! all three ways), and the 4 MiB after each (a structure may cross a
//! window's end).
//! shortcut: what a stage reaches only through pointers it reads out of
//! memory (InitHeaders' table of structure addresses, LoadInstances'
//! instance records) is not made ready unless something else named it;
//! the fault count in the log says whether that happens. The whole answer
//! is the extent of the title's allocation around an address, which the
//! memory module of the layer knows and this one does not ask yet.
//!
//! shortcut: a copy that is numbered is kept for the rest of the run, and
//! there are at most [`COPIES`] of them (4 MiB windows mostly: 4 GiB at
//! most). Every dispatch of such a shader brings all numbered copies up to
//! date and, if it can store, declares all of them written (the bytes are
//! compared on the CPU afterwards, as for any unordered access view): a
//! dirty-page mask written by the shader would bound both.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use victory_shader::emit::{MEMORY_FAULTS, MEMORY_PAGES};
use windows::Win32::Graphics::Direct3D12::*;

use crate::descriptor::host::HostHeap;
use crate::host::{HostList, host};
use crate::shadow::Shadow;

/// Copies that can be numbered (the run of heap slots is one longer). A
/// listed frame of the opening cutscene names about 50 windows.
const COPIES: u32 = 1023;
/// A window of memory outside buffers (`shadow.rs`).
const CELL: u64 = 4 << 20;
/// A directory page.
const PAGE: u64 = 1 << 16;

struct Memory {
    /// Buffer, its mapping, its GPU address.
    directory: (ID3D12Resource, *mut u32, u64),
    faults: (ID3D12Resource, *mut u32, u64),
    /// Copy `n` is `copies[n - 1]`.
    copies: Vec<Arc<Shadow>>,
    /// Retired copies remain allocated because their descriptor slots are
    /// append-only, but must never be refreshed or copied back again.
    retired: Vec<bool>,
    /// Per host view heap (by address): the first slot of its run, and how
    /// many copies have their view there.
    heaps: HashMap<usize, (u32, u32)>,
    /// The frame the faults were last read in.
    frame: u64,
    /// Pages asked for this frame that are not the title's memory.
    refused: std::collections::HashSet<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MapResult {
    Mapped,
    AlreadyMapped,
    OutsideDirectory,
    PreviouslyRefused,
    NotCommitted,
    NoWholeShadow,
    NotUnorderedAccess,
    CapacityExhausted,
}

/// Extract every page bit set by GPU misses. Bits are monotonic and are not
/// cleared: a CPU clear could race with a later in-flight GPU atomic OR.
/// Kept separate so collisions and the final partial word can be tested
/// without a D3D12 device.
fn fault_pages(words: &[u32]) -> Vec<u32> {
    let mut pages = Vec::new();
    for (word_index, &word) in words.iter().enumerate() {
        let mut bits = word;
        while bits != 0 {
            let bit = bits.trailing_zeros();
            let page = (word_index as u32) * 32 + bit;
            if page < MEMORY_PAGES {
                pages.push(page);
            }
            bits &= bits - 1;
        }
    }
    pages
}

fn page_bounds(address: u64, len: u64) -> Option<(u64, u64)> {
    if len == 0 {
        return None;
    }
    let end = address.checked_add(len)?;
    Some((address / PAGE, (end - 1) / PAGE))
}

fn retired_entry(entry: [u32; 2], copy: u32) -> Option<[u32; 2]> {
    (entry[0] == copy).then_some([0, 0])
}

// SAFETY: the mappings are of buffers the struct owns; all access is under
// the mutex.
unsafe impl Send for Memory {}

static MEMORY: Mutex<Option<Memory>> = Mutex::new(None);

/// What a draw binds.
pub(crate) struct Ready {
    pub(crate) directory: u64,
    pub(crate) faults: u64,
    /// The view heap slot of the null view; copy `n`'s view is `n` further.
    pub(crate) base: u32,
}

impl Memory {
    fn new() -> Option<Memory> {
        let host = host()?;
        let make = |dwords: u64, what: &str| {
            let (buffer, data) = host.shared_buffer(dwords * 4)?;
            crate::host::name(&buffer, || format!("title memory by address: {what}"));
            // SAFETY: a fresh mapping of at least that many bytes.
            unsafe { std::ptr::write_bytes(data, 0, (dwords * 4) as usize) };
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            Some((buffer, data.cast::<u32>(), gpu))
        };
        tracing::info!("title memory by address: a shader reads or writes through raw addresses");
        Some(Memory {
            directory: make(u64::from(MEMORY_PAGES) * 2, "directory")?,
            faults: make(u64::from(MEMORY_FAULTS), "faults")?,
            copies: Vec::new(),
            retired: Vec::new(),
            heaps: HashMap::new(),
            frame: u64::MAX,
            refused: Default::default(),
        })
    }

    /// Gives the page of `address` its copy, if it has none and the memory
    /// is the title's.
    fn map(&mut self, address: u64) -> MapResult {
        let page = address / PAGE;
        // The last entry stays empty: addresses past the directory read it.
        if page >= u64::from(MEMORY_PAGES) - 1 {
            return MapResult::OutsideDirectory;
        }
        // SAFETY: inside the directory's mapping.
        let entry = unsafe { self.directory.1.add(page as usize * 2) };
        // SAFETY: as above.
        if unsafe { entry.read() } != 0 {
            return MapResult::AlreadyMapped;
        }
        let start = page * PAGE;
        // A shader's pointer, or a constant taken for an address, may be
        // anything: only memory the title has.
        if self.refused.contains(&page) {
            return MapResult::PreviouslyRefused;
        }
        if !crate::texture::committed(start as *const u8, PAGE as usize) {
            self.refused.insert(page);
            return MapResult::NotCommitted;
        }
        // The copy a root view at that address would get; a buffer that
        // ends inside the page is passed over for what holds all of it.
        let whole = |s: &Arc<Shadow>| s.offset_in(start).is_some_and(|_| s.left(start) >= PAGE);
        let Some(shadow) = crate::shadow::find(start, 1)
            .filter(whole)
            .or_else(|| crate::shadow::find(start, PAGE).filter(whole))
        else {
            return MapResult::NoWholeShadow;
        };
        // SAFETY: a live buffer.
        let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
        if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
            return MapResult::NotUnorderedAccess;
        }
        let n = match self
            .copies
            .iter()
            .zip(&self.retired)
            .position(|(c, retired)| !retired && Arc::ptr_eq(c, &shadow))
        {
            Some(n) => n,
            None if self.copies.len() < COPIES as usize => {
                self.copies.push(shadow.clone());
                self.retired.push(false);
                let n = self.copies.len() - 1;
                // Every page the copy holds whole, not only the one asked
                // for: a structure is more than 64 KiB.
                // shortcut: a page of a title buffer inside this one that
                // has a copy of its own is given this copy too.
                let (first, len) = shadow.span();
                let mut at = first.next_multiple_of(PAGE);
                while at + PAGE <= first + len && at / PAGE < u64::from(MEMORY_PAGES) - 1 {
                    // SAFETY: inside the directory's mapping.
                    let other = unsafe { self.directory.1.add((at / PAGE) as usize * 2) };
                    // SAFETY: as above; the offset first, as below.
                    unsafe {
                        if other.read() == 0 {
                            other.add(1).write_volatile((at - first) as u32);
                            other.write_volatile(n as u32 + 1);
                        }
                    }
                    at += PAGE;
                }
                n
            }
            None => {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    tracing::warn!(COPIES, "title memory by address: no number left for another copy; its pages read zeros")
                });
                return MapResult::CapacityExhausted;
            }
        };
        let offset = shadow.offset_in(start).unwrap_or(0);
        // The offset first: a shader that sees the number sees it too.
        // SAFETY: inside the directory's mapping.
        unsafe {
            entry.add(1).write_volatile(offset as u32);
            entry.write_volatile(n as u32 + 1);
        }
        MapResult::Mapped
    }

    /// Removes a released resource's mapping from the shared directory.
    /// The caller first settles submissions so an in-flight dispatch cannot
    /// observe the directory changing under it.
    fn retire(&mut self, shadow: &Arc<Shadow>) {
        let Some(n) = self.copies.iter().position(|c| Arc::ptr_eq(c, shadow)) else {
            return;
        };
        if self.retired[n] {
            return;
        }
        self.retired[n] = true;
        let copy = n as u32 + 1;
        let (start, len) = shadow.span();
        if len == 0 {
            return;
        }
        let first = (start / PAGE) as usize;
        let last = ((start + len - 1) / PAGE) as usize;
        for page in first..=last {
            self.refused.remove(&(page as u64));
            if page >= MEMORY_PAGES as usize - 1 {
                break;
            }
            // SAFETY: inside the live directory; only this copy's pages
            // are cleared, leaving overlapping mappings intact.
            let entry = unsafe { self.directory.1.add(page * 2) };
            let old = unsafe { [entry.read_volatile(), entry.add(1).read_volatile()] };
            if let Some(empty) = retired_entry(old, copy) {
                unsafe {
                    entry.write_volatile(empty[0]);
                    entry.add(1).write_volatile(empty[1]);
                }
            }
        }
    }

    /// Pages shaders have found empty, from a monotonic atomic bitmap so
    /// concurrent GPU faults cannot overwrite one another or be lost to a
    /// CPU clear racing an in-flight submission.
    fn take_faults(&mut self) {
        // SAFETY: inside the faults' mapping. The bitmap is monotonic, so a
        // concurrent atomic OR cannot erase another page's fault. Its bits
        // are historical: only pages still empty need retrying below.
        let mut words = Vec::with_capacity(MEMORY_FAULTS as usize);
        for index in 0..MEMORY_FAULTS as usize {
            // SAFETY: inside the mapped fault buffer.
            let at = unsafe { self.faults.1.add(index) };
            // Volatile is required for memory visible to GPU atomic writes.
            words.push(unsafe { at.read_volatile() });
        }
        let pages = fault_pages(&words)
            .into_iter()
            .filter(|&page| {
                // The page may have become resident through root metadata
                // since the GPU first recorded its fault.
                // SAFETY: the decoded page is within the directory.
                unsafe { self.directory.1.add(page as usize * 2).read_volatile() == 0 }
            })
            .collect::<Vec<_>>();
        for page in pages {
            let start = u64::from(page) * PAGE;
            // This reports recovery after the shader access: any read already
            // returned zero and any write was dropped. Keep the failure reason
            // visible; page zero commonly denotes an invalid low pointer, not
            // a page that the mapper successfully made resident.
            static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let result = self.map(start);
            if SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 50 {
                tracing::warn!(
                    page_start = format_args!("{start:#x}"),
                    page_end = format_args!("{:#x}", start + PAGE),
                    mapping = ?result,
                    bitmap = "monotonic_history",
                    "title memory by address: a prior shader access to this unmapped page read zero or dropped its write; the bitmap does not identify when it happened, and mapping recovery was attempted"
                );
            }
        }
    }

    /// What a dispatch names: the copy around `address` and the 4 MiB
    /// after it.
    fn name(&mut self, address: u64) {
        let _ = self.map(address);
        let _ = self.map((address & !(CELL - 1)) + CELL);
    }

    /// Makes every page of a known resource allocation resident before a
    /// builder dispatch. Unlike fault recovery, this runs before GPU
    /// execution, so failure is safe to report as a dropped dispatch.
    fn map_range(&mut self, address: u64, len: u64) -> bool {
        if len == 0 {
            return true;
        }
        let Some((mut page, last)) = page_bounds(address, len) else {
            return false;
        };
        while page <= last {
            let at = page * PAGE;
            self.map(at);
            if page >= u64::from(MEMORY_PAGES) - 1 {
                return false;
            }
            // SAFETY: the page was range-checked against the directory.
            let copy = unsafe { self.directory.1.add(page as usize * 2).read_volatile() };
            if copy == 0 {
                return false;
            }
            page += 1;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fault_bitmap_preserves_colliding_pages_without_mutating_gpu_state() {
        // Pages 7 and 7 + 4095 collided in the old modulo fault list.
        let mut words = vec![0; MEMORY_FAULTS as usize];
        for page in [7, 7 + 4095, 31, 32, MEMORY_PAGES - 1] {
            let word = (page / 32) as usize;
            words[word] |= 1 << (page % 32);
        }
        let got = fault_pages(&words);
        let mut got = got;
        got.sort_unstable();
        assert_eq!(got, [7, 31, 32, 4102, MEMORY_PAGES - 1]);
        assert!(fault_pages(&words).contains(&(MEMORY_PAGES - 1)));
    }

    #[test]
    fn fault_bitmap_allocation_covers_exactly_the_address_directory() {
        assert_eq!(MEMORY_FAULTS, MEMORY_PAGES.div_ceil(32));
    }

    #[test]
    fn page_bounds_include_both_sides_of_crossed_boundaries() {
        assert_eq!(page_bounds(PAGE - 4, 8), Some((0, 1)));
        assert_eq!(page_bounds(PAGE, PAGE), Some((1, 1)));
        assert_eq!(page_bounds(u64::MAX - 3, 8), None);
        assert_eq!(page_bounds(0, 0), None);
    }

    #[test]
    fn retirement_clears_only_entries_owned_by_the_released_copy() {
        assert_eq!(retired_entry([4, 0x1234], 4), Some([0, 0]));
        assert_eq!(retired_entry([5, 0x5678], 4), None);
    }
}

/// Before a draw whose shader reads title memory by address: what its
/// root arguments name gets its copies (`addresses`: its root views;
/// `constants`: its root constants, see the module), every numbered copy
/// has its view in `heap`'s run and is brought up to date when `list` is
/// submitted (and, with `writes`, copied back after). None if the host
/// cannot hold it; the draw is then dropped.
pub(crate) fn ready(
    list: &HostList,
    heap: &HostHeap,
    addresses: &[u64],
    constants: &[u32],
    required: &[(u64, u64)],
    writes: bool,
) -> Option<Ready> {
    let mut guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        *guard = Memory::new();
    }
    let m = guard.as_mut()?;
    let frame = crate::frame::current();
    if m.frame != frame {
        m.frame = frame;
        m.refused.clear();
        m.take_faults();
    }
    let numbered = m.copies.len();
    for &address in addresses {
        m.name(address);
    }
    // A title address in two constants (low dword, then bits 32 to 39), or
    // in one in 256-byte units. A constant that only looks like one names
    // memory nothing reads; one that is not the title's memory is refused.
    for (i, &c) in constants.iter().enumerate() {
        if let Some(&high) = constants.get(i + 1)
            && (1..=0xFF).contains(&high)
            && c >= 0x1_0000
        {
            m.name(u64::from(c) | u64::from(high) << 32);
        }
        if c >= 0x0100_0000 {
            m.name(u64::from(c) << 8);
        }
    }
    for &(address, len) in required {
        if !m.map_range(address, len) {
            tracing::warn!(
                address = format_args!("{address:#x}"),
                len,
                "builder dispatch has a descriptor-backed allocation that could not be made resident; dispatch dropped"
            );
            return None;
        }
    }
    let key = std::ptr::from_ref(heap) as usize;
    let (base, made) = match m.heaps.get(&key) {
        Some(&run) => run,
        None => {
            // Held for the heap's life: not given back with a list's slots.
            let base = heap.dynamic(COPIES + 1)?;
            heap.null_raw_uav_at(base);
            (base, 0)
        }
    };
    for (n, shadow) in m.copies.iter().enumerate().skip(made as usize) {
        let (_, len) = shadow.span();
        heap.raw_uav_at(base + 1 + n as u32, &shadow.buffer, 0, len);
    }
    m.heaps.insert(key, (base, m.copies.len() as u32));
    // Once per list, and for copies numbered since: the builder alone
    // dispatches hundreds of times a frame.
    // shortcut: a copy another list numbers meanwhile is brought up to date
    // by that list's submission, which may run after this one's.
    const READ: usize = usize::MAX - 0x6d;
    const WRITE: usize = usize::MAX - 0x6e;
    let first_read = list.first_use(READ);
    let first_write = writes && list.first_use(WRITE);
    for (n, shadow) in m.copies.iter().enumerate() {
        if m.retired[n] {
            continue;
        }
        let (start, len) = shadow.span();
        if first_read || n >= numbered {
            shadow.refresh(list, start, len);
        }
        if writes && (first_write || n >= numbered) {
            shadow.gpu_writes(list, start, len);
        }
    }
    if first_read {
        list.keep(&m.directory.0);
        list.keep(&m.faults.0);
    }
    Some(Ready {
        directory: m.directory.2,
        faults: m.faults.2,
        base,
    })
}

/// Retires raw-address mappings for a title buffer that has been released.
/// Outstanding submissions are settled before the global page directory is
/// changed; a reused virtual address will receive a new copy number.
pub(crate) fn retire(shadow: &Arc<Shadow>) {
    let mut guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    let Some(m) = guard.as_mut() else {
        return;
    };
    if !m.copies.iter().any(|c| Arc::ptr_eq(c, shadow)) {
        return;
    }
    // Hold the memory lock while settling so no later list can record a
    // dispatch using the mapping we're about to retire.
    crate::host::settle_for_present();
    m.retire(shadow);
}

/// Invalidates title virtual pages whose backing changed. Raw-address
/// directory changes are made only after submissions using the previous
/// mapping have completed.
pub(crate) fn invalidate(address: u64, len: u64) {
    let Some((first, last)) = page_bounds(address, len) else {
        return;
    };
    let mut guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    let Some(m) = guard.as_mut() else {
        return;
    };
    let limit = u64::from(MEMORY_PAGES) - 1;
    if first >= limit {
        return;
    }
    let last = last.min(limit - 1);
    for page in first..=last {
        m.refused.remove(&page);
    }
    let has_entries = (first..=last).any(|page| {
        // SAFETY: this range is clamped to the directory.
        unsafe { m.directory.1.add(page as usize * 2).read_volatile() != 0 }
    });
    if !has_entries {
        return;
    }
    crate::host::settle_for_present();
    for page in first..=last {
        // SAFETY: this range is clamped to the directory. The queue is
        // settled, so no GPU command can observe a partially cleared range.
        let entry = unsafe { m.directory.1.add(page as usize * 2) };
        unsafe {
            entry.write_volatile(0);
            entry.add(1).write_volatile(0);
        }
    }
}
