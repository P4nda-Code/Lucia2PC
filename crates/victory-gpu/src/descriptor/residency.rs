//! Logical-to-physical descriptor residency over the host heap's shared
//! dynamic-slot allocator. Mapping publication is transactional: a caller
//! installs the real D3D12 descriptor before the new mapping becomes visible.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub(crate) const UNMAPPED_DESCRIPTOR: u32 = u32::MAX;

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mapping {
    Resident(u32),
    Unresident,
    Invalid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidencyError {
    InvalidLogicalIndex,
    CapacityExhausted,
    DuplicateIdentity,
    DescriptorChanged,
    DescriptorCopyFailed,
}

#[derive(Debug)]
struct PoolState {
    next: u32,
    taken: Vec<bool>,
    free: u32,
    too_long: u32,
}

/// The single allocation domain for relocatable root tables and direct
/// descriptor residency. Indices are absolute in the shader-visible heap.
pub(crate) struct PhysicalSlots {
    first: u32,
    count: u32,
    state: Mutex<PoolState>,
    on_release: Mutex<Option<Arc<dyn Fn(u32) + Send + Sync>>>,
}

impl std::fmt::Debug for PhysicalSlots {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PhysicalSlots")
            .field("first", &self.first)
            .field("count", &self.count)
            .finish_non_exhaustive()
    }
}

impl PhysicalSlots {
    pub(crate) fn new(first: u32, count: u32) -> Option<Self> {
        first.checked_add(count)?;
        Some(Self {
            first,
            count,
            state: Mutex::new(PoolState {
                next: 0,
                taken: vec![false; count as usize],
                free: count,
                too_long: u32::MAX,
            }),
            on_release: Mutex::new(None),
        })
    }

    pub(crate) fn on_release(&self, callback: Arc<dyn Fn(u32) + Send + Sync>) {
        *self.on_release.lock().unwrap_or_else(|e| e.into_inner()) = Some(callback);
    }

    pub(crate) fn allocate(&self, count: u32) -> Option<u32> {
        if count == 0 || count > self.count {
            return None;
        }
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let enough = state.free >= count && count < state.too_long;
        let n = count as usize;
        let starts = (state.next as usize..self.count as usize)
            .chain(0..state.next as usize);
        let (mut run, mut last, mut found) = (0, usize::MAX, None);
        for at in starts.take(if enough { self.count as usize } else { 0 }) {
            if at != last.wrapping_add(1) {
                run = 0;
            }
            last = at;
            run = if state.taken[at] { 0 } else { run + 1 };
            if run == n {
                found = Some(at + 1 - n);
                break;
            }
        }
        let first = found?;
        state.taken[first..first + n].fill(true);
        state.free -= count;
        state.next = ((first + n) % self.count as usize) as u32;
        Some(self.first + first as u32)
    }

    pub(crate) fn release(&self, first: u32, count: u32) {
        let Some(offset) = first.checked_sub(self.first) else { return };
        let Some(end) = offset.checked_add(count) else { return };
        if count == 0 || end > self.count {
            return;
        }
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let slots = &mut state.taken[offset as usize..end as usize];
        let newly_free = slots.iter().filter(|taken| **taken).count() as u32;
        if newly_free != 0
            && let Some(callback) = self.on_release.lock().unwrap_or_else(|e| e.into_inner()).clone()
        {
            for (relative, taken) in slots.iter().enumerate() {
                if *taken {
                    callback(first + relative as u32);
                }
            }
        }
        slots.fill(false);
        state.free += newly_free;
        state.too_long = u32::MAX;
    }

    #[cfg(test)]
    pub(crate) fn free_count(&self) -> u32 {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).free
    }

    pub(crate) fn note_fragmented(&self, count: u32) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.free >= count {
            state.too_long = state.too_long.min(count);
        }
    }

}

#[derive(Debug)]
struct Resident {
    identity: u128,
    snapshots: u64,
    current: bool,
}

#[derive(Debug)]
struct State {
    logical: Vec<Option<u32>>,
    physical: HashMap<u32, Resident>,
    identity: HashMap<u128, u32>,
    version: u64,
}

struct Inner {
    logical_count: u32,
    slots: Arc<PhysicalSlots>,
    state: Mutex<State>,
}

/// Bounded residency table for one logical title view heap.
#[derive(Clone)]
pub(crate) struct Residency {
    inner: Arc<Inner>,
}

impl Residency {
    pub(crate) fn new(logical_count: u32, slots: Arc<PhysicalSlots>) -> Self {
        Self {
            inner: Arc::new(Inner {
                logical_count,
                slots,
                state: Mutex::new(State {
                    logical: vec![None; logical_count as usize],
                    physical: HashMap::new(),
                    identity: HashMap::new(),
                    version: 0,
                }),
            }),
        }
    }

    /// Installs a real descriptor into a shared physical slot, then publishes
    /// its mapping. On copy failure the prior mapping remains unchanged.
    pub(crate) fn map_with(
        &self,
        logical: u32,
        identity: u128,
        copy: impl FnOnce(u32) -> Result<(), ResidencyError>,
    ) -> Result<u32, ResidencyError> {
        if logical >= self.inner.logical_count {
            return Err(ResidencyError::InvalidLogicalIndex);
        }
        let mut state = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(physical) = state.logical[logical as usize]
            && state.physical.get(&physical).is_some_and(|slot| slot.identity == identity)
        {
            return Ok(physical);
        }
        if state.identity.get(&identity).is_some_and(|&physical| {
            state.physical.get(&physical).is_some_and(|slot| slot.current)
        }) {
            return Err(ResidencyError::DuplicateIdentity);
        }

        let old = state.logical[logical as usize];
        let reuse = old.is_some_and(|physical| {
            state.physical.get(&physical).is_some_and(|slot| slot.snapshots == 0)
        });
        let physical = if reuse {
            old.expect("reusable old slot")
        } else {
            self.inner
                .slots
                .allocate(1)
                .ok_or(ResidencyError::CapacityExhausted)?
        };

        if let Err(error) = copy(physical) {
            if !reuse {
                self.inner.slots.release(physical, 1);
            }
            return Err(error);
        }

        if let Some(old) = state.logical[logical as usize].take() {
            let resident = state.physical.get_mut(&old).expect("logical mapping exists");
            resident.current = false;
            let old_identity = resident.identity;
            let old_unreferenced = resident.snapshots == 0;
            state.identity.remove(&old_identity);
            if old != physical && old_unreferenced {
                state.physical.remove(&old);
                self.inner.slots.release(old, 1);
            }
        }
        state.physical.insert(
            physical,
            Resident {
                identity,
                snapshots: 0,
                current: true,
            },
        );
        state.logical[logical as usize] = Some(physical);
        state.identity.insert(identity, physical);
        state.version = state.version.wrapping_add(1);
        Ok(physical)
    }

    #[cfg(test)]
    pub(crate) fn map(&self, logical: u32, identity: u128) -> Result<u32, ResidencyError> {
        self.map_with(logical, identity, |_| Ok(()))
    }

    pub(crate) fn invalidate(&self, logical: u32) -> Result<(), ResidencyError> {
        if logical >= self.inner.logical_count {
            return Err(ResidencyError::InvalidLogicalIndex);
        }
        let mut state = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
        let Some(physical) = state.logical[logical as usize].take() else {
            return Ok(());
        };
        state.version = state.version.wrapping_add(1);
        let resident = state.physical.get_mut(&physical).expect("logical mapping exists");
        resident.current = false;
        let identity = resident.identity;
        let unreferenced = resident.snapshots == 0;
        state.identity.remove(&identity);
        if unreferenced {
            state.physical.remove(&physical);
            self.inner.slots.release(physical, 1);
        }
        Ok(())
    }

    pub(crate) fn version(&self) -> u64 {
        self.inner.state.lock().unwrap_or_else(|e| e.into_inner()).version
    }

    pub(crate) fn snapshot(&self) -> MappingSnapshot {
        let mut state = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
        let mut map = Vec::with_capacity(state.logical.len());
        let mut slots = Vec::new();
        let logical = state.logical.clone();
        for (index, physical) in logical.iter().enumerate() {
            if let Some(physical) = physical {
                state.physical.get_mut(physical).expect("logical mapping exists").snapshots += 1;
                map.push(*physical);
                slots.push(*physical);
            } else if (index as u32) < self.inner.slots.first {
                // The host heap's title prefix is mirrored in place. It does
                // not consume a dynamic slot and therefore is not in the
                // residency lease's release list.
                map.push(index as u32);
            } else {
                map.push(UNMAPPED_DESCRIPTOR);
            }
        }
        MappingSnapshot {
            lease: Arc::new(SnapshotLease {
                map: map.into_boxed_slice(),
                slots,
                version: state.version,
                inner: Arc::clone(&self.inner),
            }),
            keepalive: None,
        }
    }
}

struct SnapshotLease {
    map: Box<[u32]>,
    slots: Vec<u32>,
    version: u64,
    inner: Arc<Inner>,
}

#[derive(Clone)]
pub(crate) struct MappingSnapshot {
    lease: Arc<SnapshotLease>,
    keepalive: Option<Arc<dyn std::any::Any + Send + Sync>>,
}

impl MappingSnapshot {
    pub(crate) fn with_keepalive<T: std::any::Any + Send + Sync + 'static>(
        mut self,
        keepalive: T,
    ) -> Self {
        self.keepalive = Some(Arc::new(keepalive));
        self
    }

    pub(crate) fn logical_count(&self) -> u32 {
        self.lease.map.len() as u32
    }

    pub(crate) fn version(&self) -> u64 {
        self.lease.version
    }

    #[cfg(test)]
    pub(crate) fn encoded(&self) -> &[u32] {
        &self.lease.map
    }

    #[cfg(test)]
    pub(crate) fn lookup(&self, logical: u32) -> Mapping {
        match self.lease.map.get(logical as usize) {
            Some(&UNMAPPED_DESCRIPTOR) => Mapping::Unresident,
            Some(&physical) => Mapping::Resident(physical),
            None => Mapping::Invalid,
        }
    }

    /// Copies the complete logical-to-physical table into a GPU upload
    /// mapping. The caller must keep this snapshot alive until the GPU has
    /// finished reading the copied table, so its physical descriptor slots
    /// cannot be recycled while commands still refer to them.
    ///
    /// Returns the required dword count without modifying the destination
    /// when it is too short. Any destination tail beyond the logical heap
    /// is left untouched.
    pub(crate) fn copy_encoded_to(&self, destination: &mut [u32]) -> Result<(), usize> {
        let required = self.lease.map.len();
        if destination.len() < required {
            return Err(required);
        }
        destination[..required].copy_from_slice(&self.lease.map);
        Ok(())
    }

    pub(crate) fn physical_slots(&self) -> &[u32] {
        &self.lease.slots
    }
}

impl Drop for SnapshotLease {
    fn drop(&mut self) {
        let inner = &self.inner;
        let mut state = inner.state.lock().unwrap_or_else(|e| e.into_inner());
        for &physical in &self.slots {
            let resident = state.physical.get_mut(&physical).expect("snapshot slot retained");
            resident.snapshots = resident.snapshots.checked_sub(1).expect("snapshot count underflow");
            if resident.snapshots == 0 && !resident.current {
                state.physical.remove(&physical);
                inner.slots.release(physical, 1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Mapping, PhysicalSlots, Residency, ResidencyError, UNMAPPED_DESCRIPTOR};
    use std::sync::Arc;

    const LOGICAL: u32 = 1_245_184;
    const PREFIX: u32 = 737_855;

    fn manager(capacity: u32) -> Residency {
        Residency::new(LOGICAL, Arc::new(PhysicalSlots::new(64, capacity).unwrap()))
    }

    #[test]
    fn shared_pool_bounds_allocations_and_releases_slots() {
        let slots = Arc::new(PhysicalSlots::new(64, 2).unwrap());
        let r = Residency::new(LOGICAL, slots.clone());
        let a = r.map(7, 10).unwrap();
        let b = slots.allocate(1).unwrap(); // Same pool as root-table relocation.
        assert_ne!(a, b);
        assert_eq!(r.map(9, 12), Err(ResidencyError::CapacityExhausted));
        r.invalidate(7).unwrap();
        assert_eq!(slots.free_count(), 1);
        slots.release(b, 1);
        assert_eq!(r.map(9, 12).unwrap(), a);
    }

    #[test]
    fn descriptor_copy_failure_keeps_old_mapping_and_releases_candidate() {
        let r = manager(2);
        let old = r.map(4, 40).unwrap();
        let before = r.snapshot();
        assert_eq!(
            r.map_with(4, 41, |_| Err(ResidencyError::DescriptorCopyFailed)),
            Err(ResidencyError::DescriptorCopyFailed)
        );
        assert_eq!(before.lookup(4), Mapping::Resident(old));
        assert_eq!(r.snapshot().lookup(4), Mapping::Resident(old));
        assert_eq!(r.map(5, 50).unwrap(), old + 1);
    }

    #[test]
    fn complete_logical_space_encodes_unresident_and_invalid_indices() {
        let r = manager(2);
        let high = LOGICAL - 1;
        r.map(PREFIX, 1).unwrap();
        r.map(high, 2).unwrap();
        let snapshot = r.snapshot();
        assert_eq!(snapshot.logical_count(), LOGICAL);
        assert_eq!(snapshot.lookup(PREFIX), Mapping::Resident(64));
        assert_eq!(snapshot.lookup(high), Mapping::Resident(65));
        // The mirrored physical prefix is an identity mapping even
        // though it is not allocated by the dynamic residency pool.
        assert_eq!(snapshot.lookup(0), Mapping::Resident(0));
        assert_eq!(snapshot.encoded()[0], 0);
        assert_eq!(snapshot.lookup(PREFIX + 1), Mapping::Unresident);
        assert_eq!(snapshot.encoded()[(PREFIX + 1) as usize], UNMAPPED_DESCRIPTOR);
        assert_eq!(snapshot.lookup(LOGICAL), Mapping::Invalid);
    }

    #[test]
    fn replacement_retains_old_descriptor_for_older_snapshot() {
        let r = manager(2);
        let old = r.map(PREFIX, 100).unwrap();
        let first = r.snapshot();
        let new = r.map(PREFIX, 101).unwrap();
        let second = r.snapshot();
        assert_ne!(old, new);
        assert_eq!(first.lookup(PREFIX), Mapping::Resident(old));
        assert_eq!(second.lookup(PREFIX), Mapping::Resident(new));
        drop(first);
        assert_eq!(r.map(12, 12).unwrap(), old);
    }

    #[test]
    fn mapping_versions_change_only_when_the_logical_map_changes() {
        let r = manager(2);
        assert_eq!(r.version(), 0);

        let physical = r.map(PREFIX, 100).unwrap();
        let first = r.snapshot();
        assert_eq!(first.version(), 1);
        assert_eq!(r.version(), 1);

        assert_eq!(r.map(PREFIX, 100).unwrap(), physical);
        assert_eq!(r.version(), 1, "an unchanged descriptor does not invalidate uploads");

        r.invalidate(PREFIX).unwrap();
        assert_eq!(r.version(), 2);
        assert_eq!(first.version(), 1, "submitted snapshots remain immutable");
    }

    #[test]
    fn gpu_table_copy_preserves_resident_and_unmapped_entries() {
        let r = Residency::new(8, Arc::new(PhysicalSlots::new(128, 2).unwrap()));
        let first = r.map(2, 20).unwrap();
        let second = r.map(7, 70).unwrap();
        let snapshot = r.snapshot();

        let mut destination = [0xfeed_beef; 10];
        snapshot.copy_encoded_to(&mut destination).unwrap();

        assert_eq!(destination[0], UNMAPPED_DESCRIPTOR);
        assert_eq!(destination[2], first);
        assert_eq!(destination[7], second);
        assert_eq!(destination[8..], [0xfeed_beef; 2]);
    }

    #[test]
    fn gpu_table_copy_rejects_short_destination_without_partial_write() {
        let r = Residency::new(8, Arc::new(PhysicalSlots::new(128, 1).unwrap()));
        let snapshot = r.snapshot();
        let mut destination = [0xfeed_beef; 7];

        assert_eq!(snapshot.copy_encoded_to(&mut destination), Err(8));
        assert_eq!(destination, [0xfeed_beef; 7]);
    }

    #[test]
    fn invalidation_and_in_flight_slots_release_only_after_lease_drop() {
        let r = manager(1);
        let physical = r.map(4, 40).unwrap();
        let submission = r.snapshot();
        r.invalidate(4).unwrap();
        assert_eq!(r.map(5, 50), Err(ResidencyError::CapacityExhausted));
        assert_eq!(submission.lookup(4), Mapping::Resident(physical));
        drop(submission);
        assert_eq!(r.map(5, 50).unwrap(), physical);
    }

    #[test]
    fn invalid_indices_and_duplicate_identities_fail_closed() {
        let r = manager(2);
        assert_eq!(r.map(LOGICAL, 1), Err(ResidencyError::InvalidLogicalIndex));
        r.map(1, 44).unwrap();
        assert_eq!(r.map(2, 44), Err(ResidencyError::DuplicateIdentity));
        assert_eq!(r.snapshot().lookup(2), Mapping::Unresident);
    }
}
