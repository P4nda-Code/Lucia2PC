# Ray tracing in Lucia2PC

**Status as of 2026-11-25: ray-traced lighting is not yet active in a
complete game run.** Much of the translation and emulation work described
below has been implemented and unit-tested, but the full path has not yet
been demonstrated end to end in the game. The related functionality is
disabled by default.

## How the title performs ray tracing

The title uses ray tracing for indirect light (including probes, bounce
lighting and reflections) and for shadows from lamps and the sun. Without
those effects, surfaces not reached by direct sunlight can appear black,
including interiors, shadowed sides of characters and rooms lit only by
lamps.

The title does not use a dedicated ray-tracing API for this work:

- Ray traversal is implemented as ordinary compute-shader code. Each thread
  uses a stack in shared memory and loads through raw 40-bit addresses in
  title memory.
- The static world's acceleration structures are stored pre-built in the
  title's data, using the console GPU's native node format. The formats
  include 64-byte box nodes with four children and 16-bit float bounds,
  128-byte box nodes with 32-bit float bounds, triangle nodes and instance
  nodes. These structures are unpacked on the GPU.
- A group of builder programs from the title's `XboxBvhBuilder.dll` builds
  and refits dynamic structures on the GPU each frame. The title contains
  52 compute programs for this work.
- Hardware is used for the per-node tests through
  `image_bvh64_intersect_ray` and `image_bvh_intersect_ray`. A call tests a
  ray against the four children of a box node, returning hit children in
  nearest-first order, or against a triangle node, returning the intersection
  numerator, denominator and barycentric numerators.

Using the host's DirectX Raytracing (DXR) API as a replacement is not a
practical direct mapping. The host layer does not receive the geometry
description behind the title's native structures, and traversal is embedded
in 64 of the title's own programs. Lucia2PC therefore translates the title's
builder and traversal shaders and emulates the per-node ray-test instruction.

## Work implemented

1. **Builder shader translation.** 49 of the 52 builder programs translate;
   these were all the builder programs created in the scenes tested. They
   pass buffer addresses through a spilled parameter table rather than
   descriptors. Support was added for the spilled user-data table, root
   unordered-access views in the translator and host binding, byte and
   16-bit buffer accesses, and several control-flow patterns.

2. **Raw-address access to title memory.** A directory with one entry per
   64 KiB of title address space lets a shader reach Lucia2PC's host copy of
   that memory. Loads, stores, atomic add/subtract and floating-point
   min/max use this mapping. Before dispatch, the host prepares the memory
   referenced by root views and root constants.

3. **Ray-test instruction emulation in DXIL.** Box-node tests use four slab
   tests and a sorting network. Triangle-node tests use a watertight
   ray/triangle intersection test. Both instruction forms are handled. A
   plain Rust reference implementation is in
   `crates/victory-shader/src/bvh.rs`. Tests execute the emitted DXIL on the
   GPU for 311 ray/node cases and compare the results bit for bit with the
   reference.

4. **Trace and shading shader translation.** 59 of the 64 trace programs and
   the irradiance-cache shading program translate. This required treating
   lane masks as ordinary numbers on some paths and handling exact patterns
   used by programs written for 64-lane waves on GPUs with 32-lane waves.

5. **Host-side builder safety rule.** Builder stages run only when every
   builder pipeline created by the title also has a host pipeline.
   `VICTORY_BVH_BUILDER=1` requests builder execution.
   `VICTORY_BVH_QUARANTINE=1` additionally keeps builder stores out of title
   memory for an initial test. This guard exists because a partially working
   builder previously wrote invalid bounds into title memory and crashed
   the title.

## Known gaps and risks

- Even with `VICTORY_BVH_BUILDER=1`, the builder remains disabled in a real
  run because some pipelines are refused by the host (they still request
  64-lane waves on a 32-lane GPU) or fail translation.
- Five shadow-mask trace programs do not yet translate.
- Accesses to title-memory pages without a host copy are not fully handled:
  reads return zero and stores are dropped.
- Some instruction details are not publicly documented and remain assumed:
  rounding behavior, ordering when two boxes are entered at the same
  distance, and the result for a ray passing exactly through an edge.
- GPU execution cost has not yet been measured.

## Relevant implementation files

| File | Area |
|---|---|
| `crates/victory-shader/src/bvh.rs` | Reference implementation of the node-intersection test |
| `crates/victory-shader/src/emit/lower.rs` | Lowering of the two ray-test instructions and raw-address memory access (`Lower::memory`) |
| `crates/victory-shader/src/resource.rs` | Descriptor and raw-address tracking |
| `crates/victory-gpu/src/pipeline.rs` | Builder pipeline safety rule |
| `crates/victory-gpu/src/draw.rs` | Host-side root views and title-memory handling |

This document records the implementation status above; it does not imply that
ray-traced lighting is currently working in normal gameplay.
