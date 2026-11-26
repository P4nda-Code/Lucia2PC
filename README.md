# Lucia2PC

**An experimental Xbox-to-Windows compatibility and graphics project.**

Lucia2PC is an independent development project exploring the compatibility work required to run a user-provided Xbox Series game build on a Windows PC. Its main areas of work are CPU/runtime compatibility, graphics translation, shader handling, resource management, visual correctness, stability, and performance.

Lucia2PC is unofficial and is not affiliated with, endorsed by, or sponsored by the game's developers, publisher, or Microsoft.

## Current status — 2026-11-26

Lucia2PC is an active experimental project. The current development work is concentrated on the Direct3D 12 graphics path, especially descriptor handling and shader resource access.

**Hardware investigation:** The NVIDIA GeForce RTX 3060 was used for the recent D3D12 capability and descriptor-heap investigation. On that adapter and driver, a shader-visible view heap allocation of 1,000,000 descriptors succeeded, while 1,000,001 was rejected with `E_INVALIDARG`. The adapter reported Feature Level 12_2, Resource Binding Tier 3, Shader Model 6.8, and Mesh Shader Tier 1.0.

This is a hardware/API probe, not proof that the full game renderer works on the GPU. This README intentionally makes no claim about a verified playable frame rate or a completed end-to-end gameplay session.

**Latest local verification (2026-11-26):** on the current pulled `main` revision, the user reported `cargo test --workspace --locked` passing (546 passed, 7 ignored), `cargo clippy --workspace --all-targets --locked -- -D warnings` passing with no warnings, and `cargo build --release --workspace --locked` completing successfully. These checks verify Rust compilation, linting, and unit tests; they do not establish that the game launches or renders correctly.

### Descriptor work implemented

The latest published implementation adds groundwork for handling a title descriptor heap that is larger than the host's accepted shader-visible heap:

- A bounded descriptor-residency bookkeeper tracks logical-to-physical slots, invalidation, replacement, and mapping snapshots.
- The D3D12 host heap can copy a title descriptor from CPU staging storage into a slot in a shared physical pool.
- A mapping is published only after the descriptor copy succeeds.
- Descriptor updates and copies invalidate affected resident mappings.
- Mapping snapshots retain the referenced resources and physical slots across recorded GPU work.
- Direct SM 6.6 `ResourceDescriptorHeap` accesses on a capped host are translated through a GPU-visible logical-to-physical remapping table.
- The remapping table is uploaded per command list, refreshed when the residency generation changes, and backed by snapshots that pin referenced physical slots through submission.
- The shader clamps logical indices to the table bounds, resolves resident entries to physical descriptors, and sends unmapped entries to a dedicated null slot.
- Draw preparation attempts to populate residency for high descriptors discovered through heap-wide buffer/texture preparation and descriptors selected from relocated root tables.
- Root descriptor-table relocation remains a separate mechanism from direct global-heap indexing.

**Important remaining limitation:** this is an implemented code path, not yet a verified end-to-end rendering result. The current unit tests and Windows release build do not prove that every descriptor type, shader access pattern, resource lifetime, or GPU synchronization case works in the retail game. Unmapped direct-heap accesses currently share a generic null view descriptor, so type-specific null behaviour may still need work. Keep the runtime's fail-closed checks for shaders that were translated without remapping metadata.

### Other recent graphics/runtime changes

The latest implementation commit also includes:

- Normalization of blend state for integer render targets, which D3D12 does not permit to use blending.
- A translation-cache identity that includes the active view-heap limit.
- Changes to raw title-memory fault tracking and retirement of mapped resources.
- A guard against 32-bit title allocations occupying the Windows zero page.

These are implementation changes; they should not be read as proof that all affected rendering paths have been validated in a complete game run.

## Project direction

The goal is not simply to make the game start. The goal is to make it render correctly, behave reliably, and perform as well as the underlying compatibility work allows.

Development priorities:

1. **Descriptor and resource correctness** — validate the new logical-to-physical mapping path on the GPU and in the retail game; address descriptor typing, residency coverage, resource lifetime, and synchronization gaps found by runtime testing.
2. **Shader translation** — improve coverage and correctness for the console's shader instructions and execution model.
3. **Lighting and image quality** — bring up the title's ray-traced lighting, shadows, reflections, and related effects.
4. **Runtime compatibility** — implement missing console interfaces and correct memory, file, and system behaviour.
5. **Performance and stability** — measure CPU, GPU, memory, and synchronization costs; optimize bottlenecks without needlessly reducing visual quality.

Visual quality and performance are both important. Performance changes should be measured against repeatable scenes and should not silently trade away important rendering features.

## Requirements

- Windows 10/11, 64-bit, with an AVX2-capable CPU.
- A Direct3D 12 GPU supporting Shader Model 6.6 and mesh shaders (NVIDIA RTX 20 series or newer).
- Rust, using the version specified in `rust-toolchain.toml`.
- Several GB of free Windows commit memory. If `crashes.log` reports error 1455, increase the Windows paging file.

## Building

From the repository root:

```powershell
cargo build --release -p victory
```

The release executable is produced under:

```text
target/release/victory.exe
```

Use the toolchain pinned by `rust-toolchain.toml`.

## Running

The repository includes a PowerShell launcher:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\play.ps1 -Game "<your game folder>\<the game's exe>"
```

The launcher supports `-Cache` and `-Data` to choose where the shader cache and save data are stored. It does not press buttons for you.

The game files must be supplied by the user and remain outside this repository. Keep the executable inside its complete game package so the runtime can find the accompanying resources. Lucia2PC does not include or distribute game files.

### Controller

XInput controllers are supported. A DualShock or DualSense controller can be used through DS4Windows. When no controller is connected, the keyboard acts as a controller while the game window has focus.

## Performance settings

These optional settings are off by default. Their effect depends on the system and workload:

```powershell
$env:VICTORY_RELAY = "2"
```

Or:

```powershell
$env:VICTORY_LATE_WAIT = "3"
$env:VICTORY_WIDE_BACK = "1"
```

Set environment variables in the PowerShell session before launching the game.

## Debugging

| Switch | Effect |
|---|---|
| `VICTORY_CAPTURE=<dir>` and `VICTORY_CAPTURE_FRAMES=<list>` | Save selected frames as PNG (diagnostics build) |
| `VICTORY_KEYS=30:Enter,35:X` | Press keys at the specified seconds (diagnostics build) |
| `VICTORY_D3D12_DRED=1` | Report commands in flight after a GPU failure |
| `VICTORY_PIPELINE_WAIT_MS=24` | Use a shorter per-frame wait for pipelines still being built |

Build with diagnostics enabled:

```powershell
cargo build --profile diag -p victory --features diagnostics
```

## Repository overview

Lucia2PC retains the existing Rust codebase and its technical component names. Those names are identifiers used by the source tree and build system.

| Area | Responsibility |
|---|---|
| `victory-gpu` | Host GPU resources, graphics commands, descriptors, and pipelines |
| `victory-shader` | Shader translation and related tests |
| Other workspace crates | Runtime, compatibility, formats, and supporting functionality |

For the detailed component layout and runtime flow, see [Architecture](docs/architecture.md).

## Graphics development

Graphics work is guided by a few principles:

- Diagnose the source of a rendering failure before applying a visual workaround.
- Treat shader translation, descriptor handling, resource lifetime, pipeline creation, and synchronization as connected parts of the rendering path.
- Compare the same repeatable in-game scenes when evaluating visual changes.
- Record frame time and relevant CPU/GPU measurements; do not rely on a single FPS number.
- Preserve visual quality when optimizing unless a deliberate trade-off is documented.

See [Ray-tracing development](docs/fork-ray-tracing.md) for the implementation status, test coverage, and known gaps.

## Assets and project hygiene

Game executables, proprietary assets, extracted game data, private keys, account information, and personal crash dumps do not belong in this repository.

See [Asset policy](docs/asset-policy.md) before adding, sharing, or referencing game-related files.

## Contributing and reporting issues

When reporting a problem, include as much of the following as can safely be shared:

- Lucia2PC commit or build being tested.
- Windows version, CPU, GPU, driver version, and available memory.
- The exact launch method and whether the complete game package was used.
- The scene or action where the issue occurs.
- Relevant emulator logs and a concise description of expected versus actual behaviour.

Do not attach proprietary game files or sensitive account data to public issues.

## Credits, origin, and licensing

Lucia2PC is an independent project with its own name and repository identity. The source tree retains technical component names and upstream work from the existing codebase, including work from the original VIctory project (Marvisak), victory-fork (fxceless), and other contributors. This repository does not claim authorship over upstream work.

The original project identifies its licence as **GPL-3.0-or-later**; see [`LICENSE`](LICENSE). Preserve copyright notices, attribution, and licence terms from the original project and its dependencies. The repository licence does not grant rights to proprietary game files or third-party assets.
