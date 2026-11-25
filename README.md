# Lucia2PC

**An experimental Xbox-to-Windows compatibility and graphics project.**

Lucia2PC is an independent development project built on the ZTRX codebase. It
explores the compatibility work required to run a user-provided Xbox Series
game build on a Windows PC, with particular attention to graphics translation,
visual accuracy, stability, and performance.

Lucia2PC is unofficial and is not affiliated with, endorsed by, or sponsored
by the game's developers, publisher, or Microsoft.

## Current status — 2026-11-25

The following results were recorded on Windows 11 with an **NVIDIA GeForce
RTX 3050 (8 GB)**:

| Area | Recorded result |
|---|---|
| Boot and introduction | Boot, first-boot screens, intro and opening credits work |
| Prologue | Playable from the cell to the boat chase, including the cell, cafeteria, robbery, police chase, wanted level, minimap, boat and binoculars |
| Stability | More than 1 hour 30 minutes in one session without a crash |
| Frame rate | Approximately 1–3 FPS in the world; the first run is slower while shaders translate, then translated shaders are cached |
| Known problems | Shadowed areas are black and bright scenes overexpose because the game's ray-traced lighting does not run yet; some blocky artifacts; boats sit low in the water |

These are recorded test results, not a guarantee of compatibility on other
hardware, drivers, game builds, or scenes. The project remains experimental.

## Project direction

The goal is not simply to make the game start. The goal is to make it render
correctly, behave reliably, and perform as well as the underlying compatibility
work allows.

Development priorities:

1. **Lighting and image quality** — improve lighting, shadows, reflections,
   ambient occlusion, and other features that define the scene.
2. **Rendering correctness** — improve geometry, materials, textures,
   characters, animation, and visual effects.
3. **Compatibility** — implement missing console interfaces and graphics
   behaviour, and correct resource and shader handling.
4. **Performance and stability** — measure CPU, GPU, memory, and
   synchronization costs; optimize bottlenecks without needlessly reducing
   visual quality.
5. **Future extensibility** — establish a dependable runtime before building
   higher-level mod support.

Visual quality and performance are both important. Performance changes should
be measured against repeatable scenes and should not silently trade away
important rendering features.

## Requirements

- Windows 10/11, 64-bit, with AVX2-capable CPU.
- A Direct3D 12 GPU supporting Shader Model 6.6 and mesh shaders
  (NVIDIA RTX 20 series or newer).
- Rust, using the version specified in `rust-toolchain.toml`.
- Several GB of free Windows commit memory. If `crashes.log` reports error
  1455, increase the Windows paging file.

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

The launcher supports `-Cache` and `-Data` to choose where the shader cache
and save data are stored. It does not press buttons for you.

The game files must be supplied by the user and remain outside this repository.
Keep the executable inside its complete game package so the runtime can find
the accompanying resources. Lucia2PC does not include or distribute game
files.

### Controller

XInput controllers are supported. A DualShock or DualSense controller can be
used through DS4Windows. When no controller is connected, the keyboard acts
as a controller while the game window has focus.

## Performance settings

These optional settings are off by default. They were reported to reduce frame
time on the original test machines; results may vary:

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

Lucia2PC retains the existing Rust codebase and its technical component names.
Those names are identifiers used by the source tree and build system.

| Area | Responsibility |
|---|---|
| `victory-gpu` | Host GPU resources, graphics commands, descriptors, and pipelines |
| `victory-shader` | Shader translation and related tests |
| Other workspace crates | Runtime, compatibility, formats, and supporting functionality |

For the detailed component layout and runtime flow, see
[Architecture](docs/architecture.md).

## Graphics development

Graphics work is guided by a few principles:

- Diagnose the source of a rendering failure before applying a visual
  workaround.
- Treat shader translation, descriptor handling, resource lifetime, pipeline
  creation, and synchronization as connected parts of the rendering path.
- Compare the same repeatable in-game scenes when evaluating visual changes.
- Record frame time and relevant CPU/GPU measurements; do not rely on a single
  FPS number.
- Preserve visual quality when optimizing unless a deliberate trade-off is
  documented.

See [Ray-tracing development](docs/fork-ray-tracing.md) for the current
implementation, test results, and known gaps.

## Assets and project hygiene

Game executables, proprietary assets, extracted game data, private keys,
account information, and personal crash dumps do not belong in this repository.

See [Asset policy](docs/asset-policy.md) before adding, sharing, or referencing
game-related files.

## Contributing and reporting issues

When reporting a problem, include as much of the following as can safely be
shared:

- Lucia2PC commit or build being tested.
- Windows version, CPU, GPU, driver version, and available memory.
- The exact launch method and whether the complete game package was used.
- The scene or action where the issue occurs.
- Relevant emulator logs and a concise description of expected versus actual
  behaviour.

Do not attach proprietary game files or sensitive account data to public
issues.

## Credits, origin, and licensing

Lucia2PC is based on the ZTRX codebase and retains work from the original
VIctory project (Marvisak) and victory-fork (fxceless), as well as other
upstream authors and contributors. This repository rename does not claim
authorship over upstream work.

The original project identifies its licence as **GPL-3.0-or-later**; see
[`LICENSE`](LICENSE). Preserve copyright notices, attribution, and licence
terms from the original project and its dependencies. The repository licence
does not grant rights to proprietary game files or third-party assets.
