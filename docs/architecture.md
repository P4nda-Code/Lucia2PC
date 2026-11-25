# Lucia2PC architecture

Lucia2PC runs a locally supplied Xbox title on Windows by translating and
implementing the console-facing parts of its runtime. The game executable and
game data remain outside the repository; Lucia2PC builds and uses its own
runtime components and caches converted data on the user's machine.

## Runtime overview

```text
User-supplied files (outside the repository)
  ├── game assets ──> victory-pipeline / victory-game
  │                    └── victory-formats / victory-shader
  │                          └── converted-data cache
  │
  └── game executable ──> victory-loader ──> native x86-64 title code
                              │                    │
                              │                    └── imported APIs
                              │                           │
                              └──────────────────────────> victory-hle
                                                           │
                                                           └── Windows host APIs

Title GPU work ──> victory-gpu ──> host D3D12
                       │
                       ├── D3D12.X objects and interfaces
                       │       └── exposed through victory-hle
                       │
                       └── shader creation ──> victory-shader
                                                   │
                                                   └── shader cache
```

Shaders are translated when the title creates a pipeline, using the shader
bytes supplied at that point, regardless of whether the title loaded them from
its executable, an archive, or memory. Translating shader files ahead of time
is useful as a warm-up, but it does not replace translation during pipeline
creation.

## Main components

| Component | Responsibility |
|---|---|
| `victory-core` | Configuration, asset-policy guard and logging |
| `victory-pe` | Read-only PE32+ and import-library analysis |
| `victory-loader` | Prepare, map, bind and start the title image |
| `victory-hle` | Console API implementations and host pass-through policy |
| `victory-gpu` | D3D12.X objects, command lists and GPU memory mapped to host D3D12 |
| `victory-shader` | Xbox shader containers translated to host shaders |
| `victory-formats` | GPU texture layouts, sizes, tiling and detiling; audio codecs; BCPack support is planned |
| `victory-pipeline` | Converter interface and content-addressed cache |
| `victory-game` | Title-specific formats, build identities and hooks |
| `apps/victory` | Command-line front end |
| `xtask` | Asset-policy checks and Git hooks |
| `tests/fixtures/hle-probe` | A test-only stand-in title made from project-owned code |

Everything outside `victory-game` is intended to be platform-level and
testable with other Xbox Series titles as well.

## Import resolution

For each imported `dll!function`, Lucia2PC resolves the function in this order:

1. An implementation in `victory-hle::modules` takes priority.
2. The Windows host function is used if `HostPolicy` allows that DLL.
3. An export from a DLL shipped beside the title executable is used when
   neither of the first two options provides the function. The DLL is mapped
   like the executable.
4. If no implementation is available, a generated stub identifies the import
   when called. With `--unimplemented trap`, execution breaks into the
   debugger and aborts; with `zero`, the stub logs once and returns `0`.

`LoadLibrary*` calls for console-only DLLs implemented by Lucia2PC (for
example, `xgameruntime.dll`) return pseudo-handles. `GetProcAddress` on those
handles resolves to the project's implementations.

When a title DLL is loaded, Lucia2PC maps its image if it has not already been
mapped. Host DLLs are exposed only when permitted by `HostPolicy`, and
`GetProcAddress` returns the project's override wherever import binding would
do so.

`QueryApiImpl` provides runtime objects by provider and interface ID. The
implemented areas are listed under `victory-hle::modules::xgameruntime`.
Online-facing areas behave as an offline console. Game saves are stored as
files beneath `VICTORY_DATA_DIR`.

## File paths and title storage

A title treats storage as drives on its console and constructs paths using
those drive names. `victory-hle::modules::files` handles path-taking imports
from the Windows kernel and C/C++ runtimes, translating paths on virtual
drives to the host directories behind them.

Persistent local storage is one such virtual drive backed by a directory
under `VICTORY_DATA_DIR`. Paths that do not refer to a virtual drive pass
through to the host unchanged.

## Loader support

An executable's static TLS is accessed through slot 0. When that image is
mapped as a DLL, it receives another slot, so `victory-loader::tls` redirects
the relevant accesses.

Running `victory run --execute` installs a crash reporter. It identifies
fault locations as `module+rva` and unwinds the faulting thread's stack
through `victory-loader::crash`.

`victory census` reports import classification without running the title. It
does not map title DLLs; `victory run` without `--execute` does.

## Shader translation

`victory_shader::ir::build` converts a native shader program into SSA, with
one instruction per machine instruction. `victory_shader::structure` then
removes EXEC-dependent control flow:

- Lane masks become per-lane booleans.
- Wave-wide mask tests become per-lane branches.
- Vector instructions retain a guard only when a lane can reach them with
  its EXEC bit clear.

`victory_shader::resource` traces each descriptor and scalar memory address
back to its root parameter, such as a table, heap, root descriptor or
constant.

For descriptor access inside a waterfall loop, the title may read the
descriptor from the first lane with `v_readfirstlane_b32`. The translator
tracks the per-lane load or choice that produced the descriptor, including
`v_cndmask_b32` selections and values assigned along if/else paths. It can
then access the descriptor using a non-uniform heap index in one pass.

`victory translate` processes the shader inputs it finds, including containers
and SGD5 files, and reports constructs it cannot yet handle.
