---
description: How to run LTspice simulations from Python in this project's WSL environment (e.g. for parameter fitting against LTspice models). Use whenever writing or debugging code that invokes LTspice, PyLTSpice, or spicelib, or that reads .raw simulation output, in this repo.
---

# Running LTspice from WSL in this project

LTspice XVII lives at the native Windows path:
`/mnt/c/Program Files/LTC/LTspiceXVII/XVIIx64.exe`
It's invoked directly from WSL via binfmt_misc interop (no wine — this is real Windows LTspice).

## Don't use PyLTSpice's AscEditor / SimRunner for running sims here

The installed `spicelib`/`PyLTSpice` versions don't work cleanly in this environment:

- `AscEditor(path)` asserts the schematic's `Version` header is exactly `"4"`. LTspice XVII always
  writes `"Version 4.1"`, so opening any real `.asc` throws `AssertionError`.
- Even after patching the version string, `AscEditor` needs the LTspice symbol library to resolve
  components (`cap.asy`, `res.asy`, ...). `get_default_library_paths()` returns `[]` on Linux/WSL
  (it only auto-detects on native Windows) — you'd have to call
  `AscEditor.set_custom_library_paths('/mnt/c/Program Files/LTC/LTspiceXVII/lib/sym')`.
- This spicelib version's `AscEditor` doesn't even have `save_as()` or `get_all_parameter_names()`
  — different API generation than a lot of example code assumes. It has `set_parameter`,
  `save_netlist`, `write_netlist` instead.
- `SimRunner.run_now()` silently fails (raw/log "not found") in this environment, even with a
  correct `simulator=` path override. Root cause not fully diagnosed — `LTspice.get_default_library_paths`
  / the simulator-detection logic auto-prepends `["wine", exe_path]` on `sys.platform == "linux"`,
  which plausibly fights any override.

## What actually works

**Editing parameters:** copy the `.asc` (and any `.inc` it includes) into a scratch directory and
do plain text/regex substitution of `.param X=...` lines yourself. Don't rely on `AscEditor.set_parameter`
for anything that lives inside an `.include`d file — it can't reach those at all.

**Running the simulation:** plain subprocess, no PyLTSpice wrapper for the run step:

```python
import subprocess
subprocess.run(
    ["/mnt/c/Program Files/LTC/LTspiceXVII/XVIIx64.exe", "-b", "-Run", "relative_name.asc"],
    cwd=scratch_dir,   # MUST be under /mnt/c/... (see below)
    timeout=120,
)
```

- Use a **relative path** to the `.asc`, with `cwd` set to its directory. This is the pattern
  confirmed to work reliably.
- The scratch directory **must be under `/mnt/c/...`**, not a pure-Linux path like `/tmp`. The
  native `XVIIx64.exe` process can't reliably read/write a pure-Linux path via WSL interop —
  confirmed it silently produces no `.raw`/`.log` there.

**Reading results:** `PyLTSpice.RawRead` (pure Python `.raw` parser, no subprocess involved) works
fine regardless of how the sim was launched:

```python
from PyLTSpice import RawRead
raw = RawRead(f"{scratch_dir}/relative_name.raw")
t = raw.get_trace("time").get_wave()
i = raw.get_trace("I(R1)").get_wave()
```

## Diagnostics / gotchas worth knowing

- LTspice's own `.log` ("Total elapsed time", `totiter`/`tranpoints`) is a reliable,
  environment-independent timing/iteration count — prefer it over wall-clock `time` when comparing
  simulation cost, since WSL process-spawn overhead is real but small (~1s) next to actual solve
  time for stiff circuits.
- If a schematic uses `delay()`-based behavioral sources to break an algebraic feedback loop (common
  in this project's multi-stage PMT dynode models), a very small `tdelay` combined with
  `.options method=gear` can force millions of tiny transient timesteps and make batch runs take
  30-40s+ even though the same file might run near-instantly through the LTspice GUI on the same
  machine (observed, unresolved discrepancy — not caused by `.save` directives or thread count,
  which were checked and matched). If you hit this, don't assume it's a WSL artifact by default —
  first check whether the schematic is this kind of stiff feedback-loop circuit; a plain
  non-stiff schematic runs in ~0.1s through the identical batch path.
- Changing `tdelay` to speed things up trades off against the fidelity of transient dynamics
  (verified: loosening `tdelay` 10x changed one integrated-charge test value by ~0.4%, but that
  does not guarantee the full current-vs-time *shape* is preserved) — don't do this without
  explicit sign-off, and validate against the full waveform, not just an integrated total, if you do.
