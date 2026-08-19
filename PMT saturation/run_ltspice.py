#!/usr/bin/env python3
"""
Run an LTspice .asc schematic in batch mode and report timing.

Copies the schematic and any locally-included .inc files into a scratch dir
(must be under /mnt/c/... -- native LTspice.exe can't reliably read/write
pure-Linux paths over WSL interop), runs LTspice batch mode there, and prints
wall-clock time plus the LTspice .log contents (which includes solver-reported
"Total elapsed time" and, for stiff transients, totiter/tranpoints/accept/
rejected counts).

Usage:
    python3 run_ltspice.py pmt_saturation_fit_v4.asc
    python3 run_ltspice.py pmt_saturation_fit_v4.asc --scratch-dir perf_test_scratch
"""
import argparse
import re
import shutil
import subprocess
import time
from pathlib import Path

# LTspice 24 (current default engine). LTspice 17 also exists on this machine at
# "/mnt/c/Program Files/LTC/LTspiceXVII/XVIIx64.exe" -- only needed for specific legacy
# compatibility cases (e.g. some TI opamp SPICE models only work under v17), not this project.
LTSPICE = "/mnt/c/Users/bglen/AppData/Local/Programs/ADI/LTspice/LTspice.exe"
INCLUDE_RE = re.compile(r"^\s*!\.include\s+(\S+)", re.IGNORECASE | re.MULTILINE)


def find_local_includes(asc_path):
    text = asc_path.read_text(errors="replace")
    return [asc_path.parent / name for name in INCLUDE_RE.findall(text)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asc_file", type=Path, help="Path to the .asc schematic to run")
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=None,
        help="Scratch dir to run in (default: <asc_dir>/ltspice_run_scratch). "
        "Must resolve to a path under /mnt/c/...",
    )
    args = parser.parse_args()

    asc_src = args.asc_file.resolve()
    if not asc_src.exists():
        raise SystemExit(f"No such file: {asc_src}")

    scratch = (args.scratch_dir or asc_src.parent / "ltspice_run_scratch").resolve()
    scratch.mkdir(parents=True, exist_ok=True)

    asc_dst = scratch / asc_src.name
    shutil.copy(asc_src, asc_dst)
    for inc_src in find_local_includes(asc_src):
        if inc_src.exists():
            shutil.copy(inc_src, scratch / inc_src.name)

    # clear stale outputs from a previous run in this scratch dir
    for ext in (".log", ".raw", ".op.raw", ".net"):
        stale = asc_dst.with_suffix(ext)
        if stale.exists():
            stale.unlink()

    t0 = time.time()
    result = subprocess.run(
        [LTSPICE, "-b", "-Run", asc_dst.name],
        cwd=str(scratch),
        capture_output=True,
        text=True,
    )
    wall_clock = time.time() - t0

    print(f"subprocess return code: {result.returncode}")
    if result.stdout.strip():
        print("stdout:", result.stdout)
    if result.stderr.strip():
        print("stderr:", result.stderr)
    print(f"wall-clock time: {wall_clock:.2f} s")

    log_path = asc_dst.with_suffix(".log")
    if log_path.exists():
        print("\n--- LTspice .log ---")
        print(log_path.read_text(errors="replace"))
    else:
        print(f"\nNo .log produced at {log_path}")


if __name__ == "__main__":
    main()
