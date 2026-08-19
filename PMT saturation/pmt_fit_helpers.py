"""
Fit helpers for rewiring model_fitting.ipynb onto pmt_model_v4.inc.

Runs pmt_saturation_fit_v4.asc (which .includes pmt_model_v4.inc) through LTspice batch
mode for a given V1/N_pe/parameter set, and extracts the integrated anode charge. Parameter
injection is done by regex-patching text copies of the .asc/.inc rather than AscEditor, which
doesn't work reliably in this environment (see reference-ltspice-wsl-runner memory).
"""
import re
import subprocess
from pathlib import Path

import numpy as np
from PyLTSpice import RawRead

LTSPICE = "/mnt/c/Users/bglen/AppData/Local/Programs/ADI/LTspice/LTspice.exe"

PROJECT_DIR = Path(__file__).resolve().parent
BASE_ASC = PROJECT_DIR / "pmt_saturation_fit_v4.asc"
BASE_INC = PROJECT_DIR / "pmt_model_v4.inc"
SCRATCH_DIR = PROJECT_DIR / "fit_scratch"

T0, T1 = 10e-6, 20e-6  # anode charge integration window (matches PULSE delay=10u, Ton=10u)
ELECTRON_CHARGE = 1.6e-19

SIM_TIMEOUT = 60  # seconds; kill any single LTspice batch run that hangs rather than
                  # blocking the whole optimizer indefinitely
LOSS_PENALTY = 1e6  # returned by stage*_loss when a candidate point fails/times out,
                    # well above any real loss value seen so far, so the optimizer
                    # steers away from that region instead of crashing


class SimTimeoutError(RuntimeError):
    """Raised when a single LTspice batch run exceeds SIM_TIMEOUT."""


def _fmt(value):
    return f"{value:.10g}" if isinstance(value, float) else str(value)


_SPICE_SUFFIX = {
    "f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3,
    "k": 1e3, "meg": 1e6, "g": 1e9, "t": 1e12,
}


def get_param(text, name):
    """Read a `.param NAME = VALUE` value out of .inc/.asc text as a float, resolving
    SPICE unit suffixes (150m -> 0.15, 1Meg -> 1e6, etc)."""
    m = re.search(r"\.param\s+" + re.escape(name) + r"(?!\w)\s*=\s*([^\s;\\]+)", text)
    if not m:
        raise ValueError(f"Parameter {name!r} not found")
    num_match = re.match(r"^([+-]?[\d.]+(?:[eE][+-]?\d+)?)([a-zA-Z]*)$", m.group(1))
    if not num_match:
        raise ValueError(f"Can't parse SPICE value {m.group(1)!r} for {name!r}")
    number, suffix = num_match.groups()
    return float(number) * (_SPICE_SUFFIX.get(suffix.lower(), 1.0) if suffix else 1.0)


def set_param(text, name, value):
    """Regex-substitute a `.param NAME = VALUE` token, whether it's a bare .inc line or
    embedded in a `!.param ...` directive inside an .asc TEXT line. Raises if the
    parameter isn't found exactly once, so silent typos/renames can't inject nothing."""
    pattern = re.compile(r"(\.param\s+" + re.escape(name) + r")(?!\w)(\s*=\s*)([^\s;\\]+)")
    new_text, n = pattern.subn(lambda m: m.group(1) + m.group(2) + _fmt(value), text)
    if n != 1:
        raise ValueError(f"Parameter {name!r} matched {n} times in text (expected 1)")
    return new_text


def set_voltage_source(text, inst_name, value):
    """Regex-substitute the SYMATTR Value line immediately following
    `SYMATTR InstName <inst_name>` for a plain DC voltage source."""
    pattern = re.compile(
        r"(SYMATTR InstName " + re.escape(inst_name) + r"\r?\nSYMATTR Value )([^\r\n]+)"
    )
    new_text, n = pattern.subn(lambda m: m.group(1) + _fmt(value), text)
    if n != 1:
        raise ValueError(f"Voltage source {inst_name!r} matched {n} times in text (expected 1)")
    return new_text


def read_anode_charge(raw_path):
    """Integrated I(R1) over the pulse window, in Coulombs. Signed (I(R1) integrates
    negative for the anode sense-resistor's current convention)."""
    raw = RawRead(str(raw_path))
    t = raw.get_trace("time").get_wave()
    i_r1 = raw.get_trace("I(R1)").get_wave()
    mask = (t >= T0) & (t <= T1)
    return float(np.trapz(i_r1[mask], t[mask]))


def gain_from_charge(Q, n_pe):
    return abs(Q) / (n_pe * ELECTRON_CHARGE)


def peak_abs_trace(raw_path, trace_name, window=None):
    """Peak absolute value of an arbitrary trace, optionally restricted to a time window."""
    raw = RawRead(str(raw_path))
    t = raw.get_trace("time").get_wave()
    y = np.abs(raw.get_trace(trace_name).get_wave())
    if window is not None:
        w0, w1 = window
        mask = (t >= w0) & (t <= w1)
        y = y[mask]
    return float(np.max(y))


def add_save_nodes(text, extra_nodes):
    """Append extra traces to the schematic's `!.save V(ANODE) I(R1)` directive so a
    one-off diagnostic run can read more than the fit loop's minimal default set."""
    pattern = re.compile(r"(!\.save\s+[^\r\n]+)")
    new_text, n = pattern.subn(lambda m: m.group(1) + " " + " ".join(extra_nodes), text)
    if n != 1:
        raise ValueError(f"'.save' directive matched {n} times in text (expected 1)")
    return new_text


def _execute(*, v1, n_pe, inc_params=None, extra_save_nodes=None, scratch_dir=SCRATCH_DIR, run_name="run"):
    """Patch scratch copies of the base .asc/.inc with the given V1, N_pe, and any
    pmt_model_v4.inc parameter overrides, run through LTspice batch mode, and return
    the resulting .raw path for the caller to read whichever traces it needs."""
    scratch_dir = Path(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    asc_text = BASE_ASC.read_text()
    asc_text = set_voltage_source(asc_text, "V1", v1)
    asc_text = set_param(asc_text, "N_pe", n_pe)
    if extra_save_nodes:
        asc_text = add_save_nodes(asc_text, extra_save_nodes)
    asc_name = f"{run_name}.asc"
    (scratch_dir / asc_name).write_text(asc_text)

    inc_text = BASE_INC.read_text()
    for name, value in (inc_params or {}).items():
        inc_text = set_param(inc_text, name, value)
    (scratch_dir / BASE_INC.name).write_text(inc_text)

    for ext in (".log", ".raw", ".op.raw", ".net"):
        stale = scratch_dir / f"{run_name}{ext}"
        if stale.exists():
            stale.unlink()

    try:
        result = subprocess.run(
            [LTSPICE, "-b", "-Run", asc_name],
            cwd=str(scratch_dir),
            capture_output=True,
            text=True,
            timeout=SIM_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        raise SimTimeoutError(
            f"LTspice run {run_name!r} exceeded {SIM_TIMEOUT}s timeout (v1={v1}, n_pe={n_pe})"
        ) from exc
    raw_path = scratch_dir / f"{run_name}.raw"
    if result.returncode != 0 or not raw_path.exists():
        raise RuntimeError(
            f"LTspice run failed (rc={result.returncode}).\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return raw_path


def run_sim(*, v1, n_pe, inc_params=None, scratch_dir=SCRATCH_DIR, run_name="run"):
    """Same as _execute, but returns the integrated anode charge (Coulombs, signed)
    directly -- the common case for fit-loop evaluations."""
    raw_path = _execute(v1=v1, n_pe=n_pe, inc_params=inc_params, scratch_dir=scratch_dir, run_name=run_name)
    return read_anode_charge(raw_path)


def simulate_gain(v1, n_pe, inc_params=None, scratch_dir=SCRATCH_DIR, run_name="run"):
    Q = run_sim(v1=v1, n_pe=n_pe, inc_params=inc_params, scratch_dir=scratch_dir, run_name=run_name)
    return gain_from_charge(Q, n_pe)


# --- Stage 1: fit k, a against gain-vs-HV data (run at low N_pe to stay in the linear regime) ---

def stage1_loss(params, target_voltage, target_gain, n_pe=10, scratch_dir=SCRATCH_DIR):
    k, a = params
    sim_gains = []
    for v1 in target_voltage:
        try:
            sim_gains.append(
                simulate_gain(v1, n_pe, inc_params={"k": k, "a": a}, scratch_dir=scratch_dir))
        except (SimTimeoutError, RuntimeError) as exc:
            print(f"stage1_loss: sim failed for v1={v1}, k={k}, a={a}: {exc}")
            return LOSS_PENALTY
    sim_gains = np.array(sim_gains)
    error = np.sum((np.log10(target_gain) - np.log10(sim_gains)) ** 2)
    return error


# --- Stage 2: fit saturation params against the PandaX Fig 8 charge-linearity curve ---

def stage2_loss(params, k, a, n_pe_values, target_charge_kpe, v1=1060, scratch_dir=SCRATCH_DIR):
    Vknee, Vslope, Isc8, Isc9, Isc10, m = params
    inc_params = {
        "k": k, "a": a,
        "Vknee": Vknee, "Vslope": Vslope,
        "Isc8": Isc8, "Isc9": Isc9, "Isc10": Isc10, "m": m,
    }
    try:
        gain_ref = simulate_gain(v1, 10, inc_params=inc_params, scratch_dir=scratch_dir, run_name="ref")
    except (SimTimeoutError, RuntimeError) as exc:
        print(f"stage2_loss: reference gain sim failed for params={params}: {exc}")
        return LOSS_PENALTY

    n_pe_obs = []
    for n_pe in n_pe_values:
        try:
            n_pe_obs.append(
                abs(run_sim(v1=v1, n_pe=n_pe, inc_params=inc_params, scratch_dir=scratch_dir, run_name="pt"))
                / (ELECTRON_CHARGE * gain_ref))
        except (SimTimeoutError, RuntimeError) as exc:
            print(f"stage2_loss: sim failed for n_pe={n_pe}, params={params}: {exc}")
            return LOSS_PENALTY
    n_pe_obs = np.array(n_pe_obs)
    error = np.sum((n_pe_obs / 1e3 - target_charge_kpe) ** 2)
    return error
