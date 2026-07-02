import os
import re
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


def ingest_nu_global_fit_data(path: str = "nu_global_fit_data.txt") -> Dict[str, object]:
    """Parse the global-fit table and return central values with asymmetric 1-sigma errors."""
    line_pattern = re.compile(
        r"^(?P<parameter>.*?)\t+(?P<bestfit>.*?)\t+(?P<sigma2>.*?)\t+(?P<sigma3>.*?)$"
    )
    symmetric_pattern = re.compile(
        r"^(?P<central>[-+]?\d*\.?\d+)(?:\\pm|±)(?P<error>[-+]?\d*\.?\d+)$"
    )
    asymmetric_pattern = re.compile(
        r"^(?P<central>[-+]?\d*\.?\d+)\^\{\+(?P<plus>[-+]?\d*\.?\d+)\}_\{-(?P<minus>[-+]?\d*\.?\d+)\}$"
    )

    parameters = []
    central_values = []
    err_minus = []
    err_plus = []
    sigma2_ranges = []
    sigma3_ranges = []
    raw_rows = []

    data_path = os.path.expanduser(path)
    with open(data_path, encoding="utf-8") as infile:
        lines = infile.read().splitlines()

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("parameter"):
            continue

        match = line_pattern.match(line)
        if not match:
            continue

        best_fit = match.group("bestfit").strip()
        symmetric = symmetric_pattern.match(best_fit)
        if symmetric:
            central = float(symmetric.group("central"))
            minus = plus = float(symmetric.group("error"))
        else:
            asymmetric = asymmetric_pattern.match(best_fit)
            if not asymmetric:
                raise ValueError(f"Unrecognized best-fit format: {best_fit}")
            central = float(asymmetric.group("central"))
            plus = float(asymmetric.group("plus"))
            minus = float(asymmetric.group("minus"))

        parameter = match.group("parameter").strip()
        sigma2 = match.group("sigma2").strip()
        sigma3 = match.group("sigma3").strip()

        parameters.append(parameter)
        central_values.append(central)
        err_minus.append(minus)
        err_plus.append(plus)
        sigma2_ranges.append(sigma2)
        sigma3_ranges.append(sigma3)
        raw_rows.append(
            {
                "parameter": parameter,
                "central": central,
                "err_minus": minus,
                "err_plus": plus,
                "sigma2_range": sigma2,
                "sigma3_range": sigma3,
            }
        )

    return {
        "parameter": parameters,
        "central": np.array(central_values),
        "err_minus": np.array(err_minus),
        "err_plus": np.array(err_plus),
        "yerr": np.vstack([err_minus, err_plus]),
        "sigma2_range": sigma2_ranges,
        "sigma3_range": sigma3_ranges,
        "rows": raw_rows,
    }


def get_best_fit_and_errors(
    parameter_name: str,
    ordering: str = "NO",
    data_path: str = "./ahep_globalfit_chi2/1-dim/",
):
    datafile_name = os.path.join(data_path, f"{parameter_name}-{ordering}.dat")
    data = np.genfromtxt(datafile_name)

    indices = np.where(data[:, 1] <= 3)[0]
    x_fit = data[indices, 0]
    y_fit = data[indices, 1]
    p = np.polyfit(x_fit, y_fit, 2)

    a, b, c = p
    minimum = -0.5 * b / a
    offset = c - b**2 / (4 * a)
    coeffs = [a, b, c + offset - 1]
    roots = np.roots(coeffs)

    output_dict = {
        "Param": parameter_name,
        "BestFit": float(minimum),
        "ErrorMinus": float(np.min(roots)),
        "ErrorPlus": float(np.max(roots)),
    }
    return data, p, output_dict


def get_best_fit_dict(parameter_name: str, ordering: str = "NO", data_path: str = "./ahep_globalfit_chi2/1-dim/"):
    _, _, output_dict = get_best_fit_and_errors(parameter_name, ordering, data_path)
    return output_dict


def build_globalfit_tables(data_path: str = "./ahep_globalfit_chi2/1-dim/") -> Tuple[pd.DataFrame, pd.DataFrame]:
    files = os.listdir(data_path)
    parameters_no = set()
    parameters_io = set()

    for filename in files:
        if filename.endswith("-NO.dat"):
            parameters_no.add(filename.split("-")[0])
        elif filename.endswith("-IO.dat"):
            parameters_io.add(filename.split("-")[0])

    params_list_no = [get_best_fit_dict(param, ordering="NO", data_path=data_path) for param in sorted(parameters_no)]
    params_list_io = [get_best_fit_dict(param, ordering="IO", data_path=data_path) for param in sorted(parameters_io)]

    return pd.DataFrame(params_list_no), pd.DataFrame(params_list_io)


def params_dict_from_df(df: pd.DataFrame) -> Dict[str, float]:
    return dict(df[["Param", "BestFit"]].values)


def majorana_mass(
    params_dict: Dict[str, float],
    ordering: str = "NO",
    delta_a1: float = 0.0,
    delta_a2: float = 0.0,
    m_l=0.0,
):
    if ordering == "NO":
        m1 = m_l
        m2 = np.sqrt(m1**2 + params_dict["dm21"])
        m3 = np.sqrt(m2**2 + params_dict["dm31"])
    elif ordering == "IO":
        m3 = m_l
        m1 = np.sqrt(m3**2 + params_dict["dm31"])
        m2 = np.sqrt(m1**2 + params_dict["dm21"])
    else:
        raise ValueError("ordering must be 'NO' or 'IO'")

    theta12 = np.arcsin(np.sqrt(params_dict["sq12"]))
    theta13 = np.arcsin(np.sqrt(params_dict["sq13"]))

    ue1_sq = (np.cos(theta12) * np.cos(theta13)) ** 2
    ue2_sq = (np.sin(theta12) * np.cos(theta13)) ** 2
    ue3_sq = (np.sin(theta13)) ** 2

    return np.abs(ue1_sq * m1 + ue2_sq * m2 * np.exp(1j * delta_a1) + ue3_sq * m3 * np.exp(1j * delta_a2))


def one_sigma_ranges(df_ref: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    ranges = {}
    for _, row in df_ref.iterrows():
        low = min(row["ErrorMinus"], row["ErrorPlus"])
        high = max(row["ErrorMinus"], row["ErrorPlus"])
        ranges[row["Param"]] = (low, high)
    return ranges


def sample_params_1sigma(params_dict: Dict[str, float], df_ref: pd.DataFrame, rng=None) -> Dict[str, float]:
    if rng is None:
        rng = np.random.default_rng()

    one_sigma = one_sigma_ranges(df_ref)
    sampled = {}
    for param_name, best_fit in params_dict.items():
        if param_name in one_sigma:
            low, high = one_sigma[param_name]
            sampled[param_name] = rng.uniform(low, high)
        else:
            sampled[param_name] = best_fit
    return sampled


def sample_mbb_at_mlightest(
    m_l_value: float,
    params_dict: Dict[str, float],
    df_ref: pd.DataFrame,
    ordering: str,
    n_samples: int = 1000,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    mbb_samples = np.empty(n_samples)
    for i in range(n_samples):
        params_draw = sample_params_1sigma(params_dict, df_ref=df_ref, rng=rng)
        delta_a1 = rng.uniform(0.0, 2.0 * np.pi)
        delta_a2 = rng.uniform(0.0, 2.0 * np.pi)
        mbb_samples[i] = majorana_mass(
            params_draw,
            ordering=ordering,
            delta_a1=delta_a1,
            delta_a2=delta_a2,
            m_l=m_l_value,
        )
    return mbb_samples


def generate_mbb_density_samples(
    params_dict: Dict[str, float],
    df_ref: pd.DataFrame,
    ordering: str,
    m_l_grid=None,
    n_samples_per_m: int = 500,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()
    if m_l_grid is None:
        m_l_grid = np.logspace(-4.0, 0.0, 120)

    m_points = np.repeat(m_l_grid, n_samples_per_m)
    mbb_points = np.empty_like(m_points)

    start = 0
    for m_val in m_l_grid:
        draws = sample_mbb_at_mlightest(
            m_val,
            params_dict,
            df_ref=df_ref,
            ordering=ordering,
            n_samples=n_samples_per_m,
            rng=rng,
        )
        stop = start + n_samples_per_m
        mbb_points[start:stop] = draws
        start = stop

    return m_points, mbb_points


def save_sample_cache(cache_path: str, cache_payload: Dict[str, object]):
    with open(cache_path, "wb") as cache_file:
        import pickle

        pickle.dump(cache_payload, cache_file, protocol=pickle.HIGHEST_PROTOCOL)


def load_sample_cache(cache_path: str) -> Dict[str, object]:
    with open(cache_path, "rb") as cache_file:
        import pickle

        return pickle.load(cache_file)


def x_edges_from_grid(grid):
    logg = np.log10(grid)
    edges_log = np.empty(grid.size + 1)
    edges_log[1:-1] = 0.5 * (logg[:-1] + logg[1:])
    edges_log[0] = logg[0] - 0.5 * (logg[1] - logg[0])
    edges_log[-1] = logg[-1] + 0.5 * (logg[-1] - logg[-2])
    return 10 ** edges_log


def compute_density_histograms(m_no, mbb_no, m_io, mbb_io, x_bins, y_bins):
    h_no, x_edges, y_edges = np.histogram2d(m_no, mbb_no, bins=[x_bins, y_bins])
    h_io, _, _ = np.histogram2d(m_io, mbb_io, bins=[x_bins, y_bins])

    colsum_no = h_no.sum(axis=1, keepdims=True)
    colsum_io = h_io.sum(axis=1, keepdims=True)
    h_no_pdf = np.divide(h_no, colsum_no, out=np.zeros_like(h_no), where=colsum_no > 0)
    h_io_pdf = np.divide(h_io, colsum_io, out=np.zeros_like(h_io), where=colsum_io > 0)

    h_no_pdf_masked = np.ma.masked_less_equal(h_no_pdf, 0.0)
    h_io_pdf_masked = np.ma.masked_less_equal(h_io_pdf, 0.0)

    positive_vals = np.concatenate([h_no_pdf[h_no_pdf > 0.0], h_io_pdf[h_io_pdf > 0.0]])
    vmin = positive_vals.min()
    vmax = positive_vals.max()

    return {
        "H_no": h_no,
        "H_io": h_io,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "H_no_pdf": h_no_pdf,
        "H_io_pdf": h_io_pdf,
        "H_no_pdf_masked": h_no_pdf_masked,
        "H_io_pdf_masked": h_io_pdf_masked,
        "vmin": vmin,
        "vmax": vmax,
    }


def neutrino_masses(m_lightest, dm21, dm31, ordering="NO"):
    ordering = ordering.upper()
    dm21 = float(dm21)
    dm31 = abs(float(dm31))

    if ordering == "NO":
        m1 = m_lightest
        m2 = np.sqrt(m1**2 + dm21)
        m3 = np.sqrt(m1**2 + dm31)
    elif ordering == "IO":
        m3 = m_lightest
        m1 = np.sqrt(m3**2 + dm31)
        m2 = np.sqrt(m3**2 + dm31 + dm21)
    else:
        raise ValueError("ordering must be 'NO' or 'IO'")

    return m1, m2, m3


def sum_masses(m_lightest, dm21, dm31, ordering="NO"):
    m1, m2, m3 = neutrino_masses(m_lightest, dm21, dm31, ordering)
    return m1 + m2 + m3


def solve_lightest_from_sum(sum_target, dm21, dm31, ordering="NO", m_hi=1.0, rtol=1e-12, maxiter=300):
    ordering = ordering.upper()

    s_min = sum_masses(0.0, dm21, dm31, ordering)
    if sum_target < s_min:
        raise ValueError(f"Target sum {sum_target:.6f} eV is below minimum {s_min:.6f} eV for {ordering}.")

    def f(m):
        return sum_masses(m, dm21, dm31, ordering) - sum_target

    lo, hi = 0.0, float(m_hi)
    while f(hi) < 0:
        hi *= 2.0
        if hi > 100:
            raise RuntimeError("Failed to bracket m_lightest solution.")

    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
        if abs(hi - lo) <= rtol * max(1.0, abs(mid)):
            break

    return 0.5 * (lo + hi)


def cosmological_lightest_limits(sum_m_nu: float, params_dict_no: Dict[str, float], params_dict_io: Dict[str, float]):
    dm21_no = params_dict_no["dm21"]
    dm31_no = params_dict_no["dm31"]
    dm21_io = params_dict_io["dm21"]
    dm31_io = params_dict_io["dm31"]

    m_l_no_max = solve_lightest_from_sum(sum_m_nu, dm21_no, dm31_no, ordering="NO")
    m_l_io_max = solve_lightest_from_sum(sum_m_nu, dm21_io, dm31_io, ordering="IO")

    return m_l_no_max, m_l_io_max


def compute_nme_bands():
    nme_ab_initio = {"lower": 1.08, "upper": 1.90}
    nme_pheno = {"lower": 1.98, "upper": 5.06}

    g = 14.58e-15
    g_a = 1.273
    m_e = 0.511e6

    halflife_80t_discovery = {"upper": 5.76e27, "lower": 3.81e27}
    halflife_60t_discovery = {"upper": 4.06e27, "lower": 2.64e27}

    mbb_ab_initio_80t = {
        "upper": np.sqrt(1.0 / (g * g_a**4 * nme_ab_initio["upper"] ** 2 * halflife_80t_discovery["upper"])) * m_e,
        "lower": np.sqrt(1.0 / (g * g_a**4 * nme_ab_initio["lower"] ** 2 * halflife_80t_discovery["lower"])) * m_e,
    }
    mbb_ab_initio_60t = {
        "upper": np.sqrt(1.0 / (g * g_a**4 * nme_ab_initio["upper"] ** 2 * halflife_60t_discovery["upper"])) * m_e,
        "lower": np.sqrt(1.0 / (g * g_a**4 * nme_ab_initio["lower"] ** 2 * halflife_60t_discovery["lower"])) * m_e,
    }
    mbb_pheno_80t = {
        "upper": np.sqrt(1.0 / (g * g_a**4 * nme_pheno["upper"] ** 2 * halflife_80t_discovery["upper"])) * m_e,
        "lower": np.sqrt(1.0 / (g * g_a**4 * nme_pheno["lower"] ** 2 * halflife_80t_discovery["lower"])) * m_e,
    }
    mbb_pheno_60t = {
        "upper": np.sqrt(1.0 / (g * g_a**4 * nme_pheno["upper"] ** 2 * halflife_60t_discovery["upper"])) * m_e,
        "lower": np.sqrt(1.0 / (g * g_a**4 * nme_pheno["lower"] ** 2 * halflife_60t_discovery["lower"])) * m_e,
    }

    return {
        "mbb_ab_initio_80t": mbb_ab_initio_80t,
        "mbb_ab_initio_60t": mbb_ab_initio_60t,
        "mbb_pheno_80t": mbb_pheno_80t,
        "mbb_pheno_60t": mbb_pheno_60t,
    }


def sensitivity_weights(y_centers, band):
    return np.clip((y_centers - band["upper"]) / (band["lower"] - band["upper"]), 0.0, 1.0)


def discovery_probabilities(h_no_pdf, h_io_pdf, y_edges, band_80t, band_60t):
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    w_80t = sensitivity_weights(y_centers, band_80t)
    w_60t = sensitivity_weights(y_centers, band_60t)

    return {
        "no_80t": h_no_pdf @ w_80t,
        "no_60t": h_no_pdf @ w_60t,
        "io_80t": h_io_pdf @ w_80t,
        "io_60t": h_io_pdf @ w_60t,
    }


def phase_curve(params_dict, ordering, m_l_phase_eV, delta_a1, delta_a2):
    return 1e3 * majorana_mass(
        params_dict,
        ordering=ordering,
        delta_a1=delta_a1,
        delta_a2=delta_a2,
        m_l=m_l_phase_eV,
    )


def exclusion_weights(samples_eV, band):
    return np.clip((samples_eV - band["upper"]) / (band["lower"] - band["upper"]), 0.0, 1.0)


def histogram_density(samples_meV, bins_meV, weights=None):
    counts, edges = np.histogram(samples_meV, bins=bins_meV, weights=weights)
    density = counts / (samples_meV.size * np.diff(edges))
    area = np.sum(density * np.diff(edges))
    return density, edges, area


def slice_distributions(
    params_dict,
    df_ref,
    ordering,
    m_l_slice,
    n_slice_samples,
    slice_bins_meV,
    band_80t,
    band_60t,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng(11)

    samples_eV = sample_mbb_at_mlightest(
        m_l_slice,
        params_dict,
        df_ref=df_ref,
        ordering=ordering,
        n_samples=n_slice_samples,
        rng=rng,
    )
    samples_meV = samples_eV * 1e3

    raw_density, edges, raw_area = histogram_density(samples_meV, slice_bins_meV)
    weights_80t = exclusion_weights(samples_eV, band_80t)
    weights_60t = exclusion_weights(samples_eV, band_60t)
    weighted_80t, _, area_80t = histogram_density(samples_meV, slice_bins_meV, weights=weights_80t)
    weighted_60t, _, area_60t = histogram_density(samples_meV, slice_bins_meV, weights=weights_60t)

    return {
        "samples_meV": samples_meV,
        "raw_density": raw_density,
        "weighted_80t": weighted_80t,
        "weighted_60t": weighted_60t,
        "edges": edges,
        "raw_area": raw_area,
        "area_80t": area_80t,
        "area_60t": area_60t,
        "mean_80t": weights_80t.mean(),
        "mean_60t": weights_60t.mean(),
    }
