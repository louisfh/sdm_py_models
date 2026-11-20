"""Estimate call density per species and point with Bayesian Platt scaling.

For each species and point:
Scores are on the logit scale.
1. Fit P(call | score) = inverse_logit(a + b * score) to the reviewed
   windows at that point in Chorus_detection_history.csv (confirmed = 1,
   rejected = 0). Each point has its own (a, b).
2. Apply the posterior of that point's (a, b) to every unreviewed window at the
   point in the species' classifier score file.
3. Call density at a point = (sum of review labels + sum of calibrated
   probabilities of unreviewed windows) / number of windows at the point,
   computed for every posterior draw.

Run from the repository root:
    .venv/bin/python Call_density/01_SDM.py
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import sys
import warnings
from pathlib import Path

REQUIRED_PACKAGES = ["numpy", "pandas", "pymc", "arviz", "scipy", "pyarrow", "netCDF4"]
_missing = []
for _package in REQUIRED_PACKAGES:
    try:
        __import__(_package)
    except ImportError:
        _missing.append(_package)
if _missing:
    sys.exit("Missing Python packages: " + ", ".join(_missing) + "\nInstall with: pip install -r requirements.txt")

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from scipy import stats

logging.getLogger("pymc").setLevel(logging.ERROR)

# MCMC settings
MCMC_CHAINS = 4
MCMC_TUNE = 1000            # warmup draws per chain
MCMC_DRAWS = 1000           # kept draws per chain
MCMC_CORES = min(MCMC_CHAINS, os.cpu_count() or 1)
MCMC_SEED = 2026

# Convergence rule for status "converged": Rhat < RHAT_THRESHOLD and bulk and
# tail ESS >= ESS_THRESHOLD for a and b, no divergent transitions, no maximum
# tree depth hits and E-BFMI >= BFMI_THRESHOLD in every chain.
RHAT_THRESHOLD = 1.01
ESS_THRESHOLD = 400
BFMI_THRESHOLD = 0.3

# Priors: a ~ Normal(0, PRIOR_SD), b ~ Normal(0, PRIOR_SD).
PRIOR_SD = 2.5

# Review labels: confirmed = 1, rejected = 0. Uncertain windows are not used as
# labels; they are treated as unreviewed windows and calibrated from their score.
LABELS = {"confirmed": 1, "rejected": 0}

# Scores. Unlabelled score files: column "Score", logit scale, one row for every
# window. Chorus_detection_history.csv: column "score", on the scale set by
# CHORUS_SCORE_SCALE ("logit", or "confidence" for 0 to 1 scores, which are
# converted with logit(c) = log(c / (1 - c)) after clipping to CONFIDENCE_CLIP).
CHORUS_SCORE_SCALE = "confidence"
CONFIDENCE_CLIP = (0.00005, 0.99995)
# Unlabelled scores are rounded to SCORE_ROUNDING (logit units) when summing
# calibrated probabilities.
SCORE_ROUNDING = 0.01
# Labelled windows whose Chorus score and unlabelled score differ by more than
# SCORE_TOLERANCE (logit units) are counted in platt_status.csv.
SCORE_TOLERANCE = 0.01

PROJECT_DIR = Path(__file__).resolve().parent
CHORUS_FILE = PROJECT_DIR / "1_Input_from_chorus" / "Chorus_detection_history.csv"
SCORES_DIR = PROJECT_DIR / "1_Input_from_chorus" / "Unlabelled_scores"
OUTPUT_DIR = PROJECT_DIR / "4_Expected_output"
CSV_DIR = OUTPUT_DIR / "CSV"
NETCDF_DIR = OUTPUT_DIR / "NETCDF"

SCORE_COLUMNS = ["Start (s)", "Scientific name", "Score", "File", "site", "project"]
PROJECTS = {"Calima": "Calima", "San Carlos": "San_Carlos", "San_Carlos": "San_Carlos"}


def fail(message: str) -> None:
    sys.exit("Error: " + message)


def logit(confidence: np.ndarray) -> np.ndarray:
    clipped = np.clip(confidence, *CONFIDENCE_CLIP)
    return np.log(clipped) - np.log1p(-clipped)


def inverse_logit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def read_scores(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns
    missing = [column for column in SCORE_COLUMNS if column not in header]
    if missing:
        fail(f"{path.name} is missing columns: {', '.join(missing)} (expected: {', '.join(SCORE_COLUMNS)})")
    scores = pd.read_csv(path, usecols=SCORE_COLUMNS, engine="pyarrow")
    scores["filename"] = scores["File"].str.replace("\\", "/", regex=False).str.split("/").str[-1]
    project = scores["project"].map(PROJECTS)
    if project.isna().any():
        fail(f"Unknown project names in {path.name}: {sorted(scores.loc[project.isna(), 'project'].unique())}")
    scores["location_name"] = project + "_" + scores["site"].astype(str)
    scores = scores.rename(columns={"Start (s)": "start", "Score": "logit_score", "Scientific name": "species"})
    return scores[["species", "location_name", "filename", "start", "logit_score"]]


def platt_model(score: np.ndarray, label: np.ndarray, point_index: np.ndarray, points: list[str]) -> pm.Model:
    """One (a, b) per point with independent priors and no shared parameters.

    The joint posterior factorises over points, so this is equivalent to a
    separate fit at each point; it is sampled as one model per species.
    """
    with pm.Model(coords={"point": points}) as model:
        a = pm.Normal("a", mu=0, sigma=PRIOR_SD, dims="point")
        b = pm.Normal("b", mu=0, sigma=PRIOR_SD, dims="point")
        pm.Bernoulli("label", logit_p=a[point_index] + b[point_index] * score, observed=label)
    return model


def sample(model: pm.Model):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with model:
                idata = pm.sample(
                    draws=MCMC_DRAWS, tune=MCMC_TUNE, chains=MCMC_CHAINS, cores=MCMC_CORES,
                    random_seed=MCMC_SEED, progressbar=False, compute_convergence_checks=False,
                )
            error = None
        except Exception as exception:
            idata, error = None, f"{type(exception).__name__}: {exception}"
    return idata, sorted({str(warning.message) for warning in caught}), error


def prior_posterior_overlap(draws: np.ndarray) -> float:
    low = min(draws.min(), -4 * PRIOR_SD)
    high = max(draws.max(), 4 * PRIOR_SD)
    grid = np.linspace(low, high, 4000)
    prior = stats.norm.pdf(grid, 0, PRIOR_SD)
    posterior = stats.gaussian_kde(draws)(grid)
    return float(np.trapezoid(np.minimum(prior, posterior), grid))


def parameter_table(idata, point_position: int) -> pd.DataFrame:
    rows = []
    for term in ["a", "b"]:
        values = np.asarray(idata.posterior[term].values)[:, :, point_position]
        flat = values.reshape(-1)
        rows.append({
            "term": term,
            "mean": flat.mean(),
            "sd": flat.std(ddof=1),
            "lower_95": np.quantile(flat, 0.025),
            "upper_95": np.quantile(flat, 0.975),
            "rhat": float(az.rhat(values)),
            "ess_bulk": float(az.ess(values, method="bulk")),
            "ess_tail": float(az.ess(values, method="tail", prob=(0.05, 0.95))),
            "mcse_mean": float(az.mcse(values)),
            "posterior_contraction": 1.0 - flat.var(ddof=1) / PRIOR_SD**2,
            "prior_posterior_overlap": prior_posterior_overlap(flat),
        })
    return pd.DataFrame(rows)


def sampler_diagnostics(idata) -> dict:
    stats_group = idata.sample_stats
    depth = next((stats_group[name] for name in ["reached_max_treedepth", "reaching_max_treedepth"]
                  if name in stats_group), None)
    return {
        "divergent_transitions": int(np.asarray(stats_group["diverging"].values).sum()),
        "max_treedepth_hits": int(np.asarray(depth.values).sum()) if depth is not None else None,
        "min_bfmi": float(np.asarray(az.bfmi(idata)["energy"].values).min()),
    }


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name)


def summarise(draws: np.ndarray, prefix: str) -> dict:
    return {
        f"{prefix}_mean": float(draws.mean()),
        f"{prefix}_sd": float(draws.std(ddof=1)),
        f"{prefix}_lower_95": float(np.quantile(draws, 0.025)),
        f"{prefix}_upper_95": float(np.quantile(draws, 0.975)),
    }


def main() -> None:
    if not CHORUS_FILE.exists():
        fail(f"Required input file not found: {CHORUS_FILE}")
    score_files = sorted(SCORES_DIR.glob("*.csv"))
    if not score_files:
        fail(f"No score files (.csv) found in {SCORES_DIR}")
    shutil.rmtree(CSV_DIR, ignore_errors=True)
    shutil.rmtree(NETCDF_DIR, ignore_errors=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    NETCDF_DIR.mkdir(parents=True, exist_ok=True)

    # Reviewed windows
    chorus = pd.read_csv(CHORUS_FILE)
    required = ["scientific_name", "location_name", "filename", "detection_start_seconds", "score", "review_status"]
    if not set(required) <= set(chorus.columns):
        fail("Chorus input must contain: " + ", ".join(required))
    chorus["review_status"] = chorus["review_status"].fillna("").str.strip().str.lower()
    chorus["label"] = chorus["review_status"].map(LABELS)
    chorus = chorus.rename(columns={"scientific_name": "species", "detection_start_seconds": "start"})
    if CHORUS_SCORE_SCALE == "confidence":
        chorus["logit_score"] = logit(chorus["score"].to_numpy(float))
    elif CHORUS_SCORE_SCALE == "logit":
        chorus["logit_score"] = chorus["score"].astype(float)
    else:
        fail('CHORUS_SCORE_SCALE must be "logit" or "confidence"')

    print("Reading score files", flush=True)
    score_tables = {}
    for path in score_files:
        species_in_file = pd.read_csv(path, usecols=["Scientific name"], engine="pyarrow")["Scientific name"].unique()
        if len(species_in_file) != 1:
            fail(f"{path.name} must contain exactly one species; found {len(species_in_file)}")
        score_tables[species_in_file[0]] = path

    platt_rows, parameter_rows, density_rows, curve_rows = [], [], [], []
    curve_scores = np.round(np.arange(-10, 10.01, 0.1), 1)
    for index, (species, path) in enumerate(sorted(score_tables.items())):
        reviewed = chorus[chorus["species"] == species]
        labelled = reviewed[reviewed["label"].notna()]
        scores = read_scores(path).drop_duplicates(["filename", "start"])
        labelled_keys = set(zip(labelled["filename"], labelled["start"]))
        is_labelled = pd.Series([key in labelled_keys for key in zip(scores["filename"], scores["start"])],
                                index=scores.index)
        matched = labelled.merge(scores[["filename", "start", "logit_score"]], on=["filename", "start"],
                                 how="left", suffixes=("", "_score_file"))
        unreviewed = scores[~is_labelled]
        points = sorted(set(unreviewed["location_name"]) | set(reviewed["location_name"]))
        fitted_points = sorted(labelled["location_name"].unique())
        print(f"[{index + 1}/{len(score_tables)}] {species}: {len(points)} points, {len(labelled)} labelled windows, "
              f"{int(labelled['label'].sum())} confirmed", flush=True)

        idata, fit_warnings, error, sampler = None, [], None, {}
        if fitted_points:
            point_index = pd.Index(fitted_points).get_indexer(labelled["location_name"])
            idata, fit_warnings, error = sample(platt_model(
                labelled["logit_score"].to_numpy(float), labelled["label"].to_numpy(int), point_index, fitted_points))
            if idata is not None:
                sampler = sampler_diagnostics(idata)
                idata.to_netcdf(NETCDF_DIR / f"{safe_name(species)}_platt_posterior.nc")

        for point in points:
            point_reviewed = reviewed[reviewed["location_name"] == point]
            point_labels = labelled[labelled["location_name"] == point]
            point_matched = matched[matched["location_name"] == point]
            in_score_file = point_matched["logit_score_score_file"].notna()
            status_row = {
                "species": species, "point": point, "score_file": path.name,
                "reviewed_windows": len(point_reviewed), "labelled_windows": len(point_labels),
                "confirmed": int((point_labels["label"] == 1).sum()),
                "rejected": int((point_labels["label"] == 0).sum()),
                "uncertain_or_unreviewed": int(point_reviewed["label"].isna().sum()),
                "labelled_windows_in_score_file": int(in_score_file.sum()),
                "labelled_windows_not_in_score_file": int((~in_score_file).sum()),
                "labelled_score_differs_from_score_file": int(
                    (np.abs(point_matched["logit_score"] - point_matched["logit_score_score_file"])
                     > SCORE_TOLERANCE).sum()),
                "status": "", "fit_error": None,
            }
            if point not in fitted_points:
                status_row["status"] = "no_labelled_windows"
                platt_rows.append(status_row)
                continue
            if idata is None:
                status_row.update(status="failed", fit_error=error)
                platt_rows.append(status_row)
                continue
            position = fitted_points.index(point)
            parameters = parameter_table(idata, position)
            converged = (parameters["rhat"].max() < RHAT_THRESHOLD
                         and parameters[["ess_bulk", "ess_tail"]].min().min() >= ESS_THRESHOLD
                         and sampler["divergent_transitions"] == 0
                         and not sampler["max_treedepth_hits"]
                         and sampler["min_bfmi"] >= BFMI_THRESHOLD)
            status_row.update(
                status="converged" if converged else "not_converged",
                max_rhat=parameters["rhat"].max(),
                min_ess_bulk=parameters["ess_bulk"].min(), min_ess_tail=parameters["ess_tail"].min(),
                **{f"species_model_{key}": value for key, value in sampler.items()},
                fit_warnings="; ".join(fit_warnings),
            )
            parameter_rows.append(parameters.assign(species=species, point=point)[
                ["species", "point"] + list(parameters.columns)])

            a = np.asarray(idata.posterior["a"].values)[:, :, position].reshape(-1)
            b = np.asarray(idata.posterior["b"].values)[:, :, position].reshape(-1)
            curve = inverse_logit(a[:, None] + b[:, None] * curve_scores[None, :])
            for column, score in enumerate(curve_scores):
                curve_rows.append({"species": species, "point": point, "logit_score": score,
                                   **summarise(curve[:, column], "p_call")})

            # Unreviewed windows: every window in the score file at this point
            # that is not a labelled window.
            point_scores = unreviewed[unreviewed["location_name"] == point]
            rounded = np.round(point_scores["logit_score"].to_numpy(float) / SCORE_ROUNDING) * SCORE_ROUNDING
            score_values, counts = np.unique(rounded, return_counts=True)
            calibrated = inverse_logit(a[:, None] + b[:, None] * score_values[None, :])
            expected_calls = calibrated @ counts + point_labels["label"].sum()
            total_windows = len(point_scores) + len(point_labels)
            density_rows.append({
                "species": species,
                "point": point,
                "recording_files": point_scores["filename"].nunique(),
                "total_windows": total_windows,
                "labelled_windows": len(point_labels),
                "labelled_confirmed": int(point_labels["label"].sum()),
                "unreviewed_windows": len(point_scores),
                **summarise(expected_calls, "expected_calls"),
                **summarise(expected_calls / total_windows, "call_density"),
            })
            platt_rows.append(status_row)

    pd.DataFrame(platt_rows).to_csv(CSV_DIR / "platt_status.csv", index=False)
    if parameter_rows:
        pd.concat(parameter_rows).to_csv(CSV_DIR / "platt_parameters.csv", index=False)
        pd.DataFrame(curve_rows).to_csv(CSV_DIR / "calibration_curves.csv", index=False)
        pd.DataFrame(density_rows).to_csv(CSV_DIR / "call_density_by_point.csv", index=False)
    status = pd.DataFrame(platt_rows)["status"]
    print(f"Species x points: {len(status)}; converged: {(status == 'converged').sum()}; "
          f"not converged: {(status == 'not_converged').sum()}; other: {(~status.isin(['converged', 'not_converged'])).sum()}")
    print("Outputs written to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
