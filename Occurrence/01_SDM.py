"""Fit two Bayesian models per species and export site/raster predictions.

- Occupancy: single-season occupancy model, constant detection,
  occupancy ~ raster covariates. Latent occupancy is marginalised out.
- Logistic: Bernoulli logistic regression with one row per observed
  site-visit, no latent occupancy state and no site-level term.

Both models are projected over every raster cell with complete covariates.

Both models are written in PyMC with logistic(0, 1) priors on every intercept
and coefficient.

Run from the repository root:
    .venv/bin/python Occurrence/01_SDM.py
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import shutil
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

# 1. Load packages
REQUIRED_PACKAGES = ["numpy", "pandas", "pymc", "arviz", "rasterio", "netCDF4"]
_missing = []
for _package in REQUIRED_PACKAGES:
    try:
        __import__(_package)
    except ImportError:
        _missing.append(_package)
if _missing:
    sys.exit(
        "Missing Python packages: " + ", ".join(_missing)
        + "\nInstall with: pip install -r requirements.txt"
    )

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import rasterio
from rasterio.warp import Resampling, reproject, transform as transform_points
from scipy import stats

logging.getLogger("pymc").setLevel(logging.ERROR)


def log_sigmoid(x):
    """log(inverse_logit(x)), numerically stable."""
    return -pt.softplus(-x)

# MCMC settings
MCMC_CHAINS = 4
MCMC_TUNE = 500             # warmup draws per chain
MCMC_DRAWS = 500            # kept draws per chain
MCMC_CORES = min(MCMC_CHAINS, os.cpu_count() or 1)
MCMC_SEED = 2026

# Convergence: a fit counts as converged only if every coefficient has
# Rhat < RHAT_THRESHOLD and bulk and tail ESS >= ESS_THRESHOLD, and the sampler
# reports no divergences, no maximum tree depth hits and E-BFMI >= BFMI_THRESHOLD
# in every chain (Vehtari et al. 2021; Betancourt 2017).
RHAT_THRESHOLD = 1.01
ESS_THRESHOLD = 400
BFMI_THRESHOLD = 0.3

# Prior vs posterior: every coefficient has a logistic(0, 1) prior.
# overlap_above_threshold is True when the prior and posterior densities
# overlap by more than OVERLAP_THRESHOLD (threshold from Garrett and Zeger 2000).
PRIOR_MEAN = 0.0
PRIOR_SCALE = 1.0
PRIOR_SD = PRIOR_SCALE * np.pi / np.sqrt(3)
OVERLAP_THRESHOLD = 0.35

# Posterior predictive check: share of replicated data sets with a statistic at
# least as large as the observed one. outside_bounds is True when this share
# is outside PPC_BOUNDS.
PPC_BOUNDS = (0.05, 0.95)

# 2. Define input and output paths
PROJECT_DIR = Path(__file__).resolve().parent
INTERMEDIATE_DIR = PROJECT_DIR / "3_Intermediate_outputs"
CHORUS_INPUT_DIR = PROJECT_DIR / "1_Input_from_chorus"
RASTER_DIR = PROJECT_DIR / "2_Other_inputs" / "Raster_covariates"

COVARIATES_FILE = INTERMEDIATE_DIR / "Processed_covariates.csv"
DETECTIONS_FILE = INTERMEDIATE_DIR / "Processed_detection_history.csv"
CHORUS_FILE = CHORUS_INPUT_DIR / "Chorus_detection_history.csv"
ARU_COORDINATES_FILE = INTERMEDIATE_DIR / "ARU_coordinates.csv"

# Outputs: shared covariate products, then one folder per model with the same
# CSV / GEOTIFF / NETCDF / JSON layout.
OUTPUT_DIR = PROJECT_DIR / "4_Expected_output"
COVARIATE_OUTPUT_DIR = OUTPUT_DIR / "Covariates"
MODEL_OUTPUT_DIRS = {"occupancy": OUTPUT_DIR / "Occupancy", "logistic": OUTPUT_DIR / "Logistic"}
MODEL_SUBDIRS = ["CSV", "GEOTIFF", "NETCDF", "JSON"]

SITE_ID_COLUMNS = ["R_ID", "Project", "Site", "Latitude", "Longitude"]
NODATA = -9999
RASTER_LAYERS = ["mean", "sd", "lower_95", "upper_95"]


# 3. Define model and summary helper functions

def fail(message: str) -> None:
    sys.exit("Error: " + message)


def naive_occupancy(response: np.ndarray) -> dict:
    surveyed = (~np.isnan(response)).sum(axis=1) > 0
    detected = np.nansum(response == 1, axis=1) > 0
    return {
        "detected_at_site": np.where(surveyed, detected.astype(float), np.nan),
        "surveyed_sites": int(surveyed.sum()),
        "detected_sites": int((detected & surveyed).sum()),
        "naive_occupancy": float(detected[surveyed].mean()) if surveyed.any() else np.nan,
    }


def occupancy_model(response: np.ndarray, covariates: np.ndarray) -> pm.Model:
    """z_i ~ Bernoulli(psi_i), y_ij ~ Bernoulli(z_i * p), logit(psi_i) = X_i beta.

    z_i is marginalised: sites with a detection contribute
    log(psi) + k log(p) + (n - k) log(1 - p); sites without one contribute
    log(psi (1 - p)^n + 1 - psi).
    """
    observed_visits = (~np.isnan(response)).sum(axis=1)
    detections = np.nansum(response, axis=1)
    keep = observed_visits > 0
    n, k, X = observed_visits[keep], detections[keep], covariates[keep]
    detected = k > 0
    with pm.Model() as model:
        state_intercept = pm.Logistic("state_intercept", mu=0, s=1)
        state_beta = pm.Logistic("state_beta", mu=0, s=1, shape=X.shape[1])
        detection_intercept = pm.Logistic("detection_intercept", mu=0, s=1)
        logit_psi = state_intercept + pm.math.dot(X, state_beta)
        log_psi = log_sigmoid(logit_psi)
        log_not_psi = log_sigmoid(-logit_psi)
        log_p = log_sigmoid(detection_intercept)
        log_not_p = log_sigmoid(-detection_intercept)
        site_loglik = pt.switch(
            detected,
            log_psi + k * log_p + (n - k) * log_not_p,
            pt.logaddexp(log_psi + n * log_not_p, log_not_psi),
        )
        pm.Potential("likelihood", site_loglik.sum())
    return model


def logistic_model(response: np.ndarray, covariates: np.ndarray) -> pm.Model:
    """y_ij ~ Bernoulli(p_i), logit(p_i) = alpha + X_i beta, one row per observed visit."""
    site_index, _ = np.nonzero(~np.isnan(response))
    y = response[~np.isnan(response)].astype(int)
    X = covariates[site_index]
    with pm.Model() as model:
        intercept = pm.Logistic("intercept", mu=0, s=1)
        beta = pm.Logistic("beta", mu=0, s=1, shape=X.shape[1])
        pm.Bernoulli("y", logit_p=intercept + pm.math.dot(X, beta), observed=y)
    return model


def sample(model: pm.Model):
    """Sample a model, keeping warnings and turning errors into a message."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with model:
                idata = pm.sample(
                    draws=MCMC_DRAWS, tune=MCMC_TUNE, chains=MCMC_CHAINS, cores=MCMC_CORES,
                    random_seed=MCMC_SEED, progressbar=False, compute_convergence_checks=False,
                )
            error = None
        except Exception as exception:  # report and carry on with other species
            idata, error = None, f"{type(exception).__name__}: {exception}"
    messages = sorted({str(warning.message) for warning in caught})
    return idata, messages, error


def posterior_array(idata, name: str) -> np.ndarray:
    """Posterior as (chain, draw, ...) numpy array."""
    return np.asarray(idata.posterior[name].values)


def prior_posterior_overlap(draws: np.ndarray) -> float:
    """Overlap of the prior density and a kernel estimate of the posterior density (0 to 1)."""
    low = min(draws.min(), stats.logistic.ppf(0.0005, PRIOR_MEAN, PRIOR_SCALE))
    high = max(draws.max(), stats.logistic.ppf(0.9995, PRIOR_MEAN, PRIOR_SCALE))
    grid = np.linspace(low, high, 4000)
    prior = stats.logistic.pdf(grid, PRIOR_MEAN, PRIOR_SCALE)
    posterior = stats.gaussian_kde(draws)(grid)
    return float(np.trapezoid(np.minimum(prior, posterior), grid))


def parameter_table(chains: dict[str, np.ndarray], terms: list[str], submodel: str) -> pd.DataFrame:
    """Posterior summary, MCMC diagnostics and prior-vs-posterior metrics per coefficient.

    chains maps term -> (chain, draw) array.
    - posterior_contraction: 1 - posterior variance / prior variance.
    - prior_posterior_overlap: shared area of the prior and posterior densities.
    - posterior_shift: distance of the posterior mean from the prior mean, in prior SDs.
    - prob_positive: posterior probability that the coefficient is above 0.
    """
    rows = []
    for term in terms:
        values = chains[term]
        flat = values.reshape(-1)
        overlap = prior_posterior_overlap(flat)
        rows.append({
            "submodel": submodel,
            "term": term,
            "mean": flat.mean(),
            "sd": flat.std(ddof=1),
            "lower_95": np.quantile(flat, 0.025),
            "upper_95": np.quantile(flat, 0.975),
            "prob_positive": (flat > 0).mean(),
            "rhat": float(az.rhat(values)),
            "ess_bulk": float(az.ess(values, method="bulk")),
            "ess_tail": float(az.ess(values, method="tail", prob=(0.05, 0.95))),
            "mcse_mean": float(az.mcse(values)),
            "posterior_contraction": 1.0 - flat.var(ddof=1) / PRIOR_SD**2,
            "prior_posterior_overlap": overlap,
            "posterior_shift": abs(flat.mean() - PRIOR_MEAN) / PRIOR_SD,
            "overlap_above_threshold": overlap > OVERLAP_THRESHOLD,
        })
    return pd.DataFrame(rows)


def sampler_stat(idata, *names: str) -> np.ndarray | None:
    for name in names:
        if name in idata.sample_stats:
            return np.asarray(idata.sample_stats[name].values)
    return None


def mcmc_diagnostics(idata, parameters: pd.DataFrame) -> dict:
    tree_depth_hits = sampler_stat(idata, "reached_max_treedepth", "reaching_max_treedepth")
    bfmi = np.asarray(az.bfmi(idata)["energy"].values) if sampler_stat(idata, "energy") is not None else None
    diagnostics = {
        "max_rhat": float(parameters["rhat"].max()),
        "min_ess_bulk": float(parameters["ess_bulk"].min()),
        "min_ess_tail": float(parameters["ess_tail"].min()),
        "divergent_transitions": int(sampler_stat(idata, "diverging").sum()),
        "max_treedepth_hits": int(tree_depth_hits.sum()) if tree_depth_hits is not None else None,
        "min_bfmi": float(bfmi.min()) if bfmi is not None else None,
    }
    diagnostics["converged"] = bool(
        diagnostics["max_rhat"] < RHAT_THRESHOLD
        and diagnostics["min_ess_bulk"] >= ESS_THRESHOLD
        and diagnostics["min_ess_tail"] >= ESS_THRESHOLD
        and diagnostics["divergent_transitions"] == 0
        and not diagnostics["max_treedepth_hits"]
        and (diagnostics["min_bfmi"] is None or diagnostics["min_bfmi"] >= BFMI_THRESHOLD)
    )
    coefficients = parameters[parameters["term"] != "(Intercept)"]
    diagnostics["min_posterior_contraction"] = float(coefficients["posterior_contraction"].min())
    diagnostics["max_prior_posterior_overlap"] = float(coefficients["prior_posterior_overlap"].max())
    diagnostics["terms_overlap_above_threshold"] = [
        f"{row.submodel}:{row.term}" for row in parameters.itertuples() if row.overlap_above_threshold
    ]
    return diagnostics


def posterior_predictive_check(response: np.ndarray, site_draws: np.ndarray,
                               detection_draws: np.ndarray | None, seed: int) -> dict:
    """Compare observed detection totals with data replicated from the posterior.

    site_draws: sites x draws of psi (occupancy) or p (logistic).
    detection_draws: per-draw detection probability for the occupancy model, None for logistic.
    """
    rng = np.random.default_rng(seed)
    observed_visits = (~np.isnan(response)).sum(axis=1)
    if detection_draws is None:
        replicated = rng.binomial(observed_visits[:, None], site_draws)
    else:
        occupied = rng.random(site_draws.shape) < site_draws
        replicated = rng.binomial(observed_visits[:, None], occupied * detection_draws[None, :])
    checks = {}
    for statistic, observed, replicate in [
        ("total_detections", np.nansum(response == 1), replicated.sum(axis=0)),
        ("sites_with_detections", int((np.nansum(response == 1, axis=1) > 0).sum()), (replicated > 0).sum(axis=0)),
    ]:
        p_value = float((replicate >= observed).mean())
        checks[statistic] = {
            "observed": int(observed),
            "replicated_mean": float(replicate.mean()),
            "replicated_lower_95": float(np.quantile(replicate, 0.025)),
            "replicated_upper_95": float(np.quantile(replicate, 0.975)),
            "p_value": p_value,
            "outside_bounds": not (PPC_BOUNDS[0] <= p_value <= PPC_BOUNDS[1]),
        }
    return checks


def summarise_draws(draws: np.ndarray, prefix: str) -> pd.DataFrame:
    """Rows are sites or cells, columns are draws."""
    lower, upper = np.quantile(draws, [0.025, 0.975], axis=1)
    return pd.DataFrame({
        f"{prefix}_mean": draws.mean(axis=1),
        f"{prefix}_sd": draws.std(axis=1, ddof=1),
        f"{prefix}_lower_95": lower,
        f"{prefix}_upper_95": upper,
    })


def probability_draws(covariates: np.ndarray, coefficient_draws: np.ndarray) -> np.ndarray:
    """coefficient_draws: draws x (1 + covariates), intercept first."""
    design = np.column_stack([np.ones(len(covariates)), covariates])
    return 1.0 / (1.0 + np.exp(-(design @ coefficient_draws.T)))


def conditional_occupancy_draws(response: np.ndarray, occupancy: np.ndarray, detection: np.ndarray) -> np.ndarray:
    """Occupancy given the detection history, per draw. occupancy: sites x draws."""
    observed_visits = (~np.isnan(response)).sum(axis=1)
    detected = np.nansum(response == 1, axis=1) > 0
    missed = (1.0 - detection[None, :]) ** observed_visits[:, None]
    conditioned = occupancy * missed / (occupancy * missed + 1.0 - occupancy)
    conditioned[detected, :] = 1.0
    return conditioned


def write_layers(profile: dict, shape: tuple, cell_index: np.ndarray, summaries: pd.DataFrame, path: Path) -> None:
    layer_profile = {**profile, "count": summaries.shape[1], "dtype": "float32", "nodata": np.nan}
    with rasterio.open(path, "w", **layer_profile) as destination:
        for band, column in enumerate(summaries.columns, start=1):
            values = np.full(shape[0] * shape[1], np.nan, dtype="float32")
            values[cell_index] = summaries[column].to_numpy(dtype="float32")
            destination.write(values.reshape(shape), band)
            destination.set_band_description(band, column)


def to_json_ready(value):
    """Convert numpy/pandas values to JSON types; NaN becomes null. Data frames become row lists."""
    if isinstance(value, pd.DataFrame):
        return [to_json_ready(row) for row in value.to_dict(orient="records")]
    if isinstance(value, dict):
        return {key: to_json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return [to_json_ready(item) for item in list(value)]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    return value


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name)


def main() -> None:
    # 4. Discover covariate rasters and validate source files
    # Every single-band .tif in 2_Other_inputs/Raster_covariates is a covariate.
    # The file name (without .tif) becomes the covariate name, the column name in
    # Processed_covariates.csv, and the term name in the model output.
    covariate_files = sorted(p for p in RASTER_DIR.glob("*") if p.suffix.lower() == ".tif")
    if not covariate_files:
        fail(f"No covariate rasters (.tif) found in {RASTER_DIR}")
    covariate_rasters = {path.stem: path for path in covariate_files}
    covariate_names = list(covariate_rasters)
    invalid = [name for name in covariate_names if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.]*", name)]
    if invalid:
        fail("Covariate file names must start with a letter and contain only letters, digits, '.', '_': "
             + ", ".join(invalid))
    reserved = set(covariate_names) & set(SITE_ID_COLUMNS + ["cell", "x", "y"])
    if reserved:
        fail("Covariate file names clash with reserved column names: " + ", ".join(sorted(reserved)))
    multiband = [name for name, path in covariate_rasters.items() if rasterio.open(path).count != 1]
    if multiband:
        fail("Each covariate raster must have exactly one band: " + ", ".join(multiband))
    if not CHORUS_FILE.exists():
        fail(f"Required input file not found: {CHORUS_FILE}")

    shutil.rmtree(COVARIATE_OUTPUT_DIR, ignore_errors=True)
    for directory in MODEL_OUTPUT_DIRS.values():
        shutil.rmtree(directory, ignore_errors=True)
    for directory in [INTERMEDIATE_DIR, COVARIATE_OUTPUT_DIR] + [
        directory / subdir for directory in MODEL_OUTPUT_DIRS.values() for subdir in MODEL_SUBDIRS
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    # 5. Load Chorus data and validate its columns
    chorus = pd.read_csv(CHORUS_FILE, dtype=str, keep_default_na=False, na_values=[""])
    required_chorus_columns = [
        "scientific_name", "location_name", "latitude", "longitude", "recording_datetime",
        "filename", "detection_start_seconds", "detection_end_seconds", "review_status",
    ]
    if not set(required_chorus_columns) <= set(chorus.columns):
        fail("Chorus input must contain: " + ", ".join(required_chorus_columns))
    if chorus[["scientific_name", "location_name"]].isna().any().any():
        fail("Chorus rows must include a scientific name and location name.")

    # 6. Generate ARU coordinates and detection history from Chorus
    location = chorus["location_name"].str.strip()
    project = np.select(
        [location.str.startswith("Calima_"), location.str.startswith("San_Carlos_")],
        ["Calima", "San_Carlos"], default="",
    )
    aru = pd.DataFrame({
        "Project": project,
        "Site": location,
        "Latitude": pd.to_numeric(chorus["latitude"], errors="coerce"),
        "Longitude": pd.to_numeric(chorus["longitude"], errors="coerce"),
    }).drop_duplicates()
    if (aru.isna().any().any() or (aru["Project"] == "").any() or (aru["Site"] == "").any()
            or (~aru["Latitude"].between(-90, 90)).any() or (~aru["Longitude"].between(-180, 180)).any()
            or aru.duplicated(["Project", "Site"]).any()):
        fail("ARU coordinate input contains missing, invalid, or duplicate Project/Site rows.")
    aru = aru.sort_values(["Project", "Site"]).reset_index(drop=True)
    aru.to_csv(ARU_COORDINATES_FILE, index=False)
    aru["R_ID"] = np.arange(len(aru))

    recording_time = pd.to_datetime(chorus["recording_datetime"], format="%Y-%m-%dT%H:%M:%S", errors="coerce")
    start = pd.to_numeric(chorus["detection_start_seconds"], errors="coerce")
    end = pd.to_numeric(chorus["detection_end_seconds"], errors="coerce")
    status = chorus["review_status"].fillna("").str.strip().str.lower()
    unexpected = set(status[status != ""]) - {"confirmed", "rejected", "uncertain", "unreviewed"}
    if recording_time.isna().any() or start.isna().any() or end.isna().any() or unexpected:
        fail("Chorus contains invalid timestamps, detection intervals, or review statuses.")
    detections = pd.DataFrame({
        "site": location,
        "species": chorus["scientific_name"],
        "recording_time": recording_time,
        "filename": chorus["filename"].fillna(""),
        "detection_start_seconds": start,
        "detection_end_seconds": end,
        # confirmed = 1, rejected / uncertain = 0, unreviewed = missing
        "detected": status.map({"confirmed": 1, "rejected": 0, "uncertain": 0}).astype("Int64"),
        "source_row": np.arange(len(chorus)),
    })
    if (detections["site"] == "").any():
        fail("Chorus rows must include a site and species for every detection record.")
    detections = detections.sort_values(
        ["site", "species", "recording_time", "filename", "detection_start_seconds",
         "detection_end_seconds", "source_row"], kind="mergesort",
    )
    detections["visit"] = detections.groupby(["site", "species"]).cumcount() + 1
    detections = detections[["site", "visit", "species", "detected"]]
    detections.to_csv(DETECTIONS_FILE, index=False, na_rep="")
    detections = pd.read_csv(DETECTIONS_FILE, dtype={"site": str, "species": str})
    if not detections["site"].isin(aru["Site"]).all():
        fail("Detection history contains sites missing from ARU_coordinates.csv.")

    # 7. Sample the model covariates from the external rasters
    sampled = {}
    for name, path in covariate_rasters.items():
        with rasterio.open(path) as source:
            xs, ys = aru["Longitude"].to_numpy(), aru["Latitude"].to_numpy()
            if source.crs and source.crs.to_epsg() != 4326:
                xs, ys = transform_points("EPSG:4326", source.crs, xs, ys)
            rows, cols = rasterio.transform.rowcol(source.transform, xs, ys)
            band = source.read(1).astype(float)
            rows, cols = np.asarray(rows), np.asarray(cols)
            inside = (rows >= 0) & (rows < source.height) & (cols >= 0) & (cols < source.width)
            values = np.full(len(aru), np.nan)
            values[inside] = band[rows[inside], cols[inside]]
            if source.nodata is not None:
                values[values == source.nodata] = np.nan
            sampled[name] = values
    covariates = pd.concat([aru[SITE_ID_COLUMNS], pd.DataFrame(sampled)], axis=1)
    covariate_values = covariates[covariate_names].to_numpy()
    if (~np.isfinite(covariate_values)).any() or (covariate_values == NODATA).any():
        fail("Raster covariates must produce finite, non-nodata values for every ARU location.")
    covariates.to_csv(COVARIATES_FILE, index=False)
    covariates = pd.read_csv(COVARIATES_FILE)
    # Covariate names are taken from the header row: every column that is not a
    # site identifier column is a model covariate.
    if not set(SITE_ID_COLUMNS) <= set(covariates.columns):
        fail("Covariate table must contain: " + ", ".join(SITE_ID_COLUMNS))
    covariate_names = [column for column in covariates.columns if column not in SITE_ID_COLUMNS]
    if set(covariate_names) != set(covariate_rasters):
        fail("Covariate table columns must match the raster covariates: " + ", ".join(covariate_rasters))

    # 8. Validate detections and map sites to projects and covariates
    if detections[["site", "visit", "species"]].isna().any().any() or \
            (~detections["detected"].dropna().isin([0, 1])).any():
        fail("Detection rows must have valid site, visit, and species values; detected must be 0, 1, or missing.")
    if covariates["R_ID"].isna().any() or covariates["R_ID"].duplicated().any():
        fail("Covariate R_ID values must be present and unique.")
    detections = (detections.assign(detected=detections["detected"].fillna(-1))
                  .groupby(["site", "visit", "species"], as_index=False)["detected"].max())
    detections.loc[detections["detected"] < 0, "detected"] = np.nan

    sites = sorted(detections["site"].unique())
    visits = sorted(detections["visit"].unique())
    species = sorted(detections["species"].unique())
    site_lookup = covariates.assign(
        Project=covariates["Project"].str.strip(),
        label=[s if s.startswith(p + "_") else f"{p}_{s}" for p, s in
               zip(covariates["Project"].str.strip(), covariates["Site"].str.strip())],
    ).set_index("label")
    if site_lookup.index.duplicated().any():
        fail("Covariates must include a unique, non-missing Project/Site label for each R_ID.")
    missing_sites = [site for site in sites if site not in site_lookup.index]
    if missing_sites:
        fail("Missing site covariates for location labels: " + ", ".join(missing_sites))
    site_projects = site_lookup.loc[sites, "Project"].to_numpy()
    species_projects = (pd.DataFrame({"species": chorus["scientific_name"], "site": location})
                        .drop_duplicates()
                        .assign(project=lambda frame: frame["site"].map(site_lookup["Project"])))
    if species_projects["project"].isna().any():
        fail("Chorus contains species or locations that cannot be mapped to the covariates.")
    species_projects = species_projects[["species", "project"]].drop_duplicates()
    if species_projects["species"].duplicated().any() or \
            (~species_projects["project"].isin(["Calima", "San_Carlos"])).any():
        fail("Each Chorus species must belong to exactly one project: Calima or San_Carlos.")
    species_project = species_projects.set_index("species")["project"]

    # 9. Standardize site-level covariates (across all detection sites)
    site_covariates = site_lookup.loc[sites, covariate_names].to_numpy(dtype=float)
    site_covariates[site_covariates == NODATA] = np.nan
    if (~np.isfinite(site_covariates)).any():
        fail("Model covariates contain missing or non-finite values for detection sites.")
    centers = site_covariates.mean(axis=0)
    scales = site_covariates.std(axis=0, ddof=1)
    if (~np.isfinite(scales)).any() or (scales == 0).any():
        fail("At least one site covariate has zero or non-finite variance.")
    site_covariates = (site_covariates - centers) / scales

    # 10. Build the detection-history array (site x visit x species, NaN = not observed)
    y = np.full((len(sites), len(visits), len(species)), np.nan)
    y[
        pd.Index(sites).get_indexer(detections["site"]),
        pd.Index(visits).get_indexer(detections["visit"]),
        pd.Index(species).get_indexer(detections["species"]),
    ] = detections["detected"].to_numpy(dtype=float)

    # 11. Align rasters and prepare prediction cells
    with rasterio.open(covariate_files[0]) as template:
        profile = template.profile.copy()
        shape = (template.height, template.width)
        raster_transform = template.transform
        raster_crs = template.crs
    stack = np.empty((len(covariate_names),) + shape)
    for index, name in enumerate(covariate_names):
        with rasterio.open(covariate_rasters[name]) as source:
            band = source.read(1).astype(float)
            if source.nodata is not None:
                band[band == source.nodata] = np.nan
            if source.crs != raster_crs or source.transform != raster_transform or source.shape != shape:
                aligned = np.full(shape, np.nan)
                reproject(band, aligned, src_transform=source.transform, src_crs=source.crs,
                          dst_transform=raster_transform, dst_crs=raster_crs,
                          src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.bilinear)
                band = aligned
        band[band == NODATA] = np.nan
        stack[index] = band
    normalized = (stack - centers[:, None, None]) / scales[:, None, None]
    stack_profile = {**profile, "count": len(covariate_names), "dtype": "float32", "nodata": np.nan}
    for array, file_name in [(stack, "covariates_raw.tif"), (normalized, "covariates_standardized.tif")]:
        with rasterio.open(COVARIATE_OUTPUT_DIR / file_name, "w", **stack_profile) as destination:
            destination.write(array.astype("float32"))
            for band, name in enumerate(covariate_names, start=1):
                destination.set_band_description(band, name)

    raw_cells = stack.reshape(len(covariate_names), -1).T
    normalized_cells = normalized.reshape(len(covariate_names), -1).T
    complete_cells = np.isfinite(normalized_cells).all(axis=1)
    prediction_cells = np.flatnonzero(complete_cells)
    prediction_data = normalized_cells[complete_cells]
    if not len(prediction_data):
        fail("No raster cells have complete, finite covariates.")
    cell_rows, cell_cols = np.divmod(prediction_cells, shape[1])
    cell_x, cell_y = rasterio.transform.xy(raster_transform, cell_rows, cell_cols, offset="center")
    pd.concat([
        pd.DataFrame({"cell": prediction_cells + 1, "x": cell_x, "y": cell_y}),  # 1-based, row-major cells
        pd.DataFrame(raw_cells[complete_cells], columns=covariate_names),
        pd.DataFrame(prediction_data, columns=[f"{name}_scaled" for name in covariate_names]),
    ], axis=1).to_csv(COVARIATE_OUTPUT_DIR / "raster_covariates.csv", index=False)
    bounds = rasterio.transform.array_bounds(shape[0], shape[1], raster_transform)
    raster_metadata = {
        "projection_scope": "all complete cells in the buffered raster extent, including the other project",
        "crs": raster_crs.to_wkt() if raster_crs else None,
        "rows": shape[0], "columns": shape[1],
        "total_cells": shape[0] * shape[1], "predicted_cells": int(complete_cells.sum()),
        "extent": {"xmin": bounds[0], "xmax": bounds[2], "ymin": bounds[1], "ymax": bounds[3]},
        "resolution": {"x": raster_transform.a, "y": -raster_transform.e},
        "layers": RASTER_LAYERS,
        "layer_description": "posterior mean, posterior SD and 95% credible interval",
    }

    # 12. Fit both models per species and export species-specific results
    model_terms = ["(Intercept)"] + covariate_names
    model_definitions = {
        "occupancy": {
            "formula": "~ 1 ~ " + " + ".join(covariate_names),
            "description": "Single-season occupancy model: z_i ~ Bernoulli(psi_i), y_ij ~ Bernoulli(z_i * p); "
                           "logit(psi_i) = X_i * beta_state; constant detection p. z_i is marginalised out.",
            "prediction": "psi: unconditional occupancy probability",
            "file_suffix": "psi",
        },
        "logistic": {
            "formula": "y_ij ~ Bernoulli(p_i); logit(p_i) = " + " + ".join(model_terms),
            "description": "Logistic regression with one row per observed site-visit: y_ij ~ Bernoulli(p_i), "
                           "logit(p_i) = X_i * beta. No latent occupancy state and no site-level term.",
            "prediction": "p: fitted Bernoulli probability, logit(p) = X * beta",
            "file_suffix": "p",
        },
    }
    priors = "logistic(0, 1) on every intercept and coefficient"
    status_columns = ["detection_records", "observed_site_visits", "positive_detections", "missing_site_visits"]
    model_status = {
        name: pd.DataFrame({
            "species": species, **{column: 0 for column in status_columns}, "status": "",
            "max_rhat": np.nan, "min_ess_bulk": np.nan, "min_ess_tail": np.nan,
            "divergent_transitions": pd.NA, "max_treedepth_hits": pd.NA, "min_bfmi": np.nan,
            "min_posterior_contraction": np.nan, "max_prior_posterior_overlap": np.nan,
            "terms_overlap_above_threshold": "",
            "ppc_total_detections_p": np.nan, "ppc_sites_with_detections_p": np.nan,
        }) for name in model_definitions
    }
    site_tables = {name: [] for name in model_definitions}
    parameter_tables = {name: [] for name in model_definitions}
    detection_records = detections.groupby("species").size()
    runtime = {
        "python_version": platform.python_version(),
        "pymc_version": pm.__version__,
        "arviz_version": az.__version__,
        "numpy_version": np.__version__,
    }

    for index, species_name in enumerate(species):
        name = safe_name(species_name)
        project_sites = np.flatnonzero(site_projects == species_project[species_name])
        response = y[project_sites, :, index]
        project_covariates = site_covariates[project_sites]
        observed = naive_occupancy(response)
        counts = {
            "detection_records": int(detection_records.get(species_name, 0)),
            "observed_site_visits": int((~np.isnan(response)).sum()),
            "positive_detections": int(np.nansum(response == 1)),
            "missing_site_visits": int(np.isnan(response).sum()),
        }
        print(f"[{index + 1}/{len(species)}] {species_name}: {counts['positive_detections']} detections", flush=True)

        for model_name, definition in model_definitions.items():
            model_dir = MODEL_OUTPUT_DIRS[model_name]
            status_table = model_status[model_name]
            for column, value in counts.items():
                status_table.loc[index, column] = value
            json_path = Path("JSON") / f"{name}_{definition['file_suffix']}.json"
            summary = {
                "schema_version": "3.0",
                "model_type": model_name,
                "species": species_name,
                "project": species_project[species_name],
                "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "status": None,
                "fit_error": None,
                "output": {"path_base": f"4_Expected_output/{model_dir.name}", "json": str(json_path),
                           "posterior_netcdf": None, "geotiff": None},
                "data": {
                    **{key: value for key, value in counts.items() if key != "detection_records"},
                    "number_of_sites": len(project_sites),
                    "number_of_visits": len(visits),
                    "surveyed_sites": observed["surveyed_sites"],
                    "detected_sites": observed["detected_sites"],
                    "naive_occupancy": observed["naive_occupancy"],
                    "site_ids": [sites[site] for site in project_sites],
                    "visit_ids": visits,
                },
                "covariates": {
                    "scaling": pd.DataFrame({"covariate": covariate_names, "model_name": covariate_names,
                                             "mean": centers, "sd": scales}),
                    "normalization_population": "detection sites across both projects",
                    "raster_sources": [f"2_Other_inputs/Raster_covariates/{covariate_rasters[c].name}"
                                       for c in covariate_names],
                },
                "model": None, "raster": None, "predictions": None,
                "inputs": {
                    "ARU_coordinates": f"3_Intermediate_outputs/{ARU_COORDINATES_FILE.name}",
                    "site_covariates": f"3_Intermediate_outputs/{COVARIATES_FILE.name}",
                    "detection_history": f"3_Intermediate_outputs/{DETECTIONS_FILE.name}",
                    "chorus": f"1_Input_from_chorus/{CHORUS_FILE.name}",
                },
                "runtime": runtime,
            }

            if counts["positive_detections"] == 0:
                summary["status"] = "skipped_no_detections"
            else:
                build = occupancy_model if model_name == "occupancy" else logistic_model
                idata, fit_warnings, error = sample(build(response, project_covariates))
                if idata is None:
                    summary["fit_error"] = error
                    summary["status"] = f"failed: {error}"
                else:
                    if model_name == "occupancy":
                        state = np.concatenate([posterior_array(idata, "state_intercept")[..., None],
                                                posterior_array(idata, "state_beta")], axis=-1)
                        detection = posterior_array(idata, "detection_intercept")
                        parameters = pd.concat([
                            parameter_table({term: state[..., k] for k, term in enumerate(model_terms)},
                                            model_terms, "state"),
                            parameter_table({"(Intercept)": detection}, ["(Intercept)"], "detection"),
                        ], ignore_index=True)
                        coefficient_draws = state.reshape(-1, state.shape[-1])
                        detection_draws = 1.0 / (1.0 + np.exp(-detection.reshape(-1)))
                        site_draws = probability_draws(project_covariates, coefficient_draws)
                        site_predictions = pd.concat([
                            summarise_draws(site_draws, "occupancy"),
                            pd.DataFrame({
                                "detection_probability_mean": detection_draws.mean(),
                                "detection_probability_lower_95": np.quantile(detection_draws, 0.025),
                                "detection_probability_upper_95": np.quantile(detection_draws, 0.975),
                            }, index=range(len(project_sites))),
                            summarise_draws(conditional_occupancy_draws(response, site_draws, detection_draws),
                                            "conditional_occupancy"),
                        ], axis=1)
                    else:
                        coefficients = np.concatenate([posterior_array(idata, "intercept")[..., None],
                                                       posterior_array(idata, "beta")], axis=-1)
                        parameters = parameter_table({term: coefficients[..., k] for k, term in enumerate(model_terms)},
                                                     model_terms, "detection_per_visit")
                        coefficient_draws = coefficients.reshape(-1, coefficients.shape[-1])
                        site_draws = probability_draws(project_covariates, coefficient_draws)
                        site_predictions = summarise_draws(site_draws, "detection_probability")
                        detection_draws = None
                    cell_draws = probability_draws(prediction_data, coefficient_draws)

                    diagnostics = mcmc_diagnostics(idata, parameters)
                    summary["status"] = "converged" if diagnostics["converged"] else "not_converged"
                    for column in ["max_rhat", "min_ess_bulk", "min_ess_tail", "divergent_transitions",
                                   "max_treedepth_hits", "min_bfmi", "min_posterior_contraction",
                                   "max_prior_posterior_overlap"]:
                        status_table.loc[index, column] = diagnostics[column]
                    status_table.loc[index, "terms_overlap_above_threshold"] = "; ".join(
                        diagnostics["terms_overlap_above_threshold"])
                    ppc = posterior_predictive_check(response, site_draws, detection_draws, MCMC_SEED + index)
                    for statistic, check in ppc.items():
                        status_table.loc[index, f"ppc_{statistic}_p"] = check["p_value"]

                    netcdf_path = Path("NETCDF") / f"{name}_posterior.nc"
                    geotiff_path = Path("GEOTIFF") / f"{name}_{definition['file_suffix']}.tif"
                    idata.to_netcdf(model_dir / netcdf_path)
                    cell_summary = summarise_draws(cell_draws, "value")
                    cell_summary.columns = RASTER_LAYERS
                    write_layers(profile, shape, prediction_cells, cell_summary, model_dir / geotiff_path)
                    summary["output"]["posterior_netcdf"] = str(netcdf_path)
                    summary["output"]["geotiff"] = str(geotiff_path)

                    site_table = pd.concat([
                        pd.DataFrame({
                            "species": species_name,
                            "site": [sites[site] for site in project_sites],
                            "observed_visits": (~np.isnan(response)).sum(axis=1),
                            "positive_detections": np.nansum(response == 1, axis=1).astype(int),
                            "detected_at_site": observed["detected_at_site"],
                            "naive_occupancy": observed["naive_occupancy"],
                        }),
                        site_predictions,
                    ], axis=1)
                    site_tables[model_name].append(site_table)
                    parameter_tables[model_name].append(parameters.assign(species=species_name)[
                        ["species"] + list(parameters.columns)])

                    summary["model"] = {
                        "description": definition["description"],
                        "formula": definition["formula"],
                        "priors": priors,
                        "coefficient_scale": "logit (covariates standardized)",
                        "mcmc": {"chains": MCMC_CHAINS, "tune": MCMC_TUNE, "draws": MCMC_DRAWS,
                                 "seed": MCMC_SEED, "posterior_draws": len(coefficient_draws)},
                        "diagnostics": {**diagnostics, "rhat_threshold": RHAT_THRESHOLD,
                                        "ess_threshold": ESS_THRESHOLD, "bfmi_threshold": BFMI_THRESHOLD,
                                        "overlap_threshold": OVERLAP_THRESHOLD},
                        "prior": {"distribution": "logistic", "location": PRIOR_MEAN, "scale": PRIOR_SCALE,
                                  "sd": PRIOR_SD},
                        "posterior_predictive_check": ppc,
                        "parameter_estimates": parameters,
                        "fit_warnings": fit_warnings,
                    }
                    summary["raster"] = {
                        "value": definition["prediction"], **raster_metadata,
                        "posterior_mean_summary": {"min": cell_summary["mean"].min(),
                                                   "mean": cell_summary["mean"].mean(),
                                                   "max": cell_summary["mean"].max()},
                    }
                    summary["predictions"] = {
                        "site_scope": "original species search project only",
                        "value": definition["prediction"],
                        "interval": "95% equal-tailed posterior credible interval",
                        "naive_occupancy_scope": "Proportion of surveyed sites with at least one detection in "
                                                 "this species' project; repeated across site rows. "
                                                 "detected_at_site is the site-level 0/1 indicator.",
                        "conditional_occupancy": (
                            "Posterior of z_i given the detection history; detected sites have probability 1 "
                            "under the no-false-positive assumption." if model_name == "occupancy" else None),
                        "by_site": site_table,
                    }
            status_table.loc[index, "status"] = summary["status"]
            with open(model_dir / json_path, "w") as handle:
                json.dump(to_json_ready(summary), handle, indent=2)

    # 13. Write combined tables and report run results
    for model_name, model_dir in MODEL_OUTPUT_DIRS.items():
        model_status[model_name].to_csv(model_dir / "CSV" / "model_status.csv", index=False)
        if site_tables[model_name]:
            pd.concat(site_tables[model_name]).to_csv(model_dir / "CSV" / "site_predictions.csv", index=False)
            pd.concat(parameter_tables[model_name]).to_csv(model_dir / "CSV" / "parameter_estimates.csv",
                                                           index=False)
    print("Inputs:", COVARIATES_FILE, DETECTIONS_FILE, *covariate_rasters.values(), sep="\n ")
    fitted_any = False
    for model_name, table in model_status.items():
        fitted = table["status"].isin(["converged", "not_converged"])
        fitted_any |= bool(fitted.any())
        overlap = (table["terms_overlap_above_threshold"] != "").sum()
        ppc_outside = (~table["ppc_sites_with_detections_p"].between(*PPC_BOUNDS)
                       | ~table["ppc_total_detections_p"].between(*PPC_BOUNDS))[fitted].sum()
        print(f"{model_name}: fitted {fitted.sum()} of {len(species)} species; "
              f"converged: {(table['status'] == 'converged').sum()}; "
              f"failed: {table['status'].str.startswith('failed').sum()}; "
              f"with prior-posterior overlap > {OVERLAP_THRESHOLD}: {overlap}; "
              f"posterior predictive p outside {PPC_BOUNDS}: {ppc_outside}")
    print("Outputs written to:", OUTPUT_DIR)
    if not fitted_any:
        fail("No species models fitted successfully.")


if __name__ == "__main__":
    main()
