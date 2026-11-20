# Occurrence: Bayesian single-species SDMs

## Run
From the repository root:
```bash
uv venv .venv --python 3.12        # or: python3 -m venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python Occurrence/01_SDM.py
```
- Python packages (`requirements.txt`): `numpy`, `pandas`, `pymc`, `arviz`, `rasterio`, `netCDF4`
- Takes about 2 minutes for 40 species on a laptop
- MCMC settings are at the top of `01_SDM.py` (4 chains x 500 warmup + 500 draws)

## Inputs
- `1_Input_from_chorus/Chorus_detection_history.csv`
  - Required columns: `scientific_name`, `location_name`, `latitude`, `longitude`, `recording_datetime`, `filename`, `detection_start_seconds`, `detection_end_seconds`, `review_status`
  - `location_name` starts with `Calima_` or `San_Carlos_`
  - `review_status`: confirmed = 1, rejected / uncertain = 0, unreviewed = NA
- `2_Other_inputs/Raster_covariates/*.tif`
  - Every single-band .tif here is a covariate
  - File name (without .tif) = covariate name = column name = model term
  - Names must start with a letter and use only letters, digits, `.` and `_` (e.g. `ndvi_2020`, not `2020-ndvi`)
  - -9999 = nodata

## Intermediate outputs (written by the script)
- `3_Intermediate_outputs/ARU_coordinates.csv`: Project, Site, Latitude, Longitude
- `3_Intermediate_outputs/Processed_detection_history.csv`: site, visit, species, detected
- `3_Intermediate_outputs/Processed_covariates.csv`: R_ID, Project, Site, Latitude, Longitude, then one column per covariate
  - Covariates are read from this header: every column except the five ID columns

## Outputs (`4_Expected_output/`)
Previous outputs are deleted at the start of each run.

Shared:
- `Covariates/raster_covariates.csv`: one row per raster cell, raw and `_scaled` covariates
- `Covariates/covariates_raw.tif`, `covariates_standardized.tif`

One folder per model, `Occupancy/` and `Logistic/`, with the same layout:
- `CSV/model_status.csv`: one row per species: fit status, convergence diagnostics, prior-vs-posterior summary, posterior predictive check p-values
- `CSV/site_predictions.csv`: one row per species x site, posterior mean / SD / 95% interval
  - Occupancy: `occupancy_*` (psi), `detection_probability_*` (p), `conditional_occupancy_*`
  - Logistic: `detection_probability_*` (p_i)
- `CSV/parameter_estimates.csv`: one row per species x coefficient (logit scale): posterior summary, MCMC diagnostics and prior-vs-posterior metrics
- `GEOTIFF/<Species>_psi.tif` (Occupancy) or `<Species>_p.tif` (Logistic): model projected over every raster cell with complete covariates; 4 layers: `mean`, `sd`, `lower_95`, `upper_95` (fitted species only)
- `NETCDF/<Species>_posterior.nc`: full posterior (ArviZ InferenceData; fitted species only)
- `JSON/<Species>_psi.json` or `<Species>_p.json`: per-species summary, all species (schema 3.0)

## Models
Both are written in PyMC and use the same covariates (standardized across all
detection sites), the same sites (the species' own project) and the same
priors: logistic(0, 1) on every intercept and coefficient.

- **Occupancy** (`~ 1 ~ <all covariates>`): z_i ~ Bernoulli(psi_i), y_ij ~ Bernoulli(z_i p); logit(psi_i) = X_i beta; constant detection p
  - z_i is marginalised out of the likelihood; conditional occupancy is computed from the posterior draws
- **Logistic**: y_ij ~ Bernoulli(p_i); logit(p_i) = X_i beta, one row per observed site-visit
  - No latent occupancy state and no site-level term
- Species with no detections are skipped (JSON only)

## Diagnostics
Thresholds are set at the top of `01_SDM.py`.

Convergence (status `converged` only if all pass):
- Rhat < 1.01 for every coefficient (rank-normalised split-Rhat)
- Bulk and tail ESS >= 400 for every coefficient
- No divergent transitions, no maximum tree depth hits
- E-BFMI >= 0.3 in every chain
- `mcse_mean` (Monte Carlo error of the posterior mean) is reported per coefficient

Prior vs posterior (all coefficients have a logistic(0, 1) prior, SD = 1.81):
- `posterior_contraction` = 1 - posterior variance / prior variance
- `prior_posterior_overlap` = shared area of the prior and posterior densities
- `overlap_above_threshold` = overlap > 0.35 (Garrett and Zeger 2000)
- `posterior_shift` = |posterior mean - prior mean| / prior SD
- `prob_positive` = posterior probability that the coefficient is above 0

Posterior predictive checks (per species, in `model_status.csv` and the JSON):
- Data are replicated from the posterior and compared with the observed total detections and number of sites with a detection
- p-value = share of replicates >= observed
- `outside_bounds` = p-value outside 0.05–0.95

