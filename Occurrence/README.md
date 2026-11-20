# Occurrence

Fits two Bayesian models per species (PyMC) and projects them over the covariate rasters.

- Occupancy: z_i ~ Bernoulli(psi_i), y_ij ~ Bernoulli(z_i p), logit(psi_i) = X_i beta, constant p
- Logistic: y_i ~ Bernoulli(p_i), logit(p_i) = X_i beta, one row per site; y_i = naive occupancy (1 if detected on any visit, else 0). Sites with no observed visits are left out.
- Priors: logistic(0, 1) on every intercept and coefficient

Run from the repository root: `.venv/bin/python Occurrence/01_SDM.py`

## Inputs
- `1_Input_from_chorus/Chorus_detection_history.csv` (confirmed = 1, rejected / uncertain = 0, unreviewed = missing)
- `2_Other_inputs/Raster_covariates/*.tif`: one single-band raster per covariate, named after the file

## Outputs (`4_Expected_output/`)
- `Covariates/`: raw and standardized covariate rasters and table
- `Occupancy/`, `Logistic/`:
  - `GEOTIFF/<Species>_psi.tif` / `<Species>_p.tif`: projected map (mean, sd, lower_95, upper_95)
  - `CSV/`: model status and diagnostics, site predictions, parameter estimates
  - `JSON/`: per-species summary
  - `NETCDF/`: posterior draws
