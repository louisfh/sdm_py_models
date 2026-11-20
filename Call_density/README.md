# Call_density

Call density per species and point from BirdNET scores and review labels.

1. Bayesian Platt scaling on reviewed windows: logit(p) = a + b * logit(score); a, b ~ Normal(0, 2.5); confirmed = 1, rejected = 0
2. Calibrate every unreviewed window with the posterior of (a, b)
3. Call density at a point = (sum of labels + sum of calibrated probabilities) / number of windows, per posterior draw

Run from the repository root: `.venv/bin/python Call_density/01_call_density.py`

## Inputs
- `1_Input_from_chorus/Chorus_detection_history.csv`
- `1_Input_from_chorus/Unlabelled_scores/<Species>_<project>.csv`: BirdNET confidence per 3 s window (not yet exported by Chorus). Windows below 0.0001 are not listed; they are counted as 20 windows per 1-minute recording with confidence 0.00005.

## Outputs (`4_Expected_output/`)
- `CSV/call_density_by_point.csv`: expected calls and call density per species and point (mean, sd, 95% interval)
- `CSV/calibration_curves.csv`: P(call | score) per species
- `CSV/platt_parameters.csv`, `CSV/platt_status.csv`: posterior summaries and diagnostics
- `NETCDF/`: posterior draws
