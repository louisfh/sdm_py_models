# Call_density

Call density per species and point from classifier scores and review labels.

1. Bayesian Platt scaling on the reviewed windows at each point, with a separate (a, b) per species and point: logit(p) = a + b * logit(score); a, b ~ Normal(0, 2.5); confirmed = 1, rejected = 0
2. Calibrate every unreviewed window at the point with the posterior of that point's (a, b). Points without labelled windows are not calibrated.
3. Call density at a point = (sum of labels + sum of calibrated probabilities) / number of windows, per posterior draw

Run from the repository root: `.venv/bin/python Call_density/01_call_density.py`

## Inputs
- `1_Input_from_chorus/Chorus_detection_history.csv`: reviewed windows, exported by Chorus. Used to fit the calibration.
- `1_Input_from_chorus/Unlabelled_scores/*.csv`: classifier scores for all windows, used for the unreviewed windows. **These files are not yet exported by Chorus.** The script expects the format below.

### Expected format of `Unlabelled_scores/*.csv`
- One CSV per species, any file name ending in `.csv` (e.g. `<Species>_<project>.csv`)
- One row per 3 s window of each 1-minute recording (20 windows per recording)
- Windows with a score below 0.0001 may be left out; they are counted as windows with score 0.00005

| Column | Content |
|---|---|
| `Start (s)` | window start in the recording, matched to `detection_start_seconds` |
| `Scientific name` | species; one species per file, matching `scientific_name` |
| `Confidence` | classifier score (0 to 1) |
| `File` | recording path; the file name (after the last `/` or `\`) is matched to `filename` |
| `site` | site number |
| `project` | `Calima` or `San Carlos`; point = `<project>_<site>`, matching `location_name` |

Other columns are ignored.

## Outputs (`4_Expected_output/`)
- `CSV/call_density_by_point.csv`: expected calls and call density per species and point (mean, sd, 95% interval)
- `CSV/calibration_curves.csv`: P(call | score) per species and point
- `CSV/platt_parameters.csv`, `CSV/platt_status.csv`: posterior summaries and diagnostics per species and point
- `NETCDF/`: posterior draws, one file per species with a `point` dimension
