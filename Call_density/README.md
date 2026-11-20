# Call_density

Call density per species and point from classifier scores and review labels.

Scores are on the logit scale.

1. Bayesian Platt scaling on the reviewed windows at each point, with a separate (a, b) per species and point: logit(p) = a + b * score; a, b ~ Normal(0, 2.5); confirmed = 1, rejected = 0
2. Calibrate every unreviewed window at the point with the posterior of that point's (a, b). Points without labelled windows are not calibrated.
3. Call density at a point = (sum of labels + sum of calibrated probabilities) / number of windows, per posterior draw

Run from the repository root: `.venv/bin/python Call_density/01_SDM.py`

## Inputs
- `1_Input_from_chorus/Chorus_detection_history.csv`: reviewed windows, exported by Chorus. Used to fit the calibration. `score` is read on the scale set by `CHORUS_SCORE_SCALE` in `01_SDM.py`: `"confidence"` (0 to 1, converted to logit) or `"logit"`.
- `1_Input_from_chorus/Unlabelled_scores/*.csv`: logit-scale classifier scores for every window, used for the unreviewed windows. **These files are not yet exported by Chorus.** The script expects the format below.

### Expected format of `Unlabelled_scores/*.csv`
- One CSV per species, any file name ending in `.csv` (e.g. `<Species>_<project>.csv`)
- One row for every 3 s window of every recording; no windows left out
- Number of windows at a point = number of rows for that point (plus labelled windows not in the file)

| Column | Content |
|---|---|
| `Start (s)` | window start in the recording, matched to `detection_start_seconds` |
| `Scientific name` | species; one species per file, matching `scientific_name` |
| `Score` | classifier score on the logit scale |
| `File` | recording path; the file name (after the last `/` or `\`) is matched to `filename` |
| `site` | site number |
| `project` | `Calima` or `San Carlos`; point = `<project>_<site>`, matching `location_name` |

Other columns are ignored. Scores are rounded to 0.01 when summing calibrated probabilities.

## Outputs (`4_Expected_output/`)
- `CSV/call_density_by_point.csv`: expected calls and call density per species and point (mean, sd, 95% interval)
- `CSV/calibration_curves.csv`: P(call | score) per species and point, for logit scores -10 to 10 in steps of 0.1
- `CSV/platt_parameters.csv`, `CSV/platt_status.csv`: posterior summaries and diagnostics per species and point
- `NETCDF/`: posterior draws, one file per species with a `point` dimension
