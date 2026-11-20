# sdm_py_models

Code that is run on bird detection data exported from Chorus.

Each top-level directory is one type of analysis:

- `Occurrence/`: species occupancy and logistic regression models, projected over raster covariates
- `Call_density/`: call density per species and point from classifier scores and review labels
- `Point-level/`: no code yet
- `Richness/`: no code yet

## Directory structure

Each analysis directory has the same layout. Data folders are not tracked in git, except `Occurrence/2_Other_inputs/Raster_covariates`.

```
<Analysis>/
  01_SDM.py
  README.md
  1_Input_from_chorus/      files exported from Chorus
  2_Other_inputs/           other inputs (e.g. Raster_covariates/*.tif for Occurrence)
  3_Intermediate_outputs/   written by the script
  4_Expected_output/        written by the script
```

## 1_Input_from_chorus

`Chorus_detection_history.csv`: one row per reviewed detection window. Exported by Chorus.

| Column | Content |
|---|---|
| `scientific_name` | species |
| `common_name` | species common name |
| `model_class` | classifier class |
| `location_name` | point, `<project>_<site>` (e.g. `San_Carlos_10`, `Calima_14`) |
| `latitude`, `longitude` | point coordinates (WGS84) |
| `filename` | recording file name |
| `recording_datetime` | `YYYY-MM-DDTHH:MM:SS` |
| `detection_start_seconds`, `detection_end_seconds` | window within the recording |
| `detection_method` | e.g. `classifier` |
| `model_name`, `model_version` | classifier name and version |
| `score` | classifier score (0 to 1) |
| `review_status` | `confirmed`, `rejected`, `uncertain` or empty (unreviewed) |
| `reviewer` | reviewer initials |
| `license` | licence |

`Unlabelled_scores/*.csv` (Call_density only): logit-scale classifier scores for every 3 s window of every recording, one file per species, no windows left out. These files are not yet exported by Chorus; the script expects this format:

| Column | Content |
|---|---|
| `Start (s)` | window start in the recording |
| `Scientific name` | species (one per file) |
| `Score` | classifier score on the logit scale |
| `File` | recording path; the file name matches `filename` in `Chorus_detection_history.csv` |
| `site` | site number |
| `project` | `Calima` or `San Carlos` |

## 2_Other_inputs/Raster_covariates (Occurrence)

Covariates for the Occurrence models. Every `.tif` in this folder is used as one covariate.

- Single-band GeoTIFF, one per covariate
- File name without `.tif` = covariate name (must start with a letter; letters, digits, `.` and `_` only)
- Values are sampled at each point's `latitude`/`longitude` to fit the models
- The fitted models are projected over every raster cell that has a value in all covariates
- All rasters are aligned to the grid of the first file (alphabetical); nodata = -9999
- Current files: `bio01_ann_temp.tif`, `bio12_ann_precip.tif`, `ndvi_2020.tif`; EPSG:4326, 306 x 244 cells, 0.00898° resolution, extent -76.80 to -74.61 longitude, 3.70 to 6.45 latitude (covers Calima and San Carlos)

## Setup

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python Occurrence/01_SDM.py
```
