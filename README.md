# PADS Parkinson's screening — 27020 NEURO Group 5

Classify Parkinson's disease vs. healthy controls from smartwatch movement data
(PADS dataset, PhysioNet). Proof of concept for a smartphone-based screening tool.

**Poster supplement:** the screening, scale and cost calculations behind the poster are in
[CALCULATIONS.md](CALCULATIONS.md).

KPIs from the pitch: detection accuracy > 70%, false positive rate < 25%.
Because PD subjects outnumber healthy ones ~3.5:1, plain accuracy is misleading
(always guessing "PD" already scores ~78%), so we report balanced accuracy,
sensitivity, specificity / FPR and ROC AUC.

## Dataset

Varghese, J. et al. *PADS — Parkinson's Disease Smartwatch dataset* (v1.0.0), PhysioNet.
https://doi.org/10.13026/m0w9-zx22 — licence CC BY-NC-SA 4.0.

469 subjects, one assessment session each: 11 movement tasks recorded with an
Apple Watch on each wrist (accelerometer + gyroscope, 100 Hz), plus a 30-item
non-motor symptom questionnaire. Labels: healthy, Parkinson's, other movement disorders.

The data is **not** in this repo. `src/download.py` fetches the preprocessed
subset (~240 MB) into `data/raw/`.

## Setup

Use Python 3.12 so the pinned versions in `requirements.txt` install and the
numbers match:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Pipeline

Run from the repo root, in order:

| Step | Command | Output |
|------|---------|--------|
| 1. Download | `python -m src.download` | `data/raw/preprocessed/...` |
| 2. Features | `python -m src.features` | `data/processed/features.csv` |
| 3. Split    | `python -m src.split` | `splits/<target>_{train,test}_ids.txt` (committed) |
| 4. Train / CV | `python -m src.train` | `results/<target>_cv_metrics.json` |
| 4b. Bias check | `python -m src.check_confounds` | printed: performance within sex / age groups |
| 5. Test (once, done) | see [Test-set results](#test-set-results-scored-once-2026-11-07) | `results/<target>_test_<model>_<features>_<rule>_<correction>.json` |

Settings (target task, seed, test fraction) live in `src/config.py`.

`src.train` options (comma-separated lists are compared in CV mode; `--final` takes one of each):

- `--models logreg|svm|rf|hgb|logreg+rf|logreg+svm+rf`: classifier (`a+b` = average of both).
- `--features base|base+asym|base+trem|all|quest|all+quest`: feature groups.
  `all` = everything from the watch, `quest` = the 30-item symptom questionnaire.
- `--rules fixed_0.5|youden|max_fpr`: how the decision cut-off is picked (`max_fpr`:
  most PD caught while false positive rate stays at or below 20%). Chosen with nested CV.
- `--correction none|sex_feature|residualize|reweight`: optional sex-imbalance correction.
- `--repeats N`: also report ROC AUC mean/sd over N reshuffled CV runs.

## Current results (nested 5-fold CV, train subjects only, `max_fpr` cut-off)

| Model | Inputs | ROC AUC (5 repeats) | Balanced acc. | Sensitivity | FPR |
|---|---|---|---|---|---|
| always guess PD | – | – | 0.500 | 1.000 | 1.000 |
| logreg | watch | 0.833 ± 0.012 | 0.775 | 0.724 | 0.175 |
| rf | watch | 0.863 ± 0.009 | 0.769 | 0.760 | 0.222 |
| **logreg+rf** | **watch** | 0.852 ± 0.012 | **0.797** | 0.769 | **0.175** |
| rf | questionnaire | 0.937 ± 0.008 | 0.847 | 0.900 | 0.206 |
| **rf** | **watch + questionnaire** | **0.946 ± 0.007** | **0.870** | 0.914 | **0.175** |

Both models hold up within each sex, age group and disease duration
(`python -m src.check_confounds --model rf --features all+quest`). Only 63 healthy
subjects in train, so FPR moves about 1.6 points per person.

## Final models (fixed before the test set)

Chosen from the cross-validation results above, using train subjects only, and
committed **before** the test set is scored. Each is scored on the test set
exactly once and reported as-is: no re-tuning or swapping models afterwards.

| | Model | Inputs | Cut-off rule | Command |
|---|---|---|---|---|
| 1. Watch only | `logreg+rf` | `all` (558 watch features) | `max_fpr` (FPR ≤ 20% on train) | `python -m src.train --final --models logreg+rf --features all --rules max_fpr` |
| 2. Watch + questionnaire | `rf` | `all+quest` (558 + 30) | `max_fpr` | `python -m src.train --final --models rf --features all+quest --rules max_fpr` |

Task: PD (label 1) vs healthy (label 0); the 114 "other movement disorder"
subjects are excluded. Test set: 71 subjects (55 PD, 16 healthy), listed in `splits/`.

Pass criteria (from the pitch KPIs): balanced accuracy > 70% and false positive
rate < 25%. With 16 healthy test subjects each one is 6.25 points of FPR, so
expect the test FPR to vary by roughly ±10–15 points from chance alone.

`--final` refuses to overwrite an existing test result, so a model cannot be
quietly re-scored.

## Test-set results (scored once, 2026-11-07)

Both final models were scored exactly once on the 71 locked test subjects, with
the settings above. Nothing was changed afterwards.

| | Model 1: watch only | Model 2: watch + questionnaire |
|---|---|---|
| Cut-off (chosen on train) | 0.802 | 0.653 |
| Balanced accuracy (KPI > 70%) | **0.784** — pass | **0.888** — pass |
| False positive rate (KPI < 25%) | **0.250** (4/16) — on the limit, not below | **0.188** (3/16) — pass |
| Sensitivity | 0.818 (45/55) | 0.964 (53/55) |
| Specificity | 0.750 (12/16) | 0.812 (13/16) |
| ROC AUC | 0.932 | 0.953 |
| Accuracy | 0.803 | 0.930 |

95% intervals (Clopper–Pearson for rates, 5000 stratified bootstrap resamples
for balanced accuracy and AUC, `python -m src.test_intervals`):

| | Model 1 | Model 2 |
|---|---|---|
| Balanced accuracy | 0.66–0.89 | 0.78–0.97 |
| False positive rate | 0.07–0.52 | 0.04–0.46 |
| ROC AUC | 0.86–0.98 | 0.89–1.00 |

Test results agree with cross-validation (balanced accuracy 0.797 vs 0.784 and
0.870 vs 0.888), so the models did not overfit to the train subjects.

Caveat on sex: in the test set every false positive was a healthy man
(Model 1: 4/9 men, 0/7 women; Model 2: 3/9 men, 0/7 women), and Model 1 caught
fewer women with PD (11/17) than men (34/38). Cross-validation showed no sex
effect and these groups are very small, but it should not be reported as
"equal performance across sexes".

## Ground rules

- **Split by subject.** Each person is in train *or* test, never both.
- **The test set is locked.** Explore, tune and compare models with cross-validation
  on the train subjects only. Both final models have been scored (see above).
- **Preprocessing inside the pipeline.** Scaling/PCA are fit on training folds only
  (sklearn `Pipeline`), never on the full dataset.

## Layout

```
src/
  config.py    paths and constants
  download.py  fetch PADS from PhysioNet (checksum-verified)
  load.py      read labels, signals, questionnaire
  features.py  per task/wrist/sensor signal features
  split.py     subject-level stratified train/test split
  train.py     model, threshold rules, nested CV and final test evaluation
  check_confounds.py  does the model just detect sex/age? early vs late PD?
  test_intervals.py   95% intervals + per-sex counts for the scored test results
splits/        shared train/test subject IDs
notebooks/     exploration
results/       metrics (gitignored)
```
