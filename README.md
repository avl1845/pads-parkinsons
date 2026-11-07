# PADS Parkinson's screening — 27020 NEURO Group 5

Classify Parkinson's disease vs. healthy controls from smartwatch movement data
(PADS dataset, PhysioNet). Proof of concept for a smartphone-based screening tool.

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

```bash
python3 -m venv .venv
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
| 5. Test (once, at the end) | `python -m src.train --final` | `results/<target>_test_metrics.json` |

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
subjects in train, so FPR moves about 1.6 points per person. Test set not yet scored.

## Ground rules

- **Split by subject.** Each person is in train *or* test, never both.
- **The test set is locked.** Explore, tune and compare models with cross-validation
  on the train subjects only. Run `--final` once the group has picked a model.
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
splits/        shared train/test subject IDs
notebooks/     exploration
results/       metrics (gitignored)
```
