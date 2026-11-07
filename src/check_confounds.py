"""Is the model detecting Parkinson's, or just sex? TRAIN subjects only.

Healthy controls are mostly female, PD patients mostly male, and men tend to
move with more force. So a model could score well by learning "male => PD".
Checks:
  1. How well does sex ALONE predict PD? (the shortcut's ceiling)
  2. How well do the movement features predict SEX? (is the information there?)
  3. Does the PD model still work WITHIN each sex? If it is mostly a sex
     detector, it should fall to ~0.5 AUC among women only / men only.
     Repeated for every sex correction in train.CORRECTIONS. A good correction
     keeps AUC within each sex and makes the false positive rate similar for
     healthy men and healthy women.
  4. Same within age groups, split at the median.

Usage:  python -m src.check_confounds [--features all|base|...] [--rule max_fpr|...]
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import SEX_COL
from src.load import load_subjects
from src.split import load_split
from src.train import (CORRECTIONS, FEATURE_SETS, THRESHOLD_RULES, load_xy, metrics,
                       nested_cv_predictions, oof_scores)

SHOW = ["roc_auc", "balanced_accuracy", "sensitivity", "false_positive_rate"]


def by_group(groups: pd.Series, y, pred, score) -> dict[str, dict[str, float]]:
    rows = {"everyone": metrics(y, pred, score)}
    for g in sorted(groups.unique()):
        m = (groups == g).to_numpy()
        rows[g] = metrics(y[m], pred[m], score[m])
    return {g: {k: r[k] for k in SHOW} for g, r in rows.items()}


def counts(groups: pd.Series, y) -> str:
    return ", ".join(f"{g}: {(y[(groups == g).to_numpy()] == 1).sum()} PD / "
                     f"{(y[(groups == g).to_numpy()] == 0).sum()} healthy"
                     for g in sorted(groups.unique()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", choices=list(FEATURE_SETS), default="all")
    parser.add_argument("--rule", choices=THRESHOLD_RULES, default="max_fpr")
    args = parser.parse_args()

    train_ids, _ = load_split()
    info = load_subjects().set_index("id").loc[train_ids]
    X, y = load_xy(train_ids, args.features)
    sex = info["gender"]
    male = (sex == "male").astype(int).to_numpy()

    print(f"Train subjects by sex -- {counts(sex, y)}")
    print(f"\n1. Sex alone -> PD:            AUC {roc_auc_score(y, male):.3f}")
    # Movement features only (never the sex column itself!), cross-validated.
    X_blind = X.copy()
    X_blind[SEX_COL] = 0  # PDModel("none") ignores it anyway; zeroed to make the point explicit
    sex_score = oof_scores(X_blind, male)
    print(f"2. Movement features -> sex:   AUC {roc_auc_score(male, sex_score):.3f}")

    print(f"\n3. PD model [{args.features} @ {args.rule}], nested CV, per sex correction:")
    rows = {}
    for correction in CORRECTIONS:
        score, pred, _ = nested_cv_predictions(X, y, args.rule, correction)
        for group, r in by_group(sex, y, pred, score).items():
            rows[(correction, group)] = r
    table = pd.DataFrame(rows).T
    table.index.names = ["correction", "group"]
    print(table.round(3).to_string())

    older = pd.Series(np.where(info["age"] >= info["age"].median(), "older", "younger"), index=info.index)
    print(f"\n4. Within age groups (median {info['age'].median():.0f}; {counts(older, y)}), no correction:")
    score, pred, _ = nested_cv_predictions(X, y, args.rule, "none")
    print(pd.DataFrame(by_group(older, y, pred, score)).T.round(3).to_string())


if __name__ == "__main__":
    main()
