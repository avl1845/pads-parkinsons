"""Train and evaluate a baseline model. PLACEHOLDER: model choice is still open.

Default: cross-validation on the TRAIN subjects only. Scaling and PCA sit inside
the sklearn Pipeline, so they are re-fit on each training fold and never see the
held-out fold.

Decision threshold: the model outputs P(PD); we flag PD when P >= threshold.
Picking the threshold is itself "training", so it is done with NESTED CV:
  outer 5 folds -> on the 4 training folds, an inner 5-fold CV produces
  out-of-fold scores, the threshold is chosen on those, then applied unchanged
  to the 5th (outer) fold. The outer-fold numbers are therefore a fair estimate.

--final picks the threshold with CV on all train subjects, fits on all train
subjects, and scores the locked TEST subjects once. Only run it when the group
has agreed on the final model and threshold rule.

CV mode compares every feature set in FEATURE_SETS against every threshold rule.

Usage:  python -m src.train [--final] [--rule max_fpr|...] [--features all|...] [--correction none|...]
"""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import PROCESSED_DIR, RANDOM_SEED, RESULTS_DIR, SEX_COL, TARGET
from src.split import load_split

N_FOLDS = 5
# Aim below the 25% FPR KPI with some margin: a threshold tuned on ~50 healthy
# subjects is noisy and tends to do a bit worse on unseen people.
MAX_FPR = 0.20
THRESHOLD_RULES = ["fixed_0.5", "youden", "max_fpr"]
# Feature groups are column prefixes written by src/features.py.
FEATURE_SETS = {
    "base": ["base"],
    "base+asym": ["base", "asym"],
    "base+trem": ["base", "trem"],
    "all": ["base", "trem", "asym"],
}
CORRECTIONS = ["none", "sex_feature", "residualize", "reweight"]


class PDModel(ClassifierMixin, BaseEstimator):
    """Scale -> PCA (keep 95% of variance) -> logistic regression, with an optional
    correction for the sex imbalance (healthy group mostly female, PD mostly male).

    correction:
      none         ignore sex entirely (the original baseline).
      sex_feature  give the model sex as an extra input next to the PCA components,
                   so it can learn "men move more anyway" and allow for it.
      residualize  remove the average male/female difference from every movement
                   feature. The difference is estimated on the training fold with
                   PD status held fixed (feature ~ sex + PD), so PD signal is kept.
      reweight     weight training subjects so the four groups (female/male x
                   healthy/PD) count equally: in the weighted data, sex no longer
                   predicts PD, so the shortcut is worthless.

    All of it is learned in fit() from the training fold only.
    """

    def __init__(self, correction: str = "none"):
        self.correction = correction

    def fit(self, X: pd.DataFrame, y):
        if self.correction not in CORRECTIONS:
            raise ValueError(f"unknown correction {self.correction!r}")
        y = np.asarray(y)
        move, male = self._split(X)
        if self.correction == "residualize":
            design = np.column_stack([np.ones_like(male), male, y])
            self.sex_effect_ = np.linalg.lstsq(design, move, rcond=None)[0][1]
        self.reduce_ = Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=0.95, random_state=RANDOM_SEED)),
        ]).fit(self._correct(move, male))
        weight = None
        if self.correction == "reweight":
            cell = male * 2 + y                      # 0..3: (sex, label) group
            weight = len(y) / (4 * np.bincount(cell, minlength=4)[cell])
        self.clf_ = LogisticRegression(class_weight=None if weight is not None else "balanced",
                                       max_iter=5000)
        self.clf_.fit(self._inputs(move, male), y, sample_weight=weight)
        self.classes_ = self.clf_.classes_
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.clf_.predict_proba(self._inputs(*self._split(X)))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    @staticmethod
    def _split(X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return X.drop(columns=SEX_COL).to_numpy(float), X[SEX_COL].to_numpy(int)

    def _correct(self, move, male):
        return move - np.outer(male, self.sex_effect_) if self.correction == "residualize" else move

    def _inputs(self, move, male):
        z = self.reduce_.transform(self._correct(move, male))
        return np.column_stack([z, male]) if self.correction == "sex_feature" else z


def build_model(correction: str = "none") -> PDModel:
    return PDModel(correction)


def metrics(y_true, y_pred, y_score) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "sensitivity": tp / (tp + fn),       # PD correctly flagged
        "specificity": tn / (tn + fp),       # non-PD correctly cleared
        "false_positive_rate": fp / (fp + tn),  # KPI: should be < 0.25
        "roc_auc": roc_auc_score(y_true, y_score) if len(set(y_score)) > 1 else float("nan"),
    }


def load_xy(ids: list[str], feature_set: str) -> tuple[pd.DataFrame, np.ndarray]:
    feats = pd.read_csv(PROCESSED_DIR / "features.csv", dtype={"id": str}).set_index("id")
    subset = feats.loc[ids]
    groups = tuple(f"{g}__" for g in FEATURE_SETS[feature_set])
    # SEX_COL always rides along; PDModel only uses it when a correction asks for it.
    keep = subset.columns.str.startswith(groups) | (subset.columns == SEX_COL)
    return subset.loc[:, keep], subset["y"].to_numpy()


def folds() -> StratifiedKFold:
    return StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)


def oof_scores(X, y, correction: str = "none") -> np.ndarray:
    """Out-of-fold P(PD) for every subject in X: each is scored by a model that never saw it."""
    return cross_val_predict(build_model(correction), X, y, cv=folds(), method="predict_proba")[:, 1]


def choose_threshold(y, score, rule: str) -> float:
    """Pick a cut-off on P(PD) from (out-of-fold) scores."""
    if rule == "fixed_0.5":
        return 0.5
    fpr, tpr, thr = roc_curve(y, score)  # predict PD when score >= thr
    if rule == "youden":                 # best sensitivity + specificity (= balanced accuracy)
        i = np.argmax(tpr - fpr)
    elif rule == "max_fpr":              # catch as many PD as possible while FPR <= MAX_FPR
        ok = np.flatnonzero(fpr <= MAX_FPR)
        i = ok[np.argmax(tpr[ok])]
    else:
        raise ValueError(f"unknown rule {rule!r}")
    return float(min(thr[i], 1.0))


def nested_cv(X, y, rule: str, correction: str = "none") -> tuple[dict[str, float], list[float]]:
    """Fair estimate of model + threshold rule: threshold chosen on inner folds only."""
    score, pred, thresholds = nested_cv_predictions(X, y, rule, correction)
    return metrics(y, pred, score), thresholds


def nested_cv_predictions(X, y, rule: str, correction: str = "none") -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Per-subject P(PD) and yes/no prediction, each from a model + threshold that never saw them."""
    score = np.empty(len(y))
    pred = np.empty(len(y), dtype=int)
    thresholds = []
    for tr, te in folds().split(X, y):
        X_tr, y_tr = X.iloc[tr], y[tr]
        t = choose_threshold(y_tr, oof_scores(X_tr, y_tr, correction), rule)
        model = build_model(correction).fit(X_tr, y_tr)
        score[te] = model.predict_proba(X.iloc[te])[:, 1]
        pred[te] = (score[te] >= t).astype(int)
        thresholds.append(round(t, 3))
    return score, pred, thresholds


def print_threshold_sweep(X, y, correction: str = "none") -> None:
    """For intuition only: how sensitivity/FPR trade off as the cut-off moves.
    Read off the same scores it describes, so it is NOT a fair estimate -- use nested_cv for that."""
    score = oof_scores(X, y, correction)
    rows = {}
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98]:
        m = metrics(y, (score >= t).astype(int), score)
        rows[f"{t:.2f}"] = {k: m[k] for k in ("sensitivity", "specificity", "false_positive_rate",
                                             "balanced_accuracy")}
    print("\nThreshold sweep (out-of-fold scores, intuition only):")
    print(pd.DataFrame(rows).T.round(3).to_string())


def cross_validate(train_ids: list[str], correction: str = "none") -> dict[str, dict]:
    """Compare every feature set x threshold rule with nested CV on the train subjects."""
    X, y = load_xy(train_ids, "base")
    dummy = DummyClassifier(strategy="most_frequent")
    s = cross_val_predict(dummy, X, y, cv=folds(), method="predict_proba")[:, 1]
    results = {"always_PD (sanity check)": metrics(y, (s >= 0.5).astype(int), s)}
    for feature_set in FEATURE_SETS:
        X, y = load_xy(train_ids, feature_set)
        for rule in THRESHOLD_RULES:
            m, thresholds = nested_cv(X, y, rule, correction)
            results[f"{feature_set} ({X.shape[1] - 1}) @ {rule}"] = {**m, "thresholds_per_fold": thresholds}
    return results


def print_table(results: dict[str, dict]) -> None:
    table = {name: {k: v for k, v in r.items() if k != "thresholds_per_fold"}
             for name, r in results.items()}
    print(pd.DataFrame(table).T.round(3).to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true", help="score the locked test set (do this once)")
    parser.add_argument("--rule", choices=THRESHOLD_RULES, default="max_fpr",
                        help="threshold rule used by --final")
    parser.add_argument("--features", choices=list(FEATURE_SETS), default="all",
                        help="feature set used by --final and the threshold sweep")
    parser.add_argument("--correction", choices=CORRECTIONS, default="none",
                        help="sex correction (see PDModel); compare them with src.check_confounds")
    args = parser.parse_args()

    train_ids, test_ids = load_split()
    X_train, y_train = load_xy(train_ids, args.features)
    print(f"target={TARGET}  train: {len(y_train)} subjects, {y_train.mean():.0%} PD")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not args.final:
        results = cross_validate(train_ids, args.correction)
        print(f"\nNested {N_FOLDS}-fold CV on train subjects (MAX_FPR={MAX_FPR}), "
              f"correction={args.correction}, feature set (n features) @ threshold rule:")
        print_table(results)
        print(f"\n[{args.features}]", end="")
        print_threshold_sweep(X_train, y_train, args.correction)
        out = RESULTS_DIR / f"{TARGET}_cv_metrics.json"
    else:
        threshold = choose_threshold(y_train, oof_scores(X_train, y_train, args.correction), args.rule)
        model = build_model(args.correction).fit(X_train, y_train)
        X_test, y_test = load_xy(test_ids, args.features)
        score = model.predict_proba(X_test)[:, 1]
        pred = (score >= threshold).astype(int)
        results = {f"{args.features} @ {args.rule}, correction={args.correction} (TEST)": {**metrics(y_test, pred, score),
                                                             "threshold": threshold}}
        print(f"\nLocked test set: {len(y_test)} subjects, threshold {threshold:.3f}")
        print(pd.DataFrame(results).round(3).to_string())
        out = RESULTS_DIR / f"{TARGET}_test_metrics.json"

    out.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
