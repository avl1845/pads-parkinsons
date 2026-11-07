"""Train and evaluate models for PD vs healthy.

Default: cross-validation on the TRAIN subjects only. All preprocessing (scaling,
PCA, sex correction) is learned inside PDModel.fit, so it is re-fit on each
training fold and never sees the held-out fold.

Decision threshold: the model outputs P(PD); we flag PD when P >= threshold.
Picking the threshold is itself "training", so it is done with NESTED CV:
  outer 5 folds -> on the 4 training folds, an inner 5-fold CV produces
  out-of-fold scores, the threshold is chosen on those, then applied unchanged
  to the 5th (outer) fold. The outer-fold numbers are therefore a fair estimate.

CV mode compares every combination of --models x --features x --rules
(comma-separated lists).

--final picks the threshold with CV on all train subjects, fits on all train
subjects, and scores the locked TEST subjects once. Only run it when the group
has agreed on the final model. Needs exactly one model / feature set / rule.

Usage:
  python -m src.train --models logreg,rf,logreg+rf --features all,all+quest --repeats 5
  python -m src.train --final --models rf --features all --rules max_fpr
"""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import PROCESSED_DIR, RANDOM_SEED, RESULTS_DIR, SEX_COL, TARGET
from src.split import load_split

N_FOLDS = 5
# Aim below the 25% FPR KPI with some margin: a threshold tuned on ~50 healthy
# subjects is noisy and tends to do a bit worse on unseen people.
MAX_FPR = 0.20
THRESHOLD_RULES = ["fixed_0.5", "youden", "max_fpr"]
# Feature groups are column prefixes written by src/features.py.
WATCH = ["base", "trem", "asym"]
FEATURE_SETS = {
    "base": ["base"],
    "base+asym": ["base", "asym"],
    "base+trem": ["base", "trem"],
    "all": WATCH,                   # everything from the watch
    "quest": ["quest"],             # questionnaire only
    "all+quest": WATCH + ["quest"],  # watch + questionnaire
}
# Questionnaire answers skip scaling/PCA and go straight to the classifier,
# so 30 yes/no columns are not drowned out among ~560 movement columns.
PASSTHROUGH = ("quest__",)
CORRECTIONS = ["none", "sex_feature", "residualize", "reweight"]

# name: (classifier factory, preprocessing of movement features)
# "balanced" class weights: healthy subjects are rarer, so their errors count more.
MODELS = {
    "logreg": (lambda balanced: LogisticRegression(class_weight=balanced, max_iter=5000), "scale+pca"),
    # SVMs output a distance, not a probability; calibration turns it into one.
    "svm": (lambda balanced: CalibratedClassifierCV(SVC(kernel="rbf", class_weight=balanced),
                                                    ensemble=False), "scale"),
    "rf": (lambda balanced: RandomForestClassifier(
        n_estimators=500, min_samples_leaf=2, class_weight=balanced and "balanced_subsample",
        n_jobs=-1, random_state=RANDOM_SEED), "none"),
    "hgb": (lambda balanced: HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=200, max_leaf_nodes=15, l2_regularization=1.0,
        class_weight=balanced, random_state=RANDOM_SEED), "none"),
}
# Averages of two models (soft vote) can also be requested, e.g. "logreg+rf".
MODEL_CHOICES = list(MODELS) + ["logreg+rf", "logreg+svm+rf"]


class PDModel(ClassifierMixin, BaseEstimator):
    """Movement preprocessing -> classifier, with an optional sex correction.

    model: a key of MODELS.
      logreg  logistic regression on PCA components (keep 95% of variance).
      svm     support vector machine (RBF kernel) on scaled features.
      rf      random forest: 500 decision trees, each on a random subset, voting.
      hgb     gradient boosting: small trees added one by one, each fixing the
              previous ones' mistakes.

    correction (the healthy group is mostly female, PD mostly male):
      none         ignore sex entirely.
      sex_feature  give the model sex as an extra input.
      residualize  remove the average male/female difference from every movement
                   feature, estimated with PD status held fixed (feature ~ sex + PD).
      reweight     weight training subjects so female/male x healthy/PD count equally.
    """

    def __init__(self, model: str = "logreg", correction: str = "none"):
        self.model = model
        self.correction = correction

    def fit(self, X: pd.DataFrame, y):
        if self.correction not in CORRECTIONS:
            raise ValueError(f"unknown correction {self.correction!r}")
        make_clf, prep = MODELS[self.model]
        y = np.asarray(y)
        move, extra, male = self._split(X)
        if self.correction == "residualize" and move.shape[1]:
            design = np.column_stack([np.ones_like(male), male, y])
            self.sex_effect_ = np.linalg.lstsq(design, move, rcond=None)[0][1]
        steps = {"scale+pca": [("scale", StandardScaler()),
                               ("pca", PCA(n_components=0.95, random_state=RANDOM_SEED))],
                 "scale": [("scale", StandardScaler())],
                 "none": []}[prep]
        self.reduce_ = Pipeline(steps).fit(self._correct(move, male)) if steps and move.shape[1] else None
        weight = None
        if self.correction == "reweight":
            cell = male * 2 + y                      # 0..3: (sex, label) group
            weight = len(y) / (4 * np.bincount(cell, minlength=4)[cell])
        self.clf_ = make_clf(None if weight is not None else "balanced")
        self.clf_.fit(self._inputs(move, extra, male), y, sample_weight=weight)
        self.classes_ = self.clf_.classes_
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.clf_.predict_proba(self._inputs(*self._split(X)))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    @staticmethod
    def _split(X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """-> movement features, passthrough features, sex (1 = male)."""
        is_extra = X.columns.str.startswith(PASSTHROUGH)
        is_sex = X.columns == SEX_COL
        return (X.loc[:, ~is_extra & ~is_sex].to_numpy(float), X.loc[:, is_extra].to_numpy(float),
                X[SEX_COL].to_numpy(int))

    def _correct(self, move, male):
        return move - np.outer(male, self.sex_effect_) if hasattr(self, "sex_effect_") else move

    def _inputs(self, move, extra, male):
        z = self._correct(move, male)
        z = self.reduce_.transform(z) if self.reduce_ is not None else z
        parts = [z, extra] + ([male[:, None]] if self.correction == "sex_feature" else [])
        return np.column_stack(parts)


class AverageModel(ClassifierMixin, BaseEstimator):
    """Soft vote: average the P(PD) of several PDModels (e.g. "logreg+rf")."""

    def __init__(self, models: tuple[str, ...] = ("logreg", "rf"), correction: str = "none"):
        self.models = models
        self.correction = correction

    def fit(self, X, y):
        self.members_ = [PDModel(m, self.correction).fit(X, y) for m in self.models]
        self.classes_ = self.members_[0].classes_
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.members_], axis=0)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def build_model(model: str = "logreg", correction: str = "none"):
    """`model` is a MODELS key, or several joined by "+" to average them."""
    if "+" in model:
        return AverageModel(tuple(model.split("+")), correction)
    return PDModel(model, correction)


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


def oof_scores(X, y, correction: str = "none", model: str = "logreg") -> np.ndarray:
    """Out-of-fold P(PD) for every subject in X: each is scored by a model that never saw it."""
    return cross_val_predict(build_model(model, correction), X, y, cv=folds(),
                             method="predict_proba")[:, 1]


def repeated_auc(X, y, correction: str = "none", model: str = "logreg", repeats: int = 5):
    """ROC AUC needs no threshold, so plain out-of-fold scores are fair for it. Repeating
    with different fold shuffles shows how much a number moves just from the split."""
    aucs = []
    for r in range(repeats):
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED + r)
        s = cross_val_predict(build_model(model, correction), X, y, cv=cv, method="predict_proba")[:, 1]
        aucs.append(roc_auc_score(y, s))
    return float(np.mean(aucs)), float(np.std(aucs))


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


def nested_cv(X, y, rule: str, correction: str = "none", model: str = "logreg"):
    """Fair estimate of model + threshold rule: threshold chosen on inner folds only."""
    score, pred, thresholds = nested_cv_predictions(X, y, rule, correction, model)
    return metrics(y, pred, score), thresholds


def nested_cv_predictions(X, y, rule: str, correction: str = "none", model: str = "logreg"):
    """Per-subject P(PD) and yes/no prediction, each from a model + threshold that never saw them."""
    score = np.empty(len(y))
    pred = np.empty(len(y), dtype=int)
    thresholds = []
    for tr, te in folds().split(X, y):
        X_tr, y_tr = X.iloc[tr], y[tr]
        t = choose_threshold(y_tr, oof_scores(X_tr, y_tr, correction, model), rule)
        fitted = build_model(model, correction).fit(X_tr, y_tr)
        score[te] = fitted.predict_proba(X.iloc[te])[:, 1]
        pred[te] = (score[te] >= t).astype(int)
        thresholds.append(round(t, 3))
    return score, pred, thresholds


def print_threshold_sweep(X, y, correction: str = "none", model: str = "logreg") -> None:
    """For intuition only: how sensitivity/FPR trade off as the cut-off moves.
    Read off the same scores it describes, so it is NOT a fair estimate -- use nested_cv for that."""
    score = oof_scores(X, y, correction, model)
    rows = {}
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98]:
        m = metrics(y, (score >= t).astype(int), score)
        rows[f"{t:.2f}"] = {k: m[k] for k in ("sensitivity", "specificity", "false_positive_rate",
                                             "balanced_accuracy")}
    print("\nThreshold sweep (out-of-fold scores, intuition only):")
    print(pd.DataFrame(rows).T.round(3).to_string())


def cross_validate(train_ids, models, feature_sets, rules, correction="none", repeats=0) -> dict[str, dict]:
    """Nested CV of every model x feature set x threshold rule on the train subjects."""
    X, y = load_xy(train_ids, "base")
    s = cross_val_predict(DummyClassifier(strategy="most_frequent"), X, y, cv=folds(),
                          method="predict_proba")[:, 1]
    results = {"always_PD (sanity check)": metrics(y, (s >= 0.5).astype(int), s)}
    for feature_set in feature_sets:
        X, y = load_xy(train_ids, feature_set)
        for model in models:
            extra = {}
            if repeats:
                mean, sd = repeated_auc(X, y, correction, model, repeats)
                extra = {"auc_repeated_mean": mean, "auc_repeated_sd": sd}
            for rule in rules:
                m, thresholds = nested_cv(X, y, rule, correction, model)
                name = f"{model} | {feature_set} ({X.shape[1] - 1}) | {rule}"
                results[name] = {**m, **extra, "thresholds_per_fold": thresholds}
                print(f"  done: {name}", flush=True)
    return results


def print_table(results: dict[str, dict]) -> None:
    table = {name: {k: v for k, v in r.items() if k != "thresholds_per_fold"}
             for name, r in results.items()}
    print(pd.DataFrame(table).T.round(3).to_string())


def csv_list(choices):
    def parse(text):
        items = text.split(",")
        bad = [i for i in items if i not in choices]
        if bad:
            raise argparse.ArgumentTypeError(f"unknown {bad}; choose from {list(choices)}")
        return items
    return parse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true", help="score the locked test set (do this once)")
    parser.add_argument("--models", type=csv_list(MODEL_CHOICES), default=list(MODELS))
    parser.add_argument("--features", type=csv_list(FEATURE_SETS), default=["all"])
    parser.add_argument("--rules", type=csv_list(THRESHOLD_RULES), default=["max_fpr"])
    parser.add_argument("--correction", choices=CORRECTIONS, default="none",
                        help="sex correction (see PDModel); compare them with src.check_confounds")
    parser.add_argument("--repeats", type=int, default=0,
                        help="also report ROC AUC mean/sd over this many reshuffled CV runs")
    parser.add_argument("--sweep", action="store_true", help="print a threshold sweep (one combination)")
    args = parser.parse_args()

    train_ids, test_ids = load_split()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    single = len(args.models) == len(args.features) == len(args.rules) == 1

    if not args.final:
        print(f"target={TARGET}  nested {N_FOLDS}-fold CV on {len(train_ids)} train subjects, "
              f"MAX_FPR={MAX_FPR}, correction={args.correction}")
        results = cross_validate(train_ids, args.models, args.features, args.rules, args.correction,
                                 args.repeats)
        print("\nmodel | feature set (n features) | threshold rule:")
        print_table(results)
        if args.sweep and single:
            X, y = load_xy(train_ids, args.features[0])
            print_threshold_sweep(X, y, args.correction, args.models[0])
        out = RESULTS_DIR / f"{TARGET}_cv_metrics.json"
    else:
        if not single:
            raise SystemExit("--final needs exactly one --models, --features and --rules value")
        model, feature_set, rule = args.models[0], args.features[0], args.rules[0]
        X_train, y_train = load_xy(train_ids, feature_set)
        threshold = choose_threshold(y_train, oof_scores(X_train, y_train, args.correction, model), rule)
        fitted = build_model(model, args.correction).fit(X_train, y_train)
        X_test, y_test = load_xy(test_ids, feature_set)
        score = fitted.predict_proba(X_test)[:, 1]
        pred = (score >= threshold).astype(int)
        name = f"{model} | {feature_set} | {rule} | correction={args.correction} (TEST)"
        results = {name: {**metrics(y_test, pred, score), "threshold": threshold}}
        print(f"Locked test set: {len(y_test)} subjects, threshold {threshold:.3f}")
        print(pd.DataFrame(results).round(3).to_string())
        out = RESULTS_DIR / f"{TARGET}_test_metrics.json"

    out.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
