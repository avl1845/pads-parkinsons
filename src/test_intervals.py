"""Confidence intervals and per-sex counts for the two final models on the test set.

Re-creates the exact --final predictions (same code path, same seed); nothing is
tuned. Run after --final:  python -m src.test_intervals
"""
import numpy as np
from scipy.stats import beta
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from src.train import load_xy, choose_threshold, oof_scores, build_model
from src.split import load_split
from src.config import SEX_COL

def cp(k, n):  # Clopper-Pearson exact 95% CI
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return lo, hi

train_ids, test_ids = load_split()
rng = np.random.default_rng(0)
for model, fs in [("logreg+rf", "all"), ("rf", "all+quest")]:
    Xtr, ytr = load_xy(train_ids, fs)
    thr = choose_threshold(ytr, oof_scores(Xtr, ytr, "none", model), "max_fpr")
    m = build_model(model, "none").fit(Xtr, ytr)
    Xte, yte = load_xy(test_ids, fs)
    s = m.predict_proba(Xte)[:, 1]; p = (s >= thr).astype(int)
    tp = int(((p == 1) & (yte == 1)).sum()); npd = int((yte == 1).sum())
    fp = int(((p == 1) & (yte == 0)).sum()); nh = int((yte == 0).sum())
    ba, auc = [], []
    for _ in range(5000):
        i = np.concatenate([rng.choice(np.where(yte == 1)[0], npd), rng.choice(np.where(yte == 0)[0], nh)])
        ba.append(balanced_accuracy_score(yte[i], p[i])); auc.append(roc_auc_score(yte[i], s[i]))
    print(f"\n{model} | {fs}  thr={thr:.3f}")
    print(f"  sens {tp}/{npd}={tp/npd:.3f}  CI {cp(tp,npd)[0]:.2f}-{cp(tp,npd)[1]:.2f}")
    print(f"  FPR  {fp}/{nh}={fp/nh:.3f}  CI {cp(fp,nh)[0]:.2f}-{cp(fp,nh)[1]:.2f}")
    print(f"  BA {balanced_accuracy_score(yte,p):.3f}  boot CI {np.percentile(ba,2.5):.2f}-{np.percentile(ba,97.5):.2f}")
    print(f"  AUC {roc_auc_score(yte,s):.3f}  boot CI {np.percentile(auc,2.5):.2f}-{np.percentile(auc,97.5):.2f}")
    sex = Xte[SEX_COL].to_numpy() if SEX_COL in Xte else None
    if sex is not None:
        for v, lab in [(1, "male"), (0, "female")]:
            k = sex == v
            print(f"  {lab}: n={k.sum()} (PD {int((yte[k]==1).sum())}, H {int((yte[k]==0).sum())})  "
                  f"sens {((p[k]==1)&(yte[k]==1)).sum()}/{(yte[k]==1).sum()}  "
                  f"FP {((p[k]==1)&(yte[k]==0)).sum()}/{(yte[k]==0).sum()}")
