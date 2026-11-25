# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# A12_seed_robustness.py -- is the test-set AUC a property of the data or of
# seed 42?
#
# The whole 70/30 split is repeated for ten seeds. Everything else is held at
# the primary specification: 48-h feature matrix, 0.1/99.9 winsorisation and
# median imputation fitted on the training split, C fixed at 0.03 (the value
# chosen by cross-validation in the primary analysis). The penalty is NOT
# re-selected per seed, so that the spread reflects the split alone.
# ---------------------------------------------------------------------------
import os, sys, time
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from joblib import Parallel, delayed

ICU_DATA_DIR = os.environ.get("ICU_DATA_DIR", "data")
BUILD = os.environ.get("ICU_BUILD_DIR", os.path.join(ICU_DATA_DIR, "_ICU预警成稿", "build"))
OUT = os.environ.get("ICU_OUT_DIR", os.path.join(BUILD, "..", "02_结果输出", "v1.32_全队列"))
os.makedirs(OUT, exist_ok=True)
C_FIXED = 0.03
SEEDS = [1, 7, 42, 99, 123, 777, 2024, 2026, 2718, 31415]
LOG = os.path.join(OUT, "A12_seed.log"); open(LOG, "w", encoding="utf-8").close()
def P(s):
    print(s, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(str(s) + "\n")

lab = pd.read_parquet(os.path.join(BUILD, "labels.parquet"))
F = pd.read_parquet(os.path.join(BUILD, "feat_48h_orig.parquet"))
ids = F.index.intersection(lab.index)
lab = lab.loc[ids]; F = F.loc[ids]
y = lab["case_30d"].astype(int).values
n = len(y)
P(f"eligible {n:,} | cases {int(y.sum()):,} | C fixed at {C_FIXED} | seeds {SEEDS}")

def one(seed):
    itr, ite = train_test_split(np.arange(n), test_size=0.30, stratify=y, random_state=seed)
    Xtr_df = F.iloc[itr]
    keep = [c for c in F.columns if Xtr_df[c].nunique(dropna=True) > 2]
    X = F[keep].astype("float64")
    lo = X.iloc[itr].quantile(0.001); hi = X.iloc[itr].quantile(0.999)
    X = X.clip(lower=lo, upper=hi, axis=1)
    X = X.fillna(X.iloc[itr].median())
    A = X.values
    sc = StandardScaler().fit(A[itr]); A = sc.transform(A)
    m = LogisticRegression(penalty="l1", solver="liblinear", C=C_FIXED, max_iter=1000,
                           tol=1e-4, random_state=42).fit(A[itr], y[itr])
    p = m.predict_proba(A[ite])[:, 1]
    yte = y[ite]
    return dict(seed=seed, n_test=len(yte), n_test_cases=int(yte.sum()),
                retained=int((np.abs(m.coef_[0]) > 1e-8).sum()),
                test_auc=round(float(roc_auc_score(yte, p)), 4),
                test_ap=round(float(average_precision_score(yte, p)), 4),
                oe=round(float(yte.sum()/p.sum()), 4),
                brier=round(float(brier_score_loss(yte, p)), 6))

t = time.time()
res = Parallel(n_jobs=4)(delayed(one)(s) for s in SEEDS)
d = pd.DataFrame(res).sort_values("seed")
for _, r in d.iterrows():
    P(f"  seed {r.seed:>6} | retained {r.retained:>3} | AUC {r.test_auc:.4f} | "
      f"AP {r.test_ap:.4f} | O/E {r.oe:.3f} | Brier {r.brier:.6f}")
d.to_csv(os.path.join(OUT, "res_a12_seed_robustness.csv"), index=False, encoding="utf-8-sig")
P(f"\ntest AUC across seeds: mean {d.test_auc.mean():.4f} | SD {d.test_auc.std(ddof=1):.4f} | "
  f"min {d.test_auc.min():.4f} | max {d.test_auc.max():.4f} | range {d.test_auc.max()-d.test_auc.min():.4f}")
P(f"AP: mean {d.test_ap.mean():.4f} SD {d.test_ap.std(ddof=1):.4f} | "
  f"O/E: mean {d.oe.mean():.3f} SD {d.oe.std(ddof=1):.3f} | "
  f"retained: {d.retained.min()}-{d.retained.max()} (median {int(d.retained.median())})")
P(f"seed 42 value: AUC {d.loc[d.seed==42,'test_auc'].iloc[0]:.4f} "
  f"(z = {(d.loc[d.seed==42,'test_auc'].iloc[0]-d.test_auc.mean())/d.test_auc.std(ddof=1):+.2f})")
P(f"({time.time()-t:.0f}s)")
