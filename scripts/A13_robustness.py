# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# A13_robustness.py -- two perturbation checks that had not been run cleanly.
#
# (1) Winsorisation-threshold scan. Four arms at a FIXED C = 0.03 on the same
#     70/30 split: none, p1/p99, p0.5/p99.5, p0.1/p99.9. Only the clipping
#     threshold changes, so any difference is attributable to preprocessing.
#
# (2) Leave-one-department-out. The primary model (fitted once on the full
#     training set) is scored on the test set with each first-test department
#     removed in turn. The model is NOT refitted, so the only thing that changes
#     is which admissions are evaluated.
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
SEED, C_FIXED = 42, 0.03
LOG = os.path.join(OUT, "A13_robustness.log"); open(LOG, "w", encoding="utf-8").close()
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
itr, ite = train_test_split(np.arange(n), test_size=0.30, stratify=y, random_state=SEED)
Xtr_df = F.iloc[itr]
keep = [c for c in F.columns if Xtr_df[c].nunique(dropna=True) > 2]
BASE = F[keep].astype("float64")
P(f"cohort {n:,} | cases {int(y.sum()):,} | features {len(keep)} | C fixed at {C_FIXED}")

def arm(q):
    t = time.time()
    X = BASE.copy()
    if q is not None:
        lo = X.iloc[itr].quantile(q[0]); hi = X.iloc[itr].quantile(q[1])
        X = X.clip(lower=lo, upper=hi, axis=1)
    X = X.fillna(X.iloc[itr].median())
    A = X.values
    sc = StandardScaler().fit(A[itr]); A = sc.transform(A)
    m = LogisticRegression(penalty="l1", solver="liblinear", C=C_FIXED, max_iter=1000,
                           tol=1e-4, random_state=SEED).fit(A[itr], y[itr])
    p = m.predict_proba(A[ite])[:, 1]
    yte = y[ite]
    return dict(winsor="none" if q is None else f"p{q[0]*100:g}/p{q[1]*100:g}",
                retained=int((np.abs(m.coef_[0]) > 1e-8).sum()),
                test_auc=round(float(roc_auc_score(yte, p)), 4),
                test_ap=round(float(average_precision_score(yte, p)), 4),
                oe=round(float(yte.sum()/p.sum()), 4),
                brier=round(float(brier_score_loss(yte, p)), 6),
                seconds=round(time.time()-t, 0))

ARMS = [None, (0.01, 0.99), (0.005, 0.995), (0.001, 0.999)]
P("\n=== (1) winsorisation-threshold scan, C fixed at 0.03 ===")
res = Parallel(n_jobs=4)(delayed(arm)(q) for q in ARMS)
d = pd.DataFrame(res)
for _, r in d.iterrows():
    P(f"  {r.winsor:<14} retained {r.retained:>3} | AUC {r.test_auc:.4f} | AP {r.test_ap:.4f} | "
      f"O/E {r.oe:.3f} | Brier {r.brier:.6f}")
P(f"  AUC range across thresholds: {d.test_auc.min():.4f}-{d.test_auc.max():.4f} "
  f"(spread {d.test_auc.max()-d.test_auc.min():.4f})")
d.to_csv(os.path.join(OUT, "res_a13_winsor_scan.csv"), index=False, encoding="utf-8-sig")

# ---- (2) leave-one-department-out, model fixed ----
P("\n=== (2) leave-one-department-out (model fixed, evaluation set restricted) ===")
X = BASE.copy()
lo = X.iloc[itr].quantile(0.001); hi = X.iloc[itr].quantile(0.999)
X = X.clip(lower=lo, upper=hi, axis=1)
X = X.fillna(X.iloc[itr].median())
A = X.values
sc = StandardScaler().fit(A[itr]); A = sc.transform(A)
m = LogisticRegression(penalty="l1", solver="liblinear", C=C_FIXED, max_iter=1000,
                       tol=1e-4, random_state=SEED).fit(A[itr], y[itr])
p = m.predict_proba(A[ite])[:, 1]
yte = y[ite]
ids_te = ids[ite]
dept = lab.loc[ids_te, "first_dept"].astype(str).values
base_auc = roc_auc_score(yte, p)
P(f"  reference: full test set AUC {base_auc:.4f}, n {len(yte):,}, cases {int(yte.sum()):,}")

rows = []
for dv in pd.unique(dept):
    mask = dept != dv
    if mask.sum() < 1000 or len(np.unique(yte[mask])) < 2: continue
    k = int(yte[mask].sum())
    if k < 5: continue
    a = roc_auc_score(yte[mask], p[mask])
    rows.append(dict(department=dv, n_eval=int(mask.sum()), cases_eval=k,
                     auc=round(float(a), 4), delta_vs_full=round(float(a-base_auc), 4)))
t = pd.DataFrame(rows).sort_values("delta_vs_full")
t.to_csv(os.path.join(OUT, "res_a13_leave_one_dept.csv"), index=False, encoding="utf-8-sig")
P(f"  departments evaluated: {len(t)}")
P("  largest AUC changes after removing one department:")
for _, r in pd.concat([t.head(6), t.tail(6)]).drop_duplicates().iterrows():
    P(f"    drop {r.department:<28} n {r.n_eval:>6,} cases {r.cases_eval:>4} "
      f"AUC {r.auc:.4f} (delta {r.delta_vs_full:+.4f})")
P(f"  |delta| range: {t.delta_vs_full.abs().min():.4f} to {t.delta_vs_full.abs().max():.4f}")
P("wrote res_a13_winsor_scan.csv / res_a13_leave_one_dept.csv")
