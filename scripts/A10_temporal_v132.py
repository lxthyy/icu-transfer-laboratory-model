# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# A10_temporal_v132.py -- temporal validation under the revised (full-cohort,
# no matching, no class weights) design.
#
# Split   : training  t0 in [2026-01-01, 2026-03-31]
#           validation t0 in [2026-04-01, 2026-05-31]
#           (t0 on or before 31 May leaves >= 30 days of follow-up before the
#            extraction ends on 30 June 2026)
# Same 48-h feature matrix, same 0.1/99.9 winsorisation, same median imputation
# and standardisation, both fitted on the training period only; C chosen by
# five-fold CV inside the training period.
# ---------------------------------------------------------------------------
import os, sys, time
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from joblib import Parallel, delayed

ICU_DATA_DIR = os.environ.get("ICU_DATA_DIR", "data")
BUILD = os.environ.get("ICU_BUILD_DIR", os.path.join(ICU_DATA_DIR, "_ICU预警成稿", "build"))
OUT = os.environ.get("ICU_OUT_DIR", os.path.join(BUILD, "..", "02_结果输出", "v1.32_全队列"))
os.makedirs(OUT, exist_ok=True)
SEED, NJOBS = 42, 4
CGRID = [0.001, 0.003, 0.01, 0.03, 0.1]
LOG = os.path.join(OUT, "A10_temporal.log"); open(LOG, "w", encoding="utf-8").close()
def P(s):
    print(s, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(str(s) + "\n")

lab = pd.read_parquet(os.path.join(BUILD, "labels.parquet"))
F = pd.read_parquet(os.path.join(BUILD, "feat_48h_orig.parquet"))
ids = F.index.intersection(lab.index)
lab = lab.loc[ids]; F = F.loc[ids]
lab["t0"] = pd.to_datetime(lab["t0"])
P(f"eligible {len(ids):,} | t0 range {lab['t0'].min().date()} to {lab['t0'].max().date()}")

tr_mask = (lab["t0"] >= "2026-01-01") & (lab["t0"] <= "2026-03-31")
va_mask = (lab["t0"] >= "2026-04-01") & (lab["t0"] <= "2026-05-31")
P(f"training period  {int(tr_mask.sum()):,} admissions, cases {int(lab.loc[tr_mask,'case_30d'].sum()):,}")
P(f"validation period {int(va_mask.sum()):,} admissions, cases {int(lab.loc[va_mask,'case_30d'].sum()):,}")

itr = np.where(tr_mask.values)[0]
iva = np.where(va_mask.values)[0]
X = F.astype("float64")
y = lab["case_30d"].astype(int).values

Xtr_df = X.iloc[itr]
keep = [c for c in X.columns if Xtr_df[c].nunique(dropna=True) > 2]
X = X[keep]
lo = X.iloc[itr].quantile(0.001); hi = X.iloc[itr].quantile(0.999)
X = X.clip(lower=lo, upper=hi, axis=1)
med = X.iloc[itr].median()
X = X.fillna(med)
A = X.values; sc = StandardScaler().fit(A[itr]); A = sc.transform(A)
Xtr, Xva = A[itr], A[iva]
ytr, yva = y[itr], y[iva]
P(f"features {len(keep)} | train {Xtr.shape} cases {int(ytr.sum()):,} ({ytr.mean()*100:.3f}%) | "
  f"validation {Xva.shape} cases {int(yva.sum()):,} ({yva.mean()*100:.3f}%)")

def cv(C, a, b):
    m = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=1000, tol=1e-4,
                           random_state=SEED).fit(Xtr[a], ytr[a])
    return (C, roc_auc_score(ytr[b], m.predict_proba(Xtr[b])[:, 1]))

folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xtr, ytr))
t0 = time.time()
out = Parallel(n_jobs=NJOBS)(delayed(cv)(C, a, b) for C in CGRID for a, b in folds)
s = pd.DataFrame(out, columns=["C", "auc"]).groupby("C")["auc"].agg(["mean", "std"]).reset_index()
for _, r in s.iterrows():
    P(f"  CV C={r.C:<7g} {r['mean']:.4f} +/- {r['std']:.4f}")
C = float(s.loc[s["mean"].idxmax(), "C"])
P(f"selected C = {C}  ({time.time()-t0:.0f}s)")

m = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=1000, tol=1e-4,
                       random_state=SEED).fit(Xtr, ytr)
ptr = m.predict_proba(Xtr)[:, 1]; pva = m.predict_proba(Xva)[:, 1]
nz = int((np.abs(m.coef_[0]) > 1e-8).sum())

def boot(fn):
    rng = np.random.default_rng(2026); v = []
    for _ in range(1000):
        b = rng.integers(0, len(yva), len(yva))
        if len(np.unique(yva[b])) < 2: continue
        v.append(fn(yva[b], pva[b]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

auc = roc_auc_score(yva, pva); alo, ahi = boot(roc_auc_score)
ap = average_precision_score(yva, pva); apl, aph = boot(average_precision_score)
br = brier_score_loss(yva, pva)
eps = 1e-6
lg = np.log(np.clip(pva, eps, 1-eps)/(1-np.clip(pva, eps, 1-eps)))
lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(lg.reshape(-1, 1), yva)
slope, icept = float(lr.coef_[0][0]), float(lr.intercept_[0])
oe = float(yva.sum())/float(pva.sum())

P(f"training AUC {roc_auc_score(ytr, ptr):.4f} | retained {nz}/{len(keep)}")
P(f"VALIDATION AUC {auc:.4f} ({alo:.4f}-{ahi:.4f}) | AP {ap:.4f} ({apl:.4f}-{aph:.4f}) | "
  f"Brier {br:.5f}")
P(f"validation calibration slope {slope:.3f} | intercept {icept:.3f} | O/E {oe:.3f} "
  f"(obs {int(yva.sum())}, exp {pva.sum():.1f})")

pd.DataFrame([
    ("design", "full cohort, no matching, no class weights"),
    ("training period", "2026-01-01 to 2026-03-31"),
    ("validation period", "2026-04-01 to 2026-05-31"),
    ("n_train", len(itr)), ("n_train_cases", int(ytr.sum())),
    ("n_validation", len(iva)), ("n_validation_cases", int(yva.sum())),
    ("C_selected", C), ("n_features", len(keep)), ("n_retained", nz),
    ("train_auc", round(float(roc_auc_score(ytr, ptr)), 4)),
    ("validation_auc", round(float(auc), 4)), ("validation_auc_lo", round(alo, 4)),
    ("validation_auc_hi", round(ahi, 4)),
    ("validation_ap", round(float(ap), 4)), ("validation_ap_lo", round(apl, 4)),
    ("validation_ap_hi", round(aph, 4)),
    ("validation_brier", round(float(br), 6)),
    ("validation_calibration_slope", round(slope, 4)),
    ("validation_calibration_intercept", round(icept, 4)),
    ("validation_oe_ratio", round(oe, 4)),
], columns=["item", "value"]).to_csv(os.path.join(OUT, "res_a10_temporal.csv"),
                                     index=False, encoding="utf-8-sig")
P("wrote res_a10_temporal.csv")
