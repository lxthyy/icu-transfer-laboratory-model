# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# A6_nocount_v132.py -- does the model only measure how much testing was done?
#
# Reviewer 1's challenge. The count-type predictors are:
#   检验项数   number of laboratory results in the 48-h window
#   异常项数   number of results outside the reference interval
#   异常比例   the ratio of the two
#   *_次       number of measurements of a given analyte
# This script refits the full-cohort model after removing all of them, on the
# same split, the same winsorisation and the same C grid.
# ---------------------------------------------------------------------------
import os, sys, time
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from joblib import Parallel, delayed

ICU_DATA_DIR = os.environ.get("ICU_DATA_DIR", "data")
BUILD = os.environ.get("ICU_BUILD_DIR", os.path.join(ICU_DATA_DIR, "_ICU预警成稿", "build"))
OUT = os.environ.get("ICU_OUT_DIR", os.path.join(BUILD, "..", "02_结果输出", "v1.32_全队列"))
os.makedirs(OUT, exist_ok=True)
SEED, NJOBS = 42, 4
CGRID = [0.001, 0.003, 0.01, 0.03, 0.1]
LOG = os.path.join(OUT, "A6_nocount.log")
open(LOG, "w", encoding="utf-8").close()
def P(s):
    print(s, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(str(s) + "\n")

lab = pd.read_parquet(os.path.join(BUILD, "labels.parquet"))
F = pd.read_parquet(os.path.join(BUILD, "feat_48h_orig.parquet"))
ids = F.index.intersection(lab.index)
lab = lab.loc[ids]; F = F.loc[ids]
y = lab["case_30d"].astype(int).values
n = len(ids)
itr, ite = train_test_split(np.arange(n), test_size=0.30, stratify=y, random_state=SEED)
P(f"full cohort n={n:,} cases={int(y.sum()):,} | train {len(itr):,} test {len(ite):,}")

Xtr_df = F.iloc[itr]
keep0 = [c for c in F.columns if Xtr_df[c].nunique(dropna=True) > 2]
COUNT_FEATS = ["检验项数", "异常项数", "异常比例"]
keep1 = [c for c in keep0 if c not in COUNT_FEATS and not c.endswith("_次")]
dropped = [c for c in keep0 if c not in keep1]
P(f"features: {len(keep0)} -> {len(keep1)} after removing count-type predictors")
P(f"removed ({len(dropped)}): {dropped}")

def cv(Xtr, ytr, C, a, b):
    m = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=1000, tol=1e-4, random_state=SEED)
    m.fit(Xtr[a], ytr[a])
    return (C, roc_auc_score(ytr[b], m.predict_proba(Xtr[b])[:, 1]))

def fit_eval(cols, tag):
    X = F[cols].astype("float64")
    lo = X.iloc[itr].quantile(0.001); hi = X.iloc[itr].quantile(0.999)
    X = X.clip(lower=lo, upper=hi, axis=1).fillna(X.iloc[itr].median())
    A = X.values
    sc = StandardScaler().fit(A[itr]); A = sc.transform(A)
    Xtr, Xte = A[itr], A[ite]
    ytr, yte = y[itr], y[ite]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xtr, ytr))
    tasks = [(C, a, b) for C in CGRID for a, b in folds]
    t = time.time()
    out = Parallel(n_jobs=NJOBS)(delayed(cv)(Xtr, ytr, C, a, b) for C, a, b in tasks)
    s = pd.DataFrame(out, columns=["C", "auc"]).groupby("C")["auc"].agg(["mean", "std"]).reset_index()
    for _, r in s.iterrows():
        P(f"  {tag} CV C={r.C:<7g} {r['mean']:.4f} +/- {r['std']:.4f}")
    C = float(s.loc[s["mean"].idxmax(), "C"])
    m = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=1000, tol=1e-4, random_state=SEED)
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    nz = int((np.abs(m.coef_[0]) > 1e-8).sum())
    P(f"{tag:<12} features {len(cols):>3} | C={C:<6g} retained {nz:>3} | "
      f"TEST AUC {roc_auc_score(yte, p):.4f} | AP {average_precision_score(yte, p):.4f} | "
      f"Brier {brier_score_loss(yte, p):.5f} | {time.time()-t:.0f}s")
    return dict(arm=tag, n_features=len(cols), C=C, retained=nz,
                test_auc=round(float(roc_auc_score(yte, p)), 4),
                test_ap=round(float(average_precision_score(yte, p)), 4),
                test_brier=round(float(brier_score_loss(yte, p)), 5))

rows = [fit_eval(keep1, "no-count")]
pd.DataFrame(rows).to_csv(os.path.join(OUT, "res_a6_nocount.csv"), index=False, encoding="utf-8-sig")
P("wrote res_a6_nocount.csv")
