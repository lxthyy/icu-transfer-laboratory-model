# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# A5_matched_v132.py -- matched-sample arm of the revision
#
# Three arms, same 70/30 split (seed 42) and same 1:4 match files:
#   A  "as-submitted"      : no winsorisation, C grid 0.02-1.0, class weights
#                            1.0/0.25  -> reproduction check against the
#                            submitted test AUC of 0.8160
#   B  "matched, no weights": winsorised at training p0.1/p99.9, C grid rescaled
#                            to 0.001-0.1, no class weights -> isolates the
#                            effect of the class weights (Reviewer 1 point 3)
#   C  "matched, weights"   : winsorised, same grid, weights kept -> isolates the
#                            effect of the preprocessing change
# ---------------------------------------------------------------------------
import os, sys, time
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

ICU_DATA_DIR = os.environ.get("ICU_DATA_DIR", "data")
BUILD = os.environ.get("ICU_BUILD_DIR", os.path.join(ICU_DATA_DIR, "_ICU预警成稿", "build"))
OUT = os.environ.get("ICU_OUT_DIR", os.path.join(BUILD, "..", "02_结果输出", "v1.32_全队列"))
os.makedirs(OUT, exist_ok=True)
SEED = 42

def P(s):
    print(s, flush=True)
    with open(os.path.join(OUT, "A5_matched.log"), "a", encoding="utf-8") as f:
        f.write(str(s) + "\n")
open(os.path.join(OUT, "A5_matched.log"), "w", encoding="utf-8").close()

F = pd.read_parquet(os.path.join(BUILD, "feat_48h_orig.parquet"))
lab = pd.read_parquet(os.path.join(BUILD, "labels.parquet"))
case = np.load(os.path.join(BUILD, "match_w30_r4_case.npy"), allow_pickle=True).tolist()
ctrl = np.load(os.path.join(BUILD, "match_w30_r4_ctrl.npy"), allow_pickle=True).tolist()
ids = case + ctrl
y = np.array([1]*len(case) + [0]*len(ctrl))
w = np.array([1.0]*len(case) + [0.25]*len(ctrl))
X = F.reindex(ids)
P(f"matched cohort n={len(ids):,} cases={len(case):,} controls={len(ctrl):,}")

itr, ite = train_test_split(np.arange(len(ids)), test_size=0.30, stratify=y, random_state=SEED)
Xtr_df = X.iloc[itr]
keep = [c for c in X.columns if Xtr_df[c].nunique(dropna=True) > 2]
X = X[keep].astype("float64")
ytr, yte = y[itr], y[ite]

def prep(winsor):
    Z = X.copy()
    if winsor:
        lo = Z.iloc[itr].quantile(0.001); hi = Z.iloc[itr].quantile(0.999)
        Z = Z.clip(lower=lo, upper=hi, axis=1)
    Z = Z.fillna(Z.iloc[itr].median())
    A = Z.values
    sc = StandardScaler().fit(A[itr]); A = sc.transform(A)
    return A[itr], A[ite]

def run(tag, winsor, grid, weighted):
    t = time.time()
    Xt, Xv = prep(winsor)
    best = None
    for C in grid:
        aucs = []
        for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xt, ytr):
            m = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=1000,
                                   tol=1e-4, random_state=SEED)
            m.fit(Xt[a], ytr[a], sample_weight=w[itr][a] if weighted else None)
            aucs.append(roc_auc_score(ytr[b], m.predict_proba(Xt[b])[:, 1]))
        mu = float(np.mean(aucs))
        if best is None or mu > best[1]:
            best = (C, mu, float(np.std(aucs)))
    C = best[0]
    m = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=1000,
                           tol=1e-4, random_state=SEED)
    m.fit(Xt, ytr, sample_weight=w[itr] if weighted else None)
    p = m.predict_proba(Xv)[:, 1]
    auc = roc_auc_score(yte, p)
    P(f"{tag:<34} C={C:<6g} cvAUC={best[1]:.4f}+/-{best[2]:.4f} nnz={int((np.abs(m.coef_[0])>1e-8).sum()):>3d} "
      f"| test AUC {auc:.4f} | AP {average_precision_score(yte, p):.4f} | Brier {brier_score_loss(yte, p):.4f} | {time.time()-t:.0f}s")
    return dict(arm=tag, C=C, cv_auc=round(best[1], 4), n_features=int((np.abs(m.coef_[0])>1e-8).sum()),
                test_auc=round(float(auc), 4), test_ap=round(float(average_precision_score(yte, p)), 4),
                test_brier=round(float(brier_score_loss(yte, p)), 4), weighted=weighted, winsorised=winsor)

P("features " + str(len(keep)))
rows = [
    run("A as-submitted (no winsor, weights)", False, [0.02, 0.05, 0.1, 0.2, 0.5, 1.0], True),
    run("B matched, no weights", True, [0.001, 0.003, 0.01, 0.03, 0.1], False),
    run("C matched, weights kept", True, [0.001, 0.003, 0.01, 0.03, 0.1], True),
]
pd.DataFrame(rows).to_csv(os.path.join(OUT, "res_a5_matched_arms.csv"), index=False, encoding="utf-8-sig")
P("\nwrote res_a5_matched_arms.csv")
