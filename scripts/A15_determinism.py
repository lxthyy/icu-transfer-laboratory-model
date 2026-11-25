# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# A15_determinism.py -- ten consecutive executions of the fitted model at the
# primary specification. Records an MD5 of the coefficient vector and of the
# test-set predictions for each run, to show the pipeline is deterministic.
# ---------------------------------------------------------------------------
import os, sys, time, hashlib, json
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

ICU_DATA_DIR = os.environ.get("ICU_DATA_DIR", "data")
BUILD = os.environ.get("ICU_BUILD_DIR", os.path.join(ICU_DATA_DIR, "_ICU预警成稿", "build"))
OUT = os.environ.get("ICU_OUT_DIR", os.path.join(BUILD, "..", "02_结果输出", "v1.32_全队列"))
SEED, C_FIXED, N_RUNS = 42, 0.03, 10
LOG = os.path.join(OUT, "A15_determinism.log"); open(LOG, "w", encoding="utf-8").close()
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
X = F[keep].astype("float64")
lo = X.iloc[itr].quantile(0.001); hi = X.iloc[itr].quantile(0.999)
X = X.clip(lower=lo, upper=hi, axis=1)
X = X.fillna(X.iloc[itr].median())
A = X.values
sc = StandardScaler().fit(A[itr]); A = sc.transform(A)
P(f"cohort {n:,} | train {A[itr].shape} | test {A[ite].shape} | C = {C_FIXED} | runs = {N_RUNS}")

rows = []
for k in range(N_RUNS):
    t = time.time()
    m = LogisticRegression(penalty="l1", solver="liblinear", C=C_FIXED, max_iter=1000,
                           tol=1e-4, random_state=SEED).fit(A[itr], y[itr])
    p = m.predict_proba(A[ite])[:, 1]
    coef = np.concatenate([m.coef_.ravel(), [float(m.intercept_[0])]])
    md5c = hashlib.md5(coef.tobytes()).hexdigest()
    md5p = hashlib.md5(p.tobytes()).hexdigest()
    rows.append(dict(run=k + 1, md5_coefficients=md5c, md5_predictions=md5p,
                     nonzero=int((np.abs(m.coef_[0]) > 1e-8).sum()),
                     sum_predictions=round(float(p.sum()), 10), seconds=round(time.time()-t, 1)))
    P(f"  run {k+1:>2}: coef MD5 {md5c[:16]}... | pred MD5 {md5p[:16]}... | "
      f"nnz {rows[-1]['nonzero']} | sum p {rows[-1]['sum_predictions']:.10f} | {rows[-1]['seconds']}s")

t = pd.DataFrame(rows)
t.to_csv(os.path.join(OUT, "res_a15_determinism.csv"), index=False, encoding="utf-8-sig")
P(f"\ncoefficient MD5 unique values : {t.md5_coefficients.nunique()} of {N_RUNS}")
P(f"prediction  MD5 unique values : {t.md5_predictions.nunique()} of {N_RUNS}")
P(f"deterministic: {t.md5_coefficients.nunique() == 1 and t.md5_predictions.nunique() == 1}")
