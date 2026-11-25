# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# A17_split_and_48h.py
#
# Two requests that were still open after the first revision round:
#
# 1. Reviewer 1 asked for a 48-hour outcome window. The primary analysis used
#    a 30-day window, and 7- and 14-day horizons were already reported, but
#    the shortest window the reviewer named was never estimated.
#
# 2. Reviewer 2 objected that the ablation which removed "count-type and
#    test-ordering variables" removed ordering intensity AND abnormality
#    burden at the same time, so the paper could not claim that ordering,
#    rather than abnormal results, explained the discrimination. This script
#    separates the two groups:
#        ordering   = 检验项数 + every '*_次' column   (48 features)
#        abnormality= 异常项数 + 异常比例               ( 2 features)
#    The two together are exactly the 50 features removed by A6.
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
OUT = os.environ.get("ICU_OUT_DIR",
                     os.path.join(ICU_DATA_DIR, "_ICU预警成稿", "02_结果输出", "v1.32_全队列"))
os.makedirs(OUT, exist_ok=True)
SEED, NJOBS = 42, 4
CGRID = [0.001, 0.003, 0.01, 0.03, 0.1]
LOG = os.path.join(OUT, "A17_split_and_48h.log")
open(LOG, "w", encoding="utf-8").close()

def P(s):
    s = str(s)
    print(s, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(s + "\n")

def boot_ci(y, p, fn, n=1000, seed=2026):
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n):
        b = rng.integers(0, len(y), len(y))
        if len(np.unique(y[b])) < 2:
            continue
        v.append(fn(y[b], p[b]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

def calib_slope(y, p):
    eps = 1e-6
    lp = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(lp.reshape(-1, 1), y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])

lab = pd.read_parquet(os.path.join(BUILD, "labels.parquet"))
F = pd.read_parquet(os.path.join(BUILD, "feat_48h_orig.parquet"))
ids = F.index.intersection(lab.index)
lab, F = lab.loc[ids], F.loc[ids]
y30 = lab["case_30d"].astype(int).values
n = len(ids)
P(f"cohort n={n:,} | 30-day cases {int(y30.sum()):,}")

# ======================================================================= part 1
P("\n" + "=" * 70)
P("PART 1  48-hour outcome window")
P("=" * 70)
gap = pd.to_numeric(lab["gap_d"], errors="coerce")
case30 = lab["case_30d"].astype(bool)
P("gap_d summary (days):")
P(gap.describe().to_string())
for d in (1, 2, 3, 7, 14, 30):
    P(f"  gap <= {d:2d} d : {int((case30 & (gap <= d)).sum()):5d} cases")
P("check: case_7d total = %d, gap<=7 = %d" %
  (int(lab['case_7d'].sum()), int((case30 & (gap <= 7)).sum())))

y48 = (case30 & (gap <= 2)).astype(int).values
P(f"48-h cases {int(y48.sum()):,} ({y48.mean()*100:.3f}%)")

itr, ite = train_test_split(np.arange(n), test_size=0.30, stratify=y30, random_state=SEED)
X = F.copy().astype("float64")
keep = [c for c in X.columns if X.iloc[itr][c].nunique(dropna=True) > 2]
X = X[keep]
lo = X.iloc[itr].quantile(0.001); hi = X.iloc[itr].quantile(0.999)
X = X.clip(lower=lo, upper=hi, axis=1).fillna(X.iloc[itr].median())
A = StandardScaler().fit_transform(X.values[itr]) if False else None
sc = StandardScaler().fit(X.values[itr])
A = sc.transform(X.values)
Xtr, Xte = A[itr], A[ite]
P(f"features {len(keep)} | train {Xtr.shape} | test {Xte.shape}")

rows = []
for tag, yv in [("48-hour", y48)]:
    ytr, yte = yv[itr], yv[ite]
    P(f"\n--- {tag}: train cases {int(ytr.sum()):,} ({ytr.mean()*100:.3f}%) | "
      f"test cases {int(yte.sum()):,} ({yte.mean()*100:.3f}%)")
    m = LogisticRegression(penalty="l1", solver="liblinear", C=0.03, max_iter=1000,
                           tol=1e-4, random_state=SEED)
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p)
    auc_lo, auc_hi = boot_ci(yte, p, lambda a, b: roc_auc_score(a, b))
    ap = average_precision_score(yte, p)
    ap_lo, ap_hi = boot_ci(yte, p, lambda a, b: average_precision_score(a, b))
    slope, icept = calib_slope(yte, p)
    oe = yte.sum() / p.sum()
    brier = brier_score_loss(yte, p)
    nz = int((np.abs(m.coef_[0]) > 1e-8).sum())
    P(f"    C = 0.03 | retained {nz} | test AUC {auc:.4f} ({auc_lo:.4f}-{auc_hi:.4f}) | "
      f"AP {ap:.4f} ({ap_lo:.4f}-{ap_hi:.4f}) | slope {slope:.3f} | O/E {oe:.3f} | Brier {brier:.6f}")
    rows.append(dict(horizon=tag, C=0.03, n_train=len(itr), n_test=len(ite),
                     train_cases=int(ytr.sum()), test_cases=int(yte.sum()),
                     test_prevalence_pct=round(yte.mean() * 100, 4), retained=nz,
                     test_auc=round(auc, 4), auc_lo=round(auc_lo, 4), auc_hi=round(auc_hi, 4),
                     test_ap=round(ap, 4), ap_lo=round(ap_lo, 4), ap_hi=round(ap_hi, 4),
                     calibration_slope=round(slope, 4), calibration_intercept=round(icept, 4),
                     oe_ratio=round(oe, 4), brier=round(brier, 6)))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "res_a17_horizon48.csv"), index=False)
P("\nwrote res_a17_horizon48.csv")

# ======================================================================= part 2
P("\n" + "=" * 70)
P("PART 2  separate ablations: ordering intensity vs abnormality burden")
P("=" * 70)
Xtr_df = F.iloc[itr]
base = [c for c in F.columns if Xtr_df[c].nunique(dropna=True) > 2]
ABNORMAL = ["异常项数", "异常比例"]
ORDERING = ["检验项数"] + [c for c in base if c.endswith("_次")]
assert all(c in base for c in ABNORMAL), "abnormality columns missing"
assert all(c in base for c in ORDERING), "ordering columns missing"
P(f"base features {len(base)}")
P(f"ORDERING  ({len(ORDERING)}): 检验项数 + {sum(1 for c in ORDERING if c.endswith('_次'))} '*_次' columns")
P(f"ABNORMAL  ({len(ABNORMAL)}): {ABNORMAL}")
P(f"together  {len(ORDERING)+len(ABNORMAL)}  (A6 removed 50)")

def cv(Xtr, ytr, C, a, b):
    m = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=1000,
                           tol=1e-4, random_state=SEED)
    m.fit(Xtr[a], ytr[a])
    return (C, roc_auc_score(ytr[b], m.predict_proba(Xtr[b])[:, 1]))

def fit_eval(cols, tag):
    Xd = F[cols].astype("float64")
    lo2 = Xd.iloc[itr].quantile(0.001); hi2 = Xd.iloc[itr].quantile(0.999)
    Xd = Xd.clip(lower=lo2, upper=hi2, axis=1).fillna(Xd.iloc[itr].median())
    M = Xd.values
    s2 = StandardScaler().fit(M[itr])
    M = s2.transform(M)
    Xa, Xb = M[itr], M[ite]
    ya, yb = y30[itr], y30[ite]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xa, ya))
    t0 = time.time()
    out = Parallel(n_jobs=NJOBS)(delayed(cv)(Xa, ya, C, a, b) for C in CGRID for a, b in folds)
    s = pd.DataFrame(out, columns=["C", "auc"]).groupby("C")["auc"].agg(["mean", "std"]).reset_index()
    for _, r in s.iterrows():
        P(f"  {tag} CV C={r.C:<7g} {r['mean']:.4f} +/- {r['std']:.4f}")
    C = float(s.loc[s["mean"].idxmax(), "C"])
    m = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=1000,
                           tol=1e-4, random_state=SEED)
    m.fit(Xa, ya)
    p = m.predict_proba(Xb)[:, 1]
    nz = int((np.abs(m.coef_[0]) > 1e-8).sum())
    P(f"{tag:<16} features {len(cols):>3} | C={C:<6g} retained {nz:>3} | "
      f"TEST AUC {roc_auc_score(yb, p):.4f} | AP {average_precision_score(yb, p):.4f} | "
      f"Brier {brier_score_loss(yb, p):.5f} | {time.time()-t0:.0f}s")
    return dict(arm=tag, n_features=len(cols), C=C, retained=nz,
                test_auc=round(float(roc_auc_score(yb, p)), 4),
                test_ap=round(float(average_precision_score(yb, p)), 4),
                test_brier=round(float(brier_score_loss(yb, p)), 5))

arms = []
arms.append(fit_eval([c for c in base if c not in ORDERING], "no-ordering"))
arms.append(fit_eval([c for c in base if c not in ABNORMAL], "no-abnormality"))
pd.DataFrame(arms).to_csv(os.path.join(OUT, "res_a17_split_ablation.csv"), index=False)
P("\nwrote res_a17_split_ablation.csv")
P(pd.DataFrame(arms).to_string(index=False))
