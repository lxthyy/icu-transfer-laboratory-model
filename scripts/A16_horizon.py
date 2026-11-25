# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# A16_horizon.py -- the same model pipeline evaluated at 7-day and 14-day
# horizons, so that the choice of a 30-day window is supported by numbers
# rather than asserted. The split, the winsorisation, the imputation and the
# penalty (C = 0.03) are held at their primary values; only the outcome
# definition changes, so differences are attributable to the horizon alone.
# ---------------------------------------------------------------------------
import os, sys, time
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

ICU_DATA_DIR = os.environ.get("ICU_DATA_DIR", "data")
BUILD = os.environ.get("ICU_BUILD_DIR", os.path.join(ICU_DATA_DIR, "_ICU预警成稿", "build"))
OUT = os.environ.get("ICU_OUT_DIR",
                     os.path.join(ICU_DATA_DIR, "_ICU预警成稿", "02_结果输出", "v1.32_全队列"))
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "A16_horizon.log")
open(LOG, "w", encoding="utf-8").close()

SEED = 42
C_FIXED = 0.03
WINSOR = (0.001, 0.999)

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
lab = lab.loc[ids]
F = F.loc[ids]
P(f"cohort {len(ids):,} | 7-day cases {int(lab['case_7d'].sum()):,} | "
  f"14-day cases {int(lab['case_14d'].sum()):,} | 30-day cases {int(lab['case_30d'].sum()):,}")

# split fixed by the primary outcome so that the three horizons share a test set
y30 = lab["case_30d"].astype(int).values
itr, ite = train_test_split(np.arange(len(y30)), test_size=0.30, stratify=y30, random_state=SEED)
P(f"split from the primary analysis: train {len(itr):,} | test {len(ite):,}")

X = F.copy().astype("float64")
Xtr_df = X.iloc[itr]
keep = [c for c in X.columns if Xtr_df[c].nunique(dropna=True) > 2]
X = X[keep]
lo = X.iloc[itr].quantile(WINSOR[0]); hi = X.iloc[itr].quantile(WINSOR[1])
X = X.clip(lower=lo, upper=hi, axis=1)
med = X.iloc[itr].median()
X = X.fillna(med)
A = X.values
sc = StandardScaler().fit(A[itr])
A = sc.transform(A)
Xtr, Xte = A[itr], A[ite]
P(f"features {len(keep)} | train {Xtr.shape} | test {Xte.shape}")

rows = []
for tag, col in [("7-day", "case_7d"), ("14-day", "case_14d"), ("30-day", "case_30d")]:
    y = lab[col].astype(int).values
    ytr, yte = y[itr], y[ite]
    P(f"\n--- horizon {tag}: train cases {int(ytr.sum()):,} ({ytr.mean()*100:.3f}%) | "
      f"test cases {int(yte.sum()):,} ({yte.mean()*100:.3f}%)")
    m = LogisticRegression(penalty="l1", solver="liblinear", C=C_FIXED, max_iter=1000,
                           tol=1e-4, random_state=SEED)
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p)
    lo_ci, hi_ci = boot_ci(yte, p, lambda a, b: roc_auc_score(a, b))
    ap = average_precision_score(yte, p)
    ap_lo, ap_hi = boot_ci(yte, p, lambda a, b: average_precision_score(a, b))
    slope, icept = calib_slope(yte, p)
    oe = yte.sum() / p.sum()
    brier = brier_score_loss(yte, p)
    nz = int((np.abs(m.coef_[0]) > 1e-8).sum())
    P(f"    C = {C_FIXED} | retained {nz} | test AUC {auc:.4f} ({lo_ci:.4f}-{hi_ci:.4f}) | "
      f"AP {ap:.4f} ({ap_lo:.4f}-{ap_hi:.4f}) | slope {slope:.3f} | O/E {oe:.3f} | Brier {brier:.6f}")
    rows.append(dict(horizon=tag, outcome=col, C=C_FIXED, n_train=len(itr), n_test=len(ite),
                     train_cases=int(ytr.sum()), test_cases=int(yte.sum()),
                     test_prevalence_pct=round(yte.mean() * 100, 4),
                     retained=nz, test_auc=round(auc, 4), auc_lo=round(lo_ci, 4), auc_hi=round(hi_ci, 4),
                     test_ap=round(ap, 4), ap_lo=round(ap_lo, 4), ap_hi=round(ap_hi, 4),
                     calibration_slope=round(slope, 4), calibration_intercept=round(icept, 4),
                     oe_ratio=round(oe, 4), brier=round(brier, 6)))

out = pd.DataFrame(rows)
out.to_csv(os.path.join(OUT, "res_a16_horizon.csv"), index=False)
P("\nwrote res_a16_horizon.csv")
P(out.to_string(index=False))
