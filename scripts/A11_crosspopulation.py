# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# A11_crosspopulation.py -- decompose the discrimination gap properly.
#
# The full-cohort model (C = 0.03, 46 retained coefficients) is scored, with its
# own winsorisation / median / standardisation parameters, on the held-out test
# set of the full cohort. That same set of predictions is then split by the
# department of the first laboratory result. Differences between the resulting
# AUCs can only come from the composition of the test population, because the
# model, its coefficients and its preprocessing are identical.
#
# This replaces the comparison of two separately refitted models (full vs ward),
# which confounds the population with the penalty and the number of variables.
# ---------------------------------------------------------------------------
import os, sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

ICU_DATA_DIR = os.environ.get("ICU_DATA_DIR", "data")
BUILD = os.environ.get("ICU_BUILD_DIR", os.path.join(ICU_DATA_DIR, "_ICU预警成稿", "build"))
OUT = os.environ.get("ICU_OUT_DIR", os.path.join(BUILD, "..", "02_结果输出", "v1.32_全队列"))
SEED = 42
WARD_KEYS = ["病房", "病区", "诊疗中心", "ICU"]
ICU_EXACT = {"重症医学科病房", "神经重症医学科(ICU)", "急诊ICU病房", "呼吸与危重症医学科ICU病区", "中心ICU"}
LOG = os.path.join(OUT, "A11_crosspopulation.log"); open(LOG, "w", encoding="utf-8").close()
def P(s):
    print(s, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(str(s) + "\n")

lab = pd.read_parquet(os.path.join(BUILD, "labels.parquet"))
F = pd.read_parquet(os.path.join(BUILD, "feat_48h_orig.parquet"))
spec = pd.read_csv(os.path.join(OUT, "res_a4_model_spec_full.csv"))

icpt_row = spec[spec["feature"] == "(intercept, standardized scale)"]
intercept = float(icpt_row["standardized_coefficient"].iloc[0])
spec = spec[spec["feature"] != "(intercept, standardized scale)"].reset_index(drop=True)
feats = spec["feature"].tolist()
beta = spec["standardized_coefficient"].values.astype(float)
wlo = spec["winsor_lower"].values.astype(float)
whi = spec["winsor_upper"].values.astype(float)
med = spec["training_median"].values.astype(float)
mu = spec["training_mean"].values.astype(float)
sd = spec["training_sd"].values.astype(float)
P(f"model spec: {len(feats)} features, {int((np.abs(beta)>1e-8).sum())} non-zero, intercept {intercept:.4f}")

ids = F.index.intersection(lab.index)
lab = lab.loc[ids]; F = F.loc[ids]
y = lab["case_30d"].astype(int).values
n = len(y)
itr, ite = train_test_split(np.arange(n), test_size=0.30, stratify=y, random_state=SEED)

X = F[feats].astype("float64")
X = X.clip(lower=pd.Series(wlo, index=feats), upper=pd.Series(whi, index=feats), axis=1)
X = X.fillna(pd.Series(med, index=feats))
Z = (X.values - mu) / sd
lp = Z @ beta + intercept
p = 1.0 / (1.0 + np.exp(-lp))

yte = y[ite]; pte = p[ite]
ids_te = ids[ite]
cat = lab.loc[ids_te, "first_dept"].map(
    lambda s: "门诊" if isinstance(s, str) and "门诊" in s
    else ("病房/病区" if isinstance(s, str) and any(k in s for k in WARD_KEYS) else "其他"))
is_ward = (cat == "病房/病区").values & (~lab.loc[ids_te, "first_dept"].isin(ICU_EXACT)).values

P(f"\ntest set n = {len(yte):,} cases {int(yte.sum()):,} ({yte.mean()*100:.3f}%)")
P(f"  outpatient-first {int((cat=='门诊').sum()):,} | ward-first {int(is_ward.sum()):,} | other {int((cat=='其他').sum()):,}")
P(f"  cases among outpatients {int(yte[(cat=='门诊').values].sum())} | cases among ward {int(yte[is_ward].sum())}")

rows = []
def rep(tag, mask):
    yy, pp = yte[mask], pte[mask]
    if len(np.unique(yy)) < 2:
        P(f"{tag}: single class"); return
    auc = roc_auc_score(yy, pp)
    ap = average_precision_score(yy, pp)
    br = brier_score_loss(yy, pp)
    oe = yy.sum()/pp.sum()
    rows.append(dict(population=tag, n=len(yy), cases=int(yy.sum()),
                     prevalence_pct=round(float(yy.mean()*100), 4),
                     auc=round(float(auc), 4), ap=round(float(ap), 4),
                     brier=round(float(br), 6), oe=round(float(oe), 4)))
    P(f"{tag:<28} n {len(yy):>7,} cases {int(yy.sum()):>4,} ({yy.mean()*100:5.3f}%) | "
      f"AUC {auc:.4f} | AP {ap:.4f} | Brier {br:.6f} | O/E {oe:.3f}")

rep("FULL test set (as scored)", np.ones(len(yte), dtype=bool))
rep("outpatient-first only", (cat == "门诊").values)
rep("ward-first only (no ICU ward)", is_ward)
rep("ward-first only (incl ICU ward)", (cat == "病房/病区").values & ~is_ward | is_ward)

t = pd.DataFrame(rows)
t.to_csv(os.path.join(OUT, "res_a11_crosspopulation.csv"), index=False, encoding="utf-8-sig")

f = t[t.population == "FULL test set (as scored)"].auc.iloc[0]
w = t[t.population == "ward-first only (no ICU ward)"].auc.iloc[0]
P(f"\nexcess over chance, full test set  : {f-0.5:.4f}")
P(f"excess over chance, ward-only      : {w-0.5:.4f}")
P(f"AUC drop attributable to population composition alone: {f-w:.4f} "
  f"({(f-w)/(f-0.5)*100:.1f}% of the excess over chance)")
P("wrote res_a11_crosspopulation.csv")
