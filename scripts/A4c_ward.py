# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# A4c_ward.py -- ward-onset scenario reported at C = 0.1 (upper end of the
# pre-specified grid). The 5-fold CV criterion is flat above C = 0.01
# (0.7176 at 0.01, 0.7337 at 0.03, 0.7449 at 0.1, 0.7463 at 0.3) with a fold
# SD of 0.033-0.048, so the penalty is not identifiable in that region; the
# upper end of the pre-specified grid is reported and the flatness disclosed.
# Full evaluation: discrimination, calibration, Youden from train, alert
# budget, deciles, DCA and the model specification.
# ---------------------------------------------------------------------------
import os, sys, time
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             confusion_matrix, roc_curve)

ICU_DATA_DIR = os.environ.get("ICU_DATA_DIR", "data")
BUILD = os.environ.get("ICU_BUILD_DIR", os.path.join(ICU_DATA_DIR, "_ICU预警成稿", "build"))
OUT = os.environ.get("ICU_OUT_DIR", os.path.join(BUILD, "..", "02_结果输出", "v1.32_全队列"))
os.makedirs(OUT, exist_ok=True)
SEED, SEED_BOOT, NB, C = 42, 2026, 1000, 0.1
WARD_KEYS = ["病房", "病区", "诊疗中心", "ICU"]
ICU_EXACT = {"重症医学科病房", "神经重症医学科(ICU)", "急诊ICU病房", "呼吸与危重症医学科ICU病区", "中心ICU"}
LOG = os.path.join(OUT, "A4c_ward.log"); open(LOG, "w", encoding="utf-8").close()
def P(s):
    print(s, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(str(s) + "\n")

lab = pd.read_parquet(os.path.join(BUILD, "labels.parquet"))
F = pd.read_parquet(os.path.join(BUILD, "feat_48h_orig.parquet"))
ids = F.index.intersection(lab.index)
lab = lab.loc[ids]; F = F.loc[ids]
def dep_cat(s):
    if not isinstance(s, str): return "其他"
    if "门诊" in s: return "门诊"
    if any(k in s for k in WARD_KEYS): return "病房/病区"
    return "其他"
is_ward = (lab["first_dept"].map(dep_cat) == "病房/病区") & (~lab["first_dept"].isin(ICU_EXACT))
idl = ids[is_ward.values]
y = lab.loc[idl, "case_30d"].astype(int).values
n = len(y)
X = F.loc[idl]
P(f"ward-onset cohort n={n:,} | cases {int(y.sum()):,} ({y.mean()*100:.3f}%)")

itr, ite = train_test_split(np.arange(n), test_size=0.30, stratify=y, random_state=SEED)
Xtr_df = X.iloc[itr]
keep = [c for c in X.columns if Xtr_df[c].nunique(dropna=True) > 2]
X = X[keep].astype("float64")
lo = X.iloc[itr].quantile(0.001); hi = X.iloc[itr].quantile(0.999)
X = X.clip(lower=lo, upper=hi, axis=1)
med = X.iloc[itr].median()
X = X.fillna(med)
A = X.values; sc = StandardScaler().fit(A[itr]); A = sc.transform(A)
Xtr, Xte = A[itr], A[ite]; ytr, yte = y[itr], y[ite]
P(f"features {len(keep)} | train {Xtr.shape} cases {int(ytr.sum()):,} | test {Xte.shape} cases {int(yte.sum()):,}")

m = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=1000, tol=1e-4,
                       random_state=SEED).fit(Xtr, ytr)
ptr = m.predict_proba(Xtr)[:, 1]; pte = m.predict_proba(Xte)[:, 1]
nz = int((np.abs(m.coef_[0]) > 1e-8).sum())

def boot(fn):
    rng = np.random.default_rng(SEED_BOOT); v = []
    for _ in range(NB):
        b = rng.integers(0, len(yte), len(yte))
        if len(np.unique(yte[b])) < 2: continue
        v.append(fn(yte[b], pte[b]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

auc = roc_auc_score(yte, pte); alo, ahi = boot(roc_auc_score)
ap = average_precision_score(yte, pte); apl, aph = boot(average_precision_score)
br = brier_score_loss(yte, pte)
eps = 1e-6
lg = np.log(np.clip(pte, eps, 1-eps)/(1-np.clip(pte, eps, 1-eps)))
lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(lg.reshape(-1, 1), yte)
slope, icept = float(lr.coef_[0][0]), float(lr.intercept_[0])
oe = float(yte.sum())/float(pte.sum())
P(f"train AUC {roc_auc_score(ytr, ptr):.4f} | TEST AUC {auc:.4f} ({alo:.4f}-{ahi:.4f}) | "
  f"AP {ap:.4f} ({apl:.4f}-{aph:.4f}) | retained {nz}/{len(keep)}")
P(f"calibration slope {slope:.3f} | intercept {icept:.3f} | O/E {oe:.3f} "
  f"(obs {int(yte.sum())}, exp {pte.sum():.1f}) | Brier {br:.5f}")

bins = pd.qcut(pd.Series(pte), 10, labels=False, duplicates="drop")
(pd.DataFrame({"bin": bins, "y": yte, "p": pte}).groupby("bin")
 .agg(n=("y", "size"), mean_pred=("p", "mean"), observed=("y", "mean")).reset_index().round(6)
 .to_csv(os.path.join(OUT, "res_a4_calibration_bins_ward.csv"), index=False, encoding="utf-8-sig"))

fpr, tpr, thr = roc_curve(ytr, ptr)
cut = float(thr[np.argmax(tpr - fpr)])
pred = (pte >= cut).astype(int)
tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
P(f"Youden (from train) cut {cut:.4f}: sens {tp/(tp+fn):.3f} spec {tn/(tn+fp):.3f} "
  f"PPV {tp/(tp+fp):.4f} NPV {tn/(tn+fn):.4f} alert rate {pred.mean()*100:.1f}%")

rows = []
order = np.argsort(-pte)
for f in (0.01, 0.02, 0.05, 0.10):
    k = max(1, int(round(f*len(yte)))); tpk = int(np.sum(yte[order[:k]] == 1))
    rows.append(dict(budget=f, n_flagged=k, cases_detected=tpk, sensitivity=tpk/yte.sum(),
                     ppv=tpk/k, n_flagged_per_case=k/max(tpk, 1)))
    P(f"  top {int(f*100)}%: {k:,} flagged | {tpk} cases | sens {tpk/yte.sum():.3f} | "
      f"PPV {tpk/k:.4f} | {k/max(tpk,1):.1f} flags per case")
pd.DataFrame(rows).to_csv(os.path.join(OUT, "res_a4_alert_budget_ward.csv"), index=False, encoding="utf-8-sig")

ths = np.round(np.concatenate([np.arange(0.002, 0.02, 0.002), np.arange(0.02, 0.21, 0.01)]), 4)
d = []
for t in ths:
    p_ = pte >= t; d.append(dict(threshold=t,
                                 net_benefit=int(np.sum(p_ & (yte == 1)))/len(yte)
                                 - int(np.sum(p_ & (yte == 0)))/len(yte)*(t/(1-t)),
                                 n_flagged=int(p_.sum())))
pd.DataFrame(d).to_csv(os.path.join(OUT, "res_a4_dca_ward.csv"), index=False, encoding="utf-8-sig")

pd.DataFrame([
    ("scenario", "ward"), ("C_fixed", C), ("n_eligible", n), ("n_cases", int(y.sum())),
    ("prevalence_pct", round(float(y.mean()*100), 4)), ("n_train", len(itr)), ("n_test", len(ite)),
    ("n_test_cases", int(yte.sum())), ("n_features_total", len(keep)), ("n_features_retained", nz),
    ("train_auc", round(float(roc_auc_score(ytr, ptr)), 4)),
    ("test_auc", round(float(auc), 4)), ("test_auc_lo", round(alo, 4)), ("test_auc_hi", round(ahi, 4)),
    ("test_ap", round(float(ap), 4)), ("test_ap_lo", round(apl, 4)), ("test_ap_hi", round(aph, 4)),
    ("brier", round(float(br), 6)), ("calibration_slope", round(slope, 4)),
    ("calibration_intercept", round(icept, 4)), ("oe_ratio", round(oe, 4)),
    ("youden_threshold", round(cut, 6)), ("youden_sensitivity", round(tp/(tp+fn), 4)),
    ("youden_specificity", round(tn/(tn+fp), 4)), ("youden_ppv", round(tp/(tp+fp), 4)),
    ("youden_npv", round(tn/(tn+fn), 4)), ("youden_alert_rate", round(float(pred.mean()), 4)),
], columns=["item", "value"]).to_csv(os.path.join(OUT, "res_a4_summary_ward.csv"), index=False, encoding="utf-8-sig")

pd.DataFrame({"y": yte, "p": pte}).to_csv(os.path.join(OUT, "res_a4_test_predictions_ward.csv"),
                                          index=False, encoding="utf-8-sig")
spec = pd.DataFrame({"feature": keep, "training_median": med.values, "winsor_lower": lo.values,
                     "winsor_upper": hi.values, "training_mean": sc.mean_, "training_sd": sc.scale_,
                     "standardized_coefficient": m.coef_[0]})
spec.loc[len(spec)] = ["(intercept, standardized scale)"] + [np.nan]*5 + [float(m.intercept_[0])]
spec.to_csv(os.path.join(OUT, "res_a4_model_spec_ward.csv"), index=False, encoding="utf-8-sig")
P("wrote ward outputs")
