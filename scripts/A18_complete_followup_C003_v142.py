# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# A18_complete_followup_C003_v142.py
#
# Purpose
#   Second-revision sensitivity analysis for IJMEDI-D-26-05531R1.
#   Repeats the primary analysis (A4_v131.py) in the 155,031 admissions whose
#   landmark allowed a full 30 days of observation (t0 <= 2026-05-31), with the
#   L1 penalty FIXED at the primary value C = 0.03.  The penalty is not
#   reselected, because re-selecting it would confound the restriction of the
#   population with a change of penalty.
#
# Byte-level provenance
#   This file differs from the script that produced the archived results only
#   in this header block and in the two edited lines marked "[v1.42]".
#   The archived outputs in ..\结果\ were regenerated with this exact file and
#   compared byte for byte; see ..\README_冻结说明.md.
#
# Environment
#   ICU_DATA_DIR   root containing _ICU预警成稿\build\*.parquet
#   ICU_BUILD_DIR  (optional) explicit build directory
#   ICU_OUT_DIR    output directory for res_a4_*.csv and A4_v132.log
#
# Derived from A4_v131.py (IJMI major revision).  The analysis logic,
# preprocessing, split, bootstrap and metrics are unchanged.
# ---------------------------------------------------------------------------
import os, sys, time
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             confusion_matrix, roc_curve)
from joblib import Parallel, delayed

sys.stdout.reconfigure(encoding="utf-8")

ICU_DATA_DIR = os.environ.get("ICU_DATA_DIR", "data")
BUILD = os.environ.get("ICU_BUILD_DIR", os.path.join(ICU_DATA_DIR, "_ICU预警成稿", "build"))
OUT = os.environ.get("ICU_OUT_DIR",
                     os.path.join(ICU_DATA_DIR, "_ICU预警成稿", "02_结果输出", "v1.42_完整随访"))
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "A4_v132.log")
open(LOG, "w", encoding="utf-8").close()

SEED, SEED_BOOT, NB, NJOBS = 42, 2026, 1000, 4
# [v1.42] penalty fixed at the primary value; no CV selection
CGRID_BY_SCEN = {"full": [0.03],
                 "ward": [0.03]}
WINSOR = (0.001, 0.999)
WARD_KEYS = ["病房", "病区", "诊疗中心", "ICU"]
ICU_EXACT = {"重症医学科病房", "神经重症医学科(ICU)", "急诊ICU病房",
             "呼吸与危重症医学科ICU病区", "中心ICU"}

T0 = time.time()
def P(s):
    s = str(s)
    print(s, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(s + "\n")

# ---------------- load ----------------
lab = pd.read_parquet(os.path.join(BUILD, "labels.parquet"))
F = pd.read_parquet(os.path.join(BUILD, "feat_48h_orig.parquet"))
ids = F.index.intersection(lab.index)
lab = lab.loc[ids]; F = F.loc[ids]
# [v1.42] restrict to admissions with complete 30-day observation
lab = lab[lab['t0'] <= pd.Timestamp('2026-05-31')]
ids = F.index.intersection(lab.index)
lab = lab.loc[ids]; F = F.loc[ids]
P(f"[v1.42] complete-follow-up cohort {len(ids):,} | cases {int(lab['case_30d'].sum()):,}")

P(f"eligible cohort {len(ids):,} | 30-day cases {int(lab['case_30d'].sum()):,} "
  f"({lab['case_30d'].mean()*100:.3f}%)")

def dep_cat(s):
    if not isinstance(s, str): return "其他"
    if "门诊" in s: return "门诊"
    if any(k in s for k in WARD_KEYS): return "病房/病区"
    return "其他"

dept_cat = lab["first_dept"].map(dep_cat)
is_ward = (dept_cat == "病房/病区") & (~lab["first_dept"].isin(ICU_EXACT))
P(f"ward-onset population {int(is_ward.sum()):,} | cases "
  f"{int(lab.loc[is_ward,'case_30d'].sum()):,} ({lab.loc[is_ward,'case_30d'].mean()*100:.3f}%)")

def calibration_stats(y, p):
    eps = 1e-6
    lp = np.log(np.clip(p, eps, 1-eps) / (1-np.clip(p, eps, 1-eps)))
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(lp.reshape(-1, 1), y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])

def boot_ci(y, p, fn, n=NB):
    rng = np.random.default_rng(SEED_BOOT)
    v = []
    for _ in range(n):
        b = rng.integers(0, len(y), len(y))
        if len(np.unique(y[b])) < 2: continue
        v.append(fn(y[b], p[b]))
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (np.nan, np.nan)

def dca(y, p, ths):
    n = len(y); rows = []
    for t in ths:
        pred = p >= t
        tp = int(np.sum(pred & (y == 1))); fp = int(np.sum(pred & (y == 0)))
        rows.append(dict(threshold=t, net_benefit=tp/n - fp/n*(t/(1-t)), n_flagged=int(pred.sum())))
    return pd.DataFrame(rows)

def alert_budget(y, p, fracs=(0.01, 0.02, 0.05, 0.10)):
    order = np.argsort(-p); n = len(y); rows = []
    for f in fracs:
        k = max(1, int(round(f*n)))
        tp = int(np.sum(y[order[:k]] == 1))
        rows.append(dict(budget=f, n_flagged=k, cases_detected=tp,
                         sensitivity=tp/max(1, int(y.sum())), ppv=tp/k,
                         n_flagged_per_case_detected=(k/tp) if tp else np.nan))
    return pd.DataFrame(rows)

def _cv_fit(Xtr, ytr, C, a, b):
    m = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=1000,
                           tol=1e-4, random_state=SEED)
    m.fit(Xtr[a], ytr[a])
    return (C, roc_auc_score(ytr[b], m.predict_proba(Xtr[b])[:, 1]))

def run_scenario(tag, mask):
    P("\n" + "=" * 78)
    P(f"SCENARIO {tag}")
    idl = ids[mask.values]
    X = F.loc[idl]
    y = lab.loc[idl, "case_30d"].astype(int).values
    n = len(y)
    P(f"n = {n:,} | cases {int(y.sum()):,} | prevalence {y.mean()*100:.3f}%")

    itr, ite = train_test_split(np.arange(n), test_size=0.30, stratify=y, random_state=SEED)
    Xtr_df = X.iloc[itr]
    keep = [c for c in X.columns if Xtr_df[c].nunique(dropna=True) > 2]
    X = X[keep].astype("float64")

    # ---- winsorise at training-set 0.1 / 99.9 percentiles (leak-free) ----
    lo = X.iloc[itr].quantile(WINSOR[0]); hi = X.iloc[itr].quantile(WINSOR[1])
    n_clip_lo = int((X.lt(lo, axis=1)).sum().sum()); n_clip_hi = int((X.gt(hi, axis=1)).sum().sum())
    P(f"winsorised at training p{WINSOR[0]*100:g} / p{WINSOR[1]*100:g}: "
      f"{n_clip_lo:,} values raised, {n_clip_hi:,} values capped "
      f"({(n_clip_lo+n_clip_hi)/(n*len(keep))*100:.3f}% of all cells)")
    X = X.clip(lower=lo, upper=hi, axis=1)
    wide = (hi - lo).sort_values(ascending=False)
    P("widest winsorised ranges (feature: lower .. upper):")
    for c in wide.index[:12]:
        P(f"    {c:<26} {lo[c]:>12.2f} .. {hi[c]:>12.2f}")
    med = X.iloc[itr].median()
    X = X.fillna(med)
    A = X.values
    sc = StandardScaler().fit(A[itr]); A = sc.transform(A)
    Xtr, Xte = A[itr], A[ite]
    ytr, yte = y[itr], y[ite]
    P(f"features {len(keep)} | max|z| after winsorisation {np.abs(A).max():.2f} | "
      f"train {Xtr.shape} cases {int(ytr.sum()):,} | test {Xte.shape} cases {int(yte.sum()):,}")
    zmax = np.abs(A).max(axis=0)
    if zmax.max() > 15:
        P("  residual extreme standardised values (verify these are physiologically possible):")
        for i in np.argsort(-zmax)[:8]:
            P(f"    {keep[i]:<26} max|z|={zmax[i]:8.1f}  winsor range {lo[keep[i]]:.2f} .. {hi[keep[i]]:.2f}")
    del A, X

    # ---- C fixed at the primary value; CV kept only for the record ----
    CGRID = CGRID_BY_SCEN[tag]
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(Xtr, ytr))
    tasks = [(C, a, b) for C in CGRID for a, b in folds]
    t_cv = time.time()
    out = Parallel(n_jobs=NJOBS)(delayed(_cv_fit)(Xtr, ytr, C, a, b) for C, a, b in tasks)
    cv = pd.DataFrame(out, columns=["C", "cv_auc"])
    s = cv.groupby("C")["cv_auc"].agg(["mean", "std"]).reset_index().rename(
        columns={"mean": "cv_auc", "std": "cv_sd"}).round(4)
    s.to_csv(os.path.join(OUT, f"res_a4_cv_C_{tag}.csv"), index=False, encoding="utf-8-sig")
    for _, r in s.iterrows():
        P(f"  CV C={r.C:<7g} AUC {r.cv_auc:.4f} +/- {r.cv_sd:.4f}")
    P(f"  (CV {time.time()-t_cv:.0f} s, {len(tasks)} fits on {NJOBS} workers)")
    Cbest = float(s.loc[s["cv_auc"].idxmax(), "C"])
    P(f"selected C = {Cbest}")

    model = LogisticRegression(penalty="l1", solver="liblinear", C=Cbest, max_iter=1000,
                               tol=1e-4, random_state=SEED)
    tf = time.time(); model.fit(Xtr, ytr); P(f"  final fit {time.time()-tf:.0f} s")
    ptr = model.predict_proba(Xtr)[:, 1]; pte = model.predict_proba(Xte)[:, 1]
    nz = int((np.abs(model.coef_[0]) > 1e-8).sum())

    auc = roc_auc_score(yte, pte); alo, ahi = boot_ci(yte, pte, roc_auc_score)
    ap = average_precision_score(yte, pte); aplo, aphi = boot_ci(yte, pte, average_precision_score)
    br = brier_score_loss(yte, pte)
    slope, icept = calibration_stats(yte, pte)
    oe = float(yte.sum())/float(pte.sum())
    P(f"train AUC {roc_auc_score(ytr, ptr):.4f} | TEST AUC {auc:.4f} ({alo:.4f}-{ahi:.4f}) | "
      f"AP {ap:.4f} ({aplo:.4f}-{aphi:.4f}) | L1 retained {nz}/{len(keep)}")
    P(f"calibration slope {slope:.3f} | intercept {icept:.3f} | O/E {oe:.3f} "
      f"(obs {int(yte.sum())}, exp {pte.sum():.1f}) | Brier {br:.5f}")

    bins = pd.qcut(pd.Series(pte), 10, labels=False, duplicates="drop")
    (pd.DataFrame({"bin": bins, "y": yte, "p": pte}).groupby("bin")
     .agg(n=("y", "size"), mean_pred=("p", "mean"), observed=("y", "mean"))
     .reset_index().round(6)
     .to_csv(os.path.join(OUT, f"res_a4_calibration_bins_{tag}.csv"), index=False, encoding="utf-8-sig"))

    fpr, tpr, thr = roc_curve(ytr, ptr)
    cut = float(thr[np.argmax(tpr - fpr)])
    pred = (pte >= cut).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
    P(f"Youden (from train) cut {cut:.4f}: sens {tp/(tp+fn):.3f} spec {tn/(tn+fp):.3f} "
      f"PPV {tp/(tp+fp):.4f} NPV {tn/(tn+fn):.4f} alert rate {pred.mean()*100:.1f}%")

    ab = alert_budget(yte, pte)
    ab.to_csv(os.path.join(OUT, f"res_a4_alert_budget_{tag}.csv"), index=False, encoding="utf-8-sig")
    for _, r in ab.iterrows():
        P(f"  top {int(r.budget*100)}%: {int(r.n_flagged):,} flagged | {int(r.cases_detected)} cases | "
          f"sens {r.sensitivity:.3f} | PPV {r.ppv:.4f} | {r.n_flagged_per_case_detected:.1f} flags per case")

    ths = np.round(np.concatenate([np.arange(0.002, 0.02, 0.002), np.arange(0.02, 0.21, 0.01)]), 4)
    dca(yte, pte, ths).to_csv(os.path.join(OUT, f"res_a4_dca_{tag}.csv"), index=False, encoding="utf-8-sig")

    pd.DataFrame([
        ("scenario", tag), ("n_eligible", n), ("n_cases", int(y.sum())),
        ("prevalence_pct", round(float(y.mean()*100), 4)),
        ("n_train", len(itr)), ("n_test", len(ite)), ("n_test_cases", int(yte.sum())),
        ("C_selected", Cbest), ("n_features_total", len(keep)), ("n_features_retained", nz),
        ("winsor_p_lo", WINSOR[0]), ("winsor_p_hi", WINSOR[1]),
        ("winsor_cells_clipped", n_clip_lo + n_clip_hi),
        ("train_auc", round(float(roc_auc_score(ytr, ptr)), 4)),
        ("test_auc", round(float(auc), 4)), ("test_auc_lo", round(alo, 4)), ("test_auc_hi", round(ahi, 4)),
        ("test_ap", round(float(ap), 4)), ("test_ap_lo", round(aplo, 4)), ("test_ap_hi", round(aphi, 4)),
        ("brier", round(float(br), 6)),
        ("calibration_slope", round(slope, 4)), ("calibration_intercept", round(icept, 4)),
        ("oe_ratio", round(oe, 4)),
        ("youden_threshold", round(cut, 6)), ("youden_sensitivity", round(tp/(tp+fn), 4)),
        ("youden_specificity", round(tn/(tn+fp), 4)),
        ("youden_ppv", round(tp/(tp+fp), 4) if (tp+fp) else np.nan),
        ("youden_npv", round(tn/(tn+fn), 4) if (tn+fn) else np.nan),
        ("youden_alert_rate", round(float(pred.mean()), 4)),
    ], columns=["item", "value"]).to_csv(os.path.join(OUT, f"res_a4_summary_{tag}.csv"),
                                        index=False, encoding="utf-8-sig")
    pd.DataFrame({"y": yte, "p": pte}).to_csv(
        os.path.join(OUT, f"res_a4_test_predictions_{tag}.csv"), index=False, encoding="utf-8-sig")

    spec = pd.DataFrame({"feature": keep, "training_median": med.values,
                         "winsor_lower": lo.values, "winsor_upper": hi.values,
                         "training_mean": sc.mean_, "training_sd": sc.scale_,
                         "standardized_coefficient": model.coef_[0]})
    spec.loc[len(spec)] = ["(intercept, standardized scale)"] + [np.nan]*5 + [float(model.intercept_[0])]
    spec.to_csv(os.path.join(OUT, f"res_a4_model_spec_{tag}.csv"), index=False, encoding="utf-8-sig")

    return dict(tag=tag, n=n, cases=int(y.sum()), auc=auc, alo=alo, ahi=ahi, ap=ap,
                slope=slope, oe=oe, C=Cbest, nz=nz)

res = [run_scenario("full", pd.Series(True, index=ids))]
P("\n" + "=" * 78)
P("SUMMARY")
for r in res:
    P(f"  {r['tag']:<5} n {r['n']:>7,} cases {r['cases']:>5,} | AUC {r['auc']:.4f} "
      f"({r['alo']:.4f}-{r['ahi']:.4f}) | AP {r['ap']:.4f} | slope {r['slope']:.3f} | "
      f"O/E {r['oe']:.3f} | C={r['C']} | retained {r['nz']}")
P(f"elapsed {time.time()-T0:.0f} s")
