# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# Z3_ipcw_v142.py -- inverse-probability-of-censoring weighting (IPCW) for
# administrative censoring of the 30-day outcome.
#
# Produces: Kaplan-Meier 30-day incidence, IPCW incidence, IPCW observed-to-
# expected ratio with a bootstrap interval, and the weighted calibration slope,
# for the full cohort and for the test set of the primary model.
#
# Environment: ICU_BUILD_DIR (build folder), ICU_OUT_DIR (output folder).
# The test-set predictions are read from predictions_primary.csv, either in
# ICU_OUT_DIR or next to this script.
# ---------------------------------------------------------------------------
import os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(encoding="utf-8")
BUILD = os.environ.get("ICU_BUILD_DIR", "build")
OUT = os.environ.get("ICU_OUT_DIR", "results")
os.makedirs(OUT, exist_ok=True)
CUT = pd.Timestamp(os.environ.get("ICU_END_DATE", "2026-06-30"))
SEED = int(os.environ.get("ICU_SEED", 42))

lab = pd.read_parquet(os.path.join(BUILD, "labels.parquet"))
F = pd.read_parquet(os.path.join(BUILD, "feat_48h_orig.parquet"))
ids = F.index.intersection(lab.index)
lab = lab.loc[ids].copy()
y = lab["case_30d"].astype(int).values
n = len(y)
fu = (CUT - lab["t0"]).dt.days.values
t = np.where(y == 1, lab["gap_d"].fillna(999).values, np.minimum(fu, 30)).astype(float)
t = np.minimum(t, 30.0)


def g_curve(tt, ee):
    order = np.argsort(tt)
    tt_s, ee_s = tt[order], ee[order]
    out, G, i = {}, 1.0, 0
    while i < len(tt_s):
        j = i
        while j + 1 < len(tt_s) and tt_s[j + 1] == tt_s[i]:
            j += 1
        d = int((1 - ee_s[i:j + 1]).sum())
        out[tt_s[i]] = G
        if d:
            G *= (1 - d / (len(tt_s) - i))
        i = j + 1
    return out


gmap = g_curve(t, y)
Gv = np.array([gmap.get(x, 1.0) for x in t])
observed = (y == 1) | (t >= 30)
w = np.where(observed, 1.0 / np.maximum(Gv, 1e-12), 0.0)


def km_inc(tt, ee):
    order = np.argsort(tt); tt, ee = tt[order], ee[order]
    S, i = 1.0, 0
    while i < len(tt):
        j = i
        while j + 1 < len(tt) and tt[j + 1] == tt[i]:
            j += 1
        d = int(ee[i:j + 1].sum())
        if d:
            S *= (1 - d / (len(tt) - i))
        i = j + 1
    return 1 - S


def boot(p, yy, ww, nb=500, seed=2026):
    rng = np.random.default_rng(seed); v = []
    for _ in range(nb):
        i = rng.integers(0, len(p), len(p))
        den = (p[i] * ww[i]).sum()
        if den > 0:
            v.append((ww[i] * yy[i]).sum() / den)
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def report(tag, yy, pp, tt, ww):
    lp = np.log(np.clip(pp, 1e-6, 1 - 1e-6) / (1 - np.clip(pp, 1e-6, 1 - 1e-6)))
    sw = np.where(ww > 0, ww, 1e-12)
    m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(
        lp.reshape(-1, 1), yy, sample_weight=sw)
    lo, hi = boot(pp, yy, ww)
    print(f"{tag}: n={len(yy):,} events={int(yy.sum()):,} | KM30 {km_inc(tt,yy)*100:.5f}% "
          f"| IPCW incidence {(ww*yy).mean()*100:.5f}% | O/E {(ww*yy).sum()/(ww*pp).sum():.4f} "
          f"({lo:.3f}-{hi:.3f}) | weighted slope {float(m.coef_[0][0]):.3f}")
    return dict(population=tag, n=len(yy), events=int(yy.sum()),
                km30_pct=round(km_inc(tt, yy) * 100, 5),
                ipcw_incidence_pct=round(float((ww * yy).mean() * 100), 5),
                oe_ipcw=round(float((ww * yy).sum() / (ww * pp).sum()), 4),
                oe_lo=round(float(lo), 4), oe_hi=round(float(hi), 4),
                weighted_slope=round(float(m.coef_[0][0]), 4))


rows = [dict(population="full cohort (incidence only)", n=n, events=int(y.sum()),
             km30_pct=round(km_inc(t, y) * 100, 5),
             ipcw_incidence_pct=round(float((w * y).mean() * 100), 5),
             oe_ipcw="", oe_lo="", oe_hi="", weighted_slope="")]
print(f"full: n={n:,} | KM30 {km_inc(t,y)*100:.5f}% | IPCW incidence {(w*y).mean()*100:.5f}%")

itr, ite = train_test_split(np.arange(n), test_size=0.30, stratify=y, random_state=SEED)
HERE = os.path.dirname(os.path.abspath(__file__))
cand = [os.path.join(OUT, "predictions_primary.csv"),
        os.path.join(HERE, "predictions_primary.csv"),
        os.path.join(os.path.dirname(HERE), "predictions_primary.csv")]
path = next((p for p in cand if os.path.exists(p)), None)
assert path, "predictions_primary.csv not found (set ICU_OUT_DIR or place it next to the script)"
pred = pd.read_csv(path)
assert len(pred) == len(ite) and np.array_equal(pred["y"].values.astype(int), y[ite])
rows.append(report("test set (primary model)", y[ite], pred["p"].values, t[ite], w[ite]))

pd.DataFrame(rows).to_csv(os.path.join(OUT, "res_z3_ipcw.csv"), index=False, encoding="utf-8-sig")
print("wrote", os.path.join(OUT, "res_z3_ipcw.csv"))
