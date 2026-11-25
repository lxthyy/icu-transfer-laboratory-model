# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# Z2_censoring_km_v142.py
#
# Purpose
#   Quantify the administrative censoring in the primary cohort for
#   IJMEDI-D-26-05531R1 and the sensitivity analysis added in the second
#   revision.
#     * cohort                 : every admission with a 48-h feature row
#     * administrative end     : 30 June 2026 (end of the extraction window)
#     * unobservable for 30 d  : landmark after 31 May 2026
#     * estimator              : Kaplan-Meier cumulative incidence at 30 days,
#                                with the end of the extract as the only
#                                censoring time
#   Censoring is administrative by construction, so the KM estimate is
#   non-informative-censoring corrected.  Discharge is not recorded in this
#   extract, so if patients discharged alive cannot transfer within the same
#   admission, the reported rate remains a lower bound.
#
# Environment
#   ICU_DATA_DIR / ICU_BUILD_DIR as in A18_complete_followup_C003_v142.py
#   ICU_OUT_DIR  directory for censoring_report_v142.txt
#
# No randomness: output is deterministic.
# ---------------------------------------------------------------------------
import os, sys
import numpy as np, pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ICU_DATA_DIR = os.environ.get("ICU_DATA_DIR", "data")
BUILD = os.environ.get("ICU_BUILD_DIR", os.path.join(ICU_DATA_DIR, "_ICU预警成稿", "build"))
OUT = os.environ.get("ICU_OUT_DIR", ".")
os.makedirs(OUT, exist_ok=True)
CUT = pd.Timestamp(os.environ.get("ICU_END_DATE", "2026-06-30"))

lab = pd.read_parquet(os.path.join(BUILD, "labels.parquet"))
F = pd.read_parquet(os.path.join(BUILD, "feat_48h_orig.parquet"))
ids = F.index.intersection(lab.index)
lab = lab.loc[ids].copy()

lines = []
def P(s):
    s = str(s)
    lines.append(s)


def km_ci(t, e):
    """Kaplan-Meier cumulative incidence at the last event time <= 30."""
    n = len(t)
    order = np.argsort(t, kind="mergesort")
    t, e = t[order], e[order]
    S, i, out = 1.0, 0, 1.0
    while i < len(t):
        j = i
        while j + 1 < len(t) and t[j + 1] == t[i]:
            j += 1
        d = int(e[i:j + 1].sum())
        at_risk = n - i
        if d:
            S *= (1 - d / at_risk)
            if t[i] <= 30:
                out = S
        i = j + 1
    return 1 - out


P("Censoring audit for the complete-follow-up sensitivity analysis (v1.42)")
P("=" * 70)
P(f"analysis cohort (48-h feature row) : {len(lab):,}")
P(f"30-day events observed             : {int(lab.case_30d.sum()):,}")
P(f"naive 30-day rate                  : {lab.case_30d.mean()*100:.5f}%")
P(f"extraction end date                : {CUT.date()}")
lab["fu_d"] = (CUT - lab["t0"]).dt.days
P(f"landmark after 31 May 2026         : {int((lab.t0 > pd.Timestamp('2026-05-31')).sum()):,}")
P(f"follow-up < 30 days                : {int((lab.fu_d < 30).sum()):,}")
P(f"complete 30-day observation        : {int((lab.t0 <= pd.Timestamp('2026-05-31')).sum()):,}")
P("")

t = np.where(lab.case_30d.values, lab.gap_d.fillna(999).values, np.minimum(lab.fu_d.values, 30))
t = np.minimum(t, 30.0)
e = lab.case_30d.values.astype(int)
f30 = km_ci(t, e)
P(f"Kaplan-Meier 30-day cumulative incidence : {f30*100:.5f}%")
P(f"implied 30-day events                    : {f30*len(t):.1f}")
P(f"deficit versus observed                  : {f30*len(t)-e.sum():.1f} events "
  f"({(f30*len(t)-e.sum())/(f30*len(t))*100:.2f}% of the corrected total)")
P("")

for cut in ["2026-06-30", "2026-07-01"]:
    fu = (pd.Timestamp(cut) - lab["t0"]).dt.days
    tt = np.where(lab.case_30d.values, lab.gap_d.fillna(999).values, np.minimum(fu.values, 30))
    tt = np.minimum(tt, 30.0)
    P(f"[sensitivity] end date {cut}: follow-up<30 = {int((fu < 30).sum()):,}, "
      f"KM = {km_ci(tt, e)*100:.5f}%")

txt = "\n".join(lines)
print(txt)
with open(os.path.join(OUT, "censoring_report_v142.txt"), "w", encoding="utf-8") as f:
    f.write(txt + "\n")
