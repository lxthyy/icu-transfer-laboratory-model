# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# B1b_table1_fullcohort.py -- baseline table for the revised (full-cohort)
# analysis. Cases vs non-cases among all 181,947 admissions with a 48-h feature
# row. Reports median (IQR) or n (%), with standardised mean differences.
#
# This is a NEW script; B1_table1.py (the matched-sample version that anchors
# the v1.30 reproduction record) is left untouched.
# ---------------------------------------------------------------------------
import os, sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd

ICU_DATA_DIR = os.environ.get("ICU_DATA_DIR", "data")
BUILD = os.environ.get("ICU_BUILD_DIR", os.path.join(ICU_DATA_DIR, "_ICU预警成稿", "build"))
OUT = os.environ.get("ICU_OUT_DIR", os.path.join(BUILD, "..", "02_结果输出", "v1.32_全队列"))
os.makedirs(OUT, exist_ok=True)

lab = pd.read_parquet(os.path.join(BUILD, "labels.parquet"))
F = pd.read_parquet(os.path.join(BUILD, "feat_48h_orig.parquet"))
ids = F.index.intersection(lab.index)
lab = lab.loc[ids]; F = F.loc[ids]
case = lab["case_30d"].astype(bool)

WARD_KEYS = ["病房", "病区", "诊疗中心", "ICU"]
def dep_cat(s):
    if not isinstance(s, str): return "Other"
    if "门诊" in s: return "Outpatient clinic"
    if any(k in s for k in WARD_KEYS): return "Inpatient ward"
    return "Other"

rows = []
def fmt_med(x):
    s = pd.to_numeric(x, errors="coerce").dropna()
    if len(s) == 0: return "NA"
    return f"{s.median():.2f} ({s.quantile(.25):.2f}-{s.quantile(.75):.2f})"

def smd_cont(a, b):
    a = pd.to_numeric(a, errors="coerce").dropna(); b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2: return np.nan
    sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan

def smd_bin(pa, pb):
    return (pa - pb) / np.sqrt((pa*(1-pa) + pb*(1-pb)) / 2) if 0 < pa < 1 and 0 < pb < 1 else np.nan

n_c, n_n = int(case.sum()), int((~case).sum())
def add(name, vc, vn, smd, note=""):
    rows.append(dict(variable=name, cases=f"{vc} (n={n_c:,})", non_cases=f"{vn} (n={n_n:,})",
                     smd=round(float(smd), 3) if pd.notna(smd) else np.nan, note=note))

add("Age, years, median (IQR)",
    fmt_med(lab.loc[case, "age"]), fmt_med(lab.loc[~case, "age"]),
    smd_cont(lab.loc[case, "age"], lab.loc[~case, "age"]))

pm = (lab["sex"] == "男")
add("Male sex, n (%)", f"{int(pm[case].sum()):,} ({pm[case].mean()*100:.1f}%)",
    f"{int(pm[~case].sum()):,} ({pm[~case].mean()*100:.1f}%)", smd_bin(pm[case].mean(), pm[~case].mean()))

cat = lab["first_dept"].map(dep_cat)
for lvl in ["Outpatient clinic", "Inpatient ward", "Other"]:
    m = cat == lvl
    add(f"First laboratory result obtained in: {lvl}, n (%)",
        f"{int(m[case].sum()):,} ({m[case].mean()*100:.1f}%)",
        f"{int(m[~case].sum()):,} ({m[~case].mean()*100:.1f}%)",
        smd_bin(m[case].mean(), m[~case].mean()))

CONT = [
    ("检验项数", "Laboratory results in the 48-h window, n"),
    ("异常项数", "Results outside the reference interval, n"),
    ("异常比例", "Proportion of results outside the reference interval"),
    ("最早抽血偏移h", "Interval to first blood draw, h"),
    ("白细胞计数_首", "White cell count, x10^9/L"),
    ("血红蛋白_首", "Haemoglobin, g/L"),
    ("血小板计数_首", "Platelet count, x10^9/L"),
    ("中性粒细胞百分比_首", "Neutrophil percentage, %"),
    ("淋巴细胞绝对值_首", "Lymphocyte count, x10^9/L"),
    ("肌酐_首", "Creatinine, umol/L"),
    ("白蛋白_首", "Albumin, g/L"),
    ("总胆红素_首", "Total bilirubin, umol/L"),
    ("凝血酶原时间_首", "Prothrombin time, s"),
    ("纤维蛋白原_首", "Fibrinogen, g/L"),
    ("钠_首", "Sodium, mmol/L"),
    ("钾_首", "Potassium, mmol/L"),
]
for col, label in CONT:
    if col not in F.columns:
        print(f"  (skip, absent) {col}"); continue
    a, b = F.loc[case, col], F.loc[~case, col]
    add(label + ", median (IQR)", fmt_med(a), fmt_med(b), smd_cont(a, b),
        note=f"missing {a.isna().mean()*100:.1f}% vs {b.isna().mean()*100:.1f}%")

t = pd.DataFrame(rows)
p = os.path.join(OUT, "Table1_baseline_fullcohort.csv")
t.to_csv(p, index=False, encoding="utf-8-sig")
print(f"cases {n_c:,} | non-cases {n_n:,} | rows {len(t)}")
print(t.to_string(index=False))
print("\n->", p)
