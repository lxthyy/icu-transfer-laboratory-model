# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# A8e_map_audit_table.py -- final audit table for all EXACT_MAP entries.
# Verdict rules:
#   identical-name       alias == canonical, no merge happens
#   alias-absent         alias not present in the extract
#   UNIT-CONFLICT        quantile ratios consistent (CV<0.25) and median ratio
#                        outside [0.5, 2]  -> same quantity on a different scale
#   SHAPE-DIFFERS        median ratio outside [0.5, 2] but ratios scatter
#                        (CV>=0.25) -> different patient mix / different assay
#                        generation; not a scale error, kept
#   compatible           median ratio within [0.5, 2]
#   DEFECT-FIXED         removed from the map (白细胞)
# Output: res_a8_map_audit_final.csv
# ---------------------------------------------------------------------------
import os, sys, ast, re
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd, numpy as np

SCRIPT = os.environ.get("ICU_A1_SCRIPT",
                        os.path.join(os.path.dirname(os.path.abspath(__file__)), "A1_build.py"))
src = open(SCRIPT, encoding="utf-8").read()
MAP = ast.literal_eval(re.search(r"EXACT_MAP\s*=\s*(\{.*?\n\})", src, re.S).group(1))

SOURCE = os.environ.get("ICU_SOURCE_DIR",
                        os.path.join(os.environ.get("ICU_DATA_DIR", "data"), "source"))
PANEL = os.path.join(SOURCE, "住院_面板长表.parquet")
pan = pd.read_parquet(PANEL, columns=["检验项目", "结果"])
pan["检验项目"] = pan["检验项目"].astype(str).str.replace(" ", "")
pan["结果"] = pd.to_numeric(pan["结果"], errors="coerce")
pan = pan.dropna(subset=["结果"])
by = pan.groupby("检验项目")["结果"]
Q = [0.10, 0.25, 0.50, 0.75, 0.90, 0.99]

cache = {}
def prof(name):
    if name in cache: return cache[name]
    s = by.get_group(name) if name in by.groups else None
    if s is None or len(s) == 0:
        cache[name] = (None, 0)
    else:
        cache[name] = (np.array([s.quantile(q) for q in Q]), len(s))
    return cache[name]

rows = []
for alias, canon in MAP.items():
    if alias == canon:
        pa, na = prof(alias)
        rows.append(dict(alias=alias, canonical=canon, verdict="identical-name",
                         n_alias=na, n_canon=na, med_ratio=np.nan, ratio_cv=np.nan, note=""))
        continue
    pa, na = prof(alias); pc, nc = prof(canon)
    if pa is None:
        rows.append(dict(alias=alias, canonical=canon, verdict="alias-absent",
                         n_alias=0, n_canon=nc, med_ratio=np.nan, ratio_cv=np.nan,
                         note="alias not present in this extract"))
        continue
    if pc is None:
        rows.append(dict(alias=alias, canonical=canon, verdict="alias-absent",
                         n_alias=na, n_canon=0, med_ratio=np.nan, ratio_cv=np.nan,
                         note="canonical not present in this extract"))
        continue
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(pc != 0, pa / pc, np.nan)
    rr = r[~np.isnan(r) & (r > 0)]
    med = float(np.median(rr)) if len(rr) else np.nan
    cv = float(np.std(np.log(rr))) if len(rr) >= 3 else np.nan
    if na < 30:
        verdict, note = "SHAPE-DIFFERS", "small alias n (<30); not interpretable"
    elif pd.notna(cv) and cv < 0.25 and pd.notna(med) and (med > 2 or med < 0.5):
        verdict, note = "UNIT-CONFLICT", "consistent factor across quantiles"
    elif pd.notna(med) and (med > 2 or med < 0.5):
        verdict, note = "SHAPE-DIFFERS", "ratios scatter across quantiles; assay generation / mix"
    else:
        verdict, note = "compatible", ""
    rows.append(dict(alias=alias, canonical=canon, verdict=verdict, n_alias=na, n_canon=nc,
                     med_ratio=round(med, 3) if pd.notna(med) else np.nan,
                     ratio_cv=round(cv, 3) if pd.notna(cv) else np.nan, note=note))

# the one entry removed after the audit
rows.append(dict(alias="白细胞", canonical="白细胞计数", verdict="DEFECT-FIXED",
                 n_alias=int(len(by.get_group("白细胞"))) if "白细胞" in by.groups else 0,
                 n_canon=int(len(by.get_group("白细胞计数"))) if "白细胞计数" in by.groups else 0,
                 med_ratio=round(float(by.get_group("白细胞").median()) /
                                 float(by.get_group("白细胞计数").median()), 3),
                 ratio_cv=np.nan,
                 note="removed: merged with the blood count; reference band would collapse to 2.20-52.97 instead of 3.39-9.28"))

d = pd.DataFrame(rows).sort_values(["verdict", "alias"])
OUT = os.environ.get("ICU_OUT_DIR", ".")
d.to_csv(os.path.join(OUT, "res_a8_map_audit_final.csv"), index=False, encoding="utf-8-sig")
print(f"entries audited: {len(d)}")
print(d["verdict"].value_counts().to_string())
print("\nentries needing a decision:")
print(d[d.verdict.isin(["UNIT-CONFLICT", "SHAPE-DIFFERS", "DEFECT-FIXED", "alias-absent"])]
      [["alias", "canonical", "verdict", "n_alias", "med_ratio", "ratio_cv", "note"]].to_string(index=False))
