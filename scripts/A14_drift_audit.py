# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# A14_drift_audit.py -- column-level comparison of every artefact between the
# submitted build (v1.30, folder "build") and the revised build (v1.32, folder
# "build_v132"). Answers one question only: did anything change apart from the
# corrections that were declared in advance?
#
# Output: res_a14_drift_audit.csv          one row per file x column
#         res_a14_drift_summary.csv        one row per artefact
# ---------------------------------------------------------------------------
import os, sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd

LEGACY = os.environ.get("ICU_LEGACY_ROOT", "")
if (not LEGACY or not os.path.isdir(os.path.join(LEGACY, "build"))
        or not os.path.isdir(os.path.join(LEGACY, "build_v132"))):
    print("SKIP A14: set ICU_LEGACY_ROOT to a folder containing the pre-correction\n"
          "build (sub-folder build) and the corrected build (sub-folder build_v132).")
    sys.exit(0)
OLD, NEW = os.path.join(LEGACY, "build"), os.path.join(LEGACY, "build_v132")
OUT = os.environ.get("ICU_OUT_DIR", "results")
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "A14_drift.log"); open(LOG, "w", encoding="utf-8").close()
def P(s):
    print(s, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(str(s) + "\n")

rows, summary = [], []

# ---------- 1. labels ----------
a = pd.read_parquet(os.path.join(OLD, "labels.parquet"))
b = pd.read_parquet(os.path.join(NEW, "labels.parquet"))
same_index = a.index.equals(b.index)
ncol_changed = 0
for c in a.columns:
    if a[c].dtype.kind in "fi":
        eq = np.allclose(a[c].fillna(-9e9).values, b[c].fillna(-9e9).values, equal_nan=True)
    else:
        eq = a[c].astype(str).equals(b[c].astype(str))
    rows.append(dict(artefact="labels.parquet", column=c, identical=bool(eq), n_changed=0 if eq else -1,
                     max_abs_diff=0.0, mean_diff=0.0))
    ncol_changed += (not eq)
summary.append(dict(artefact="labels.parquet", shape_old=str(a.shape), shape_new=str(b.shape),
                    index_identical=bool(same_index), columns_changed=ncol_changed,
                    columns_total=len(a.columns)))
P(f"labels.parquet: {a.shape} vs {b.shape} | index identical {same_index} | "
  f"columns changed {ncol_changed}/{len(a.columns)}")

# ---------- 2. match files ----------
n_match, n_diff = 0, 0
for f in sorted(os.listdir(OLD)):
    if f.startswith("match_") and f.endswith(".npy"):
        x = np.load(os.path.join(OLD, f), allow_pickle=True)
        y = np.load(os.path.join(NEW, f), allow_pickle=True)
        n_match += 1
        ok = (len(x) == len(y)) and np.array_equal(x, y)
        n_diff += (not ok)
        rows.append(dict(artefact=f, column="(members)", identical=bool(ok),
                         n_changed=0 if ok else abs(len(x)-len(y)), max_abs_diff=0.0, mean_diff=0.0))
summary.append(dict(artefact="match_*.npy (9 files)", shape_old=f"{n_match} files",
                    shape_new=f"{n_match} files", index_identical=bool(n_diff == 0),
                    columns_changed=n_diff, columns_total=n_match))
P(f"match files: {n_match} compared, {n_diff} differ")

# ---------- 3. feature matrices ----------
for f in ["feat_48h_orig.parquet", "feat_48h_canon.parquet",
          "feat_24h_orig.parquet", "feat_24h_canon.parquet"]:
    p1, p2 = os.path.join(OLD, f), os.path.join(NEW, f)
    A = pd.read_parquet(p1); B = pd.read_parquet(p2)
    cols_a, cols_b = list(A.columns), list(B.columns)
    only_old = [c for c in cols_a if c not in set(cols_b)]
    only_new = [c for c in cols_b if c not in set(cols_a)]
    shared = [c for c in cols_a if c in set(cols_b)]
    if only_old or only_new:
        P(f"{f}: COLUMN SET CHANGED | only in v1.30 ({len(only_old)}): {only_old[:12]}"
          f" | only in v1.32 ({len(only_new)}): {only_new[:12]}")
        for c in only_old:
            rows.append(dict(artefact=f, column=c, identical=False, n_changed=-1,
                             max_abs_diff=np.nan, mean_diff=np.nan))
    changed = []
    for c in shared:
        x, y = A[c].values.astype("float64"), B[c].values.astype("float64")
        nan_same = np.array_equal(np.isnan(x), np.isnan(y))
        if nan_same:
            d = y - x
            d = d[~np.isnan(d)]
            nz = int((d != 0).sum())
            mx = float(np.abs(d).max()) if len(d) else 0.0
            mn = float(d.mean()) if len(d) else 0.0
        else:
            nz = int((np.isnan(x) != np.isnan(y)).sum())
            mx, mn = float("nan"), float("nan")
        if nz:
            changed.append(c)
            rows.append(dict(artefact=f, column=c, identical=False, n_changed=nz,
                             max_abs_diff=round(mx, 4) if mx == mx else np.nan,
                             mean_diff=round(mn, 6) if mn == mn else np.nan))
    summary.append(dict(artefact=f, shape_old=str(A.shape), shape_new=str(B.shape),
                        index_identical=bool(A.index.equals(B.index)),
                        columns_changed=len(changed) + len(only_old) + len(only_new),
                        columns_total=len(cols_a)))
    P(f"{f}: {A.shape} vs {B.shape} | shared {len(shared)} | value-changed {len(changed)}"
      + (f" -> {changed}" if changed else "") +
      (f" | columns removed {len(only_old)}, added {len(only_new)}" if (only_old or only_new) else ""))

t = pd.DataFrame(rows); s = pd.DataFrame(summary)
t.to_csv(os.path.join(OUT, "res_a14_drift_audit.csv"), index=False, encoding="utf-8-sig")
s.to_csv(os.path.join(OUT, "res_a14_drift_summary.csv"), index=False, encoding="utf-8-sig")
P("\n--- summary ---")
P(s.to_string(index=False))
P("\n--- every column that differs ---")
P(t[~t.identical].to_string(index=False))
P("\nwrote res_a14_drift_audit.csv / res_a14_drift_summary.csv")
