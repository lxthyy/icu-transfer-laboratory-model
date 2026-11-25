# -*- coding: utf-8 -*-
"""Run the full analysis pipeline in order, then collect and check the key numbers.

v1.42.1 wiring fixes (no model logic changed):
  * collect() now emits all 37 items in expected_numbers.csv, including AUC CIs,
    horizon, ablation and the censoring deficit.
  * "deficit vs" -> "deficit" (Z2 writes "deficit versus observed").
  * --check-only re-collects from an existing results tree and compares to
    expected_numbers.csv; exits non-zero on any FAIL/MISSING.
  * a full run also checks at the end; --no-check disables that.
  * --skip-heavy removed (it left A11 without the A4 model specification).
  * default fail-fast; --keep-going restores the old continue-on-error behaviour.
  * A4_v131's internal ward files are excluded from the merged folder, so they
    cannot overwrite A4c_ward's Table-2 ward results.
  * A14 SKIP is shown on the END line.
  * BLAS thread counts are pinned to 1 for each worker.

Usage
-----
    python run_all.py --data-root D:\\work [--source-dir D:\\work\\source]
                      [--legacy-root D:\\work\\legacy]
    python run_all.py --data-root D:\\work --check-only
"""
import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCR = os.path.join(HERE, "scripts")
EXPECTED = os.path.join(HERE, "expected_numbers.csv")

ORDER = [
    "A1_build.py",
    "A4_v131.py",
    "A4c_ward.py",
    "A11_crosspopulation.py",
    "A6_nocount_v132.py",
    "A5_matched_v132.py",
    "B1b_table1_fullcohort.py",
    "A10_temporal_v132.py",
    "A16_horizon.py",
    "A17_split_and_48h.py",
    "A12_seed_robustness.py",
    "A13_robustness.py",
    "A15_determinism.py",
    "A18_complete_followup_C003_v142.py",
    "Z2_censoring_km_v142.py",
    "Z3_ipcw_v142.py",
    "A8e_map_audit_table.py",
    "A14_drift_audit.py",
    "A7_numbers_v132.py",
]

# folders whose CSV outputs are pooled for A7_numbers. A18/Z2 are excluded because
# they reuse the res_a4_* file names for a different analysis.
MERGE = ["A4_v131", "A4c_ward", "A11_crosspopulation", "A6_nocount_v132", "A5_matched_v132",
         "B1b_table1_fullcohort", "A10_temporal_v132", "A16_horizon", "A17_split_and_48h",
         "A12_seed_robustness", "A13_robustness", "A15_determinism", "A8e_map_audit_table"]


def load_expected():
    import pandas as pd
    d = pd.read_csv(EXPECTED)
    return {str(q): str(v) for q, v in zip(d["quantity"], d["value"])}


def decimals(s):
    m = re.search(r"-?\d+\.(\d+)", str(s).replace(",", ""))
    return len(m.group(1)) if m else 0


def parse_num(s):
    m = re.search(r"-?\d+(?:\.\d+)?", str(s).replace(",", ""))
    return float(m.group(0)) if m else None


def fmt(q, v, exp):
    """Format an observed value to the precision used in expected_numbers.csv."""
    if str(v) == "MISSING":
        return "MISSING"
    e = exp.get(q, "")
    if "%" in e:
        try:
            return f"{float(str(v).replace('%', '')):.{decimals(e)}f}%"
        except Exception:
            return str(v)
    if any(ch.isalpha() for ch in e):
        return str(v)
    if "." not in e:
        try:
            return str(int(round(float(v))))
        except Exception:
            return str(v)
    try:
        return f"{float(v):.{decimals(e)}f}"
    except Exception:
        return str(v)


def collect(root, results):
    """Pull every expected value out of the result files."""
    import pandas as pd
    exp = load_expected()
    out = []

    def one(path, key=None, col="value"):
        try:
            d = pd.read_csv(path)
            if key is None:
                v = d.iloc[0][col]
            else:
                d["item"] = d["item"].astype(str)
                v = d.set_index("item").loc[key, col]
            if isinstance(v, pd.Series):
                v = v.iloc[0]
            return v
        except Exception:
            return "MISSING"

    def first(path, **conds):
        try:
            d = pd.read_csv(path)
            for k, v in conds.items():
                d = d[d[k].astype(str) == str(v)]
            return d.iloc[0] if len(d) else "MISSING"
        except Exception:
            return "MISSING"

    def row(name, value, source):
        out.append([name, fmt(name, value, exp), source])

    s = lambda *p: os.path.join(results, *p)

    for k in ["n_eligible", "n_cases", "prevalence_pct", "C_selected",
              "n_features_retained", "test_auc", "test_auc_lo", "test_auc_hi",
              "test_ap", "calibration_slope", "oe_ratio", "brier"]:
        row("primary." + k, one(s("A4_v131", "res_a4_summary_full.csv"), k), "A4_v131")

    for k in ["C_fixed", "n_test", "n_features_retained", "test_auc",
              "calibration_slope", "oe_ratio"]:
        row("ward." + k, one(s("A4c_ward", "res_a4_summary_ward.csv"), k), "A4c_ward")

    for k in ["train_auc", "validation_auc"]:
        row("temporal." + k, one(s("A10_temporal_v132", "res_a10_temporal.csv"), k), "A10_temporal")

    h16 = s("A16_horizon", "res_a16_horizon.csv")
    for tag, q in [("7-day", "7d"), ("14-day", "14d"), ("30-day", "30d")]:
        r = first(h16, horizon=tag)
        row(f"horizon.{q}_auc", "MISSING" if isinstance(r, str) else r["test_auc"], "A16_horizon")
    r = first(s("A17_split_and_48h", "res_a17_horizon48.csv"), horizon="48-hour")
    row("horizon.48h_auc", "MISSING" if isinstance(r, str) else r["test_auc"], "A17_split_and_48h")

    ab = s("A17_split_and_48h", "res_a17_split_ablation.csv")
    for arm, q in [("no-ordering", "no_ordering"), ("no-abnormality", "no_abnormality")]:
        r = first(ab, arm=arm)
        row(f"ablation.{q}_auc", "MISSING" if isinstance(r, str) else r["test_auc"],
            "A17_split_and_48h")
        if q == "no_ordering":
            row("ablation.no_ordering_retained",
                "MISSING" if isinstance(r, str) else r["retained"], "A17_split_and_48h")
    r = first(s("A6_nocount_v132", "res_a6_nocount.csv"), arm="no-count")
    row("ablation.no_count_auc", "MISSING" if isinstance(r, str) else r["test_auc"],
        "A6_nocount_v132")

    for k in ["n_eligible", "C_selected", "n_features_retained", "test_auc",
              "test_ap", "calibration_slope", "oe_ratio"]:
        row("followup." + k,
            one(s("A18_complete_followup_C003_v142", "res_a4_summary_full.csv"), k), "A18")

    try:
        txt = open(s("Z2_censoring_km_v142", "censoring_report_v142.txt"), encoding="utf-8").read()
        for line in txt.splitlines():
            if line.startswith("Kaplan-Meier"):
                row("censoring.km_30d", line.split(":", 1)[1].strip(), "Z2")
            if line.startswith("deficit"):
                row("censoring.deficit", line.split(":", 1)[1].strip(), "Z2")
    except Exception:
        row("censoring.km_30d", "MISSING", "Z2")
        row("censoring.deficit", "MISSING", "Z2")

    with open(os.path.join(root, "key_numbers.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "value", "source"])
        w.writerows(out)
    return out


def check(root):
    """Compare key_numbers.csv with expected_numbers.csv; write reproduction_check.csv."""
    import pandas as pd
    exp = pd.read_csv(EXPECTED)
    kn = pd.read_csv(os.path.join(root, "key_numbers.csv"))
    got = {str(q): (str(v), str(src)) for q, v, src in
           zip(kn["quantity"], kn["value"], kn["source"])}
    rows = []
    npass = nfail = nmiss = 0
    for _, r in exp.iterrows():
        q, e = str(r["quantity"]), str(r["value"])
        o, src = got.get(q, ("MISSING", ""))
        if o == "MISSING":
            status, diff, dec = "MISSING", "", decimals(e)
            nmiss += 1
        elif q == "censoring.deficit":
            en = re.findall(r"\d+(?:\.\d+)?", e)
            on = re.findall(r"\d+(?:\.\d+)?", o)
            ok = len(en) == len(on) and all(
                abs(float(a) - float(b)) <= 0.5 * 10 ** (-decimals(a)) + 1e-9
                for a, b in zip(en, on))
            status, diff, dec = ("PASS" if ok else "FAIL"), "", ""
            npass += int(ok)
            nfail += int(not ok)
        else:
            ev, ov = parse_num(e), parse_num(o)
            if ev is None or ov is None:
                ok = e.strip() == o.strip()
                diff = ""
                dec = decimals(e)
            else:
                dec = decimals(e)
                ok = abs(ov - ev) <= 0.5 * 10 ** (-dec) + 1e-9
                diff = f"{ov - ev:+.6g}"
            status = "PASS" if ok else "FAIL"
            npass += int(ok)
            nfail += int(not ok)
        rows.append([q, e, o, diff, dec, status, src])

    out = pd.DataFrame(rows, columns=["quantity", "expected", "observed",
                                      "abs_diff", "decimals", "pass_", "source"])
    out.to_csv(os.path.join(root, "reproduction_check.csv"), index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))
    print(f"\nPASS {npass} / {len(out)}   FAIL {nfail}   MISSING {nmiss}")
    return nfail == 0 and nmiss == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--source-dir", default=None,
                    help="folder with 住院_面板长表.parquet and 参考区间_主表.csv")
    ap.add_argument("--legacy-root", default=None,
                    help="optional; folder containing build/ and build_v132/ for A14")
    ap.add_argument("--check-only", action="store_true",
                    help="do not run scripts; re-collect from results and compare to expected")
    ap.add_argument("--keep-going", action="store_true",
                    help="continue after a script exits non-zero (default: stop)")
    ap.add_argument("--no-check", action="store_true",
                    help="skip the final comparison (not recommended)")
    args = ap.parse_args()

    root = os.path.abspath(args.data_root)
    src = os.path.abspath(args.source_dir) if args.source_dir else os.path.join(root, "source")
    build = os.path.join(root, "build")
    results = os.path.join(root, "results")

    if args.check_only:
        if not os.path.isdir(results):
            print("ERROR: results folder not found:", results)
            sys.exit(2)
        collect(root, results)
        sys.exit(0 if check(root) else 1)

    os.makedirs(build, exist_ok=True)
    os.makedirs(results, exist_ok=True)

    env0 = dict(os.environ)
    env0.update(ICU_DATA_DIR=root, ICU_SOURCE_DIR=src, ICU_BUILD_DIR=build,
                PYTHONIOENCODING="utf-8",
                OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if args.legacy_root:
        env0["ICU_LEGACY_ROOT"] = os.path.abspath(args.legacy_root)

    rows = []
    for s in ORDER:
        name = os.path.splitext(s)[0]
        if s == "A7_numbers_v132.py":
            out = os.path.join(results, "_merged")
            os.makedirs(out, exist_ok=True)
            for d in MERGE:
                src_dir = os.path.join(results, d)
                if os.path.isdir(src_dir):
                    for f in os.listdir(src_dir):
                        if not f.endswith(".csv"):
                            continue
                        # A4_v131 also writes an internal ward scenario; it is not Table 2
                        if d == "A4_v131" and "ward" in f.lower():
                            continue
                        shutil.copy2(os.path.join(src_dir, f), os.path.join(out, f))
        else:
            out = os.path.join(results, name)
            os.makedirs(out, exist_ok=True)
        env = dict(env0)
        env["ICU_OUT_DIR"] = out
        t0 = time.time()
        print(f"===== START {s} =====", flush=True)
        with open(os.path.join(results, name + ".log"), "w", encoding="utf-8") as fh:
            p = subprocess.run([sys.executable, os.path.join(SCR, s)],
                               cwd=root, env=env, stdout=fh, stderr=subprocess.STDOUT)
        sec = int(time.time() - t0)
        suffix = ""
        if s == "A14_drift_audit.py":
            try:
                txt = open(os.path.join(results, name + ".log"), encoding="utf-8").read()
                if "SKIP A14" in txt:
                    suffix = "  (SKIP: no legacy root)"
            except Exception:
                pass
        print(f"===== END   {s}  exit={p.returncode}  {sec}s{suffix} =====")
        rows.append([s, p.returncode, sec])

        if s == "A4_v131.py":
            spec = os.path.join(results, "A4_v131", "res_a4_model_spec_full.csv")
            if os.path.exists(spec):
                os.makedirs(os.path.join(results, "A11_crosspopulation"), exist_ok=True)
                shutil.copy2(spec, os.path.join(results, "A11_crosspopulation"))

        if p.returncode != 0 and not args.keep_going:
            print(f"STOP: {s} exited {p.returncode}; see results/{name}.log")
            break

    with open(os.path.join(root, "run_manifest.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["script", "exit", "seconds"])
        w.writerows(rows)

    collect(root, results)
    if not args.no_check:
        if not check(root):
            sys.exit(1)
    print("\nDone. See run_manifest.csv, key_numbers.csv and reproduction_check.csv in", root)


if __name__ == "__main__":
    main()
