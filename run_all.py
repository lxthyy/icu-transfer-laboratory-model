# -*- coding: utf-8 -*-
"""Run the full analysis pipeline in order, then collect the key numbers.

Usage
-----
    python run_all.py --data-root D:\\work [--source-dir D:\\work\\source]
                      [--legacy-root D:\\work\\legacy] [--skip-heavy]

The script never edits itself or the data; it only writes below --data-root.
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCR = os.path.join(HERE, "scripts")

# script, needs (informational), produces-into-own-folder
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
    "A8e_map_audit_table.py",
    "A14_drift_audit.py",
    "A7_numbers_v132.py",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--source-dir", default=None,
                    help="folder with 住院_面板长表.parquet and 参考区间_主表.csv")
    ap.add_argument("--legacy-root", default=None,
                    help="optional; folder containing build/ and build_v132/ for A14")
    ap.add_argument("--skip-heavy", action="store_true",
                    help="skip A4_v131, A15 and A12 (fast structural check only)")
    args = ap.parse_args()

    root = os.path.abspath(args.data_root)
    src = os.path.abspath(args.source_dir) if args.source_dir else os.path.join(root, "source")
    build = os.path.join(root, "build")
    results = os.path.join(root, "results")
    os.makedirs(build, exist_ok=True)
    os.makedirs(results, exist_ok=True)

    env0 = dict(os.environ)
    env0.update(ICU_DATA_DIR=root, ICU_SOURCE_DIR=src, ICU_BUILD_DIR=build,
                PYTHONIOENCODING="utf-8")
    if args.legacy_root:
        env0["ICU_LEGACY_ROOT"] = os.path.abspath(args.legacy_root)

    skiplist = {"A4_v131.py", "A15_determinism.py", "A12_seed_robustness.py"} if args.skip_heavy else set()

    # folders whose CSV outputs are pooled for A7_numbers (A18/Z2 are excluded because they
    # reuse the res_a4_* file names for a different analysis)
    MERGE = ["A4_v131", "A4c_ward", "A11_crosspopulation", "A6_nocount_v132", "A5_matched_v132",
             "B1b_table1_fullcohort", "A10_temporal_v132", "A16_horizon", "A17_split_and_48h",
             "A12_seed_robustness", "A13_robustness", "A15_determinism", "A8e_map_audit_table"]

    rows = []
    for s in ORDER:
        name = os.path.splitext(s)[0]
        if s in skiplist:
            rows.append([s, "SKIPPED", 0]); print(f"SKIP {s}"); continue
        if s == "A7_numbers_v132.py":
            out = os.path.join(results, "_merged")
            os.makedirs(out, exist_ok=True)
            for d in MERGE:
                src_dir = os.path.join(results, d)
                if os.path.isdir(src_dir):
                    for f in os.listdir(src_dir):
                        if f.endswith(".csv"):
                            shutil.copy2(os.path.join(src_dir, f), os.path.join(out, f))
        else:
            out = os.path.join(results, name)
            os.makedirs(out, exist_ok=True)
        env = dict(env0); env["ICU_OUT_DIR"] = out
        t0 = time.time()
        print(f"===== START {s} =====", flush=True)
        with open(os.path.join(results, name + ".log"), "w", encoding="utf-8") as fh:
            p = subprocess.run([sys.executable, os.path.join(SCR, s)],
                               cwd=root, env=env, stdout=fh, stderr=subprocess.STDOUT)
        sec = int(time.time() - t0)
        print(f"===== END   {s}  exit={p.returncode}  {sec}s =====", flush=True)
        rows.append([s, p.returncode, sec])

        # wiring: A11 reads the model specification written by A4_v131
        if s == "A4_v131.py":
            spec = os.path.join(results, "A4_v131", "res_a4_model_spec_full.csv")
            if os.path.exists(spec):
                os.makedirs(os.path.join(results, "A11_crosspopulation"), exist_ok=True)
                shutil.copy2(spec, os.path.join(results, "A11_crosspopulation"))

    with open(os.path.join(root, "run_manifest.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["script", "exit", "seconds"]); w.writerows(rows)

    collect(root, results)
    print("\nDone. See run_manifest.csv and key_numbers.csv in", root)


def collect(root, results):
    """Pull the headline values out of the result files."""
    import pandas as pd

    def one(path, key=None, col="value"):
        try:
            d = pd.read_csv(path)
            if key is None:
                return d.iloc[0][col]
            return d.set_index("item").loc[key, col]
        except Exception:
            return ""

    def row(name, value, source):
        out.append([name, value, source])

    out = []
    s = lambda *p: os.path.join(results, *p)
    for k in ["n_eligible", "n_cases", "prevalence_pct", "C_selected", "n_features_total",
              "n_features_retained", "test_auc", "test_ap", "calibration_slope", "oe_ratio", "brier"]:
        row("primary." + k, one(s("A4_v131", "res_a4_summary_full.csv"), k), "A4_v131")
    for k in ["C_fixed", "n_test", "n_features_retained", "test_auc", "calibration_slope", "oe_ratio"]:
        row("ward." + k, one(s("A4c_ward", "res_a4_summary_ward.csv"), k), "A4c_ward")
    row("temporal.train_auc", one(s("A10_temporal_v132", "res_a10_temporal.csv"), "train_auc"), "A10_temporal")
    row("temporal.validation_auc", one(s("A10_temporal_v132", "res_a10_temporal.csv"), "validation_auc"), "A10_temporal")
    for k in ["n_eligible", "C_selected", "n_features_retained", "test_auc", "test_ap",
              "calibration_slope", "oe_ratio"]:
        row("followup." + k, one(s("A18_complete_followup_C003_v142", "res_a4_summary_full.csv"), k), "A18")
    try:
        km = open(s("Z2_censoring_km_v142", "censoring_report_v142.txt"), encoding="utf-8").read()
        for line in km.splitlines():
            if line.startswith("Kaplan-Meier"):
                row("censoring.km_30d", line.split(":")[1].strip(), "Z2")
            if line.startswith("deficit vs"):
                row("censoring.deficit", line.split(":", 1)[1].strip(), "Z2")
    except Exception:
        pass

    with open(os.path.join(root, "key_numbers.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["quantity", "value", "source"]); w.writerows(out)


if __name__ == "__main__":
    main()
