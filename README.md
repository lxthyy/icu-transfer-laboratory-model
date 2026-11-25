# Analysis code — routine laboratory data and 30-day ICU transfer

Reproduction package for the manuscript *“Routine laboratory data for predicting intensive
care unit transfer: a retrospective prediction model study with temporal validation”*
(International Journal of Medical Informatics, manuscript IJMEDI-D-26-05531).

The package contains **code only**. No patient-level data are included or required to run the
pipeline end to end when the two source tables are available locally (see *Inputs*).

---

> Repository: https://github.com/lxthyy/icu-transfer-laboratory-model
> Archived release: Zenodo DOI to be added at publication


## 1. What is here

```
scripts/A1_build.py                     cohort definition, feature matrices, matched-sample files
scripts/A4_v131.py                      primary model (full cohort, natural prevalence, L1)
scripts/A4c_ward.py                     model refitted within the ward-onset population
scripts/A11_crosspopulation.py          one fixed model scored in sub-populations of its test set
scripts/A5_matched_v132.py              matched-sample arms (reproduces the originally submitted design)
scripts/A6_nocount_v132.py              removal of all count-type and test-ordering variables
scripts/A10_temporal_v132.py            temporal validation (Jan-Mar training, Apr-May validation)
scripts/A16_horizon.py                  7-, 14- and 30-day outcome horizons on one common split
scripts/A17_split_and_48h.py            48-hour horizon and the separated ordering/abnormality ablations
scripts/A12_seed_robustness.py          ten random 70/30 splits
scripts/A13_robustness.py               four winsorisation thresholds; leave-one-department-out
scripts/A14_drift_audit.py              column-level comparison of two builds (optional, see below)
scripts/A15_determinism.py              ten consecutive fits, MD5 of coefficients and predictions
scripts/A18_complete_followup_C003_v142.py  complete 30-day follow-up sensitivity analysis
scripts/Z2_censoring_km_v142.py         Kaplan-Meier correction for administrative censoring
scripts/B1b_table1_fullcohort.py        Table 1
scripts/A8e_map_audit_table.py          audit table for the synonym (harmonisation) rules
scripts/A7_numbers_v132.py              consolidates every reported number with its source file
run_all.py                              runs the pipeline in order and checks the key numbers
expected_numbers.csv                    the values a successful run must reproduce
requirements.txt
```

Scripts that audit the manuscript files themselves (sentence audit, cross-check, typo scan) are
not part of this package, because they require the submitted documents and produce no estimate.

## 2. Inputs

Two derived tables are needed. Both are de-identified extracts; they are **not** distributed
with this package.

| Environment variable | Meaning |
|---|---|
| `ICU_DATA_DIR` | working root; the pipeline writes below it |
| `ICU_SOURCE_DIR` | folder holding `住院_面板长表.parquet` and `参考区间_主表.csv` |
| `ICU_BUILD_DIR` | output folder of `A1_build.py` (the “build”) |
| `ICU_OUT_DIR` | output folder for result tables |
| `ICU_LEGACY_ROOT` | optional; only for `A14_drift_audit.py` |
| `ICU_A1_SCRIPT` | optional; defaults to the `A1_build.py` next to the calling script |

`住院_面板长表.parquet` must contain the columns `住院号`, `科室`, `检验日期`, `年龄`, `性别`,
`检验项目`, `结果`. `参考区间_主表.csv` supplies the reference band used to flag abnormal results.

## 3. Run

```bash
pip install -r requirements.txt
python run_all.py --data-root /path/to/working/root
```

`run_all.py` creates the folder layout, sets the environment variables, runs the scripts in the
order below, and writes `run_manifest.csv` (one row per script: exit code, wall time) and
`key_numbers.csv` (the values to compare with `expected_numbers.csv`).

Order and dependencies:

```
A1_build                    -> build/
A4_v131                     -> results/A4_v131
A4c_ward                    -> results/A4c_ward          (overwrites the ward-scenario files)
A11_crosspopulation         needs res_a4_model_spec_full.csv from A4_v131
A6_nocount, A5_matched, B1b_table1, A10_temporal, A16_horizon, A17_split_and_48h,
A12_seed, A13_robustness, A15_determinism
A18_complete_followup       -> its own folder; do not merge with A4_v131 (same file names)
Z2_censoring                -> its own folder
A8e_map_audit               reads A1_build.py and the source panel
A14_drift_audit             optional; needs a pre-correction build (see below)
A7_numbers                  reads the merged result folder; run last
```

Runtime is about two hours on four cores; `A4_v131.py` (penalty grid) and `A15_determinism.py`
(ten fits) dominate.

## 4. Two points of care

**4.1 `A4_v131.py` runs a ward scenario that is not the ward column of Table 2.** That column
comes from `A4c_ward.py`, which refits the model inside the ward-onset population at C = 0.1 and
retains 64 of 193 coefficients. The ward scenario inside `A4_v131.py` selects C = 0.3 and retains
120 coefficients; it is retained only because it was part of the original run and is not used in
the manuscript.

**4.2 `A14_drift_audit.py` needs two builds.** It compares the build used for the originally
submitted analysis with the build used after the reference-interval correction. Set
`ICU_LEGACY_ROOT` to a folder that contains both (`build/` and `build_v132/`). Without it the
script prints a message and exits without error.

## 5. Expected values

`expected_numbers.csv` lists the values a successful run must reproduce, with the script that
produces each. The main ones:

| Quantity | Value |
|---|---|
| Analysis cohort / events | 181,947 / 1,109 (0.610%) |
| Primary model, penalty grid selected C | 0.03, 46 of 193 coefficients non-zero |
| Primary test AUC (95% CI) | 0.8830 (0.8666–0.8985) |
| Primary average precision | 0.1081 |
| Calibration slope / observed-to-expected / Brier | 1.1675 / 0.9582 / 0.005729 |
| Ward-onset refit (test n = 20,279) | AUC 0.7491, 64 coefficients, C = 0.1 |
| Temporal validation | 0.8850 training, 0.8589 (0.8415–0.8742) validation |
| Complete 30-day follow-up (155,031; C fixed at 0.03) | 49 coefficients, AUC 0.8783, AP 0.0730, slope 1.0981, O/E 0.9437 |
| Kaplan-Meier 30-day incidence vs naive | 0.62763% vs 0.60952% (32.9 events, 2.89%) |

## 6. Environment used for the reported results

Python 3.13.13; numpy 2.4.4; pandas 2.3.3; scipy 1.17.1; scikit-learn 1.8.0; joblib 1.5.3.
Seeds: split and cross-validation 42, bootstrap 2026 × 1,000 resamples. All scripts are
deterministic; ten consecutive fits give byte-identical coefficient and prediction vectors.

## 7. Data availability

The record-level data are not publicly available: they are derived from a hospital laboratory
information system and, although direct identifiers were removed, they retain dates and clinical
detail. De-identified data may be made available for verification through the institutional data
access committee under a data-use agreement. Everything needed to re-derive every reported
number without the record-level data is supplied here: the model specification (coefficients,
imputation values, winsorisation bounds and standardisation constants) and the test-set
predictions (outcome and predicted risk only).
