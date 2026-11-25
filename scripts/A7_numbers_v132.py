# -*- coding: utf-8 -*-
"""Assemble every number that goes into the revision, reading only from files.
No value is typed by hand; each row records the file it came from."""
import os, sys, glob
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd, numpy as np

ICU_DATA_DIR = os.environ.get("ICU_DATA_DIR", "data")
BUILD = os.environ.get("ICU_BUILD_DIR", os.path.join(ICU_DATA_DIR, "_ICU预警成稿", "build"))
OUT = os.environ.get("ICU_OUT_DIR", os.path.join(BUILD, "..", "02_结果输出", "v1.32_全队列"))
rows = []
def add(section, item, value, source):
    rows.append(dict(section=section, item=item, value=value, source=source))

# ---- cohort ----
lab = pd.read_parquet(os.path.join(BUILD, "labels.parquet"))
F = pd.read_parquet(os.path.join(BUILD, "feat_48h_orig.parquet"))
ids = F.index.intersection(lab.index)
sub = lab.loc[ids]
WARD_KEYS = ["病房", "病区", "诊疗中心", "ICU"]
ICU_EXACT = {"重症医学科病房", "神经重症医学科(ICU)", "急诊ICU病房", "呼吸与危重症医学科ICU病区", "中心ICU"}
def dep_cat(s):
    if not isinstance(s, str): return "其他"
    if "门诊" in s: return "门诊"
    if any(k in s for k in WARD_KEYS): return "病房/病区"
    return "其他"
cat = sub["first_dept"].map(dep_cat)
is_ward = (cat == "病房/病区") & (~sub["first_dept"].isin(ICU_EXACT))
src = "build_v132/labels.parquet + feat_48h_orig.parquet"
add("队列", "全部住院记录", f"{len(lab):,}", src)
add("队列", "有特征行的合格队列", f"{len(ids):,}", src)
add("队列", "30天转ICU病例", f"{int(sub['case_30d'].sum()):,}", src)
add("队列", "患病率", f"{sub['case_30d'].mean()*100:.3f}%", src)
add("队列", "7天病例", f"{int(sub['case_7d'].sum()):,}", src)
add("队列", "14天病例", f"{int(sub['case_14d'].sum()):,}", src)
for k, v in cat.value_counts().items():
    add("队列", f"首次检验科室={k}", f"{v:,}", src)
add("队列", "病区起病人群（剔除ICU病区）", f"{int(is_ward.sum()):,}", src)
add("队列", "病区起病病例", f"{int(sub.loc[is_ward,'case_30d'].sum()):,}", src)
add("队列", "病区起病患病率", f"{sub.loc[is_ward,'case_30d'].mean()*100:.3f}%", src)

# ---- full scenario ----
p = os.path.join(OUT, "res_a4_summary_full.csv")
if os.path.exists(p):
    s = pd.read_csv(p)
    for _, r in s.iterrows():
        add("全队列模型", str(r["item"]), str(r["value"]), os.path.basename(p))
def percase(r):
    for k in ("n_flagged_per_case_detected", "n_flagged_per_case"):
        if k in r.index:
            return float(r[k])
    return np.nan

p = os.path.join(OUT, "res_a4_alert_budget_full.csv")
if os.path.exists(p):
    for _, r in pd.read_csv(p).iterrows():
        add("全队列报警预算", f"top {r['budget']*100:.0f}% 报警人数", f"{int(r['n_flagged']):,}", os.path.basename(p))
        add("全队列报警预算", f"top {r['budget']*100:.0f}% 检出病例", f"{int(r['cases_detected'])}", os.path.basename(p))
        add("全队列报警预算", f"top {r['budget']*100:.0f}% 敏感度", f"{r['sensitivity']:.3f}", os.path.basename(p))
        add("全队列报警预算", f"top {r['budget']*100:.0f}% PPV", f"{r['ppv']:.4f}", os.path.basename(p))
        add("全队列报警预算", f"top {r['budget']*100:.0f}% 每例筛查数", f"{percase(r):.1f}", os.path.basename(p))

# ---- ward scenario ----
p = os.path.join(OUT, "res_a4_summary_ward.csv")
if os.path.exists(p):
    for _, r in pd.read_csv(p).iterrows():
        add("病区模型", str(r["item"]), str(r["value"]), os.path.basename(p))
p = os.path.join(OUT, "res_a4_alert_budget_ward.csv")
if os.path.exists(p):
    for _, r in pd.read_csv(p).iterrows():
        add("病区报警预算", f"top {r['budget']*100:.0f}% 报警人数", f"{int(r['n_flagged']):,}", os.path.basename(p))
        add("病区报警预算", f"top {r['budget']*100:.0f}% 检出病例", f"{int(r['cases_detected'])}", os.path.basename(p))
        add("病区报警预算", f"top {r['budget']*100:.0f}% 敏感度", f"{r['sensitivity']:.3f}", os.path.basename(p))
        add("病区报警预算", f"top {r['budget']*100:.0f}% PPV", f"{r['ppv']:.4f}", os.path.basename(p))
        add("病区报警预算", f"top {r['budget']*100:.0f}% 每例筛查数", f"{percase(r):.1f}", os.path.basename(p))

# ---- matched arms ----
p = os.path.join(OUT, "res_a5_matched_arms.csv")
if os.path.exists(p):
    for _, r in pd.read_csv(p).iterrows():
        add("匹配样本三臂", f"{r['arm']}｜C", str(r["C"]), os.path.basename(p))
        add("匹配样本三臂", f"{r['arm']}｜保留变量", str(int(r["n_features"])), os.path.basename(p))
        add("匹配样本三臂", f"{r['arm']}｜测试AUC", f"{r['test_auc']:.4f}", os.path.basename(p))
        add("匹配样本三臂", f"{r['arm']}｜AP", f"{r['test_ap']:.4f}", os.path.basename(p))
        add("匹配样本三臂", f"{r['arm']}｜Brier", f"{r['test_brier']:.4f}", os.path.basename(p))

# ---- no-count ----
p = os.path.join(OUT, "res_a6_nocount.csv")
if os.path.exists(p):
    for _, r in pd.read_csv(p).iterrows():
        add("去计数敏感性", "保留变量数", str(int(r["n_features"])), os.path.basename(p))
        add("去计数敏感性", "C", str(r["C"]), os.path.basename(p))
        add("去计数敏感性", "测试AUC", f"{r['test_auc']:.4f}", os.path.basename(p))
        add("去计数敏感性", "AP", f"{r['test_ap']:.4f}", os.path.basename(p))
        add("去计数敏感性", "Brier", f"{r['test_brier']:.5f}", os.path.basename(p))

# ---- CV curves ----
for tag in ("full", "ward"):
    p = os.path.join(OUT, f"res_a4_cv_C_{tag}.csv")
    if os.path.exists(p):
        for _, r in pd.read_csv(p).iterrows():
            add(f"{tag} C曲线", f"C={r['C']}", f"{r['cv_auc']:.4f} +/- {r['cv_sd']:.4f}", os.path.basename(p))

# ---- tail audit / data quality ----
p = os.path.join(OUT, "res_a4_tail_audit.csv")
if os.path.exists(p):
    d = pd.read_csv(p)
    add("数据质量", "超出自身p99.9五倍的变量数", f"{int((d['n_above_5xp999']>0).sum())} / {len(d)}", os.path.basename(p))
    add("数据质量", "超出自身p99.9五倍的单元格", f"{int(d['n_above_5xp999'].sum()):,}", os.path.basename(p))

# ---- temporal validation ----
p = os.path.join(OUT, "res_a10_temporal.csv")
if os.path.exists(p):
    for _, r in pd.read_csv(p).iterrows():
        add("时间验证", str(r["item"]), str(r["value"]), os.path.basename(p))

# ---- mapping audit ----
p = os.path.join(OUT, "res_a8_map_audit_final.csv")
if os.path.exists(p):
    d = pd.read_csv(p)
    for k, v in d["verdict"].value_counts().items():
        add("映射审计", f"判定={k}", str(v), os.path.basename(p))
    add("映射审计", "条目总数", str(len(d)), os.path.basename(p))

# ---- top coefficients of the new model ----
p = os.path.join(OUT, "res_a4_model_spec_full.csv")
if os.path.exists(p):
    d = pd.read_csv(p)
    nz = d[(d["feature"] != "(intercept, standardized scale)") &
           (d["standardized_coefficient"].abs() > 1e-8)]
    nz = nz.reindex(nz["standardized_coefficient"].abs().sort_values(ascending=False).index)
    for _, r in nz.head(12).iterrows():
        add("新模型系数", str(r["feature"]), f"{r['standardized_coefficient']:+.4f}", os.path.basename(p))

t = pd.DataFrame(rows)
t.to_csv(os.path.join(OUT, "v1.32_数字总表.csv"), index=False, encoding="utf-8-sig")
print(f"rows {len(t)} -> {os.path.join(OUT, 'v1.32_数字总表.csv')}")
print(t.groupby("section").size().to_string())
