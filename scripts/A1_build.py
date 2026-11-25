# -*- coding: utf-8 -*-
"""A1 构建统一数据管线：队列定义 + 特征矩阵（24h/48h；原始名/同义合并名）

主口径：t0 = 首次检验日期；病例 = 同一次住院内先有非 ICU 科室检验记录、随后 30d 内出现 ICU 病区检验记录
对照：同性别 + 年龄±3 岁，无放回按比例匹配
输出目录 build/
"""
import os, time, json
import numpy as np
import pandas as pd

DATA = os.environ.get("ICU_DATA_DIR", "data")
SRC = os.environ.get("ICU_SOURCE_DIR", os.path.join(DATA, "source"))
OUT = os.environ.get("ICU_BUILD_DIR", os.path.join(DATA, "build"))
os.makedirs(OUT, exist_ok=True)
SEED = 42
ICU_EXACT = {"重症医学科病房", "神经重症医学科(ICU)", "急诊ICU病房", "呼吸与危重症医学科ICU病区", "中心ICU"}
MAXF = 300
COV = 0.30
log = []
def logp(s):
    print(s, flush=True); log.append(str(s))

T0 = time.time()

# ---------------- 同义项目名规范化 ----------------
# 注意（2026-10-23 修正）：原版把 "白细胞" 映射到 "白细胞计数"，这是错的。
# LIS 中 "白细胞"（144,261 行，中位 3.00，p99 2,206，7.83% 的值>100）与
# "白细胞计数"（234,146 行，中位 6.28，p99 22.80，仅 0.32% 的值>100）是两个
# 分布完全不相容的项目，合并后 canon 矩阵的 白细胞计数_峰 最大值从 41,312
# 被抬到 63,878。私有库 lab 表没有标本类型字段，任何跨项目合并都是一次标本
# 误配的机会，故此条别名删除，"白细胞" 保持为独立项目（覆盖率不足会被丢弃）。
EXACT_MAP = {
    "MCV平均红细胞体积": "平均红细胞体积", "白细胞计数": "白细胞计数",
    "肌酐测定(干化学法）": "肌酐", "肌酐测定（干化学法）": "肌酐",
    "肌钙蛋白I": "肌钙蛋白I", "血清肌钙蛋白I": "肌钙蛋白I", "血清肌钙蛋白Ⅰ": "肌钙蛋白I",
    "心肌钙蛋白-I": "肌钙蛋白I", "超敏心肌肌钙蛋白i": "肌钙蛋白I",
    "高敏C反应蛋白": "高敏C反应蛋白", "超敏C反应蛋白定量": "高敏C反应蛋白", "超敏C反应蛋白测定": "高敏C反应蛋白",
    "降钙素原": "降钙素原", "降钙素原检测": "降钙素原",
    "钠": "钠", "钠测定": "钠", "钠离子": "钠", "血气钠测定": "钠",
    "钾": "钾", "钾测定": "钾", "钾离子": "钾", "血气钾测定": "钾",
    "氯": "氯", "氯测定": "氯", "氯离子": "氯", "血气氯测定": "氯",
    "钙": "钙", "钙测定": "钙",
    "葡萄糖": "葡萄糖", "血气葡萄糖": "葡萄糖",
    "二氧化碳总量": "二氧化碳总量", "血二氧化碳": "二氧化碳总量",
    "白球比": "白球比", "白蛋白/球蛋白": "白球比",
    "甲状腺球蛋白抗体": "甲状腺球蛋白抗体", "抗甲状腺球蛋白抗体": "甲状腺球蛋白抗体",
    "抗甲状腺球蛋白抗体测定": "甲状腺球蛋白抗体",
    "肌酸激酶MB同工酶": "肌酸激酶MB同工酶", "肌酸激酶－MB同工酶活性": "肌酸激酶MB同工酶",
    "血清肌酸激酶－MB同工酶活性测定": "肌酸激酶MB同工酶",
    "血清肌酸激酶－MB同工酶活性测定(CK-MB)": "肌酸激酶MB同工酶",
    "血清肌酸激酶－MB同工酶质量测定": "肌酸激酶MB同工酶",
    "γ-谷氨酰基转移酶": "γ-谷氨酰基转移酶", "血清γ-谷氨酰基转移酶测定": "γ-谷氨酰基转移酶",
    "天门冬氨酸氨基转移酶": "天门冬氨酸氨基转移酶", "血清天门冬氨酸氨基转移酶测定": "天门冬氨酸氨基转移酶",
    "总胆汁酸": "总胆汁酸", "血清总胆汁酸测定": "总胆汁酸",
    "直接胆红素": "直接胆红素", "血清直接胆红素测定": "直接胆红素",
    "N末端B型利钠肽原": "N末端B型利钠肽原", "B型钠尿肽前体测定": "N末端B型利钠肽原",
    "B型钠尿肽前体(荧光免疫法)": "N末端B型利钠肽原", "NT-pro脑钠肽": "N末端B型利钠肽原",
    "尿α1微球蛋白": "尿α1微球蛋白", "尿α1—微球蛋白测定": "尿α1微球蛋白", "尿α1微量球蛋白": "尿α1微球蛋白",
    "尿β2微球蛋白": "尿β2微球蛋白", "尿β2微球蛋白测定": "尿β2微球蛋白",
    "尿肌酐": "尿肌酐", "尿肌酐测定": "尿肌酐",
    "尿微量白蛋白": "尿微量白蛋白",
    "纤维蛋白原": "纤维蛋白原", "功能性纤维蛋白原": "纤维蛋白原",
    "氧和血红蛋白": "氧合血红蛋白",
    "过氧化物酶阳性白细胞": "过氧化物酶阳性白细胞",
}
def canon(n):
    n = str(n).replace(" ", "").replace("　", "")
    if n in EXACT_MAP:
        return EXACT_MAP[n]
    return n

# ---------------- 载入面板 ----------------
pan = pd.read_parquet(os.path.join(SRC, "住院_面板长表.parquet"),
                      columns=["住院号", "科室", "检验日期", "年龄", "性别", "检验项目", "结果"])
pan["住院号"] = pan["住院号"].astype(str)
pan["日期"] = pd.to_datetime(pan["检验日期"], errors="coerce")
pan["检验项目"] = pan["检验项目"].astype(str)
pan["结果"] = pd.to_numeric(pan["结果"], errors="coerce")
logp(f"面板 {pan.shape}｜日期 {pan['日期'].min().date()} ~ {pan['日期'].max().date()}")

g = pan.groupby("住院号")
t0s = g["日期"].min()
age_s = pd.to_numeric(pan["年龄"].astype(str).str.replace("岁", "", regex=False), errors="coerce").groupby(pan["住院号"]).first()
sex_s = g["性别"].first()
ntest_all = g.size()
first_dept = pan.sort_values("日期").groupby("住院号")["科室"].first()
n_dept = g["科室"].nunique()
n_item = g["检验项目"].nunique()

icu = pan[pan["科室"].isin(ICU_EXACT)]
icu_first = icu.groupby("住院号")["日期"].min()
icu_dept_first = icu.sort_values("日期").groupby("住院号")["科室"].first()
logp(f"住院号 {len(t0s):,}｜ICU 住院号 {len(icu_first):,}")

lab = pd.DataFrame({"t0": t0s, "age": age_s, "sex": sex_s, "n_test_all": ntest_all,
                    "first_dept": first_dept, "n_dept": n_dept, "n_item": n_item})
lab["icu_first"] = icu_first.reindex(lab.index)
lab["icu_dept"] = icu_dept_first.reindex(lab.index)
lab["gap_d"] = (lab["icu_first"] - lab["t0"]).dt.total_seconds() / 86400.0
lab["ever_icu"] = lab["icu_first"].notna()
lab["tr_like"] = lab["ever_icu"] & (lab["gap_d"] > 0)          # 口径乙：先普通科室后入 ICU
for W in (7, 14, 30):
    lab[f"case_{W}d"] = lab["tr_like"] & (lab["gap_d"] <= W)
lab["case_ever"] = lab["ever_icu"]                              # 口径甲（时序不要求，泄漏风险）

# ---------------- 参考区间 ----------------
ref = pd.read_csv(os.path.join(SRC, "参考区间_主表.csv"))
ref.columns = [str(c).strip() for c in ref.columns]
r = ref[(ref["分层类型"] == "全体")].copy()
if "备注" in r.columns:
    r = r[r["备注"].isna() | (r["备注"].astype(str).str.strip() == "")]
r["指标"] = r["指标"].astype(str).map(canon)
r = r.dropna(subset=["下限", "上限"]).groupby("指标")[["下限", "上限"]].median()
LO, HI = r["下限"], r["上限"]
logp(f"参考区间（全体层）可用 {len(r)} 项")

# ---------------- 特征矩阵构建 ----------------
OP_FEATS = ["检验项数", "异常项数", "异常比例", "最早抽血偏移h"]

def build(HRS, use_canon):
    w = pan[(pan["日期"] >= pan["住院号"].map(t0s)) &
            (pan["日期"] <= pan["住院号"].map(t0s) + pd.Timedelta(hours=HRS))].copy()
    if use_canon:
        w["项"] = w["检验项目"].map(canon)
    else:
        w["项"] = w["检验项目"]
    w = w.dropna(subset=["结果"])
    w = w.sort_values(["住院号", "项", "日期"])
    gg = w.groupby(["住院号", "项"])["结果"]
    agg = pd.DataFrame({"首": gg.first(), "峰": gg.max(), "谷": gg.min(), "次": gg.count()}).reset_index()
    wide = {}
    for c in ("首", "峰", "谷", "次"):
        p = agg.pivot_table(index="住院号", columns="项", values=c, aggfunc="first").astype("float32")
        wide[c] = p
    cv = wide["首"].notna().mean()
    keep = cv[cv >= COV].sort_values(ascending=False).head(MAXF).index.tolist()
    M = {c: wide[c][keep] for c in wide}
    F = pd.concat([M["首"].add_suffix("_首"), M["峰"].add_suffix("_峰"),
                   M["谷"].add_suffix("_谷"), M["次"].add_suffix("_次")], axis=1)
    # 异常负荷
    s = pd.Series(0.0, index=M["首"].index); nv = pd.Series(0.0, index=M["首"].index)
    for col in M["首"].columns:
        if col in LO.index:
            lo, hi = float(LO[col]), float(HI[col])
            v = M["首"][col]
            bad = ((v < lo) | (v > hi)).astype("float32")
            s = s.add(bad.where(v.notna(), np.nan).fillna(0), fill_value=0)
            nv = nv.add(v.notna().astype("float32"), fill_value=0)
    F["异常项数"] = s.astype("float32")
    F["异常比例"] = (s / nv.replace(0, np.nan)).astype("float32")
    F["检验项数"] = w.groupby("住院号").size().reindex(F.index).astype("float32")
    F["最早抽血偏移h"] = ((w.groupby("住院号")["日期"].min() - t0s.reindex(F.index)).dt.total_seconds() / 3600.0).astype("float32")
    F["年龄"] = age_s.reindex(F.index).astype("float32")
    F["男性"] = (sex_s.reindex(F.index) == "男").astype("float32")
    return F, keep

for HRS in (48, 24):
    for uc, tag in ((False, "orig"), (True, "canon")):
        F, keep = build(HRS, uc)
        p = os.path.join(OUT, f"feat_{HRS}h_{tag}.parquet")
        F.to_parquet(p)
        logp(f"[feat] {HRS}h {tag}: {F.shape}（项数 {len(keep)}）→ {os.path.basename(p)}｜{time.time()-T0:.0f}s")
        del F

lab.to_parquet(os.path.join(OUT, "labels.parquet"))
logp(f"[lab] 口径甲 ever_icu={int(lab['ever_icu'].sum()):,}｜口径乙 tr_like={int(lab['tr_like'].sum()):,}"
     f"｜case7={int(lab['case_7d'].sum()):,}｜case14={int(lab['case_14d'].sum()):,}｜case30={int(lab['case_30d'].sum()):,}")

# ---------------- 配对 ----------------
def match(F_index, case_ids, ratio, seed=SEED):
    pool = [i for i in F_index if i not in set(case_ids)]
    p = pd.DataFrame({"性别": sex_s.reindex(pool).values, "年龄": age_s.reindex(pool).values}, index=pool)
    p["used"] = False
    rng = np.random.default_rng(seed)
    picked = []
    for cid in case_ids:
        s, a = sex_s.get(cid), age_s.get(cid)
        if pd.isna(a) or pd.isna(s):
            continue
        cand = p[(~p["used"]) & (p["性别"] == s) & (p["年龄"].between(a - 3, a + 3))]
        if cand.empty:
            continue
        take = rng.choice(cand.index.values, size=min(ratio, len(cand)), replace=False)
        p.loc[take, "used"] = True
        picked += take.tolist()
    dropped = [c for c in case_ids if pd.isna(age_s.get(c)) or pd.isna(sex_s.get(c))]
    return [c for c in case_ids if c not in set(dropped)], picked

F48 = pd.read_parquet(os.path.join(OUT, "feat_48h_orig.parquet"))
idx = list(F48.index)
mt = []
for W in (7, 14, 30):
    for ratio in (4, 2, 1):
        cid = lab.index[(lab.index.isin(idx)) & (lab[f"case_{W}d"])].tolist()
        cs, ct = match(idx, cid, ratio)
        mt.append(dict(win=W, ratio=ratio, n_case=len(cs), n_ctrl=len(ct)))
        np.save(os.path.join(OUT, f"match_w{W}_r{ratio}_case.npy"), np.array(cs, dtype=object), allow_pickle=True)
        np.save(os.path.join(OUT, f"match_w{W}_r{ratio}_ctrl.npy"), np.array(ct, dtype=object), allow_pickle=True)
        logp(f"[match] 结局 {W}d 1:{ratio} → 病例 {len(cs):,} 对照 {len(ct):,}")
pd.DataFrame(mt).to_csv(os.path.join(OUT, "cohort_sizes.csv"), index=False, encoding="utf-8-sig")

with open(os.path.join(OUT, "A1_log.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(log))
logp(f"[done] {time.time()-T0:.0f}s")
