# RR 提升 POC — enrollment 污染检测自适应阈值 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: 用 superpowers:executing-plans 按 task 执行。Steps 用 `- [ ]` 复选框跟踪。

**Goal:** 验证"enrollment 污染检测→污染项自适应提 thr"能否不损 pos CER 地提 neg RR，输出 Go/No-Go + 推荐 thr_high

**Architecture:** 独立 POC 脚本 `code/poc_enrollment_pollution.py`，复用 enroll_infer.py 的 DiariZen+wespeaker 加载逻辑；分 Q1(检测器F1)/Q2(污染与漏拒相关性)/Q3(自适应thr代价)三阶段，每阶段独立可跑、独立验证、独立 commit

**Tech Stack:** PyTorch / DiariZen(pyrannote) / wespeaker emb / scipy.stats(Fisher exact) / scikit-learn(层次聚类) / pandas

**环境:** `code/.venv/Scripts/python.exe`（含 DiariZen+torch，复用 enroll_infer 同环境），GPU。参考 spec `docs/superpowers/specs/2026-07-16-rr-improvement-poc-design.md`

---

## File Structure

- **Create** `code/poc_enrollment_pollution.py` — 主脚本，含模型加载 + D1/D2 检测器 + Q1/Q2/Q3 三阶段 + 数据表输出
- **Create** `docs/POC_A_enrollment_pollution_结果_2026-07-16.md` — 结果文档（Go/No-Go + 推荐 thr_high + 数据表）
- **Read 参考** `code/enroll_infer.py:163-177`（DiariZen + get_emb 加载逻辑，复制不 import——enroll_infer 是 CLI main 不便 import）
- **Read 数据**
  - `code/neg_pairs_datasetA.json`（474 neg，字段 id/enrollment/recognition）
  - `code/pos_pairs_datasetA.json`（1364 pos）
  - `code/out_neg_vanilla_full/result.json`（neg 逐条 max_sim，join by id）
  - `code/poc_qwen_asr_full_result.json`（pos 逐条 sim + qwen 转写，join by id/uid）
  - `code/annot_pack/calibration_samples_v2.csv`（26 条，`enw_可靠性` 列作 Q1 真值）

---

## Task 1: 脚本骨架 + 模型加载 + 数据加载

**Files:** Create `code/poc_enrollment_pollution.py`

- [ ] **Step 1: 写模型加载 + 数据加载（复制 enroll_infer L163-177 接口）**

```python
# poc_enrollment_pollution.py 顶部（参考 enroll_infer.py:24-48 的 sys.path + safe_getmodule patch 必须带上, 否则 speechbrain lazy 崩）
import inspect as _inspect
_orig = _inspect.getmodule
def _safe(*a, **k):
    try: return _orig(*a, **k)
    except (ImportError, AttributeError): return None
_inspect.getmodule = _safe

import os, sys, json, csv, argparse
import numpy as np, torch, librosa
from repro import set_global_seed, resolve_model

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DICOW_INF = os.path.join(_ROOT, "code", "DiCoW-inference")
for _p in (DICOW_INF, os.path.join(DICOW_INF, "DiariZen"), os.path.join(DICOW_INF, "DiariZen", "pyannote-audio")):
    if os.path.isdir(_p): sys.path.insert(0, _p)

def load_models(device_str="cuda:0"):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    from diarizen.pipelines.inference import DiariZenPipeline
    diar = DiariZenPipeline.from_pretrained(resolve_model("DIAR")).to(device)
    def get_emb(wav_np):
        w = torch.from_numpy(np.ascontiguousarray(wav_np.astype(np.float32))).to(device)
        if w.dim() == 1: w = w[None, None]
        elif w.dim() == 2: w = w[None]
        with torch.no_grad(): emb = diar._embedding(w)
        emb = torch.as_tensor(emb, device=device, dtype=torch.float32)
        return torch.nn.functional.normalize(emb, dim=-1).squeeze(0).cpu().numpy()
    return diar, get_emb, device

def load_pairs_neg():  # → {id: enrollment_path}, join max_sim later
    rows = json.load(open(os.path.join(_HERE, "neg_pairs_datasetA.json"), encoding="utf-8"))
    return {r["id"]: r["enrollment"] for r in rows}

def load_neg_sim():  # out_neg_vanilla_full/result.json → {id: max_sim}（执行时确认字段名 uid/id + max_sim/sim）
    # 字段名执行时核对（可能是 result_eval.json），smoke 打印一条样本
    ...
```

- [ ] **Step 2: smoke 验证加载**

Run: `code/.venv/Scripts/python.exe code/poc_enrollment_pollution.py --smoke-load`
Expected: 打印 `neg pairs=474 pos pairs=1364 calibration=26` + DiariZen 加载成功

- [ ] **Step 3: Commit** `git add code/poc_enrollment_pollution.py && git commit -m "feat(poc): enrollment污染POC骨架+数据加载"`

---

## Task 2: D1 DiariZen 污染检测器

**Files:** Modify `code/poc_enrollment_pollution.py`

- [ ] **Step 1: 实现 detect_pollution_diar**

```python
def detect_pollution_diar(diar, enr_path):
    """D1: 对 enrollment 跑 DiariZen diar, speakers≥2 → 污染。返回 (is_polluted, n_speakers)"""
    try:
        diar_out = diar(enr_path)  # 接受路径, 同 enroll_infer.py:227
        n = len(diar_out.labels())
        return (n >= 2, n)
    except Exception as e:
        print(f"  [diar-fail] {os.path.basename(enr_path)}: {type(e).__name__} {str(e)[:60]}")
        return (None, None)  # 失败单独计, 不计入 F1 分母
```

- [ ] **Step 2: smoke** 对 calibration 里 1 条已知污染（`enw_可靠性` 含"污染"）+ 1 条干净，确认输出 True/False 合理
- [ ] **Step 3: Commit** `feat(poc): D1 DiariZen enrollment污染检测器`

---

## Task 3: D2 wespeaker 帧聚类检测器

**Files:** Modify `code/poc_enrollment_pollution.py`

- [ ] **Step 1: 实现 detect_pollution_cluster**

```python
from sklearn.cluster import AgglomerativeClustering
def detect_pollution_cluster(get_emb, enr_path, frame_sec=0.5, sim_thresh=0.5):
    """D2: enrollment 切 0.5s 帧 → 每帧 wespeaker emb → 余弦层次聚类(sim_thresh) → 簇≥2 污染"""
    wav, _ = librosa.load(enr_path, sr=16000)
    fl = int(frame_sec * 16000)
    frames = [wav[i:i+fl] for i in range(0, len(wav)-fl+1, fl)]
    if len(frames) < 2: return (False, 1)  # 太短无法分帧→单簇
    embs = np.stack([get_emb(f) for f in frames])  # (n_frames, dim) 已归一化
    dist = 1 - embs @ embs.T  # 余弦距离
    labels = AgglomerativeClustering(n_clusters=None, metric="precomputed",
                                     linkage="average", distance_threshold=1-sim_thresh).fit_predict(dist)
    n = len(set(labels))
    return (n >= 2, n)
```

- [ ] **Step 2: smoke** 同 Task 2 Step 2
- [ ] **Step 3: Commit** `feat(poc): D2 wespeaker帧聚类enrollment污染检测器`

---

## Task 4: Q1 — 检测器 F1（26 条 calibration）

**Files:** Modify `code/poc_enrollment_pollution.py`

- [ ] **Step 1: 加载 calibration 真值 + 二值化**

```python
def load_calib_truth():
    """calibration_samples_v2.csv → [(enr_path, is_polluted_bool)], 过短样本剔除(方向A只看污染)"""
    rows = list(csv.DictReader(open(os.path.join(_HERE, "annot_pack/calibration_samples_v2.csv"), encoding="utf-8-sig")))
    out = []
    for r in rows:
        enr = r.get("enrollment") or _uid_to_enr_path(r["uid"])  # csv 可能无 enr 列, 用 uid→kws_X.wav
        rel = r["enw_可靠性"]
        if "过短" in rel or "<1s" in rel: continue  # 剔除过短
        polluted = "污染" in rel
        out.append((enr, polluted))
    return out
```

- [ ] **Step 2: 跑 D1/D2 算 P/R/F1**

```python
def q1_f1(diar, get_emb, truth):
    for name, fn in [("D1_diar", lambda e: detect_pollution_diar(diar, e)[0]),
                     ("D2_cluster", lambda e: detect_pollution_cluster(get_emb, e)[0])]:
        tp=fp=fn_=0
        for enr, gt in truth:
            pred = fn(enr)
            if pred is None: continue
            if pred and gt: tp+=1
            elif pred and not gt: fp+=1
            elif not pred and gt: fn_+=1
        prec = tp/(tp+fp) if tp+fp else 0
        rec = tp/(tp+fn_) if tp+fn_ else 0
        f1 = 2*prec*rec/(prec+rec) if prec+rec else 0
        print(f"{name}: P={prec:.2f} R={rec:.2f} F1={f1:.2f} (tp={tp} fp={fp} fn={fn_})")
```

- [ ] **Step 3: 验证** 跑出 F1 表。Go: 至少一种 F1>0.7。都不行→记录 + 进 Q2 用表现较好者或都试
- [ ] **Step 4: Commit** `feat(poc): Q1 enrollment检测器F1(26条calibration)`

---

## Task 5: Q2 — 污染与漏拒相关性（474 neg）

**Files:** Modify `code/poc_enrollment_pollution.py`

- [ ] **Step 1: 跑 474 neg enrollment 检测 + 分组污染率**

```python
from scipy.stats import fisher_exact
def q2_correlation(diar, neg_enr_by_id, neg_sim_by_id, thr=0.27):
    polluted = {}
    for uid, enr in neg_enr_by_id.items():
        polluted[uid] = detect_pollution_diar(diar, enr)[0]  # 或 D2, 看 Q1 结果
    leak = [uid for uid, sim in neg_sim_by_id.items() if sim >= thr]  # 45 漏拒
    allneg = list(neg_sim_by_id.keys())
    def rate(uids): 
        ps = [u for u in uids if polluted.get(u) is True]
        ns = [u for u in uids if polluted.get(u) is False]
        return len(ps)/(len(ps)+len(ns)) if ps+ns else 0, len(ps), len(ns)
    rA, pA, nA = rate(leak); rB, pB, nB = rate(allneg)
    # Fisher exact: [[leak_polluted, leak_clean],[nonleak_polluted, nonleak_clean]]
    nonleak = [u for u in allneg if u not in set(leak)]
    _, nA2 = rate(leak); _, nB2 = rate(nonleak)
    table = [[pA, nA], [pB - pA if False else rate(nonleak)[1], rate(nonleak)[2]]]
    oddsr, p = fisher_exact([[pA, nA], [rate(nonleak)[1], rate(nonleak)[2]]])
    print(f"组A(45漏拒)污染率={rA:.2%}({pA}/{pA+nA}) vs 组B(全474)={rB:.2%}({pB}/{pB+nB}) 倍数={rA/max(rB,1e-9):.2f}x Fisher p={p:.4f}")
```

- [ ] **Step 2: 验证** 组A≥1.5×组B 且 p<0.05 → Go Q3；≤1.2× → No-Go 记录转方向B
- [ ] **Step 3: Commit** `feat(poc): Q2 enrollment污染与漏拒相关性(474neg)`

---

## Task 6: Q3 — 自适应 thr 的 RR/pos 代价（1364 pos hold-out）

**Files:** Modify `code/poc_enrollment_pollution.py`

- [ ] **Step 1: pos enrollment 检测 + 按 id 分组 hold-out**

```python
def q3_tradeoff(diar, pos_enr_by_id, pos_sim_by_id, pos_cer_by_uid, seed=42):
    pol = {uid: detect_pollution_diar(diar, enr)[0] for uid, enr in pos_enr_by_id.items()}
    rng = np.random.default_rng(seed)
    uids = sorted(pos_sim_by_id)
    rng.shuffle(uids); cut = int(len(uids)*0.8); val = set(uids[cut:])
    # pool CER 函数复用 recompute_qwen_official.py 的官方累计池口径(import 或复制)
    for thr_high in [0.30, 0.35, 0.40]:
        # val 上: 污染 enr 用 thr_high, 干净用 0.27; 算 ΔRR(neg侧, 复用Q2) + Δpos CER(含拒口径)
        ...
```

- [ ] **Step 2: 验证** 找 ΔRR>0 且 Δpos CER≤0.01 的 thr_high。注意 CER 口径必须用官方累计池（复用 recompute_qwen_official.py），含拒
- [ ] **Step 3: Commit** `feat(poc): Q3 自适应thr代价hold-out模拟(1364pos)`

---

## Task 7: 结果文档 + Go/No-Go + commit（含 spec）

**Files:** Create `docs/POC_A_enrollment_pollution_结果_2026-07-16.md`

- [ ] **Step 1: 写结果文档** 三子问题数据表 + Go/No-Go 判断 + 推荐 thr_high + 预期 RR 提升 pp + pos 代价（val）+ 后续路径（Go→落地 spec / No-Go→方向B）
- [ ] **Step 2: 整体 commit**（spec + poc 脚本 + 结果文档一起，用户确认身份后）：
  ```bash
  git add docs/superpowers/specs/2026-07-16-rr-improvement-poc-design.md \
          docs/superpowers/plans/2026-07-16-rr-poc-enrollment-pollution.md \
          code/poc_enrollment_pollution.py \
          docs/POC_A_enrollment_pollution_结果_2026-07-16.md
  git commit  # 身份: 待用户确认 Panda_Lorrain
  ```

---

## Self-Review

**Spec coverage**: spec §4 的 Q1/Q2/Q3 → Task 4/5/6 一一对应；§5 产物 → Task 7；§6 风险备案 → 散布在各 Task 的验证步（D1 不行用 D2、过短剔除、hold-out 防 leak）。
**Placeholder**: Task 1 的 `load_neg_sim` 和 Task 6 的 CER pool 标了"执行时确认字段/复用 recompute"——这是真实数据缺口（result.json 字段名未确认），执行 Task 1/6 时先打印样本确认，非占位偷懒。
**Type consistency**: `detect_pollution_diar` 返回 `(bool, int)`，Q1/Q2/Q3 都用 `[0]` 取 bool；`get_emb` 返回 np.ndarray（D2 用 np.stack）— 一致。
