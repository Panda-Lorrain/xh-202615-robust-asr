# 稳定性 / 鲁棒性测试 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用扰动矩阵（同种子×10 + batch/变种子/输入微扰/enrollment 四维扰动）量化 qwen 后端全量残余非确定，定位波动音频并归因 5 类根因，落地只做工程修复（R1/R2）+ 诊断归档（R3/R4/R5），不碰训练。

**Architecture:** 三个新脚本（`stability_test.py` 编排 + `perturb_audio.py` 微扰生成 + `analyze_stability.py` 分析）+ 一处改动（`enroll_infer.py` 加 `--asr-batch-size` 透传）。每遍 enroll_infer 独立产出 run-id JSON，分析器汇总→波动判定→根因决策树→报告+dashboard。

**Tech Stack:** Python 3.12（uv 管理）、PyTorch、Qwen3-ASR（.venv_qwen）、stdlib wave/json/subprocess、jiwer/editdistance（eval_metrics 复用）。

**Spec:** `docs/superpowers/specs/2026-07-19-stability-robustness-test-design.md`

**项目无 pytest 约定** — 本计划的"测试"步骤统一用 **dry-run 小样本（`--limit 5`）+ 断言 JSON 输出结构**（与 `verify_reproducibility.py` 风格一致），不写 pytest。

**Git 身份约定** — 所有 commit 用 `Panda_Lorrain` 身份（本机默认 user 是 `midea-overnight-loop`，按 memory `git-identity-mismatch`）：
```bash
git -c user.name="Panda_Lorrain" -c user.email="Panda_Lorrain@users.noreply.github.com" commit -m "..."
```

---

## File Structure

| 文件 | 责任 | 创建/改动 |
|---|---|---|
| `code/enroll_infer.py` | 加 `--asr-batch-size` 参数透传给 qwen/firered 子进程（当前 413-415 行未透传，qwen 固定 batch=16） | 改（Task 1） |
| `code/stability_test.py` | 编排器：按扰动矩阵逐遍跑 enroll_infer，断点续跑，每遍产 run-id JSON | 新（Task 2/5/6） |
| `code/perturb_audio.py` | B3 输入微扰：对 recognition 音频生成 gauss/vol/time 扰动版 + 新 pairs JSON | 新（Task 3） |
| `code/analyze_stability.py` | 分析器：汇总 run JSON → 波动判定 → 根因归因 → report/per_utt/dashboard | 新（Task 8/9） |
| `code/stability_matrix/` | 产物目录：`<run-id>.json` × 33 + `perturbed/` + `stability_report.json` + `per_utt_volatility.json` | 运行时生成 |
| `docs/稳定性测试报告_2026-07-19.md` | 人读报告（答辩弹药） | 新（Task 10） |

**关键复用**（不重写）：
- `eval_metrics.cer(hyp, ref)` — 逐条 CER（jiwer，与 verify_reproducibility 一致）
- `eval_datasetA._norm_zh(t)` — 繁→简归一
- `pos_pairs_datasetA.json` — 含 `ref` 字段（CER 标准答案），结构 `{id, enrollment, recognition, ref, kws_txt}`

---

## Task 1: enroll_infer 加 --asr-batch-size 透传（Phase 0 前置，阻塞 B1）

**Files:**
- Modify: `code/enroll_infer.py`（argparse 段 ~130 行 + qwen/firered 调用段 399-415 行）

- [ ] **Step 1: 加 argparse 参数**

在 `code/enroll_infer.py` 的 `--context` 参数（第 131-132 行）之后插入：

```python
    ap.add_argument("--asr-batch-size", type=int, default=16,
                    help="qwen/firered 后端 batch 推理大小(透传 qwen_asr_backend.py --batch-size; "
                         "0=逐条; 稳定性测试 B1 维度扫描用; 默认 16 与 qwen_asr_backend 一致)")
```

- [ ] **Step 2: 透传给 qwen/firered 子进程**

找到第 413-415 行的 subprocess 调用（当前内容）：
```python
        subprocess.check_call([_be_py, _be_script_full, "--slice-dir", args.save_target_audio,
                               "--out", uid2text_path, "--seed", str(args.seed),
                               "--context", args.context])
```
替换为（加 `--batch-size`）：
```python
        subprocess.check_call([_be_py, _be_script_full, "--slice-dir", args.save_target_audio,
                               "--out", uid2text_path, "--seed", str(args.seed),
                               "--context", args.context,
                               "--batch-size", str(args.asr_batch_size)])
```

- [ ] **Step 3: 验证参数被接受（不跑模型）**

Run: `code/.venv/Scripts/python.exe code/enroll_infer.py --help 2>&1 | grep asr-batch-size`
Expected: 输出含 `--asr-batch-size` 行

- [ ] **Step 4: 回归验证 — qwen batch=16 不破坏现有 delta=0**

Run（小样本 20 条 × 2 遍，复用 verify_reproducibility）：
```bash
source code/setenv.sh && export HF_HUB_OFFLINE=1
code/.venv/Scripts/python.exe code/verify_reproducibility.py \
  --pairs code/pos_pairs_datasetA.json --limit 20 --seed 42 --backend qwen
```
Expected: summary.json 的 `text_match_rate=1.0`、`cer_delta_max≤0.01`（与改造前一致，证明加透传参数不改变 batch=16 行为）

- [ ] **Step 5: Commit**

```bash
git add code/enroll_infer.py
git -c user.name="Panda_Lorrain" -c user.email="Panda_Lorrain@users.noreply.github.com" \
  commit -m "feat(stability): enroll_infer 加 --asr-batch-size 透传(解锁 B1 batch 扫描)"
```

---

## Task 2: stability_test.py 编排器骨架（A 阶段 + 断点续跑）

**Files:**
- Create: `code/stability_test.py`

- [ ] **Step 1: 写完整编排器（A 阶段部分）**

创建 `code/stability_test.py`：

```python
#!/usr/bin/env python
"""稳定性/鲁棒性测试编排器: 按扰动矩阵逐遍跑 enroll_infer, 每遍产 run-id JSON。

spec: docs/superpowers/specs/2026-07-19-stability-robustness-test-design.md
断点续跑: 已存在的 run-id JSON 自动跳过。

用法:
  source code/setenv.sh && export HF_HUB_OFFLINE=1
  code/.venv/Scripts/python.exe code/stability_test.py --phase A --runs 10            # 全量
  code/.venv/Scripts/python.exe code/stability_test.py --phase A --runs 2 --limit 5   # dry-run
  code/.venv/Scripts/python.exe code/stability_test.py --phase B1|B2|B3|B4|all
"""
import os, sys, json, time, subprocess, argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY_MAIN = os.environ.get("PY_MAIN") or os.path.join(_HERE, ".venv", "Scripts", "python.exe")
PAIRS = os.path.join(_HERE, "pos_pairs_datasetA.json")
OUT_DIR = os.path.join(_HERE, "stability_matrix")
SIM_THR = 0.27  # 主线提交 thr(memory unified-thr-decision), 决策翻盘维度需要


def _subset_pairs(run_id, pairs, limit):
    """limit>0 取前 N 条写临时 manifest(enroll_infer 无 --limit), 返回新 pairs 路径。"""
    if limit <= 0:
        return pairs
    rows = json.load(open(pairs, encoding="utf-8"))[:limit]
    tmp = os.path.join(OUT_DIR, f"_pairs_{run_id}.json")
    json.dump(rows, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
    return tmp


def run_once(run_id, pairs, seed, batch_size, enroll_augment=False, limit=0):
    """跑一遍 enroll_infer(qwen), 输出 stability_matrix/<run_id>.json。已存在则跳过。"""
    os.makedirs(OUT_DIR, exist_ok=True)
    out_json = os.path.join(OUT_DIR, f"{run_id}.json")
    if os.path.exists(out_json):
        print(f"[skip] {run_id} 已存在")
        return out_json
    pairs = _subset_pairs(run_id, pairs, limit)
    cmd = [_PY_MAIN, os.path.join(_HERE, "enroll_infer.py"),
           "--pairs", pairs, "--out-json", out_json,
           "--always-generate", "--reject-threshold", str(SIM_THR),
           "--asr-backend", "qwen", "--seed", str(seed),
           "--asr-batch-size", str(batch_size)]
    if enroll_augment:
        cmd += ["--enroll-augment"]
    print(f"[run] {run_id} (seed={seed}, batch={batch_size}, aug={enroll_augment})")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode != 0:
        raise RuntimeError(f"{run_id} 失败({r.returncode}): {r.stderr[-800:]}")
    print(f"      done {run_id} ({dt:.0f}s)")
    return out_json


def phase_A(runs, seed, batch, limit):
    for r in range(runs):
        run_once(f"A_s{seed}_r{r}", PAIRS, seed, batch, False, limit)


def main():
    ap = argparse.ArgumentParser(description="稳定性/鲁棒性测试编排器")
    ap.add_argument("--phase", required=True, choices=["A", "B1", "B2", "B3", "B4", "all"])
    ap.add_argument("--runs", type=int, default=10, help="A 阶段遍数(默认10)")
    ap.add_argument("--seed", type=int, default=42, help="A 阶段种子(默认42)")
    ap.add_argument("--batch", type=int, default=16, help="A 阶段 batch(默认16)")
    ap.add_argument("--limit", type=int, default=0, help="每遍只跑前 N 条(0=全量, dry-run 用 5)")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.phase == "A":
        phase_A(args.runs, args.seed, args.batch, args.limit)
    elif args.phase in ("B1", "B2", "B3", "B4", "all"):
        print(f"[TODO] phase {args.phase} 在 Task 5/6 实现")
    print("[done] stability_test phase", args.phase)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: dry-run 验证 A 阶段跑通（2 遍 × 5 条）**

Run:
```bash
source code/setenv.sh && export HF_HUB_OFFLINE=1
code/.venv/Scripts/python.exe code/stability_test.py --phase A --runs 2 --limit 5
```
Expected: 输出 `[run] A_s42_r0 ... done` + `A_s42_r1`，生成 `code/stability_matrix/A_s42_r0.json` 和 `A_s42_r1.json`

- [ ] **Step 3: 断言两个 JSON 存在且含 transcript/max_sim/rejected**

Run:
```bash
code/.venv/Scripts/python.exe -c "import json; r=json.load(open('code/stability_matrix/A_s42_r0.json')); print(len(r), 'keys:', sorted(r[0].keys())[:6])"
```
Expected: 输出条数 + keys 含 `transcript`、`max_sim`、`rejected`

- [ ] **Step 4: 验证断点续跑（重跑应 skip）**

Run: `code/.venv/Scripts/python.exe code/stability_test.py --phase A --runs 2 --limit 5`
Expected: 两个 run 都输出 `[skip] A_s42_r0 已存在` / `[skip] A_s42_r1 已存在`

- [ ] **Step 5: 清理 dry-run 产物 + Commit**

```bash
rm -rf code/stability_matrix
git add code/stability_test.py
git -c user.name="Panda_Lorrain" -c user.email="Panda_Lorrain@users.noreply.github.com" \
  commit -m "feat(stability): 编排器骨架 + A阶段同种子N遍 + 断点续跑"
```

---

## Task 3: perturb_audio.py（B3 输入微扰生成）

**Files:**
- Create: `code/perturb_audio.py`

- [ ] **Step 1: 写完整微扰脚本**

创建 `code/perturb_audio.py`：

```python
#!/usr/bin/env python
"""B3 输入微扰: 对 pos_pairs 的 recognition 音频生成扰动版(gauss/vol/time), 输出新 pairs JSON。

spec §5 B3。扰动音频缓存在 code/stability_matrix/perturbed/<perturb>/<uid>.wav。
分析时 stability_test.py --phase B3 调本脚本生成新 pairs 后跑 enroll_infer。

用法:
  code/.venv/Scripts/python.exe code/perturb_audio.py --perturb gauss
"""
import os, json, wave, argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
PAIRS = os.path.join(_HERE, "pos_pairs_datasetA.json")
OUT_BASE = os.path.join(_HERE, "stability_matrix", "perturbed")


def read_wav_mono(p):
    with wave.open(p, "rb") as w:
        n, sr, ch, sw = w.getnframes(), w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(n)
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def write_wav(p, x, sr):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    xi = (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(p, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(xi.tobytes())


def perturb_gauss(x, sr):   # 叠加 -45 dB 高斯噪(不可感知, 测数值边界)
    rng = np.random.default_rng(42)
    return x + rng.standard_normal(len(x)) * (10 ** (-45 / 20))

def perturb_vol(x, sr):     # +1 dB(测能量敏感)
    return x * (10 ** (1 / 20))

def perturb_time(x, sr):    # 前补 20ms 静音(测对齐敏感)
    shift = int(0.020 * sr)
    return np.concatenate([np.zeros(shift), x])[:len(x)]


PERTURBS = {"gauss": perturb_gauss, "vol": perturb_vol, "time": perturb_time}


def main():
    ap = argparse.ArgumentParser(description="B3 输入微扰音频生成")
    ap.add_argument("--perturb", required=True, choices=list(PERTURBS))
    ap.add_argument("--pairs", default=PAIRS)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条(0=全部, dry-run 用 5)")
    args = ap.parse_args()
    fn = PERTURBS[args.perturb]
    rows = json.load(open(args.pairs, encoding="utf-8"))
    if args.limit > 0:
        rows = rows[:args.limit]
    out_dir = os.path.join(OUT_BASE, args.perturb)
    new_rows = []
    for r in rows:
        uid = os.path.splitext(os.path.basename(r["recognition"]))[0]
        dst = os.path.join(out_dir, f"{uid}.wav")
        if not os.path.exists(dst):
            x, sr = read_wav_mono(r["recognition"])
            write_wav(dst, fn(x, sr), sr)
        new_rows.append({"id": r.get("id"), "enrollment": r["enrollment"],
                         "recognition": dst, "ref": r.get("ref", ""),
                         "kws_txt": r.get("kws_txt", "")})
    out_pairs = os.path.join(_HERE, "stability_matrix", f"_pairs_B3_{args.perturb}.json")
    os.makedirs(os.path.dirname(out_pairs), exist_ok=True)
    json.dump(new_rows, open(out_pairs, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"[perturb {args.perturb}] {len(new_rows)} 条 → {out_pairs}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: dry-run 验证 3 种扰动各生成 5 条**

Run:
```bash
code/.venv/Scripts/python.exe code/perturb_audio.py --perturb gauss --limit 5
code/.venv/Scripts/python.exe code/perturb_audio.py --perturb vol --limit 5
code/.venv/Scripts/python.exe code/perturb_audio.py --perturb time --limit 5
```
Expected: 每条输出 `[perturb gauss/vol/time] 5 条 → ...`，`code/stability_matrix/perturbed/<p>/cmd_*.wav` 各 5 个 + `_pairs_B3_<p>.json` 各 1 个

- [ ] **Step 3: 断言扰动音频可读且时长与原始一致**

Run:
```bash
code/.venv/Scripts/python.exe -c "
import wave, os
for p in ['gauss','vol','time']:
    w=wave.open(f'code/stability_matrix/perturbed/{p}/cmd_0.wav','rb')
    print(p, 'frames=', w.getnframes(), 'sr=', w.getframerate())
"
```
Expected: 3 行，每行 frames/sr 非零（time 的 frames 应略小于原始，因截断到原长）

- [ ] **Step 4: 清理 dry-run 产物 + Commit**

```bash
rm -rf code/stability_matrix
git add code/perturb_audio.py
git -c user.name="Panda_Lorrain" -c user.email="Panda_Lorrain@users.noreply.github.com" \
  commit -m "feat(stability): B3 输入微扰生成(gauss/vol/time)"
```

---

## Task 4: 跑 A 阶段全量×10（Phase 1 实运行）

**Files:** 无代码改动；产物 `code/stability_matrix/A_s42_r{0-9}.json`

- [ ] **Step 1: 启动 A 阶段全量 10 遍（~2.3h，后台跑无需守候）**

Run（前台或后台均可）:
```bash
source code/setenv.sh && export HF_HUB_OFFLINE=1
code/.venv/Scripts/python.exe code/stability_test.py --phase A --runs 10
```
Expected: 依次输出 `[run] A_s42_r0 ... done A_s42_r0 (xxxxs)` ... 到 `r9`，最终 `[done] stability_test phase A`

- [ ] **Step 2: 断言 10 个 JSON 都存在且条数 = 全量**

Run:
```bash
code/.venv/Scripts/python.exe -c "
import json, os, glob
fs=sorted(glob.glob('code/stability_matrix/A_s42_r*.json'))
print('文件数:', len(fs))
print('首文件条数:', len(json.load(open(fs[0]))))
"
```
Expected: `文件数: 10`，首文件条数 = pos_pairs 总条数（约 1362）

- [ ] **Step 3: 快速看 R1 严重度（10 遍两两 transcript 一致率）**

Run:
```bash
code/.venv/Scripts/python.exe -c "
import json, glob
from eval_metrics import cer
from eval_datasetA import _norm_zh
fs=sorted(glob.glob('code/stability_matrix/A_s42_r*.json'))
runs=[{r['recognition']: r.get('transcript','') for r in json.load(open(f))} for f in fs]
uids=list(runs[0].keys())
flip=sum(1 for u in uids if len(set(runs[i].get(u,'') for i in range(len(runs))))>1)
print(f'A 阶段 {len(uids)} 条, 10遍内文本有波动的: {flip} ({flip/len(uids):.2%})')
"
```
Expected: 输出波动条数+占比。若 `flip=0` → R1 不存在（系统确定）；`flip>0` → R1 触发，记录数字供 Task 11。

> **注意**：A 阶段产物 JSON 不 commit（体积大，加入 .gitignore；Task 8 产出的 report 才 commit）。

- [ ] **Step 4: 把 stability_matrix 加入 .gitignore（若未加）**

```bash
grep -q "stability_matrix" code/.gitignore 2>/dev/null || echo "stability_matrix/" >> code/.gitignore
git add code/.gitignore
git -c user.name="Panda_Lorrain" -c user.email="Panda_Lorrain@users.noreply.github.com" \
  commit -m "chore(stability): gitignore stability_matrix 产物目录"
```

---

## Task 5: stability_test.py 加 phase_B1/B2/B4（Phase 2）

**Files:**
- Modify: `code/stability_test.py`

- [ ] **Step 1: 替换 main() 的占位分支 + 加 3 个 phase 函数**

把 `code/stability_test.py` 中 `phase_A` 函数之后、`main` 之前，插入：

```python
def phase_B1(limit):
    # batch=1/8/16/32 × 2 遍, seed=42, 原始音频
    for b in [1, 8, 16, 32]:
        for r in range(2):
            run_once(f"B1_b{b}_r{r}", PAIRS, 42, b, False, limit)


def phase_B2(limit):
    # 变种子 42/100/200/314/555 × 2 遍, batch=16, 原始音频
    for s in [42, 100, 200, 314, 555]:
        for r in range(2):
            run_once(f"B2_s{s}_r{r}", PAIRS, s, 16, False, limit)


def phase_B4(limit):
    # --enroll-augment on/off, seed=42, batch=16
    run_once("B4_augoff", PAIRS, 42, 16, False, limit)
    run_once("B4_auguon", PAIRS, 42, 16, True, limit)
```

把 `main()` 中的占位分支替换为：

```python
    elif args.phase == "B1":
        phase_B1(args.limit)
    elif args.phase == "B2":
        phase_B2(args.limit)
    elif args.phase == "B4":
        phase_B4(args.limit)
    elif args.phase in ("B3", "all"):
        print(f"[TODO] phase {args.phase} 在 Task 6 实现")
```

- [ ] **Step 2: dry-run 验证 B1/B2/B4 各跑 1 档**

Run:
```bash
source code/setenv.sh && export HF_HUB_OFFLINE=1
code/.venv/Scripts/python.exe code/stability_test.py --phase B1 --limit 5   # 跑 B1_b1_r0(其余已起 8 遍, dry-run 只验证能起)
code/.venv/Scripts/python.exe code/stability_test.py --phase B4 --limit 5
```
Expected: B1 产出 `B1_b1_r0.json` 等（断点续跑，dry-run 后续全量跑会续上）；B4 产出 `B4_augoff.json` + `B4_auguon.json`

> B1 全量会跑 8 遍（batch×2），dry-run `--limit 5` 时 8 遍都跑 5 条，确认能起即可，不必等全。

- [ ] **Step 3: 清理 dry-run 产物 + Commit**

```bash
rm -rf code/stability_matrix
git add code/stability_test.py
git -c user.name="Panda_Lorrain" -c user.email="Panda_Lorrain@users.noreply.github.com" \
  commit -m "feat(stability): 编排器加 B1 batch扫描/B2 变种子/B4 enrollment扰动"
```

---

## Task 6: stability_test.py 加 phase_B3 + all（Phase 2）

**Files:**
- Modify: `code/stability_test.py`

- [ ] **Step 1: 加 phase_B3 + 调 perturb_audio**

在 `phase_B4` 函数后插入：

```python
def phase_B3(limit):
    # 输入微扰: 先 perturb_audio 生成 3 种扰动 pairs, 再各跑 1 遍
    for p in ["gauss", "vol", "time"]:
        perturbed_pairs = os.path.join(OUT_DIR, f"_pairs_B3_{p}.json")
        if not os.path.exists(perturbed_pairs):
            subprocess.run([_PY_MAIN, os.path.join(_HERE, "perturb_audio.py"),
                            "--perturb", p] + (["--limit", str(limit)] if limit > 0 else []),
                           check=True, cwd=_HERE)
        run_once(f"B3_p{p}", perturbed_pairs, 42, 16, False, limit)
```

把 `main()` 中 `B3`/`all` 占位分支替换为：

```python
    elif args.phase == "B3":
        phase_B3(args.limit)
    elif args.phase == "all":
        phase_A(args.runs, args.seed, args.batch, args.limit)
        phase_B1(args.limit); phase_B2(args.limit)
        phase_B3(args.limit); phase_B4(args.limit)
```

- [ ] **Step 2: dry-run 验证 B3（5 条）**

Run:
```bash
source code/setenv.sh && export HF_HUB_OFFLINE=1
code/.venv/Scripts/python.exe code/stability_test.py --phase B3 --limit 5
```
Expected: 先调 perturb_audio 生成 3 个 `_pairs_B3_*.json`，再产出 `B3_pgauss.json` / `B3_pvol.json` / `B3_ptime.json`

- [ ] **Step 3: 断言 B3 JSON 的 recognition 指向扰动音频（非原始）**

Run:
```bash
code/.venv/Scripts/python.exe -c "
import json
r=json.load(open('code/stability_matrix/B3_pgauss.json'))
print(r[0]['recognition'])
"
```
Expected: 路径含 `stability_matrix/perturbed/gauss/`（确认跑的是扰动音频）

- [ ] **Step 4: 清理 dry-run 产物 + Commit**

```bash
rm -rf code/stability_matrix
git add code/stability_test.py
git -c user.name="Panda_Lorrain" -c user.email="Panda_Lorrain@users.noreply.github.com" \
  commit -m "feat(stability): 编排器加 B3 输入微扰 + all 一键全跑"
```

---

## Task 7: 跑 B 阶段全量四维（Phase 2 实运行）

**Files:** 无代码改动；产物 `code/stability_matrix/B{1,2,3,4}_*.json`

- [ ] **Step 1: 启动 B 阶段全量（~3.7h，后台跑无需守候）**

Run:
```bash
source code/setenv.sh && export HF_HUB_OFFLINE=1
code/.venv/Scripts/python.exe code/stability_test.py --phase all --runs 10
```
（A 阶段 10 遍已存在会 skip；B1/B2/B3/B4 全量续跑）

Expected: B1 产出 8 个、B2 产出 10 个、B3 产出 3 个、B4 产出 2 个 JSON

- [ ] **Step 2: 断言全部 33 个 run JSON 就位**

Run:
```bash
code/.venv/Scripts/python.exe -c "
import glob
fs=glob.glob('code/stability_matrix/A_*.json')+glob.glob('code/stability_matrix/B*.json')
fs=[f for f in fs if '_pairs_' not in f]
print('run JSON 总数:', len(fs))
"
```
Expected: `run JSON 总数: 33`（A×10 + B1×8 + B2×10 + B3×3 + B4×2）

> 产物不 commit（已在 .gitignore）。

---

## Task 8: analyze_stability.py（汇总 + 波动判定 + 根因归因）

**Files:**
- Create: `code/analyze_stability.py`

- [ ] **Step 1: 写完整分析器**

创建 `code/analyze_stability.py`：

```python
#!/usr/bin/env python
"""稳定性/鲁棒性测试分析器: 汇总所有 run JSON → 波动判定 → 根因归因 → report + per_utt。

spec §6/§8。用法:
  code/.venv/Scripts/python.exe code/analyze_stability.py
产物: code/stability_matrix/stability_report.json + per_utt_volatility.json
"""
import os, sys, json, glob, argparse
from collections import defaultdict, Counter
_HERE = os.path.dirname(os.path.abspath(__file__))
MATRIX = os.path.join(_HERE, "stability_matrix")
sys.path.insert(0, _HERE)
from eval_metrics import cer
from eval_datasetA import _norm_zh

SIM_THR = 0.27


def _uid(rec):
    return os.path.splitext(os.path.basename(rec))[0]


def load_runs():
    """返回 {run_id: {uid: result_dict}}。只加载 A_/B1_/B2_/B3_/B4_ 开头的 result JSON。"""
    runs = {}
    for f in sorted(glob.glob(os.path.join(MATRIX, "A_*.json")) + glob.glob(os.path.join(MATRIX, "B[1234]_*.json"))):
        rid = os.path.splitext(os.path.basename(f))[0]
        try:
            rows = json.load(open(f, encoding="utf-8"))
            runs[rid] = {_uid(r["recognition"]): r for r in rows}
        except Exception as e:
            print(f"[warn] 跳过 {rid}: {e}")
    return runs


def load_refs():
    """从 pos_pairs 取 ref: {uid: ref_text}。"""
    rows = json.load(open(os.path.join(_HERE, "pos_pairs_datasetA.json"), encoding="utf-8"))
    return {_uid(r["recognition"]): r.get("ref", "") for r in rows}


def group_runs(runs):
    """按维度分组: {dim: [run_id,...]}。dim ∈ A/B1/B2/B3/B4。"""
    g = defaultdict(list)
    for rid in runs:
        dim = rid.split("_")[0]
        g[dim].append(rid)
    return g


def dim_volatility(uid, runs, dim_ids, refs):
    """该 uid 在某维度各 run 的 {transcripts: Counter, cers: list, decisions: Counter, flipped: bool}。"""
    transcripts, cers, decisions = Counter(), [], Counter()
    for rid in dim_ids:
        r = runs[rid].get(uid)
        if r is None:
            continue
        t = r.get("transcript", "") or ""
        transcripts[t] += 1
        rejected = r.get("rejected", False)
        decisions["reject" if rejected else "accept"] += 1
        ref = refs.get(uid, "")
        c = 1.0 if (rejected or not t) else cer(_norm_zh(t), _norm_zh(ref))
        cers.append(c)
    flipped = len(decisions) > 1  # accept/reject 都出现 = 翻盘
    return {"transcripts": dict(transcripts), "cers": cers,
            "decisions": dict(decisions), "flipped": flipped}


def classify_root(uid, runs, groups, refs):
    """根因归因决策树(spec §6)。返回 root_causes 列表 + fix_action。"""
    causes, fix = [], []
    a = dim_volatility(uid, runs, groups.get("A", []), refs)
    a_unstable = len(a["transcripts"]) > 1 or a["flipped"]
    if a_unstable:
        causes.append("R1_gpu_nondeterminism")
        fix.append("use_deterministic_algorithms")
    b1 = dim_volatility(uid, runs, groups.get("B1", []), refs)
    if not a_unstable and (len(b1["transcripts"]) > 1 or b1["flipped"]):
        causes.append("R2_batch_padding")
        fix.append("submit_lock_batch1")
    b2 = dim_volatility(uid, runs, groups.get("B2", []), refs)
    if not a_unstable and (len(b2["transcripts"]) > 1 or b2["flipped"]):
        causes.append("R5_numeric_boundary")
        fix.append("archive_holdout_reject")
    b3 = dim_volatility(uid, runs, groups.get("B3", []), refs)
    if len(b3["transcripts"]) > 1 or b3["flipped"]:
        causes.append("R3_input_generalization")
        fix.append("archive_external_training")
    b4 = dim_volatility(uid, runs, groups.get("B4", []), refs)
    if len(b4["transcripts"]) > 1 or b4["flipped"]:
        causes.append("R4_voice_locking")
        fix.append("archive_enroll_augment_holdout")
    return causes, fix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只分析前 N 条(0=全部, dry-run 用 5)")
    args = ap.parse_args()
    runs = load_runs()
    refs = load_refs()
    groups = group_runs(runs)
    print(f"[load] {len(runs)} runs, dims: { {k: len(v) for k,v in groups.items()} }")

    all_uids = set()
    for rid in runs:
        all_uids.update(runs[rid].keys())
    all_uids = sorted(all_uids)
    if args.limit > 0:
        all_uids = all_uids[:args.limit]

    per_utt, dim_stats = {}, {}
    # 每维度整体统计
    for dim, ids in groups.items():
        flip_n, total = 0, 0
        for uid in all_uids:
            v = dim_volatility(uid, runs, ids, refs)
            if v["cers"]:
                total += 1
                if len(v["transcripts"]) > 1 or v["flipped"]:
                    flip_n += 1
        dim_stats[dim] = {"n_runs": len(ids), "volatile_rate": round(flip_n / total, 4) if total else 0,
                          "volatile_n": flip_n, "total_n": total}

    for uid in all_uids:
        causes, fix = classify_root(uid, runs, groups, refs)
        if not causes:
            continue  # 全维度稳定, 不入波动清单
        v_all = dim_volatility(uid, runs, sum(groups.values(), [])[:len(runs)], refs)
        all_cers = []
        for dim in groups:
            all_cers += dim_volatility(uid, runs, groups[dim], refs)["cers"]
        max_sim = max((runs[rid].get(uid, {}).get("max_sim", 0) for rid in runs), default=0)
        import statistics
        per_utt[uid] = {
            "ref": refs.get(uid, ""),
            "max_sim": round(max_sim, 4),
            "n_runs": len(runs),
            "n_distinct_transcripts": len(v_all["transcripts"]),
            "top_transcripts": dict(sorted(v_all["transcripts"].items(), key=lambda x: -x[1])[:3]),
            "cer_mean": round(statistics.mean(all_cers), 4) if all_cers else None,
            "cer_std": round(statistics.pstdev(all_cers), 4) if len(all_cers) > 1 else 0.0,
            "cer_max": round(max(all_cers), 4) if all_cers else None,
            "root_causes": causes,
            "fix_action": fix,
        }

    report = {
        "n_runs_total": len(runs),
        "dim_stats": dim_stats,
        "n_volatile_utts": len(per_utt),
        "root_cause_distribution": dict(Counter(c for u in per_utt for c in per_utt[u]["root_causes"])),
    }
    out_report = os.path.join(MATRIX, "stability_report.json")
    out_per = os.path.join(MATRIX, "per_utt_volatility.json")
    json.dump(report, open(out_report, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(per_utt, open(out_per, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[done] report → {out_report}")
    print(f"[done] per_utt → {out_per} ({len(per_utt)} 条波动)")
    print("\n=== 根因分布 ===")
    for c, n in report["root_cause_distribution"].items():
        print(f"  {c}: {n}")
    print("\n=== 各维度波动率 ===")
    for dim, s in dim_stats.items():
        print(f"  {dim}: {s['volatile_rate']:.2%} ({s['volatile_n']}/{s['total_n']})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: dry-run 验证分析器（前 5 条，需先有 dry-run 产物）**

先造少量 dry-run 产物：
```bash
source code/setenv.sh && export HF_HUB_OFFLINE=1
code/.venv/Scripts/python.exe code/stability_test.py --phase all --runs 2 --limit 5
```
再跑分析：
```bash
code/.venv/Scripts/python.exe code/analyze_stability.py --limit 5
```
Expected: 输出 `[load] N runs` + 根因分布 + 各维度波动率，生成 `stability_report.json` + `per_utt_volatility.json`

- [ ] **Step 3: 断言 report 含必需字段**

Run:
```bash
code/.venv/Scripts/python.exe -c "
import json
r=json.load(open('code/stability_matrix/stability_report.json'))
print('keys:', sorted(r.keys()))
print('root_cause_distribution:', r['root_cause_distribution'])
"
```
Expected: keys 含 `n_runs_total/dim_stats/n_volatile_utts/root_cause_distribution`

- [ ] **Step 4: 清理 dry-run 产物 + Commit（只 commit 脚本，不 commit 产物）**

```bash
rm -rf code/stability_matrix
git add code/analyze_stability.py
git -c user.name="Panda_Lorrain" -c user.email="Panda_Lorrain@users.noreply.github.com" \
  commit -m "feat(stability): 分析器(波动判定+5类根因归因决策树)"
```

---

## Task 9: stability_dashboard.html 可视化

**Files:**
- Create: `code/stability_dashboard.py`（生成 HTML，用 dataviz 规范的内联 SVG/Chart）

- [ ] **Step 1: 写 dashboard 生成脚本**

创建 `code/stability_dashboard.py`：

```python
#!/usr/bin/env python
"""稳定性测试 dashboard 生成器: 读 stability_report.json + per_utt_volatility.json → 单文件 HTML。

spec §8。内联 Chart.js CDN, 4 张图: 各维度波动率/CER delta 分布/波动音频 sim 分桶/根因堆叠。
用法: code/.venv/Scripts/python.exe code/stability_dashboard.py
"""
import os, json
_HERE = os.path.dirname(os.path.abspath(__file__))
MATRIX = os.path.join(_HERE, "stability_matrix")


def main():
    report = json.load(open(os.path.join(MATRIX, "stability_report.json"), encoding="utf-8"))
    per_utt = json.load(open(os.path.join(MATRIX, "per_utt_volatility.json"), encoding="utf-8"))

    dims = sorted(report["dim_stats"].keys())
    dim_rates = [report["dim_stats"][d]["volatile_rate"] * 100 for d in dims]
    root_dist = report["root_cause_distribution"]
    # sim 分桶
    buckets = {"<0.2": 0, "[0.2,0.3)": 0, "[0.3,0.4)": 0, ">=0.4": 0}
    for u in per_utt.values():
        s = u.get("max_sim", 0)
        if s < 0.2: buckets["<0.2"] += 1
        elif s < 0.3: buckets["[0.2,0.3)"] += 1
        elif s < 0.4: buckets["[0.3,0.4)"] += 1
        else: buckets[">=0.4"] += 1

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>稳定性测试 Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>body{{font-family:system-ui;margin:24px;background:#fafafa}}
h1{{font-size:20px}} .grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
.card{{background:#fff;padding:16px;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,.1)}}</style>
</head><body><h1>稳定性/鲁棒性测试 Dashboard</h1>
<p>总 run 数: {report['n_runs_total']} | 波动音频: {report['n_volatile_utts']} 条</p>
<div class="grid">
<div class="card"><h3>各维度波动率(%)</h3><canvas id="dim"></canvas></div>
<div class="card"><h3>根因分布</h3><canvas id="root"></canvas></div>
<div class="card"><h3>波动音频 sim 分桶</h3><canvas id="sim"></canvas></div>
<div class="card"><h3>波动音频 CER std 分布</h3><canvas id="std"></canvas></div>
</div>
<script>
new Chart(document.getElementById('dim'),{{type:'bar',data:{{labels:{dims!r},datasets:[{{data:{dim_rates!r},backgroundColor:'#4f46e5'}}]}},options:{{plugins:{{legend:{{display:false}}}},scales:{{y:{{beginAtZero:true}}}}}}}});
new Chart(document.getElementById('root'),{{type:'doughnut',data:{{labels:{list(root_dist.keys())!r},datasets:[{{data:{list(root_dist.values())!r}}]}}}});
new Chart(document.getElementById('sim'),{{type:'bar',data:{{labels:{list(buckets.keys())!r},datasets:[{{data:{list(buckets.values())!r},backgroundColor:'#16a34a'}}]}},options:{{plugins:{{legend:{{display:false}}}}}}}});
new Chart(document.getElementById('std'),{{type:'bar',data:{{labels:{[u['ref'][:12] for u in list(per_utt.values())[:20]]!r},datasets:[{{data:{[u.get('cer_std',0) for u in list(per_utt.values())[:20]]!r},backgroundColor:'#dc2626'}}]}},options:{{plugins:{{legend:{{display:false}}}}}}}});
</script></body></html>"""
    out = os.path.join(MATRIX, "stability_dashboard.html")
    open(out, "w", encoding="utf-8").write(html)
    print(f"[done] dashboard → {out}")


if __name__ == "__main__":
    main()
```

> **dataviz 规范**：调色板用品牌中性（indigo/green/red），暗色背景 #fafafa，单文件无外部依赖（除 Chart.js CDN），4 图网格布局。若需更严格遵循 dataviz skill，可在执行时 invoke 它核对调色板。

- [ ] **Step 2: dry-run 验证（需先有 dry-run 产物）**

Run:
```bash
source code/setenv.sh && export HF_HUB_OFFLINE=1
code/.venv/Scripts/python.exe code/stability_test.py --phase all --runs 2 --limit 5
code/.venv/Scripts/python.exe code/analyze_stability.py --limit 5
code/.venv/Scripts/python.exe code/stability_dashboard.py
```
Expected: 输出 `[done] dashboard → .../stability_dashboard.html`

- [ ] **Step 3: 断言 HTML 含 4 个 canvas**

Run: `grep -c "canvas id" code/stability_matrix/stability_dashboard.html`
Expected: `4`

- [ ] **Step 4: 清理 dry-run 产物 + Commit**

```bash
rm -rf code/stability_matrix
git add code/stability_dashboard.py
git -c user.name="Panda_Lorrain" -c user.email="Panda_Lorrain@users.noreply.github.com" \
  commit -m "feat(stability): dashboard 生成器(4图:维度波动率/根因/sim分桶/CER std)"
```

---

## Task 10: 稳定性测试报告（人读，答辩弹药）

**Files:**
- Create: `docs/稳定性测试报告_2026-07-19.md`

> **前置**：此任务需 A+B 全量产物（Task 4/7 跑完 + Task 8 分析完）才能填实数字。若执行时尚未跑全量，先建模板留 `[待填]`，跑完后回填。

- [ ] **Step 1: 写报告（读 report.json 填数字）**

Run 先取数字：
```bash
code/.venv/Scripts/python.exe -c "
import json
r=json.load(open('code/stability_matrix/stability_report.json'))
print('dim_stats:', json.dumps(r['dim_stats'], ensure_ascii=False, indent=2))
print('root_dist:', r['root_cause_distribution'])
print('n_volatile:', r['n_volatile_utts'])
"
```

创建 `docs/稳定性测试报告_2026-07-19.md`（用上面数字填）：

```markdown
# 稳定性 / 鲁棒性测试报告（2026-07-19）

## 1. 背景与方法
- 现有 run-twice 仅 20 条 × 2 遍抽样，A 集从未全量多遍跑过
- 扰动矩阵 33 遍：A 同种子×10 + B1 batch×8 + B2 变种子×10 + B3 输入微扰×3 + B4 enrollment×2
- 5 类根因归因：R1 GPU非确定 / R2 batch padding / R3 输入泛化 / R4 声纹锁定 / R5 数值边界
- hold-out 硬边界：本次不调任何基于 A 集内容的提交规则（A 集是测试集，不训练）

## 2. 各维度波动率
[填 dim_stats 表格]

## 3. 根因分布
[填 root_cause_distribution]

## 4. 关键发现
- R1（GPU 非确定）[有/无]：[若触发] → 已修 use_deterministic_algorithms（见 Task 11）
- R2（batch padding）[显著/不显著]：[若显著] → 提交锁 batch=1（见 Task 12）
- 死区/低 sim 桶波动占比：[填 sim 分桶数字] —— [验证/推翻]「死区更易波动」假设

## 5. 修复落地（本次只工程修复）
- R1 → [已修/无需修]
- R2 → [已锁 batch=1/无需]
- R3/R4/R5 → 诊断归档，未来 hold-out 拒识 / A 集外训练输入（本次不碰）

## 6. 答辩弹药
- 诚实归因：量化了系统残余非确定，定位 N 条波动音频并归因
- 工程严谨：发现并修复 [N] 处工程缺陷，提交数字可复现性达标
```

- [ ] **Step 2: Commit 报告**

```bash
git add docs/稳定性测试报告_2026-07-19.md
git -c user.name="Panda_Lorrain" -c user.email="Panda_Lorrain@users.noreply.github.com" \
  commit -m "docs(stability): 稳定性测试报告(各维度波动率+根因分布+修复落地)"
```

---

## Task 11（条件）: R1 触发 → 修 use_deterministic_algorithms

> **执行条件**：Task 4 Step 3 的 A 阶段波动数 > 0，或 report 中 `dim_stats.A.volatile_n > 0`。若 A 阶段 0 波动，**跳过本任务**。

**Files:**
- Modify: `code/repro.py`（`set_global_seed` 加确定性开关）

- [ ] **Step 1: 在 repro.py 加确定性环境变量开关**

找到 `code/repro.py` 第 39-47 行的 torch try 块，在 `torch.backends.cudnn.benchmark = False` 之后加：

```python
        # 稳定性测试 R1 修复: 消除 GPU matmul/attention 非确定(原子加)。
        # 设 env STAB_DETERMINISTIC=1 开启(提交链路开); 不设则保持原行为(效率优先)。
        if os.environ.get("STAB_DETERMINISTIC") == "1":
            torch.use_deterministic_algorithms(True, warn_only=True)
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
```

> `warn_only=True` 避免 Qwen3-ASR 某些无确定实现的 op 直接报错（降级为警告）。qwen_asr_backend.py / firered_asr_backend.py 是独立 venv 内联 set_seed，**不在 repro.py 覆盖范围**——若 R1 主源在 qwen 后端，需同步在 qwen_asr_backend.py 第 42 行后加同样两行（执行时判断）。

- [ ] **Step 2: 验证 — A 阶段 10 遍在 STAB_DETERMINISTIC=1 下波动归零**

Run:
```bash
source code/setenv.sh && export HF_HUB_OFFLINE=1 && export STAB_DETERMINISTIC=1
# 删旧 A 阶段产物重跑(或用新 out-dir; 这里删重跑验证)
rm -f code/stability_matrix/A_s42_r*.json
code/.venv/Scripts/python.exe code/stability_test.py --phase A --runs 10 --limit 50
code/.venv/Scripts/python.exe -c "
import json, glob
fs=sorted(glob.glob('code/stability_matrix/A_s42_r*.json'))
runs=[{r['recognition']: r.get('transcript','') for r in json.load(open(f))} for f in fs]
uids=list(runs[0].keys())
flip=sum(1 for u in uids if len(set(runs[i].get(u,'') for i in range(len(runs))))>1)
print(f'STAB_DETERMINISTIC=1 下 A(50条) 波动: {flip}/{len(uids)}')
"
```
Expected: 波动数下降（理想归零；若 qwen 子进程未覆盖则可能残留，记录数字 + 决定是否同步改 qwen_asr_backend.py）

- [ ] **Step 3: 重测 RTF（确认效率退化在可接受范围）**

Run（单遍对比）:
```bash
code/.venv/Scripts/python.exe code/stability_test.py --phase A --runs 1 --limit 50  # 无 STAB
STAB_DETERMINISTIC=1 code/.venv/Scripts/python.exe code/stability_test.py --phase A --runs 1 --limit 50  # 有 STAB
```
对比两次 `done A_s42_r0 (xxxxs)` 耗时。Expected: 若退化 >20% → 只在提交链路 setenv 开、验证链路关（在报告记录）。

- [ ] **Step 4: Commit**

```bash
git add code/repro.py
git -c user.name="Panda_Lorrain" -c user.email="Panda_Lorrain@users.noreply.github.com" \
  commit -m "fix(stability): R1 修复 use_deterministic_algorithms(STAB_DETERMINISTIC=1 开启)"
```

---

## Task 12（条件）: R2 触发 → 锁 batch=1 文档勘误

> **执行条件**：report 中 `dim_stats.B1.volatile_rate > 0.05`（batch 扫描波动显著）。若 B1 不显著，**跳过本任务**。

**Files:**
- Modify: `code/run_baodi.sh`（若提交脚本默认 batch≠1）+ CLAUDE.md runbook 段

- [ ] **Step 1: 确认提交链路 batch 配置**

Run: `grep -n "batch" code/run_baodi.sh code/submit_infer.py | head`
确认提交时 enroll_infer 的 batch 默认值（submit_infer 调 enroll_infer 未传 batch-size，默认 16）。

- [ ] **Step 2: 若提交 batch≠1 且 B1 显著，锁 batch=1**

在 `code/submit_infer.py` 的 `run_enroll_infer_pairs`（第 157-175 行）命令构造里，若需锁 batch=1，加：
```python
    if asr_backend in ("qwen", "firered"):
        cmd += ["--asr-batch-size", "1"]  # R2 修复: 提交锁 batch=1(稳定性测试 B1 证 batch 敏感)
```

> ⚠️ 这会拖慢提交效率（batch=1 比 16 慢 ~5×），需权衡效率腿。若效率退化不可接受，**不改代码**，仅在报告 + CLAUDE.md 记录「batch=1 与 batch=16 CER 差异 N，提交取舍」。

- [ ] **Step 3: 验证 batch=1 vs 16 的 CER 差异（坐实 R2）**

Run:
```bash
code/.venv/Scripts/python.exe -c "
import json, glob
from eval_metrics import cer
from eval_datasetA import _norm_zh
b1={r['recognition']: r.get('transcript','') for r in json.load(open('code/stability_matrix/B1_b1_r0.json'))}
b16={r['recognition']: r.get('transcript','') for r in json.load(open('code/stability_matrix/B1_b16_r0.json'))}
diff=sum(1 for k in b1 if b1[k]!=b16[k])
print(f'batch=1 vs 16 文本差异: {diff}/{len(b1)}')
"
```
Expected: 输出差异条数（坐实 R2 严重度，决定改代码还是仅记录）

- [ ] **Step 4: Commit（若有代码改动）**

```bash
git add code/submit_infer.py code/run_baodi.sh
git -c user.name="Panda_Lorrain" -c user.email="Panda_Lorrain@users.noreply.github.com" \
  commit -m "fix(stability): R2 提交锁 batch=1(B1 证 batch padding 敏感)" 2>/dev/null || \
  echo "无代码改动(仅记录到报告), 跳过 commit"
```

---

## Self-Review（写计划后自查）

**1. Spec 覆盖**：
- §3 现状核查（use_deterministic 缺口）→ Task 11 修 ✅
- §5 扰动矩阵 A/B1/B2/B3/B4 → Task 2/5/6 + 实跑 Task 4/7 ✅
- §6 波动判定 + 5 根因决策树 → Task 8 `classify_root` ✅
- §7 hold-out 硬边界 → 计划 header + Task 10 报告声明 ✅
- §8 产物（report/per_utt/dashboard/报告）→ Task 8/9/10 ✅
- §9 Phase 0-4 → Task 1(Phase0)/2-4(P0-1)/5-7(P2)/8-10(P3)/11-12(P4) ✅
- §11 决策（不碰训练）→ 全计划无训练步骤 ✅

**2. Placeholder 扫描**：Task 10 报告有 `[待填]`（依赖实跑数字），已在任务前置说明，属合理的运行后回填，非计划缺陷。其余无 TBD/TODO 占位。✅

**3. 类型/命名一致性**：run-id 命名（`A_s42_r0` / `B1_b1_r0` / `B2_s100_r0` / `B3_pgauss` / `B4_augoff`）在 Task 2/5/6/8 一致；`classify_root` 返回的根因 key（`R1_gpu_nondeterminism` 等）在 Task 8/11 一致；`run_once` 签名在 Task 2 定义、Task 5/6 调用一致。✅

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-19-stability-robustness-test.md`. Two execution options:

1. **Subagent-Driven（推荐）** — 每个 Task 派 fresh subagent，Task 间 review，快速迭代
2. **Inline Execution** — 本 session 内 executing-plans 批量执行 + checkpoint review

> ⚠️ 注意：Task 4/7 是实运行（A 阶段 ~2.3h + B 阶段 ~3.7h，合计 ~6h），建议后台跑。Task 11/12 是条件任务，依赖 Task 4/7 结果判定是否执行。
