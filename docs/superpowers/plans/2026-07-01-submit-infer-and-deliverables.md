# 推理脚本标准化 + 交付文档骨架 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: 用 superpowers:subagent-driven-development(推荐)或 superpowers:executing-plans 按 task 实现。步骤用 `- [ ]` 跟踪。

**Goal:** 建标准化纯推理脚本 `code/submit_infer.py`(吃 enrollment+recognition → result.json+timing.json) + `交付/` 三份文档骨架 + README 修正。

**Architecture:** submit_infer.py 是仅 stdlib 的 subprocess 顶层编排器,零侵入复用 4 个已验证组件(noise_classify/se_denoise/enroll_infer/llm_reject),按 SE条件化分桶解逐条 atten-lim,llm_or_sim 融合拒识,分阶段测端到端 timing。详见 spec `docs/superpowers/specs/2026-07-01-submit-infer-and-deliverables-design.md`。

**Tech Stack:** Python 3.12 stdlib only(subprocess/json/argparse/wave);复用组件在 code/.venv、code/.venv_se、.venv_llm 三个 venv。

**TDD 策略:** 纯逻辑函数走 TDD(内联 assert,`code/.venv/Scripts/python.exe tests/test_xxx.py`,无 pytest 依赖);subprocess 编排用集成验收(Task 7,--limit 5 跑通 + schema 校验)。

---

## 文件结构

| 文件 | 责任 | 动作 |
|---|---|---|
| `code/submit_infer.py` | 编排器,仅 stdlib | 新建 |
| `tests/__init__.py` | 测试包标记(空) | 新建 |
| `tests/test_submit_infer_logic.py` | 纯函数 TDD(utt_id/decide_reject/bucket/schema) | 新建 |
| `交付/设计报告.md` | 实际实现版技术方案 | 新建骨架 |
| `交付/使用说明.md` | submit_infer 用法 + 环境 + 权重 | 新建骨架 |
| `交付/测试验证方案.md` | 评测方法 + 仿真结果 + A集流程 | 新建骨架 |
| `README.md` | 架构图/栈表理想态→实际 | 微调 |

**全局常量**(submit_infer.py 顶部,后续 task 引用):
```python
HERE = os.path.dirname(os.path.abspath(__file__))      # code/
ROOT = os.path.dirname(HERE)                            # 项目根
PY_MAIN = os.path.join(HERE, ".venv", "Scripts", "python.exe")   # enroll_infer/noise_classify
PY_SE   = os.path.join(HERE, ".venv_se", "Scripts", "python.exe") # se_denoise
PY_LLM  = os.path.join(ROOT, ".venv_llm", "Scripts", "python.exe") # llm_reject
```

---

## Task 1: submit_infer.py 骨架 + 工具函数(utt_id / audio_duration)

**Files:**
- Create: `code/submit_infer.py`
- Create: `tests/__init__.py`(空)
- Create: `tests/test_submit_infer_logic.py`

- [ ] **Step 1: 写失败测试**

`tests/test_submit_infer_logic.py`:
```python
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from submit_infer import utt_id_from_path, audio_duration_s

def test_utt_id():
    assert utt_id_from_path("E:/x/rec_001.wav") == "rec_001"
    assert utt_id_from_path("rec_002.WAV") == "rec_002"
    assert utt_id_from_path("/a/b/c.wav") == "c"
    print("test_utt_id OK")

def test_audio_duration():
    # 用一条已知测试音频验证(若存在)
    p = "E:/midea_target_asr/test_wav/zh_target_01.wav"
    if os.path.exists(p):
        d = audio_duration_s(p)
        assert d > 0, f"duration should be >0, got {d}"
        print(f"test_audio_duration OK ({d:.2f}s)")
    else:
        print("test_audio_duration SKIP (no fixture)")

if __name__ == "__main__":
    test_utt_id()
    test_audio_duration()
    print("ALL PASS")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `code/.venv/Scripts/python.exe tests/test_submit_infer_logic.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'submit_infer'`

- [ ] **Step 3: 写最小实现**

`code/submit_infer.py`(本 task 只建骨架 + 2 个工具函数,后续 task 追加):
```python
"""标准化推理脚本: enrollment + recognition -> result.json + timing.json

纯 stdlib subprocess 顶层编排器,零侵入复用 noise_classify / se_denoise /
enroll_infer / llm_reject。详见 spec:
docs/superpowers/specs/2026-07-01-submit-infer-and-deliverables-design.md
"""
import os
import sys
import json
import argparse
import subprocess
import time
import shutil
import glob
import wave
import contextlib
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))      # code/
ROOT = os.path.dirname(HERE)                            # 项目根
PY_MAIN = os.path.join(HERE, ".venv", "Scripts", "python.exe")
PY_SE   = os.path.join(HERE, ".venv_se", "Scripts", "python.exe")
PY_LLM  = os.path.join(ROOT, ".venv_llm", "Scripts", "python.exe")


def utt_id_from_path(p):
    """文件名去扩展名作 utt_id。'E:/x/rec_001.wav' -> 'rec_001'。"""
    return os.path.splitext(os.path.basename(p))[0]


def audio_duration_s(p):
    """wav 时长(秒),纯 stdlib wave。读失败返回 0.0。"""
    try:
        with contextlib.closing(wave.open(p, "rb")) as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return 0.0


def main():
    # 占位,Task 6 实现
    print("[submit_infer] skeleton only — see Task 6 for full pipeline")


if __name__ == "__main__":
    main()
```

`tests/__init__.py`:空文件(0 字节)。

- [ ] **Step 4: 跑测试确认通过**

Run: `code/.venv/Scripts/python.exe tests/test_submit_infer_logic.py`
Expected: `test_utt_id OK` / `test_audio_duration OK (或 SKIP)` / `ALL PASS`

- [ ] **Step 5: Commit**

```bash
git add code/submit_infer.py tests/__init__.py tests/test_submit_infer_logic.py
git commit -m "feat(submit_infer): 骨架+utt_id/audio_duration工具函数(TDD)"
```

---

## Task 2: 输入解析(expand_inputs / load_pairs)

**Files:**
- Modify: `code/submit_infer.py`(在 audio_duration_s 后追加)
- Modify: `tests/test_submit_infer_logic.py`(追加测试)

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_submit_infer_logic.py`(在 `if __name__` 块前加测试函数,并在 `if __name__` 块内调用):
```python
from submit_infer import expand_inputs, load_pairs

def test_load_pairs(tmp_path=None):
    import tempfile, json as _json
    d = tempfile.mkdtemp()
    pj = os.path.join(d, "pairs.json")
    _json.dump([{"enrollment": "a.wav", "recognition": "b.wav"},
                {"enrollment": "a.wav", "recognition": "c.wav"}],
               open(pj, "w"))
    pairs = load_pairs(pj)
    assert pairs == [("a.wav", "b.wav"), ("a.wav", "c.wav")], pairs
    print("test_load_pairs OK")

def test_expand_inputs_folder():
    # 模拟 args 对象
    class A: pass
    import tempfile
    d = tempfile.mkdtemp()
    for n in ("r1.wav", "r2.wav"):
        open(os.path.join(d, n), "w").close()
    a = A(); a.pairs=None; a.enrollment="e.wav"; a.recognition_folder=d
    out = expand_inputs(a)
    assert ("e.wav", os.path.join(d, "r1.wav")) in out
    assert ("e.wav", os.path.join(d, "r2.wav")) in out
    assert len(out) == 2
    print("test_expand_inputs_folder OK")
```
在 `if __name__ == "__main__":` 块内追加 `test_load_pairs()` 和 `test_expand_inputs_folder()`。

- [ ] **Step 2: 跑测试确认失败**

Run: `code/.venv/Scripts/python.exe tests/test_submit_infer_logic.py`
Expected: FAIL with `ImportError: cannot import name 'expand_inputs'`

- [ ] **Step 3: 写实现**

在 `code/submit_infer.py` 的 `audio_duration_s` 后追加:
```python
def load_pairs(pairs_json):
    """读 --pairs manifest: [{enrollment, recognition}, ...] -> [(enr, rec), ...]。"""
    rows = json.load(open(pairs_json, encoding="utf-8"))
    return [(r["enrollment"], r["recognition"]) for r in rows]


def expand_inputs(args):
    """把 CLI 输入展开为 [(enrollment, recognition), ...] 统一列表。
    --pairs 一对一; --enrollment + --recognition-folder 一对多。"""
    if args.pairs:
        return load_pairs(args.pairs)
    recs = sorted(glob.glob(os.path.join(args.recognition_folder, "*.wav")))
    return [(args.enrollment, r) for r in recs]
```

- [ ] **Step 4: 跑测试确认通过**

Run: `code/.venv/Scripts/python.exe tests/test_submit_infer_logic.py`
Expected: `ALL PASS`(含两个新测试)

- [ ] **Step 5: Commit**

```bash
git add code/submit_infer.py tests/test_submit_infer_logic.py
git commit -m "feat(submit_infer): 输入解析(expand_inputs/load_pairs, 支持pairs与folder)"
```

---

## Task 3: 融合拒识决策(decide_reject)

**Files:**
- Modify: `code/submit_infer.py`(追加 decide_reject)
- Modify: `tests/test_submit_infer_logic.py`(追加测试)

- [ ] **Step 1: 写失败测试**

追加测试函数 + `if __name__` 块内调用:
```python
from submit_infer import decide_reject

def test_decide_reject():
    # sim_only: 只看 max_sim
    assert decide_reject(0.10, "accept", "sim_only", 0.2, True) == True
    assert decide_reject(0.30, "reject", "sim_only", 0.2, True) == False
    # llm_only: 只看 llm
    assert decide_reject(0.05, "accept", "llm_only", 0.2, True) == False
    assert decide_reject(0.90, "reject", "llm_only", 0.2, True) == True
    # llm_or_sim: 拒 iff llm!=accept AND max_sim<thr (LLM救回sim误拒)
    assert decide_reject(0.10, "accept", "llm_or_sim", 0.2, True) == False   # LLM救回
    assert decide_reject(0.30, "reject", "llm_or_sim", 0.2, True) == False   # sim救回
    assert decide_reject(0.10, "reject", "llm_or_sim", 0.2, True) == True    # 双低→拒
    # 无 LLM 强制退化 sim_only
    assert decide_reject(0.10, "accept", "llm_or_sim", 0.2, False) == True
    assert decide_reject(0.30, "accept", "llm_or_sim", 0.2, False) == False
    print("test_decide_reject OK")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `code/.venv/Scripts/python.exe tests/test_submit_infer_logic.py`
Expected: FAIL `ImportError: cannot import name 'decide_reject'`

- [ ] **Step 3: 写实现**

在 `expand_inputs` 后追加:
```python
def decide_reject(max_sim, llm_verdict, strategy, sim_thr, use_llm):
    """融合拒识决策。返回 True=拒识。
    - 无 LLM 或 sim_only: 拒 iff max_sim < sim_thr
    - llm_only:           拒 iff llm != accept
    - llm_or_sim(默认):   拒 iff (llm != accept) AND (max_sim < sim_thr)
      (LLM 救回声纹误拒;fuse_eval T19 验证最优)
    """
    if not use_llm or strategy == "sim_only":
        return max_sim < sim_thr
    if strategy == "llm_only":
        return llm_verdict != "accept"
    return llm_verdict != "accept" and max_sim < sim_thr
```

- [ ] **Step 4: 跑测试确认通过**

Run: `code/.venv/Scripts/python.exe tests/test_submit_infer_logic.py`
Expected: `ALL PASS`

- [ ] **Step 5: Commit**

```bash
git add code/submit_infer.py tests/test_submit_infer_logic.py
git commit -m "feat(submit_infer): 融合拒识决策decide_reject(sim_only/llm_only/llm_or_sim)"
```

---

## Task 4: SE 条件化分桶(bucket_by_atten)

**Files:**
- Modify: `code/submit_infer.py`(追加 bucket_by_atten)
- Modify: `tests/test_submit_infer_logic.py`(追加测试)

- [ ] **Step 1: 写失败测试**

追加:
```python
from submit_infer import bucket_by_atten

def test_bucket_by_atten():
    rows = [
        {"file": "a.wav", "atten_lim_db": 0},
        {"file": "b.wav", "atten_lim_db": 6},
        {"file": "c.wav", "atten_lim_db": 0},
        {"file": "d.wav", "atten_lim_db": 6},
    ]
    b = bucket_by_atten(rows)
    assert set(b.keys()) == {0, 6}
    assert sorted(b[0]) == ["a.wav", "c.wav"]
    assert sorted(b[6]) == ["b.wav", "d.wav"]
    # 空
    assert bucket_by_atten([]) == {}
    print("test_bucket_by_atten OK")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `code/.venv/Scripts/python.exe tests/test_submit_infer_logic.py`
Expected: FAIL `ImportError: cannot import name 'bucket_by_atten'`

- [ ] **Step 3: 写实现**

在 `decide_reject` 后追加:
```python
def bucket_by_atten(noise_rows):
    """按 atten_lim_db 把识别音频分桶 -> {atten: [basename, ...]}。
    SE 条件化核心: se_denoise 只支持目录统一 atten, 分桶后每桶一次 subprocess。"""
    buckets = {}
    for r in noise_rows:
        a = int(r.get("atten_lim_db", 0))
        buckets.setdefault(a, []).append(r["file"])
    return buckets
```

- [ ] **Step 4: 跑测试确认通过**

Run: `code/.venv/Scripts/python.exe tests/test_submit_infer_logic.py`
Expected: `ALL PASS`

- [ ] **Step 5: Commit**

```bash
git add code/submit_infer.py tests/test_submit_infer_logic.py
git commit -m "feat(submit_infer): SE条件化分桶bucket_by_atten(解se_denoise目录统一atten gap)"
```

---

## Task 5: 输出 schema 构建(build_result / build_timing)

**Files:**
- Modify: `code/submit_infer.py`(追加 build_result / build_timing)
- Modify: `tests/test_submit_infer_logic.py`(追加测试)

- [ ] **Step 1: 写失败测试**

追加:
```python
from submit_infer import build_result, build_timing

def test_build_result():
    items = [{
        "utt_id": "r1", "enrollment": "e.wav", "recognition": "r1.wav",
        "text": "你好", "rejected": False, "score": 0.30,
        "max_sim": 0.30, "llm_verdict": "accept",
        "noise_type": "white", "atten_lim_db": 0, "diar_fail": False,
    }, {
        "utt_id": "r2", "enrollment": "e.wav", "recognition": "r2.wav",
        "text": "", "rejected": True, "score": 0.05,
        "max_sim": 0.05, "llm_verdict": "reject",
        "noise_type": None, "atten_lim_db": None, "diar_fail": False,
    }]
    cfg = {"se": True, "llm": True, "strategy": "llm_or_sim", "sim_thr": 0.2, "device": "cuda:0"}
    out = build_result(items, cfg)
    assert out["task_id"] == "XH-202615"
    assert out["n_utt"] == 2
    assert out["config"] == cfg
    assert len(out["results"]) == 2
    assert out["results"][0]["text"] == "你好"
    assert out["results"][1]["rejected"] == True
    assert "generated_at" in out
    print("test_build_result OK")

def test_build_timing():
    t = build_timing(
        device="cuda:0", n_utt=2,
        total_audio_sec=10.0, total_wall_sec=3.0,
        phases={"noise_classify": {"wall_sec": 0.5}, "se": {"wall_sec": 1.0, "n": 2},
                "enroll_diar_dicow": {"wall_sec": 1.0, "mean_rtf": 0.1},
                "llm": {"wall_sec": 0.5}},
        per_utt=[{"utt_id": "r1", "audio_sec": 5.0, "wall_sec": 1.5, "rtf": 0.3}])
    assert t["device"] == "cuda:0"
    assert t["n_utt"] == 2
    assert abs(t["overall_rtf"] - 0.3) < 1e-6   # 3.0/10.0
    assert t["phases"]["se"]["n"] == 2
    assert len(t["per_utt"]) == 1
    print("test_build_timing OK")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `code/.venv/Scripts/python.exe tests/test_submit_infer_logic.py`
Expected: FAIL `ImportError: cannot import name 'build_result'`

- [ ] **Step 3: 写实现**

在 `bucket_by_atten` 后追加:
```python
def build_result(items, config):
    """组装 result.json schema(见 spec §8)。"""
    return {
        "task_id": "XH-202615",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "n_utt": len(items),
        "results": items,
    }


def build_timing(device, n_utt, total_audio_sec, total_wall_sec, phases, per_utt):
    """组装 timing.json schema(见 spec §8)。overall_rtf = wall/audio。"""
    return {
        "device": device,
        "n_utt": n_utt,
        "total_audio_sec": round(total_audio_sec, 3),
        "total_wall_sec": round(total_wall_sec, 3),
        "overall_rtf": round(total_wall_sec / total_audio_sec, 4) if total_audio_sec else None,
        "phases": phases,
        "per_utt": per_utt,
    }
```

- [ ] **Step 4: 跑测试确认通过**

Run: `code/.venv/Scripts/python.exe tests/test_submit_infer_logic.py`
Expected: `ALL PASS`

- [ ] **Step 5: Commit**

```bash
git add code/submit_infer.py tests/test_submit_infer_logic.py
git commit -m "feat(submit_infer): 输出schema构建build_result/build_timing(填端到端timing空缺)"
```

---

## Task 6: subprocess 编排 + main()(串全链路)

**Files:**
- Modify: `code/submit_infer.py`(追加 4 个 run_* + 重写 main)

> 本 task 是 subprocess 编排,**用 Task 7 集成验收**(不写单元测试,因 mock subprocess 收益低于端到端验证)。

- [ ] **Step 1: 写 4 个 subprocess 包装函数**

在 `build_timing` 后追加:
```python
def _run(cmd, py):
    """subprocess 跑 [py] + cmd,返回 (wall_sec)。失败抛 RuntimeError。"""
    t0 = time.perf_counter()
    full = [py] + cmd
    print(f"  [run] {os.path.basename(py)} {' '.join(cmd[:2])} ...")
    r = subprocess.run(full, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        raise RuntimeError(f"subprocess failed ({r.returncode}): {r.stderr[-500:]}")
    return dt


def run_noise_classify(rec_dir, out_json, py=PY_MAIN):
    """阶段0: 估每条噪声类型 -> out_json(list[{file,atten_lim_db,...}])。"""
    return _run([os.path.join(HERE, "noise_classify.py"),
                 "--in-dir", rec_dir, "--out", out_json], py)


def run_se_bucket(in_dir, out_dir, atten_db, py=PY_SE):
    """阶段1(单桶): se_denoise 对 in_dir 全体用统一 atten_db 降噪 -> out_dir。"""
    return _run([os.path.join(HERE, "se_denoise.py"),
                 "--in-dir", in_dir, "--out-dir", out_dir,
                 "--atten-lim-db", str(atten_db)], py)


def run_enroll_infer(enrollment, rec_dir, out_json, device, sim_thr, py=PY_MAIN):
    """阶段2: enroll声纹锁定+diar+DiCoW批量转写 -> out_json(list[{...}])。"""
    return _run([os.path.join(HERE, "enroll_infer.py"),
                 "--enrollment", enrollment, "--recognition-folder", rec_dir,
                 "--out-json", out_json, "--always-generate",
                 "--reject-threshold", str(sim_thr),
                 "--device", device], py)


def run_llm(infer_json, out_json, device, py=PY_LLM):
    """阶段3: llm_reject 对每条 transcript 判 accept/reject。"""
    return _run([os.path.join(HERE, "llm_reject.py"),
                 "--infer-json", infer_json, "--out-json", out_json,
                 "--device", device], py)
```

- [ ] **Step 2: 写 main() 串全链路**

替换 Task 1 的占位 `main()`:
```python
def main():
    ap = argparse.ArgumentParser(description="标准化推理: enrollment+recognition -> result.json+timing.json")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pairs", help="[{enrollment,recognition}] manifest(一对一)")
    g.add_argument("--enrollment", help="目标说话人参考 wav(配 --recognition-folder)")
    ap.add_argument("--recognition-folder", help="识别 wav 目录(配 --enrollment,一对多)")
    ap.add_argument("--no-se", action="store_true", help="跳过 SE 条件化降噪")
    ap.add_argument("--no-llm", action="store_true", help="跳过 LLM 拒识")
    ap.add_argument("--sim-thr", type=float, default=0.2, help="声纹拒识阈值(T20 最优)")
    ap.add_argument("--strategy", default="llm_or_sim",
                    choices=["llm_or_sim", "sim_only", "llm_only"], help="融合策略")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "submit_out"))
    ap.add_argument("--work-dir", default=None, help="中间产物(默认 <out-dir>/_work)")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条(0=全部)")
    args = ap.parse_args()

    if not args.pairs and not args.recognition_folder:
        ap.error("--enrollment 需配 --recognition-folder")
    use_llm = not args.no_llm
    use_se = not args.no_se
    work_dir = args.work_dir or os.path.join(args.out_dir, "_work")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    pairs = expand_inputs(args)
    if args.limit > 0:
        pairs = pairs[: args.limit]

    # 识别音频目录视图(软链/复制到 work/rec_in,统一喂下游)
    rec_in = os.path.join(work_dir, "rec_in")
    os.makedirs(rec_in, exist_ok=True)
    enr0 = pairs[0][0]
    one_enrollment = all(e == enr0 for e, _ in pairs)
    rec_paths = []
    for i, (enr, rec) in enumerate(pairs):
        dst = os.path.join(rec_in, f"utt{i:04d}_{utt_id_from_path(rec)}.wav")
        if not os.path.exists(dst):
            shutil.copy(rec, dst)
        rec_paths.append((enr, rec, dst))
    # enroll_infer 一对多要求单 enrollment; --pairs 多 enrollment 时分组跑(按 enr 分组)
    t_total0 = time.perf_counter()
    phases = {}
    total_audio = sum(audio_duration_s(dst) for _, _, dst in rec_paths)

    # --- 阶段0+1: SE 条件化 ---
    rec_for_enroll = rec_in  # 默认原始
    if use_se:
        t0 = time.perf_counter()
        noise_est = os.path.join(work_dir, "noise_est.json")
        phases.setdefault("noise_classify", {})["wall_sec"] = round(run_noise_classify(rec_in, noise_est), 3)
        rows = json.load(open(noise_est, encoding="utf-8"))
        buckets = bucket_by_atten(rows)
        se_out = os.path.join(work_dir, "se_out")
        os.makedirs(se_out, exist_ok=True)
        se_wall = 0.0
        for atten, files in buckets.items():
            bin_ = os.path.join(work_dir, f"se_in_{atten}")
            os.makedirs(bin_, exist_ok=True)
            for f in files:
                src = os.path.join(rec_in, f)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(bin_, f))
            bout = os.path.join(work_dir, f"se_out_{atten}")
            se_wall += run_se_bucket(bin_, bout, atten)
            for f in files:
                s = os.path.join(bout, f)
                if os.path.exists(s):
                    shutil.copy(s, os.path.join(se_out, f))
        phases["se"] = {"wall_sec": round(se_wall, 3), "n": sum(len(v) for v in buckets.values())}
        rec_for_enroll = se_out
        t1 = time.perf_counter()
        print(f"[se] 阶段0+1 用时 {t1-t0:.1f}s ({len(rows)} 条, 桶={list(buckets.keys())})")

    # --- 阶段2: enroll_infer 转写(按 enrollment 分组,每组一次批量) ---
    from collections import defaultdict
    groups = defaultdict(list)
    for enr, rec, dst in rec_paths:
        groups[enr].append(dst)
    enroll_jsons = []
    e_wall = 0.0
    sum_rtf = 0.0
    n_rtf = 0
    for gi, (enr, dsts) in enumerate(groups.items()):
        # 为本组复制 dsts 到独立子目录(enroll_infer 吃 --recognition-folder)
        gdir = os.path.join(work_dir, f"enroll_g{gi}")
        os.makedirs(gdir, exist_ok=True)
        for d in dsts:
            shutil.copy(d, os.path.join(gdir, os.path.basename(d)))
        out_json = os.path.join(work_dir, f"enroll_g{gi}.json")
        e_wall += run_enroll_infer(enr, gdir, out_json, args.device, args.sim_thr)
        rows = json.load(open(out_json, encoding="utf-8"))
        for r in rows:
            sum_rtf += float(r.get("rtf", 0.0) or 0.0)
            n_rtf += 1
        enroll_jsons.append((enr, out_json))
    phases["enroll_diar_dicow"] = {"wall_sec": round(e_wall, 3),
                                   "mean_rtf": round(sum_rtf / n_rtf, 4) if n_rtf else None}

    # 汇总 enroll 输出(utt_id -> row)
    enr_map = {}
    for enr, out_json in enroll_jsons:
        for r in json.load(open(out_json, encoding="utf-8")):
            uid = utt_id_from_path(r.get("recognition", ""))
            enr_map[uid] = r

    # --- 阶段3: LLM 拒识 ---
    llm_map = {}
    if use_llm:
        llm_in = os.path.join(work_dir, "llm_in.json")
        llm_rows = [{"file": enr_map[uid].get("recognition", ""),
                     "text": enr_map[uid].get("transcript", "") or ""}
                    for uid in enr_map]
        json.dump(llm_rows, open(llm_in, "w", encoding="utf-8"), ensure_ascii=False)
        llm_out = os.path.join(work_dir, "llm_out.json")
        l_wall = run_llm(llm_in, llm_out, args.device)
        phases["llm"] = {"wall_sec": round(l_wall, 3)}
        for row in json.load(open(llm_out, encoding="utf-8")):
            llm_map[utt_id_from_path(row.get("file", ""))] = row.get("pred", "reject")

    # --- 阶段4: 融合 + 组装 result ---
    total_wall = time.perf_counter() - t_total0
    items, per_utt = [], []
    for enr, rec, dst in rec_paths:
        uid = utt_id_from_path(dst)
        r = enr_map.get(uid, {})
        max_sim = float(r.get("max_sim", 0.0) or 0.0)
        llm_v = llm_map.get(uid, "reject") if use_llm else "accept"
        rejected = decide_reject(max_sim, llm_v, args.strategy, args.sim_thr, use_llm)
        text = "" if rejected else (r.get("transcript", "") or "")
        items.append({
            "utt_id": uid, "enrollment": enr, "recognition": rec,
            "text": text, "rejected": rejected, "score": round(max_sim, 4),
            "max_sim": round(max_sim, 4), "llm_verdict": llm_v if use_llm else None,
            "noise_type": r.get("noise_type"), "atten_lim_db": r.get("atten_lim_db"),
            "diar_fail": bool(r.get("error")),
        })
        dur = audio_duration_s(dst)
        per_utt.append({"utt_id": uid, "audio_sec": round(dur, 3),
                        "wall_sec": None, "rtf": round(float(r.get("rtf", 0.0) or 0.0), 4)})

    cfg = {"se": use_se, "llm": use_llm, "strategy": args.strategy if use_llm else "sim_only",
           "sim_thr": args.sim_thr, "device": args.device}
    result = build_result(items, cfg)
    timing = build_timing(args.device, len(items), total_audio, total_wall, phases, per_utt)

    rj = os.path.join(args.out_dir, "result.json")
    tj = os.path.join(args.out_dir, "timing.json")
    json.dump(result, open(rj, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(timing, open(tj, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    n_rej = sum(1 for it in items if it["rejected"])
    print(f"\n[done] {len(items)} 条 ({n_rej} 拒识) -> {rj}")
    print(f"       overall_rtf={timing['overall_rtf']} (audio={total_audio:.1f}s wall={total_wall:.1f}s)")
    print(f"       phases={ {k: v.get('wall_sec') for k, v in phases.items()} }")
```

- [ ] **Step 3: py_compile 语法校验**

Run: `code/.venv/Scripts/python.exe -m py_compile code/submit_infer.py`
Expected: 无输出(EXIT=0)

- [ ] **Step 4: 逻辑测试仍通过(回归)**

Run: `code/.venv/Scripts/python.exe tests/test_submit_infer_logic.py`
Expected: `ALL PASS`(纯函数未受影响)

- [ ] **Step 5: Commit**

```bash
git add code/submit_infer.py
git commit -m "feat(submit_infer): subprocess编排+main全链路(SE条件化分桶/enroll分组/LLM/融合/timing)"
```

---

## Task 7: 集成验收(--limit 5 跑通 + 开关 + schema)

**Files:** 无新建(只跑验收)

> 前置:`code/setenv.sh` 已设 E 盘缓存/代理;`test_wav/dataset/final/` 有 450 条带噪 wav(若不存在,用 `test_wav/dataset/raw/enrollment/target_long_01.wav` + 任意几条识别 wav)。

- [ ] **Step 1: 全量模式 --limit 3 跑通**

Run:
```bash
cd /e/midea_target_asr && source code/setenv.sh && export HF_HUB_OFFLINE=1
code/.venv/Scripts/python.exe code/submit_infer.py \
  --enrollment E:/midea_target_asr/test_wav/dataset/raw/enrollment/target_long_01.wav \
  --recognition-folder E:/midea_target_asr/test_wav/dataset/final \
  --limit 3 --out-dir code/submit_out_test
```
Expected: EXIT=0;末行 `[done] 3 条 ... overall_rtf=<数值>`;`code/submit_out_test/result.json` + `timing.json` 生成。

- [ ] **Step 2: schema 校验**

Run:
```bash
code/.venv/Scripts/python.exe -c "
import json
r=json.load(open('code/submit_out_test/result.json',encoding='utf-8'))
assert r['task_id']=='XH-202615'
assert set(r['config']) >= {'se','llm','strategy','sim_thr','device'}
assert len(r['results'])==3
for it in r['results']:
    assert set(it) >= {'utt_id','text','rejected','score','max_sim'}
    if it['rejected']: assert it['text']=='', it
t=json.load(open('code/submit_out_test/timing.json',encoding='utf-8'))
assert t['overall_rtf'] is not None and t['overall_rtf']>=0
assert 'phases' in t and 'per_utt' in t
print('SCHEMA OK', 'n=',r['n_utt'],'rtf=',t['overall_rtf'])
"
```
Expected: `SCHEMA OK n= 3 rtf= <数值>`

- [ ] **Step 3: --no-llm 开关生效**

Run:
```bash
code/.venv/Scripts/python.exe code/submit_infer.py \
  --enrollment E:/midea_target_asr/test_wav/dataset/raw/enrollment/target_long_01.wav \
  --recognition-folder E:/midea_target_asr/test_wav/dataset/final \
  --limit 3 --no-llm --out-dir code/submit_out_nollm
code/.venv/Scripts/python.exe -c "
import json; r=json.load(open('code/submit_out_nollm/result.json',encoding='utf-8'))
assert r['config']['llm']==False and r['config']['strategy']=='sim_only'
assert all(it['llm_verdict'] is None for it in r['results'])
print('NO-LLM OK')
"
```
Expected: `NO-LLM OK`

- [ ] **Step 4: --no-se 开关生效**

Run:
```bash
code/.venv/Scripts/python.exe code/submit_infer.py \
  --enrollment E:/midea_target_asr/test_wav/dataset/raw/enrollment/target_long_01.wav \
  --recognition-folder E:/midea_target_asr/test_wav/dataset/final \
  --limit 3 --no-se --no-llm --out-dir code/submit_out_noseno
code/.venv/Scripts/python.exe -c "
import json; r=json.load(open('code/submit_out_noseno/result.json',encoding='utf-8'))
assert r['config']['se']==False
print('NO-SE OK')
"
```
Expected: `NO-SE OK`

- [ ] **Step 5: 清理测试产物 + Commit 验收记录**

```bash
rm -rf code/submit_out_test code/submit_out_nollm code/submit_out_noseno
git add -A && git commit -m "test(submit_infer): 集成验收通过(--limit3全量/no-llm/no-se三档开关+schema校验)" --allow-empty
```

> ⚠️ 若 Step 1 跑不通(diar crash 等),记录报错到 PROGRESS,不强求全绿——集成验收的目的是暴露问题。本机 4060 8GB 全量(SE+enroll+LLM)可能显存紧,可用 `--no-llm` 先验。

---

## Task 8: 交付/设计报告.md 骨架(实际实现版)

**Files:**
- Create: `交付/设计报告.md`

- [ ] **Step 1: 写骨架(关键章填实际,不确定 `<!--TODO-->`)**

`交付/设计报告.md`(基于 00/01 文档 + RESULTS T14-T20,**写实际实现非理想态**):
```markdown
# XH-202615 技术设计方案

## 1. 问题定义与评分
- 任务:给定 enrollment(目标说话人唤醒音频),在带噪(SNR -5~5dB)+ 多人重叠(≤2人,0-100%)识别音频中,只转写目标指令、拒识非目标。
- 评分:目标 CER 40% + 拒识率 40% + 推理效率 20%(L20 GPU)。

## 2. 总体架构(实际实现)
<!-- 数据流图: recognition -> [SE条件化降噪] -> enroll声纹锁定target(wespeaker) -> DiariZen diar -> STNO mask -> DiCoW(Whisper-large-v3-turbo+FDDT) 转写target -> [LLM语义拒识] -> llm_or_sim融合 -> result -->
- 核心组件: wespeaker声纹(非CAM++,T18证伪) / DiariZen diar / DiCoW转写 / DeepFilterNet3 SE条件化 / Qwen2.5-3B LLM拒识。

## 3. 各模块技术细节
### 3.1 目标锁定(wespeaker 声纹)
- enrollment + 各 speaker 声纹(复用 diar._embedding, 256d),余弦匹配选 target_idx。
<!-- TODO: max_sim 阈值工作点 0.2(T20) -->
### 3.2 STNO 条件化转写(DiCoW + FDDT)
- FDDT 仿射变换将 [sil/target/nontarget/overlap] 4类 mask 注入 encoder 每层;fddt_init=disparagement 抑制式初始化 → 非目标帧压零(内建拒识)。
- 基座 Whisper-large-v3-turbo(0.89G params),已修 language 强制死代码 bug(apply_dicow_langfix.py)。
### 3.3 SE 条件化(DeepFilterNet3)
- 谱平坦度估噪声类型(babble/white→atten=0全力, pink→=6温和),99.78% 准确;仿真 CER 2.82(-34%)。
### 3.4 LLM 语义拒识(Qwen2.5-3B)
- 零样本 F1=0.878/Recall=1.0(34条);救回声纹误拒。
### 3.5 融合(llm_or_sim)
- rejected = (llm≠accept) AND (max_sim<0.2);LLM 救回 sim 误拒。

## 4. 差异化策略
- D1 数据增强 / D2 中文家居微调(待) / D3 SE条件化可部署 / D4 三层融合拒识 / D5 RTF=0.058 高效。
<!-- TODO: D2/D3 真实数据验证后补 -->

## 5. 当前性能(诚实,仿真450条)
- 干净场景: CER=0;重叠 0/50/100% → CER 0.00/0.13/1.00(100%单通道死区)。
- 带噪全集: se6 正确率 15.1%,瓶颈=babble diar误检+STNO崩 + Whisper带噪中文。
- ⚠️ 以上为仿真,真实 A 集待评测。

## 6. 局限与改进路线
- babble diar误检 → STNO容错/源分离(待);100%重叠 → 多通道DSENet(待通道数确认);Whisper带噪 → 中文微调。
```

- [ ] **Step 2: Commit**

```bash
git add "交付/设计报告.md"
git commit -m "docs(交付): 设计报告骨架(实际实现版wespeaker/SE条件化/LLM融合,非理想态CAM++/PVAD)"
```

---

## Task 9: 交付/使用说明.md 骨架

**Files:**
- Create: `交付/使用说明.md`

- [ ] **Step 1: 写骨架**

`交付/使用说明.md`:
```markdown
# 使用说明

## 1. 环境要求
- GPU: NVIDIA 8GB+(本机 RTX 4060 / 评测 L20 48GB)
- Python 3.12;3 个独立 venv(依赖不可合并):
  - `code/.venv`(enroll_infer/noise_classify): torch 2.5.1+cu124 / transformers 4.42.4 / DiariZen
  - `code/.venv_se`(se_denoise): df-simple/DeepFilterNet3
  - `.venv_llm`(llm_reject): Qwen2.5-3B + vllm/transformers
- 配置: `source code/setenv.sh`(E盘缓存/代理);`export HF_HUB_OFFLINE=1`

## 2. 权重清单(全 E:/hf_cache)
- DiCoW_v3_2(0.89G)、diarizen-wavlm-large-s80-md、wespeaker-voxceleb-resnet34-LM、DeepFilterNet3(E:/df_cache)、Qwen2.5-3B
<!-- TODO: Qwen 权重路径核实 -->

## 3. 运行推理(submit_infer.py)
\`\`\`bash
source code/setenv.sh && export HF_HUB_OFFLINE=1
# 全量最优(SE条件化+enroll+DiCoW+LLM)
code/.venv/Scripts/python.exe code/submit_infer.py \
  --enrollment <enr.wav> --recognition-folder <rec_dir> --out-dir <out>
# 或一对一: --pairs <manifest.json>
# 降级: --no-se / --no-llm ; 快测: --limit N
\`\`\`
输出: `<out>/result.json`(utt_id/text/rejected/score)+`<out>/timing.json`(overall_rtf/phases/per_utt)。

## 4. 输入输出格式
- result.json: [{utt_id, text, rejected, score, max_sim, llm_verdict, noise_type, atten_lim_db}]
- rejected=true 时 text=""(拒识);score=max_sim。

## 5. 常见报错
- diar crash(恶劣音频 pyannote reconstruct) → enroll_infer 已 try/except 跳过,该条 diar_fail=true。
- HF cache 清了 → 重打 langfix:`code/apply_dicow_langfix.py`。
<!-- TODO: 补 LLM venv 显存不足时的 --no-llm 兜底说明 -->
```

- [ ] **Step 2: Commit**

```bash
git add "交付/使用说明.md"
git commit -m "docs(交付): 使用说明骨架(submit_infer用法+3venv环境+权重+格式+常见报错)"
```

---

## Task 10: 交付/测试验证方案.md 骨架

**Files:**
- Create: `交付/测试验证方案.md`

- [ ] **Step 1: 写骨架**

`交付/测试验证方案.md`:
```markdown
# 测试验证方案

## 1. 评测指标(对齐评分)
- 目标 CER(40%): 字符级 CER(jiwer),`eval_metrics.py:cer`;拒识条计 1.0(漏 target)。
- 拒识率(40%): target-absent 集真实拒识率 + target-present 集误拒率。
- 效率(20%): overall_rtf(timing.json)+ 峰值显存(L20)。

## 2. 评测脚本
- `code/eval_metrics.py`: CER/RTF/拒识(precision/recall/f1/reject_rate)/batch。
- `code/fuse_eval.py`: 多策略融合扫最优工作点(评测用,非推理)。

## 3. 当前结果(仿真 450 条,诚实)
| 维度 | 结果 | 备注 |
|---|---|---|
| 干净 CER | 0.00 | mimo-tts 合成 |
| 重叠 0/50/100% CER | 0.00/0.13/1.00 | 100% 单通道死区 |
| SE 条件化 CER | 2.82(-34%) | babble/white→0, pink→6 |
| LLM 拒识 F1 | 0.878 | 34 条零样本 |
| 拒识集真实 RR | 100% | 72 条 target-absent |
| se6 正确率 | 15.1% | ⚠️ 仿真绝对值差,瓶颈 babble diar+Whisper带噪 |
<!-- TODO: 用 RESULTS T14-T20 真实数字校准本表 -->

## 4. 真实 A 集评测流程(A 到手后)
1. `submit_infer.py --pairs <A集manifest> --out-dir A_out` 出 result.json+timing.json
2. `eval_metrics.py` 算 CER/RR(对照 A 集 ground truth)
3. L20 上重跑 timing.json 报效率分
4. 写测试报告(分档:SNR/overlap/noise_type)

## 5. 局限声明
- 仿真噪声为程序生成(white/pink/babble),真实噪声谱可能不同 → noise_classify 需再校准。
- 100% 重叠单通道死区,依赖多通道(待通道数确认)。
```

- [ ] **Step 2: Commit**

```bash
git add "交付/测试验证方案.md"
git commit -m "docs(交付): 测试验证方案骨架(评测指标+仿真结果诚实表+A集到手流程)"
```

---

## Task 11: README.md 修正(理想态→实际)

**Files:**
- Modify: `README.md`(架构图/栈表/快速开始)

- [ ] **Step 1: 改技术架构图为实际实现**

把 `README.md` 的"整体方案"代码块(L30-56)替换为实际数据流:
```
识别音频 → [SE条件化降噪(DeepFilterNet3, 谱平坦度分桶)] 
        → DiariZen diarization → wespeaker声纹锁定target(余弦匹配enrollment)
        → STNO mask(FDDT 4类) → DiCoW(Whisper-large-v3-turbo)只转target
        → [LLM语义拒识(Qwen2.5-3B)] → llm_or_sim融合 → result.json
```
注:删 CAM++/Personal VAD 描述(实际未用)。

- [ ] **Step 2: 改核心技术栈表为实际**

把 L60-67 技术栈表替换:
```markdown
| 模块 | 实际方案 | 作用 |
|---|---|---|
| 声纹锁定 | wespeaker-voxceleb-resnet34(256d) | enrollment→target 余弦匹配(CAM++ T18 证伪,未用) |
| 说话人分离 | DiariZen(wavlm-large) | 多 speaker 时间段 |
| TS-ASR | DiCoW(Whisper-large-v3-turbo + FDDT) | STNO 条件化只转 target |
| 语音增强 | DeepFilterNet3(条件化) | babble/white→全力, pink→温和 |
| 语义拒识 | Qwen2.5-3B | 指令合理性,救回声纹误拒 |
| 融合 | llm_or_sim | (llm≠accept) AND (max_sim<0.2) |
```

- [ ] **Step 3: 快速开始补 submit_infer.py**

在"快速开始"段(L106-148)追加:
```markdown
### 标准化推理(submit_infer.py)
\`\`\`bash
source code/setenv.sh && export HF_HUB_OFFLINE=1
code/.venv/Scripts/python.exe code/submit_infer.py \
  --enrollment <enr.wav> --recognition-folder <rec_dir> --out-dir out/
# → out/result.json + out/timing.json
\`\`\`
```

- [ ] **Step 4: 加交付文档链接**

在"项目结构"段补:
```markdown
├── 📦 交付/
│   ├── 设计报告.md          # 技术方案(实际实现)
│   ├── 使用说明.md          # submit_infer 用法
│   └── 测试验证方案.md      # 评测方法 + 结果
```

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs(README): 架构图/栈表理想态→实际实现(删CAM++/PVAD, 补wespeaker/SE条件化/LLM/submit_infer)"
```

---

## 收尾(全部 task 完成后)

- [ ] **更新 PROGRESS.md**:append `2026-07-01 T21: submit_infer.py 标准化推理脚本 + 交付文档骨架(对照6-30战略转向P0/P1)`
- [ ] **更新 AGENT_HANDOFF.md**:T19 待办①(submit_infer)✅;P0/P1 完成,剩 P2(babble工程兜底)/P3(显存自适应)/等A集
- [ ] **Commit 收尾**
```bash
git add PROGRESS.md AGENT_HANDOFF.md
git commit -m "docs: T21进度收尾(submit_infer+交付骨架完成,P0/P1落地)"
```

---

## Self-Review(skill 自检,写计划后跑)

**1. Spec 覆盖**:
- §4 数据流 → Task 6 main() ✅
- §5 接口(所有 flag) → Task 6 argparse ✅
- §6 SE 分桶 → Task 4 bucket + Task 6 阶段0+1 ✅
- §7 融合 → Task 3 decide_reject + Task 6 阶段4 ✅
- §8 result/timing schema → Task 5 build_* + Task 6 装填 ✅
- §9 timing 测量 → Task 6 perf_counter 包各阶段 ✅
- §10 三文档 + README → Task 8-11 ✅
- §13 验收 → Task 7 ✅

**2. Placeholder 扫描**:Task 8-10 文档内的 `<!--TODO-->` 是**有意的占位策略**(spec §10/§13 明确允许,标注不确定处供 A 集后补),非计划缺陷。计划步骤本身无 TBD/未给代码。

**3. 类型一致**:`utt_id_from_path`/`decide_reject`/`bucket_by_atten`/`build_result`/`build_timing` 签名在定义 task(1/3/4/5)与使用处(Task 6 main)一致;`run_*` 包装在 Task 6 定义即用。✅
