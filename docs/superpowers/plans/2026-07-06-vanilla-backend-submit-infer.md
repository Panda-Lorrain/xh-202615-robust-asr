# vanilla 后端集成 submit_infer 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 Phase 1 验证的 vanilla Whisper + 声纹切 target timeline 路线集成进 `submit_infer`，加 `--asr-backend {dicow,vanilla}` 开关，vanilla 作提交主线，并产出符合官方格式的 submission.json。

**Architecture:** 方案 A——`enroll_infer.py` 内加 backend 开关，diar/wespeaker/选target 共享，转写分叉（dicow 走 stno_mask 条件化、vanilla 走切 target timeline 段拼接无 mask）。纯函数（繁简归一、timeline 切割）抽到 `_text_utils.py` 供 enroll_infer 与新建的 `to_submission.py` 共用、单测覆盖。

**Tech Stack:** Python 3.12（`code/.venv`，torch 2.5.1+cu124）、transformers（Whisper-large-v3-turbo / DiCoW）、zhconv（繁简）、DiariZen（diar）、unittest 风格裸函数测试。

**Spec:** `docs/superpowers/specs/2026-07-06-vanilla-backend-submit-infer-design.md`

**测试约定（遵循 `tests/test_submit_infer_logic.py`）：** 裸函数 + assert + print，`sys.path.insert` 加 code/，`__main__` 调度，跑法 `code/.venv/Scripts/python.exe tests/test_xxx_logic.py`。

---

### Task 1: `_text_utils.py` 纯函数 + 单测

**Files:**
- Create: `code/_text_utils.py`
- Test: `tests/test_text_utils_logic.py`

- [ ] **Step 1: 写失败测试**

`tests/test_text_utils_logic.py`:
```python
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
import numpy as np
from _text_utils import to_simplified, cut_target_timeline

def test_to_simplified():
    assert to_simplified("空調開到二十六度") == "空调开到二十六度"
    assert to_simplified("") == ""
    assert to_simplified(None) is None
    assert to_simplified("已经简体") == "已经简体"
    print("test_to_simplified OK")

def test_cut_basic():
    sr = 16000
    audio = np.ones(sr * 10, dtype=np.float32)
    out = cut_target_timeline(audio, [(1.0, 3.0), (5.0, 7.0)], sr=sr)
    assert len(out) == sr * 4
    print("test_cut_basic OK")

def test_cut_keeps_overlap_region():
    sr = 16000
    audio = np.arange(sr * 5, dtype=np.float32)
    out = cut_target_timeline(audio, [(0.0, 2.0), (3.0, 5.0)], sr=sr)
    assert len(out) == sr * 4
    assert out[0] == audio[0]
    assert out[sr * 2] == audio[sr * 3]  # 段2 拼接到 2s 位置
    print("test_cut_keeps_overlap_region OK")

def test_cut_too_short_fallback():
    sr = 16000
    audio = np.ones(sr * 5, dtype=np.float32)
    out = cut_target_timeline(audio, [(0.0, 0.1)], sr=sr)  # 0.1s < 0.3s
    assert len(out) == sr * 5  # 退化整条
    print("test_cut_too_short_fallback OK")

def test_cut_empty_fallback():
    sr = 16000
    audio = np.ones(sr * 3, dtype=np.float32)
    assert len(cut_target_timeline(audio, [], sr=sr)) == sr * 3
    print("test_cut_empty_fallback OK")

if __name__ == "__main__":
    test_to_simplified()
    test_cut_basic()
    test_cut_keeps_overlap_region()
    test_cut_too_short_fallback()
    test_cut_empty_fallback()
    print("ALL PASS")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `code/.venv/Scripts/python.exe tests/test_text_utils_logic.py`
Expected: `ModuleNotFoundError: No module named '_text_utils'`

- [ ] **Step 3: 实现 `code/_text_utils.py`**

```python
"""转写文本工具: 繁简归一 + target timeline 切割(纯函数, enroll_infer 与 to_submission 共用)。"""
import numpy as np


def to_simplified(text):
    """繁→简归一(zhconv)。空值直通。转写后归一 + submission 兜底都用。"""
    if not text:
        return text
    import zhconv
    return zhconv.convert(text, "zh-cn")


def cut_target_timeline(audio, per_spk_timeline, sr=16000, min_sec=0.3):
    """切 target 的 timeline 段(含重叠区)拼接成连续音频, 喂 vanilla Whisper。

    区别 collect_clean_audio(抽独占非重叠帧做声纹, 避开污染): 本函数切 target 整条
    timeline(含重叠区)做转写 —— target 在重叠区的话也要转出来。

    audio: np.ndarray 全条; per_spk_timeline: list[(start,end)]; min_sec: 不足则退化整条。
    """
    segs = sorted((float(s), float(e)) for s, e in per_spk_timeline)
    if segs:
        out = np.concatenate([audio[int(s * sr):int(e * sr)] for s, e in segs])
    else:
        out = np.asarray(audio)
    if len(out) < sr * min_sec:
        out = np.asarray(audio)  # target 太短退化整条(避免喂 Whisper 过短片段)
    return out
```

- [ ] **Step 4: 跑测试确认通过**

Run: `code/.venv/Scripts/python.exe tests/test_text_utils_logic.py`
Expected: `ALL PASS`

- [ ] **Step 5: Commit**

```bash
git add code/_text_utils.py tests/test_text_utils_logic.py
git -c user.name="Panda_Lorrain" -c user.email="2687571939@qq.com" commit -m "feat(text-utils): 抽繁简归一+target timeline切割纯函数(单测覆盖)"
```

---

### Task 2: `enroll_infer.py` 加 `--asr-backend`

**Files:**
- Modify: `code/enroll_infer.py`（模型加载 121-125 / 转写块 231-288 / 输出 292-301 / 顶部 import）
- 集成验证在 Task 6（真实模型，此处不做单元集成测）

变量重命名：`dicow` → `asr_model`（统一两个 backend）。

- [ ] **Step 1: 顶部加 import（第 28 行 `from transformers import ...` 后）**

```python
from _text_utils import to_simplified, cut_target_timeline
```

- [ ] **Step 2: argparse 加参数（`args = ap.parse_args()` 前，约 114 行）**

```python
    ap.add_argument("--asr-backend", default="dicow", choices=["dicow", "vanilla"],
                    help="ASR 后端: dicow(FDDT/STNO 条件化, fallback) / vanilla(切target timeline+whisper, 主线 CER 减半)")
    ap.add_argument("--vanilla-model", default="E:/hf_cache/whisper-large-v3-turbo",
                    help="vanilla 后端 Whisper 模型(默认 large-v3-turbo)")
```

- [ ] **Step 3: 模型加载分叉（替换 121-125 行）**

原：
```python
    print(f"[load] DiCoW {args.dicow_model} on {device}")
    dicow = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.dicow_model, trust_remote_code=True, torch_dtype=dtype).to(device).eval()
    tok = AutoTokenizer.from_pretrained(args.dicow_model)
    fe = AutoFeatureExtractor.from_pretrained(args.dicow_model)
```
改为：
```python
    if args.asr_backend == "vanilla":
        print(f"[load] vanilla Whisper {args.vanilla_model} on {device}")
        asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            args.vanilla_model, torch_dtype=dtype).to(device).eval()
        _model_path = args.vanilla_model
    else:
        print(f"[load] DiCoW {args.dicow_model} on {device}")
        asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            args.dicow_model, trust_remote_code=True, torch_dtype=dtype).to(device).eval()
        _model_path = args.dicow_model
    tok = AutoTokenizer.from_pretrained(_model_path)
    fe = AutoFeatureExtractor.from_pretrained(_model_path)
```

- [ ] **Step 4: 转写块改 backend 分叉（替换 231-288 行的 `else:` 转写分支）**

原 `else:` 块（stno + dicow.generate + SE-DiCoW setup + langfix retry）整体替换为：
```python
        else:
            if args.asr_backend == "vanilla":
                # vanilla: 切 target timeline(含重叠区)拼接 → vanilla.generate 无条件化
                target_audio = cut_target_timeline(audio, per_spk[target_idx], sr=sr)
                ifp_v = fe(target_audio, sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype)
                am_v = torch.ones(1, ifp_v.shape[-1], dtype=torch.bool, device=device)
                with torch.no_grad():
                    out = asr_model.generate(input_features=ifp_v, attention_mask=am_v,
                                             language=args.language, task="transcribe", max_new_tokens=200)
                seqs = out["sequences"] if isinstance(out, dict) else out
                text = tok.batch_decode(seqs, skip_special_tokens=True)[0].strip()
                # vanilla 不跑 langfix(英文幻觉 0.59%, langfix 是 dicow 治标)
            else:
                # dicow: stno_mask 条件化 + SE-DiCoW 自登记 + langfix retry(原逻辑保留)
                stno = get_stno_mask(diar_mask, target_idx)
                am = torch.ones(1, ifp.shape[-1], dtype=torch.bool, device=device)
                gen_kwargs = dict(input_features=ifp, attention_mask=am,
                                  stno_mask=stno[None].to(device, dtype),
                                  language=args.language, task="transcribe", max_new_tokens=200)
                if bool(getattr(asr_model.config, "uses_enrollments", False) or getattr(asr_model.config, "use_enrollments", False)):
                    if not getattr(asr_model, "_se_setup_done", False):
                        vocab = tok.get_vocab()
                        tok.upper_cased_tokens = {}
                        for _t, _i in vocab.items():
                            if len(_t) < 1:
                                continue
                            _lo = (_t[0] + _t[1].lower() + (_t[2:] if len(_t) > 2 else '')) \
                                  if (_t[0] == 'Ġ' and len(_t) > 1) else (_t[0].lower() + _t[1:])
                            if _lo != _t and vocab.get(_lo) is not None:
                                tok.upper_cased_tokens[vocab[_lo]] = _i
                        if hasattr(asr_model, "set_tokenizer"):
                            asr_model.set_tokenizer(tok)
                        asr_model.config.model_type = "whisper"
                        asr_model._se_setup_done = True
                        print("[load] SE-DiCoW uses_enrollments=True → cross-attn 内部启用(self-enrolled)")
                with torch.no_grad():
                    out = asr_model.generate(**gen_kwargs)
                seqs = out["sequences"] if isinstance(out, dict) else out
                text = tok.batch_decode(seqs, skip_special_tokens=True)[0].strip()
                # langfix retry(仅 dicow)
                _L = [c for c in text if c.isalpha()]
                _er = sum(c.isascii() for c in _L) / len(_L) if len(_L) >= 4 else 0.0
                if _er > 0.4:
                    if not hasattr(asr_model, "_zh_prompt_ids"):
                        asr_model._zh_prompt_ids = torch.tensor(
                            tok("以下是普通话的句子。", add_special_tokens=False).input_ids, device=device)
                    retry_kwargs = dict(gen_kwargs)
                    retry_kwargs.update(num_beams=1, prompt_ids=asr_model._zh_prompt_ids)
                    with torch.no_grad():
                        out2 = asr_model.generate(**retry_kwargs)
                    seqs2 = out2["sequences"] if isinstance(out2, dict) else out2
                    text2 = tok.batch_decode(seqs2, skip_special_tokens=True)[0].strip()
                    _PFX = "以下是普通话的句子。"
                    if text2.startswith(_PFX):
                        text2 = text2[len(_PFX):].strip()
                    _L2 = [c for c in text2 if c.isalpha()]
                    _er2 = sum(c.isascii() for c in _L2) / len(_L2) if len(_L2) >= 4 else 0.0
                    if _er2 < _er:
                        text = text2
                    print(f"  [langfix-retry] 英文率 {_er:.2f}→{_er2:.2f} {'采纳重生成' if _er2 < _er else '保留首次'}")
            # 统一繁简归一(dicow + vanilla 都过)
            text = to_simplified(text)
            verdict = (f"REJECT_GEN(max_sim={max_sim:.3f}<{args.reject_threshold}, always-generate 仍转)" if rejected
                       else f"TRANSCRIBE(target={speakers[target_idx]}, backend={args.asr_backend})")
```

- [ ] **Step 5: 输出加字段（`results.append({...})` 内，约 292-301 行）**

在 `"chars": len(text), "rtf": dt / dur,` 同 dict 内加两字段：
```python
            "asr_backend": args.asr_backend,
            "infer_sec": round(dt, 3),  # 单条纯推理(不含模型加载), 对齐官方 batch=1 duration
```

- [ ] **Step 6: 冒烟确认 enroll_infer 仍能跑（不报语法/导入错）**

Run: `code/.venv/Scripts/python.exe code/enroll_infer.py --help 2>&1 | grep -E "asr-backend|vanilla-model"`
Expected: 两参数都在 help 输出里。

- [ ] **Step 7: Commit**

```bash
git add code/enroll_infer.py
git -c user.name="Panda_Lorrain" -c user.email="2687571939@qq.com commit" -m "feat(enroll-infer): 加 --asr-backend {dicow,vanilla} + 繁简归一 + infer_sec 字段"
```
（注：上面命令 `commit` 应在引号外，正确写法见 Task 1 Step 5 风格：`git -c user.name=... -c user.email=... commit -m "..."`）

---

### Task 3: `submit_infer.py` 透传 `--asr-backend` + timing 字段

**Files:**
- Modify: `code/submit_infer.py`（`run_enroll_infer_pairs` 136-151 / main 调用 264 / config 321 / timing 324）
- Test: `tests/test_submit_infer_logic.py`（追加 `test_run_enroll_pairs_backend`）

- [ ] **Step 1: 追加失败测试（`tests/test_submit_infer_logic.py` 末尾 `if __name__` 前）**

```python
def test_run_enroll_pairs_backend():
    # vanilla backend 应在 cmd 加 --asr-backend vanilla; dicow 默认不加(向后兼容)
    import submit_infer as si
    captured = {}
    def fake_run(cmd, py):
        captured["cmd"] = cmd
        captured["py"] = py
        return 0.5
    orig = si._run
    si._run = fake_run
    try:
        si.run_enroll_infer_pairs("p.json", "o.json", "cuda:0", 0.4, asr_backend="vanilla")
        assert "--asr-backend" in captured["cmd"]
        assert captured["cmd"][captured["cmd"].index("--asr-backend")+1] == "vanilla"
        si.run_enroll_infer_pairs("p.json", "o.json", "cuda:0", 0.4, asr_backend="dicow")
        assert "--asr-backend" not in captured["cmd"]  # dicow 默认不透传(向后兼容)
    finally:
        si._run = orig
    print("test_run_enroll_pairs_backend OK")
```
并在 `__main__` 调度里加 `test_run_enroll_pairs_backend()`。

- [ ] **Step 2: 跑确认失败**

Run: `code/.venv/Scripts/python.exe tests/test_submit_infer_logic.py`
Expected: 旧测试 PASS，新测试 error（`run_enroll_infer_pairs() got an unexpected keyword argument 'asr_backend'`）

- [ ] **Step 3: 改 `run_enroll_infer_pairs`（136-151 行）加 asr_backend**

签名加 `asr_backend="dicow"`，cmd 构造后加：
```python
    cmd = [os.path.join(HERE, "enroll_infer.py"),
           "--pairs", pairs_json,
           "--out-json", out_json, "--always-generate",
           "--reject-threshold", str(sim_thr),
           "--device", device]
    if asr_backend != "dicow":
        cmd += ["--asr-backend", asr_backend]
    if enroll_augment:
        ...
```

- [ ] **Step 4: argparse + main 调用 + config/timing**

(a) argparse（`--strategy` 附近）加：
```python
    ap.add_argument("--asr-backend", default="dicow", choices=["dicow", "vanilla"],
                    help="ASR 后端(透传 enroll_infer): dicow(fallback) / vanilla(主线)")
```

(b) main 调用（264 行）加 `args.asr_backend`：
```python
    e_wall = run_enroll_infer_pairs(enroll_pairs, enroll_all, args.device, args.sim_thr,
                                    args.enroll_augment, args.aug_snrs, args.aug_noise_dir,
                                    args.asr_backend)
```

(c) duration 累加（阶段4 `total_wall = ...` 后，约 298 行）：
```python
    duration_infer_sec = sum(float(r.get("infer_sec", 0.0) or 0.0) for r in all_rows)
```

(d) config（321 行 cfg）加 `"asr_backend": args.asr_backend`。

(e) timing（build_timing 后）加顶层字段：
```python
    timing["duration_infer_sec"] = round(duration_infer_sec, 3)
```

- [ ] **Step 5: 跑测试确认通过**

Run: `code/.venv/Scripts/python.exe tests/test_submit_infer_logic.py`
Expected: `ALL PASS`（含新测试）

- [ ] **Step 6: Commit**

```bash
git add code/submit_infer.py tests/test_submit_infer_logic.py
git -c user.name="Panda_Lorrain" -c user.email="2687571939@qq.com" commit -m "feat(submit-infer): 透传 --asr-backend + duration_infer_sec(batch=1 口径) + config 字段"
```

---

### Task 4: `run_baodi.sh` 默认 vanilla + `BAODI_BACKEND` 切回

**Files:**
- Modify: `code/run_baodi.sh`

- [ ] **Step 1: 改默认 backend + 注释**

文件头注释（第 4 行附近）加一行：
```bash
# vanilla 作提交主线(CER 0.664 减半, 反 cascaded); BAODI_BACKEND=dicow 切回 fallback/答辩对比基线。
```

`SET/THR` 解析后（第 27 行 case 块后）加：
```bash
BACKEND="${BAODI_BACKEND:-vanilla}"
```

echo 行（第 34）与 exec 行（35-36）改为：
```bash
echo "[baodi] backend=$BACKEND 关LLM(--no-llm) + thr=$THR + strategy=sim_only  → $OUT  (vanilla 主线 / BAODI_BACKEND=dicow 切回)"
exec code/.venv/Scripts/python.exe code/submit_infer.py \
  --pairs "$PAIRS" --out-dir "$OUT" --no-llm --sim-thr "$THR" --strategy sim_only \
  --asr-backend "$BACKEND"
```

- [ ] **Step 2: shellcheck / 语法确认**

Run: `bash -n code/run_baodi.sh && echo SYNTAX_OK`
Expected: `SYNTAX_OK`

- [ ] **Step 3: Commit**

```bash
git add code/run_baodi.sh
git -c user.name="Panda_Lorrain" -c user.email="2687571939@qq.com" commit -m "feat(run-baodi): 默认 --asr-backend vanilla(主线), BAODI_BACKEND=dicow 切回 fallback"
```

---

### Task 5: `to_submission.py` + 单测

**Files:**
- Create: `code/to_submission.py`
- Test: `tests/test_to_submission_logic.py`

- [ ] **Step 1: 写失败测试 `tests/test_to_submission_logic.py`**

```python
import os, sys, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from to_submission import convert, _utt_id_stripped

def test_utt_id_stripped():
    assert _utt_id_stripped("utt0012_cmd_3.wav") == "cmd_3"
    assert _utt_id_stripped("cmd_5.wav") == "cmd_5"
    print("test_utt_id_stripped OK")

def _write(d, name, obj):
    p = os.path.join(d, name); json.dump(obj, open(p, "w")); return p

def test_convert_pos_accept():
    d = tempfile.mkdtemp()
    rj = _write(d, "result.json", {"results": [
        {"recognition": "utt0001_cmd_10.wav", "text": "空調開到二十六度", "rejected": False}]})
    pj = _write(d, "pairs.json", [{"enrollment": "e.wav", "recognition": "cmd_10.wav",
                                   "ref": "空调开到二十六度"}])
    sub = convert(rj, pj, duration_infer_sec=12.3)
    row = sub["result"]["results"][0]
    assert row["id"] == "cmd_10"
    assert row["content"] == "空调开到二十六度"  # 繁简归一
    assert row["label"] == "accept"
    assert row["cer"] == 0.0  # 完美匹配(_norm_zh 含繁简)
    assert sub["result"]["final_cer"] == 0.0
    assert sub["result"]["duration"] == 12.3
    print("test_convert_pos_accept OK")

def test_convert_pos_rejected():
    d = tempfile.mkdtemp()
    rj = _write(d, "result.json", {"results": [
        {"recognition": "cmd_1.wav", "text": "", "rejected": True}]})
    pj = _write(d, "pairs.json", [{"enrollment": "e.wav", "recognition": "cmd_1.wav",
                                   "ref": "打开空调"}])
    sub = convert(rj, pj)
    row = sub["result"]["results"][0]
    assert row["content"] == ""
    assert row["label"] == "reject"
    assert row["cer"] == 1.0
    assert sub["result"]["final_cer"] == 1.0
    print("test_convert_pos_rejected OK")

def test_convert_neg():
    d = tempfile.mkdtemp()
    rj = _write(d, "result.json", {"results": [
        {"recognition": "cmd_2.wav", "text": "", "rejected": True}]})
    pj = _write(d, "pairs.json", [{"enrollment": "e.wav", "recognition": "cmd_2.wav",
                                   "ref": ""}])  # neg ref 空
    sub = convert(rj, pj)
    row = sub["result"]["results"][0]
    assert row["label"] == "reject"
    assert row["cer"] == ""  # neg 不评 CER 评 RR
    print("test_convert_neg OK")

if __name__ == "__main__":
    test_utt_id_stripped()
    test_convert_pos_accept()
    test_convert_pos_rejected()
    test_convert_neg()
    print("ALL PASS")
```

- [ ] **Step 2: 跑确认失败**

Run: `code/.venv/Scripts/python.exe tests/test_to_submission_logic.py`
Expected: `ModuleNotFoundError: No module named 'to_submission'`

- [ ] **Step 3: 实现 `code/to_submission.py`**

```python
"""把 submit_infer result.json 转成官方提交格式。

官方格式(2026-07-06):
  {"result":{"results":[{"id","content","label","cer"}],"final_cer","duration"}}

待主办方确认口径做成 SUBMISSION_DEFAULTS 常量, 主办方回复只改常量不改逻辑。
"""
import os, sys, json, re, argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from _text_utils import to_simplified
from eval_datasetA import _norm_zh
from eval_metrics import cer as _cer

# 待主办方确认口径(默认推测值, 见 spec §7)
SUBMISSION_DEFAULTS = {
    "label_accept": "accept",
    "label_reject": "reject",
    "pos_rejected_cer": 1.0,  # pos 被错拒的 cer
}


def _utt_id_stripped(p):
    """去 uttN_ 前缀(HANDOFF §8 坑5): utt0012_cmd_3.wav → cmd_3。"""
    base = os.path.splitext(os.path.basename(p))[0]
    return re.sub(r"^utt\d+_", "", base)


def convert(result_json, pairs_json, duration_infer_sec=None):
    """result.json + pairs manifest → 官方 submission dict。

    pairs_json 提供 ref(pos 有 / neg 空)。duration_infer_sec 为 None 时从 per-utt infer_sec 累加。
    """
    with open(result_json, encoding="utf-8") as f:
        result = json.load(f)
    with open(pairs_json, encoding="utf-8") as f:
        pair_rows = json.load(f)

    ref_map, is_neg_map = {}, {}
    for row in pair_rows:
        uid = _utt_id_stripped(row["recognition"])
        ref_map[uid] = row.get("ref", "") or ""
        is_neg_map[uid] = (not row.get("ref")) or row.get("label") == "neg"

    rows_out, pos_cers = [], []
    for r in result.get("results", []):
        uid = _utt_id_stripped(r.get("recognition", ""))
        text = to_simplified(r.get("text", "") or "")
        rejected = bool(r.get("rejected"))
        label = SUBMISSION_DEFAULTS["label_reject"] if rejected else SUBMISSION_DEFAULTS["label_accept"]
        is_neg = is_neg_map.get(uid, False)
        if is_neg:
            cer_val = ""  # neg 不评 CER(评 RR)
        else:
            ref = ref_map.get(uid, "")
            if not ref:
                cer_val = ""
            elif rejected:
                cer_val = round(SUBMISSION_DEFAULTS["pos_rejected_cer"], 3)
                pos_cers.append(SUBMISSION_DEFAULTS["pos_rejected_cer"])
            else:
                t = _norm_zh(text); rr = _norm_zh(ref)
                c = _cer(t, rr) if t else 1.0
                cer_val = round(c, 3)
                pos_cers.append(c)
        rows_out.append({"id": uid, "content": text, "label": label, "cer": cer_val})

    final_cer = round(sum(pos_cers) / len(pos_cers), 3) if pos_cers else 0.0
    if duration_infer_sec is None:
        duration_infer_sec = sum(float(r.get("infer_sec", 0) or 0) for r in result.get("results", []))
    return {"result": {"results": rows_out, "final_cer": final_cer,
                       "duration": round(duration_infer_sec, 3)}}


def main():
    ap = argparse.ArgumentParser(description="result.json → 官方提交格式")
    ap.add_argument("--result-json", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--duration", type=float, default=None,
                    help="batch=1 逐条推理总秒数(覆盖 infer_sec 累加)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    sub = convert(args.result_json, args.pairs, args.duration)
    out = args.out or args.result_json.replace("result.json", "submission.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(sub, f, ensure_ascii=False, indent=2)
    print(f"[done] {len(sub['result']['results'])} 条 → {out} "
          f"(final_cer={sub['result']['final_cer']}, duration={sub['result']['duration']}s)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `code/.venv/Scripts/python.exe tests/test_to_submission_logic.py`
Expected: `ALL PASS`
（前提：`eval_datasetA._norm_zh` 含繁简归一，使「空調開到二十六度」与「空调开到二十六度」归一后相同 → cer=0。若失败先核实 `_norm_zh` 实现。）

- [ ] **Step 5: Commit**

```bash
git add code/to_submission.py tests/test_to_submission_logic.py
git -c user.name="Panda_Lorrain" -c user.email="2687571939@qq.com" commit -m "feat(to-submission): result.json→官方提交格式(id/content/label/cer/final_cer/duration)"
```

---

### Task 6: 端到端验证（vanilla 冒烟 + dicow 回归 + 全量 pos + submission 校验）

**Files:** 无（运行验证）

- [ ] **Step 1: vanilla 冒烟 100 条**

```bash
cd E:/midea_target_asr && source code/setenv.sh && export HF_HUB_OFFLINE=1
BAODI_OK=1 code/.venv/Scripts/python.exe code/submit_infer.py \
  --pairs code/pos_pairs_datasetA.json --out-dir code/out_pos_vanilla_smoke \
  --no-llm --sim-thr 0.4 --strategy sim_only --asr-backend vanilla --limit 100
code/.venv/Scripts/python.exe code/eval_datasetA.py code/out_pos_vanilla_smoke/result.json code/pos_pairs_datasetA.json
```
Expected: pos CER ≈ 0.664（±0.05，参考 `exp_vanilla_full.json`）；输出文本为简体。

- [ ] **Step 2: dicow 回归 100 条（确认 fallback 不坏）**

```bash
BAODI_OK=1 code/.venv/Scripts/python.exe code/submit_infer.py \
  --pairs code/pos_pairs_datasetA.json --out-dir code/out_pos_dicow_smoke \
  --no-llm --sim-thr 0.4 --strategy sim_only --asr-backend dicow --limit 100
code/.venv/Scripts/python.exe code/eval_datasetA.py code/out_pos_dicow_smoke/result.json code/pos_pairs_datasetA.json
```
Expected: pos CER ≈ 1.25（与改动前一致，fallback 路径未坏）。

- [ ] **Step 3: submission.json 格式校验（vanilla 冒烟产出）**

```bash
code/.venv/Scripts/python.exe code/to_submission.py \
  --result-json code/out_pos_vanilla_smoke/result.json --pairs code/pos_pairs_datasetA.json
code/.venv/Scripts/python.exe -c "import json; d=json.load(open('code/out_pos_vanilla_smoke/submission.json'))['result']; r=d['results'][0]; assert set(r)=={'id','content','label','cer'}, r; assert not r['id'].startswith('utt'), r['id']; print('schema OK, final_cer=', d['final_cer'], 'duration=', d['duration'])"
```
Expected: `schema OK ...`，id 无 `uttN_` 前缀，duration 为正数。

- [ ] **Step 4: 全量 pos vanilla（冒烟全过则跑，产出最终提交数字）**

```bash
bash code/run_baodi.sh pos   # 默认 vanilla thr=0.4
code/.venv/Scripts/python.exe code/eval_datasetA.py code/out_pos_baodi/result.json code/pos_pairs_datasetA.json
code/.venv/Scripts/python.exe code/to_submission.py \
  --result-json code/out_pos_baodi/result.json --pairs code/pos_pairs_datasetA.json
```
Expected: 全量 pos CER（提交数字）+ submission.json 产出。

- [ ] **Step 5: 更新 RESULTS.md / memory（记最终提交数字）**

把全量 vanilla pos CER / duration / final_cer 记入 `RESULTS.md` T25 + 更新 memory `h3-dicow-conditioning-backfire-vanilla`（P2-① 完成）。commit。

---

## Self-Review（plan 自检）

- **Spec coverage**：§1 架构→Task 2；§2 透传→Task 3；§3 繁简归一→Task 1+2；§4 run_baodi→Task 4；§5 验证→Task 6；§6 测试→Task 1/3/5 单测 + Task 6 集成；§7 官方格式→Task 5。✅ 全覆盖。
- **Placeholder**：无 TBD/TODO，每步有完整代码/命令。✅
- **Type 一致**：`to_simplified` / `cut_target_timeline` / `convert` / `_utt_id_stripped` / `run_enroll_infer_pairs(asr_backend=)` / `infer_sec` / `duration_infer_sec` / `asr_backend` 跨 task 命名一致。✅
- **Task 2 Step 7 commit 命令**：模板里笔误（commit 跑进引号），实际执行用 Task 1 Step 5 同款写法。✅ 已标注。
