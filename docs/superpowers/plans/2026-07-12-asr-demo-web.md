# 目标说话人 ASR Demo 网站 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 做一个公网可访问的 demo 网站，访客浏览器录音 → 本机常驻 vanilla ASR 引擎推理 → 回显转写/匹配分/拒识/切出的目标音，支持在线混 babble/第二人声演示抗干扰。

**Architecture:** FastAPI 后端（跑在 `code/.venv`，复用 enroll_infer 的 torch/CUDA 上下文）+ 原生 HTML/JS 单页前端 + cloudflared quick tunnel 公网。推理常驻引擎 `InferenceEngine` 启动时加载一次模型（diar+wespeaker+vanilla Whisper），之后每条 5–15s。

**Tech Stack:** Python 3.12（code/.venv）、FastAPI 0.138 + uvicorn 0.49（已装）、ffmpeg 7.1（已装，PATH）、浏览器 MediaRecorder API、soundfile/librosa（已装）、复用 `enroll_infer.py`/`simulate_pipeline.py`/`text_utils.py`/`repro.py`。

**Spec:** `docs/superpowers/specs/2026-07-12-asr-demo-web-design.md`

**测试约定（重要）**：本项目无 pytest 惯例（全是研究脚本）。测试用**裸 assert 脚本**放 `code/demo_web/tests/`，`python xxx.py` 直跑，无 exit=0 即失败。纯函数（audio_utils）走 TDD（先写失败 assert 再实现）；依赖大模型的组件（inference_engine）用 `selfcheck.py` 集成冒烟。

**运行约定**：所有 `uvicorn`/`selfcheck`/测试脚本都在 **`code/` 目录下**执行（让 `demo_web` 作为包可 import，且 `enroll_infer`/`simulate_pipeline` 等命中 `code/`）。推理类脚本前必须 `source code/setenv.sh`（模型缓存指 E 盘 + 代理）。

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `code/demo_web/__init__.py` | 包标识（空） |
| `code/demo_web/audio_utils.py` | ffmpeg 转码 16k mono + `add_noise`/`mix_overlap` 混音包装 + 时长 |
| `code/demo_web/inference_engine.py` | `InferenceEngine`：`load_models()` + `infer()`，复刻 enroll_infer vanilla 流程；**不改 enroll_infer.py** |
| `code/demo_web/server.py` | FastAPI 路由 + startup 加载 + `/audio` 回放 |
| `code/demo_web/selfcheck.py` | 集成冒烟：kws_0+cmd_0 验证 inference_engine |
| `code/demo_web/run_demo.sh` | source setenv + uvicorn 启动 |
| `code/demo_web/README.md` | 启动/公网/自检/依赖说明 |
| `code/demo_web/tests/test_audio_utils.py` | audio_utils 裸 assert 单测 |
| `code/demo_web/static/index.html` | 单页 UI |
| `code/demo_web/static/app.js` | 录音/上传/混音/推理/回放逻辑 |
| `code/demo_web/static/style.css` | 样式 |
| `code/demo_web/sessions/.gitkeep` | session 音频目录占位（内容 gitignore） |

新增 `.gitignore` 条目：`code/demo_web/sessions/*.wav`（音频不入库）。

---

## Task 1: 项目骨架 + 依赖确认

**Files:**
- Create: `code/demo_web/__init__.py`
- Create: `code/demo_web/tests/__init__.py`
- Create: `code/demo_web/sessions/.gitkeep`
- Modify: `.gitignore`（根目录）

- [ ] **Step 1: 建包结构**

Run（在项目根）:
```bash
mkdir -p code/demo_web/static code/demo_web/tests code/demo_web/sessions
```

写 `code/demo_web/__init__.py`（空文件，占位）：
```python
```

写 `code/demo_web/tests/__init__.py`（空）：
```python
```

写 `code/demo_web/sessions/.gitkeep`（空）：
```
```

- [ ] **Step 2: gitignore session 音频**

在项目根 `.gitignore` 末尾追加（若已有 `code/demo_web/sessions/*.wav` 则跳过）：
```
# demo web session 音频(运行时产物, 不入库)
code/demo_web/sessions/*.wav
```

- [ ] **Step 3: 确认 python-multipart（FastAPI File/Form 依赖）**

Run:
```bash
E:/midea_target_asr/code/.venv/Scripts/python.exe -c "import multipart; print('multipart OK')" 2>&1 | tail -1
```
Expected: `multipart OK`。若 `ModuleNotFoundError`：
```bash
E:/midea_target_asr/code/.venv/Scripts/python.exe -m pip install python-multipart
```

- [ ] **Step 4: Commit**

```bash
git add code/demo_web .gitignore
git commit -m "feat(demo): demo_web 包骨架 + sessions gitignore"
```

---

## Task 2: audio_utils.to_wav_16k_mono（TDD）

**Files:**
- Create: `code/demo_web/tests/test_audio_utils.py`
- Create: `code/demo_web/audio_utils.py`

- [ ] **Step 1: 先写失败测试**

写 `code/demo_web/tests/test_audio_utils.py`：
```python
"""audio_utils 裸 assert 单测。用法: cd code && .venv/Scripts/python.exe demo_web/tests/test_audio_utils.py"""
import os, sys, tempfile, contextlib, wave
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # demo_web/
import numpy as np
import soundfile as sf
import audio_utils


def _mkwav(path, sr=16000, dur=1.0, freq=220):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    w = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(path, w, sr)


def test_to_wav_16k_mono():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "in.wav"); dst = os.path.join(d, "out.wav")
        _mkwav(src, sr=44100, dur=1.0)  # 故意 44.1k → 转码有意义
        audio_utils.to_wav_16k_mono(src, dst)
        assert os.path.exists(dst), "dst 未生成"
        with contextlib.closing(wave.open(dst, "rb")) as w:
            assert w.getframerate() == 16000, f"sr={w.getframerate()}"
            assert w.getnchannels() == 1, f"ch={w.getnchannels()}"
            assert w.getsampwidth() == 2, f"sw={w.getsampwidth()}"  # s16=2字节
    print("test_to_wav_16k_mono OK")


if __name__ == "__main__":
    test_to_wav_16k_mono()
    print("ALL OK")
```

- [ ] **Step 2: 跑测试，确认失败（audio_utils 不存在）**

Run:
```bash
cd code && .venv/Scripts/python.exe demo_web/tests/test_audio_utils.py
```
Expected: FAIL（`No module named 'audio_utils'`）。

- [ ] **Step 3: 写最小实现（仅转码 + 时长，混音下个 Task 加）**

写 `code/demo_web/audio_utils.py`：
```python
"""音频转码 + 混音工具(ffmpeg 转码 + simulate_pipeline 复用混音)。

lazy import simulate_pipeline: 转码/时长功能不依赖它, 只有 mix_* 才 import。
"""
import os, sys, subprocess, glob, random, contextlib, wave
import numpy as np
import librosa
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))      # demo_web/
_CODE = os.path.dirname(_HERE)                           # code/
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)


def to_wav_16k_mono(src_path, dst_path):
    """ffmpeg 任意格式(webm/mp4/wav/m4a) → 16k mono s16 wav。失败抛 CalledProcessError。"""
    cmd = ["ffmpeg", "-y", "-i", src_path, "-vn", "-ar", "16000",
           "-ac", "1", "-sample_fmt", "s16", dst_path]
    subprocess.run(cmd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def duration_s(wav_path):
    """wav 时长(秒), 纯 stdlib wave。读失败返回 0.0。"""
    try:
        with contextlib.closing(wave.open(wav_path, "rb")) as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return 0.0
```

- [ ] **Step 4: 跑测试，确认通过**

Run:
```bash
cd code && .venv/Scripts/python.exe demo_web/tests/test_audio_utils.py
```
Expected: `test_to_wav_16k_mono OK` + `ALL OK`，exit 0。

- [ ] **Step 5: Commit**

```bash
git add code/demo_web/audio_utils.py code/demo_web/tests/test_audio_utils.py
git commit -m "feat(demo): audio_utils.to_wav_16k_mono(ffmpeg 16k mono 转码) + 单测"
```

---

## Task 3: audio_utils 混音（mix_babble / mix_voice）（TDD）

**Files:**
- Modify: `code/demo_web/audio_utils.py`（追加 mix 函数）
- Modify: `code/demo_web/tests/test_audio_utils.py`（追加用例）

- [ ] **Step 1: 追加失败测试**

在 `test_audio_utils.py` 的 `if __name__` 块**之前**追加：
```python
def test_mix_babble_duration():
    with tempfile.TemporaryDirectory() as d:
        test = os.path.join(d, "t.wav"); out = os.path.join(d, "m.wav")
        _mkwav(test, sr=16000, dur=1.0)
        audio_utils.mix_babble(test, out, snr_db=0.0, babble_pool=d)  # 空 pool → 白噪 fallback
        assert os.path.exists(out)
        d_test = audio_utils.duration_s(test); d_out = audio_utils.duration_s(out)
        assert abs(d_test - d_out) < 0.05, (d_test, d_out)  # 时长不变
    print("test_mix_babble_duration OK")


def test_mix_voice_overlap_zero():
    # overlap_ratio=0 应等于 target(simulate_pipeline.mix_overlap 语义)
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "a.wav"); b = os.path.join(d, "b.wav"); out = os.path.join(d, "o.wav")
        _mkwav(a, freq=220); _mkwav(b, freq=440)
        audio_utils.mix_voice(a, b, out, overlap_ratio=0.0)
        ta, _ = sf.read(a); to_, _ = sf.read(out)
        assert len(ta) == len(to_)
        assert np.allclose(ta, to_, atol=1e-5), "overlap=0 应 == target"
    print("test_mix_voice_overlap_zero OK")
```

并把 `if __name__ == "__main__":` 块改为：
```python
if __name__ == "__main__":
    test_to_wav_16k_mono()
    test_mix_babble_duration()
    test_mix_voice_overlap_zero()
    print("ALL OK")
```

- [ ] **Step 2: 跑测试，确认新用例失败**

Run:
```bash
cd code && .venv/Scripts/python.exe demo_web/tests/test_audio_utils.py
```
Expected: FAIL（`AttributeError: module 'audio_utils' has no attribute 'mix_babble'`）。

- [ ] **Step 3: 实现 mix_babble / mix_voice**

在 `audio_utils.py` 末尾追加：
```python
def _load_mono(path, sr=16000):
    w, _ = librosa.load(path, sr=sr)
    return w.astype(np.float32)


def mix_babble(test_wav, out_wav, snr_db, babble_pool):
    """test + babble 噪声(从 babble_pool/*.wav 随机采一段, 不足 tile) @ snr_db, 写到 out_wav。

    babble_pool 为空目录 → 退化白噪 fallback。snr_db 越低越吵(-5 很吵, 10 轻微)。
    """
    from simulate_pipeline import add_noise  # lazy: 隔离重 import
    audio = _load_mono(test_wav)
    wavs = sorted(glob.glob(os.path.join(babble_pool, "*.wav")))
    if not wavs:
        noise = np.random.standard_normal(len(audio)).astype(np.float32)
    else:
        nw, _ = librosa.load(random.choice(wavs), sr=16000)
        if len(nw) < len(audio):
            nw = np.tile(nw, len(audio) // len(nw) + 1)
        noise = nw[:len(audio)].astype(np.float32)
    mixed = add_noise(audio, noise, snr_db)
    sf.write(out_wav, mixed.astype(np.float32), 16000)


def mix_voice(test_wav, interferer_wav, out_wav, overlap_ratio):
    """test + 第二人声重叠 @ overlap_ratio(0~1, 1.0=完全重叠), 写到 out_wav。"""
    from simulate_pipeline import mix_overlap  # lazy
    target = _load_mono(test_wav)
    interf = _load_mono(interferer_wav)
    mixed = mix_overlap(target, interf, overlap_ratio=overlap_ratio)
    sf.write(out_wav, mixed.astype(np.float32), 16000)
```

- [ ] **Step 4: 跑测试，确认全过**

Run:
```bash
cd code && .venv/Scripts/python.exe demo_web/tests/test_audio_utils.py
```
Expected: 三个 `OK` + `ALL OK`，exit 0。

- [ ] **Step 5: Commit**

```bash
git add code/demo_web/audio_utils.py code/demo_web/tests/test_audio_utils.py
git commit -m "feat(demo): audio_utils.mix_babble/mix_voice(复用 simulate_pipeline) + 单测"
```

---

## Task 4: inference_engine.py（常驻推理引擎）

> 此组件依赖大模型，无法单元 TDD。本 Task 完成**可 import 且逻辑正确**的实现，下个 Task 用 selfcheck 集成验证。代码严格复刻 `enroll_infer.py` 的 vanilla 单条流程（已逐行核对 137–368 行）。

**Files:**
- Create: `code/demo_web/inference_engine.py`

- [ ] **Step 1: 写完整实现**

写 `code/demo_web/inference_engine.py`：
```python
"""常驻推理引擎: 复刻 enroll_infer.py 的 vanilla 单条流程, 供 demo server 复用。

零修改 enroll_infer.py: import 复用其模块级工具(get_diarization_mask/collect_clean_audio)
+ text_utils(cut_target_timeline/to_simplified/digit_postproc) + repro。
顶部 inspect.getmodule patch + sys.path 注入原样照搬 enroll_infer(speechbrain lazy 固化点)。
"""
# ---- 顶部: speechbrain lazy patch(原样照搬 enroll_infer.py 24-29) ----
import inspect as _inspect
_orig_getmodule = _inspect.getmodule
def _safe_getmodule(*a, **k):
    try:
        return _orig_getmodule(*a, **k)
    except (ImportError, AttributeError):
        return None
_inspect.getmodule = _safe_getmodule

import os, sys, time, threading
import torch
import numpy as np
import librosa
import soundfile as sf
import pyarrow  # 预热: 避免 import pyannote 时扫 DiariZen 目录触发 WinError 6714(同 enroll_infer)
from transformers import AutoModelForSpeechSeq2Seq, AutoTokenizer, AutoFeatureExtractor

_HERE = os.path.dirname(os.path.abspath(__file__))   # demo_web/
_CODE = os.path.dirname(_HERE)                        # code/
_ROOT = os.path.dirname(_CODE)                        # 项目根
sys.path.insert(0, _CODE)                             # 让 enroll_infer/text_utils/repro 命中 code/
# DiCoW-inference / DiariZen / pyannote sys.path(同 enroll_infer, DiariZenPipeline 需要)
DICOW_INF = os.path.join(_CODE, "DiCoW-inference")
for _p in (DICOW_INF, os.path.join(DICOW_INF, "DiariZen"),
           os.path.join(DICOW_INF, "DiariZen", "pyannote-audio")):
    if os.path.isdir(_p):
        sys.path.insert(0, _p)

import enroll_infer  # 复用模块级工具
from text_utils import cut_target_timeline, to_simplified, digit_postproc
from repro import resolve_model, set_global_seed, reset_peak_gpu, peak_gpu_mib


class InferenceEngine:
    """启动 load_models() 一次; 之后 infer() 5-15s/条。线程安全(单 GPU 串行锁)。"""

    def __init__(self, device="cuda:0", reject_threshold=0.5,
                 vanilla_model=None, diarization_model=None, language="zh", seed=42):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16
        self.reject_threshold = reject_threshold
        self.language = language
        self.vanilla_model = vanilla_model or resolve_model("VANILLA")
        self.diarization_model = diarization_model or resolve_model("DIAR")
        self.seed = seed
        self.asr_model = None
        self.tok = None
        self.fe = None
        self.diar = None
        self._lock = threading.Lock()  # 单 GPU 串行

    def load_models(self):
        set_global_seed(self.seed)
        t0 = time.time()
        print(f"[demo-engine] loading vanilla Whisper {self.vanilla_model} on {self.device}")
        self.asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.vanilla_model, torch_dtype=self.dtype).to(self.device).eval()
        self.tok = AutoTokenizer.from_pretrained(self.vanilla_model)
        self.fe = AutoFeatureExtractor.from_pretrained(self.vanilla_model)
        print(f"[demo-engine] loading DiariZen {self.diarization_model}")
        from diarizen.pipelines.inference import DiariZenPipeline
        self.diar = DiariZenPipeline.from_pretrained(self.diarization_model).to(self.device)
        print(f"[demo-engine] models loaded in {time.time() - t0:.1f}s")

    def _get_emb(self, wav_np):
        """wespeaker embedding(复用 diar._embedding), L2 归一化。照搬 enroll_infer 162-171。"""
        w = torch.from_numpy(np.ascontiguousarray(wav_np.astype(np.float32))).to(self.device)
        if w.dim() == 1:
            w = w[None, None]
        elif w.dim() == 2:
            w = w[None]
        with torch.no_grad():
            emb = self.diar._embedding(w)
        emb = torch.as_tensor(emb, device=self.device, dtype=torch.float32)
        return torch.nn.functional.normalize(emb, dim=-1).squeeze(0)

    def infer(self, enroll_wav, rec_wav, target_out_path=None):
        """复刻 enroll_infer vanilla 单条流程。返回 dict。

        target_out_path: 给定时, 把切出的 target timeline 写到此 wav(供前端回放)。
        """
        with self._lock:  # 单 GPU 串行
            return self._infer_impl(enroll_wav, rec_wav, target_out_path)

    def _infer_impl(self, enroll_wav, rec_wav, target_out_path):
        reset_peak_gpu()
        t0 = time.time()
        try:
            # enrollment embedding
            enroll_audio, _ = librosa.load(enroll_wav, sr=16000)
            enroll_emb = self._get_emb(enroll_audio)
            # recognition
            audio, sr = librosa.load(rec_wav, sr=16000)
            dur = len(audio) / sr
            ifp = self.fe(audio, sampling_rate=16000, return_tensors="pt").input_features.to(self.device, self.dtype)
            audio_len = ifp.shape[-1] // 2
            # diar
            diar_out = self.diar(rec_wav)
            speakers = list(diar_out.labels())
            per_spk = [diar_out.label_timeline(s) for s in speakers]
            diar_mask = enroll_infer.get_diarization_mask(per_spk, audio_len)
            # 各 speaker 声纹(照搬 enroll_infer 237-246)
            spk_embs = []
            for i in range(len(speakers)):
                seg = enroll_infer.collect_clean_audio(audio, diar_mask, i, sr)
                if seg is None or len(seg) < sr * 0.3:
                    segs = [audio[int(s * sr):int(e * sr)] for s, e in per_spk[i]]
                    seg = np.concatenate(segs) if segs else np.zeros(sr, dtype=np.float32)
                min_len = sr * 1
                if len(seg) < min_len:
                    seg = np.tile(seg, min_len // len(seg) + 1)[:min_len]
                spk_embs.append(self._get_emb(seg))
            # 余弦匹配
            sims = torch.stack([torch.dot(enroll_emb, e) for e in spk_embs])
            target_idx = int(torch.argmax(sims))
            max_sim = float(sims[target_idx])
            rejected = max_sim < self.reject_threshold
            target_audio_path = None
            text = ""
            if rejected:
                verdict = f"REJECT(max_sim={max_sim:.3f}<{self.reject_threshold})"
            else:
                # vanilla: 切 target timeline → generate
                target_audio = cut_target_timeline(audio, per_spk[target_idx], sr=sr)
                if target_out_path:
                    os.makedirs(os.path.dirname(os.path.abspath(target_out_path)), exist_ok=True)
                    sf.write(target_out_path, target_audio.astype(np.float32), sr)
                    target_audio_path = target_out_path
                ifp_v = self.fe(target_audio, sampling_rate=16000, return_tensors="pt").input_features.to(self.device, self.dtype)
                am_v = torch.ones(1, ifp_v.shape[-1], dtype=torch.bool, device=self.device)
                with torch.no_grad():
                    out = self.asr_model.generate(input_features=ifp_v, attention_mask=am_v,
                                                  language=self.language, task="transcribe", max_new_tokens=200)
                seqs = out["sequences"] if isinstance(out, dict) else out
                text = self.tok.batch_decode(seqs, skip_special_tokens=True)[0].strip()
                text = digit_postproc(to_simplified(text))
                verdict = f"TRANSCRIBE(target={speakers[target_idx]}, sim={max_sim:.3f})"
            dt = time.time() - t0
            print(f"[demo-engine] {verdict} {len(text)}字 ({dt:.1f}s, RTF={dt/dur:.3f}): {text}")
            return {
                "transcript": text,
                "max_sim": max_sim,
                "rejected": rejected,
                "sims": {speakers[i]: float(sims[i]) for i in range(len(speakers))},
                "target_idx": target_idx,
                "target_speaker": speakers[target_idx],
                "infer_sec": round(dt, 3),
                "rtf": round(dt / dur, 3),
                "duration_s": round(dur, 2),
                "target_audio_path": target_audio_path,
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": f"{type(e).__name__}: {str(e)[:200]}",
                    "rejected": True, "transcript": "", "max_sim": 0.0, "sims": {}}
```

- [ ] **Step 2: 验证可 import（不加载模型，快速）**

Run:
```bash
cd code && source ../code/setenv.sh 2>/dev/null; .venv/Scripts/python.exe -c "from demo_web.inference_engine import InferenceEngine; e=InferenceEngine(); print('import OK', e.device, e.vanilla_model)"
```
Expected: `import OK cuda:0 E:/hf_cache/whisper-large-v3-turbo`（import 触发 enroll_infer/torch 加载，约 10–20s，但不加载 ASR/diar 模型本体）。

若报 `ImportError: DiariZen` 等，检查 `code/DiCoW-inference/DiariZen` 目录存在（enroll_infer 依赖它，已就绪则应通过）。

- [ ] **Step 3: Commit**

```bash
git add code/demo_web/inference_engine.py
git commit -m "feat(demo): InferenceEngine 常驻推理引擎(复刻 enroll_infer vanilla 流程, 零修改参赛链路)"
```

---

## Task 5: selfcheck.py（集成冒烟，验证 inference_engine）

**Files:**
- Create: `code/demo_web/selfcheck.py`

- [ ] **Step 1: 写 selfcheck**

写 `code/demo_web/selfcheck.py`：
```python
"""集成冒烟: datasetA/pos kws_0+cmd_0 跑 InferenceEngine.infer, 断言产物。

用法(必须 source setenv, 在 code/ 下跑):
  source code/setenv.sh
  cd code && .venv/Scripts/python.exe -m demo_web.selfcheck
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))  # code/

from inference_engine import InferenceEngine

ROOT = os.path.dirname(os.path.dirname(_HERE))   # 项目根
ENR = os.path.join(ROOT, "datasetA", "pos", "kws_0.wav")
REC = os.path.join(ROOT, "datasetA", "pos", "cmd_0.wav")
OUT_DIR = os.path.join(_HERE, "sessions", "selfcheck")
os.makedirs(OUT_DIR, exist_ok=True)
TARGET = os.path.join(OUT_DIR, "target.wav")


def main():
    assert os.path.exists(ENR), f"缺 enrollment seed: {ENR}"
    assert os.path.exists(REC), f"缺 recognition seed: {REC}"
    eng = InferenceEngine()
    eng.load_models()
    r = eng.infer(ENR, REC, target_out_path=TARGET)
    show = {k: v for k, v in r.items() if k != "sims"}
    print("result:", show)
    assert "error" not in r, f"infer 出错: {r.get('error')}"
    assert r["transcript"], "transcript 为空"
    assert r["max_sim"] > 0, "max_sim<=0"
    assert os.path.exists(TARGET), "target.wav 未生成"
    print(f"\nSELFCHECK OK  transcript={r['transcript']!r}  max_sim={r['max_sim']:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑 selfcheck（首次加载模型 ~40s + 推理 ~10s）**

Run:
```bash
cd E:/midea_target_asr
source code/setenv.sh
cd code
.venv/Scripts/python.exe -m demo_web.selfcheck
```
Expected: 日志 `[demo-engine] loading vanilla Whisper ...` → `models loaded in ~40s` → `[demo-engine] TRANSCRIBE(...) N字 (Xs, ...)` → `SELFCHECK OK transcript='...' max_sim=0.xxx`，exit 0。

若 `max_sim` 很低或 transcript 异常，对照直接跑 `enroll_infer.py --asr-backend vanilla --enrollment <ENR> --recognition <REC>` 的输出排查（确认是引擎复刻偏差还是 seed 本身问题）。

- [ ] **Step 3: Commit**

```bash
git add code/demo_web/selfcheck.py
git commit -m "test(demo): selfcheck 集成冒烟(kws_0+cmd_0 验证 inference_engine)"
```

---

## Task 6: server.py（FastAPI 后端）

**Files:**
- Create: `code/demo_web/server.py`

- [ ] **Step 1: 写 server（完整路由）**

写 `code/demo_web/server.py`：
```python
"""目标说话人 ASR demo web server (FastAPI)。

启动(在 code/ 下, 让 demo_web 作包 import):
  source code/setenv.sh
  cd code && .venv/Scripts/python.exe -m uvicorn demo_web.server:app --host 0.0.0.0 --port 7860
"""
import os, sys, uuid, asyncio, shutil, glob
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

_HERE = os.path.dirname(os.path.abspath(__file__))   # demo_web/
_CODE = os.path.dirname(_HERE)                        # code/
_ROOT = os.path.dirname(_CODE)                        # 项目根
for _p in (_HERE, _CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import audio_utils
from inference_engine import InferenceEngine

SESSIONS = os.path.join(_HERE, "sessions")
os.makedirs(SESSIONS, exist_ok=True)
STATIC = os.path.join(_HERE, "static")
os.makedirs(STATIC, exist_ok=True)

app = FastAPI(title="目标说话人 ASR Demo")
engine = InferenceEngine()


@app.on_event("startup")
async def _startup():
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg 不在 PATH。装到 E:\\Tools\\ffmpeg 并加入 PATH, 或确认现有 ffmpeg 在 PATH。")
    print("[demo-server] startup: 加载模型(约 40s)...")
    await asyncio.get_event_loop().run_in_executor(None, engine.load_models)
    print("[demo-server] ready, 监听 0.0.0.0:7860")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.post("/upload/enroll")
async def upload_enroll(file: UploadFile = File(...)):
    eid = "enroll_" + uuid.uuid4().hex[:12]
    dst = os.path.join(SESSIONS, eid + ".wav")
    tmp = dst + ".in"
    raw = await file.read()
    with open(tmp, "wb") as f:
        f.write(raw)
    try:
        audio_utils.to_wav_16k_mono(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    d = audio_utils.duration_s(dst)
    if d < 0.3:
        raise HTTPException(400, f"enrollment 过短 ({d:.2f}s < 0.3s)")
    return {"enroll_id": eid, "duration_s": round(d, 2), "audio_url": f"/audio/{eid}.wav"}


@app.post("/upload/test")
async def upload_test(file: UploadFile = File(...),
                      mix_mode: str = Form("none"),
                      snr_db: float = Form(0.0),
                      interferer_id: str = Form(""),
                      overlap_ratio: float = Form(0.8)):
    tid = "test_" + uuid.uuid4().hex[:12]
    clean = os.path.join(SESSIONS, tid + ".wav")
    mixed = os.path.join(SESSIONS, tid + "_mixed.wav")
    tmp = clean + ".in"
    raw = await file.read()
    with open(tmp, "wb") as f:
        f.write(raw)
    try:
        audio_utils.to_wav_16k_mono(tmp, clean)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    d = audio_utils.duration_s(clean)
    if d < 0.3:
        raise HTTPException(400, f"test 过短 ({d:.2f}s < 0.3s)")
    if mix_mode == "babble":
        pool = os.environ.get("DEMO_BABBLE_DIR", os.path.join(_ROOT, "datasetA", "neg"))
        audio_utils.mix_babble(clean, mixed, snr_db, babble_pool=pool)
    elif mix_mode == "voice":
        intf = os.path.join(SESSIONS, interferer_id + ".wav") if interferer_id else ""
        if not (intf and os.path.exists(intf)):
            raise HTTPException(400, "voice 模式需有效 interferer_id(用 /interferers 取)")
        audio_utils.mix_voice(clean, intf, mixed, overlap_ratio)
    else:  # none
        shutil.copy(clean, mixed)
    return {"test_id": tid, "clean_url": f"/audio/{tid}.wav",
            "mixed_url": f"/audio/{tid}_mixed.wav",
            "duration_s": round(d, 2), "mix_mode": mix_mode}


@app.get("/interferers")
async def interferers():
    neg_dir = os.path.join(_ROOT, "datasetA", "neg")
    out = []
    for w in sorted(glob.glob(os.path.join(neg_dir, "*.wav")))[:50]:
        out.append({"id": os.path.splitext(os.path.basename(w))[0],
                    "name": os.path.basename(w),
                    "duration_s": round(audio_utils.duration_s(w), 2)})
    return out


@app.post("/infer")
async def do_infer(body: dict):
    enroll_id = body.get("enroll_id", "")
    test_id = body.get("test_id", "")
    enr = os.path.join(SESSIONS, enroll_id + ".wav")
    rec = os.path.join(SESSIONS, test_id + "_mixed.wav")
    if not os.path.exists(enr):
        raise HTTPException(404, f"enrollment 不存在: {enroll_id}")
    if not os.path.exists(rec):
        raise HTTPException(404, f"test 不存在: {test_id}")
    infer_id = "inf_" + uuid.uuid4().hex[:12]
    target_out = os.path.join(SESSIONS, infer_id + "_target.wav")
    try:
        result = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None, lambda: engine.infer(enr, rec, target_out_path=target_out)),
            timeout=180)
    except asyncio.TimeoutError:
        raise HTTPException(504, "推理超时 (180s, 可能排队或模型慢)")
    result["target_audio_url"] = f"/audio/{infer_id}_target.wav" if result.get("target_audio_path") else None
    return result


@app.get("/audio/{name}")
async def get_audio(name: str):
    # 防路径穿越: 只取 basename
    name = os.path.basename(name)
    p = os.path.join(SESSIONS, name)
    if not os.path.exists(p):
        raise HTTPException(404, "audio not found")
    return FileResponse(p, media_type="audio/wav")
```

> **注意**：`upload_enroll` / `upload_test` 里 `await file.read()` 只调一次（`UploadFile.read()` 读完即空，重复调用得空 bytes）；转码前先落临时文件 `.in`，转码成功后删临时文件。

- [ ] **Step 2: 启动 server，验证 startup + 基础路由**

Run（分两个终端，或先跑后台）:
```bash
cd E:/midea_target_asr
source code/setenv.sh
cd code
.venv/Scripts/python.exe -m uvicorn demo_web.server:app --host 0.0.0.0 --port 7860
```
Expected: 启动日志加载模型 ~40s 后打印 `ready, 监听 0.0.0.0:7860`。

另开终端验证路由：
```bash
curl -s http://127.0.0.1:7860/interferers | head -c 200
```
Expected: JSON 列表 `[{"id":"cmd_1000","name":"cmd_1000.wav","duration_s":...}, ...]`。

```bash
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:7860/
```
Expected: `200`（返回 index.html，下个 Task 才有内容，但路由要通；若 404 说明 static/index.html 还没建，先跳过这条，Task 7 后再测）。

> `/upload/*` 和 `/infer` 涉及音频文件上传，用前端或 curl `-F` 测；本 Step 只验证 startup + interferers，完整链路 Task 8 前端后端到端测。

- [ ] **Step 3: Commit**

```bash
git add code/demo_web/server.py
git commit -m "feat(demo): FastAPI server(upload/test/infer/interferers/audio + startup 加载引擎)"
```

---

## Task 7: 前端（index.html + app.js + style.css）

**Files:**
- Create: `code/demo_web/static/index.html`
- Create: `code/demo_web/static/app.js`
- Create: `code/demo_web/static/style.css`

- [ ] **Step 1: 写 index.html**

写 `code/demo_web/static/index.html`：
```html
<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>目标说话人 ASR Demo</title>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
  <header>
    <h1>🎯 目标说话人 ASR 现场演示</h1>
    <p class="hint">朗读声纹注册 → 朗读测试(可混噪声/人声) → 现场推理。仅转写目标说话人, 拒识非目标。</p>
  </header>

  <main>
    <section class="card">
      <h2>① Enrollment(声纹注册)</h2>
      <p class="hint">朗读 1~3 秒, 比如唤醒词或报数字。</p>
      <div class="row">
        <button id="enrRec">● 录音</button>
        <button id="enrPlay" disabled>▶ 试听</button>
        <span id="enrStatus" class="status"></span>
      </div>
      <audio id="enrAudio" controls></audio>
    </section>

    <section class="card">
      <h2>② Test(测试音频)</h2>
      <p class="hint">朗读一句家居指令, 如「把空调调到 26 度」。</p>
      <div class="row">
        <button id="testRec">● 录音</button>
        <button id="testPlay" disabled>▶ 试听</button>
        <span id="testStatus" class="status"></span>
      </div>
      <audio id="testAudio" controls></audio>

      <div class="mix">
        <label><input type="radio" name="mix" value="none" checked> 不混(干净)</label>
        <label><input type="radio" name="mix" value="babble"> 混 babble 噪声</label>
        <label><input type="radio" name="mix" value="voice"> 混第二人声</label>
      </div>
      <div id="babbleCtrl" class="ctrl hidden">
        SNR: <input id="snr" type="range" min="-5" max="10" value="0" step="1">
        <span id="snrVal">0</span> dB(越低越吵)
      </div>
      <div id="voiceCtrl" class="ctrl hidden">
        干扰人声: <select id="interferer"><option value="">加载中...</option></select>
        重叠度: <input id="overlap" type="range" min="0" max="100" value="80" step="5">
        <span id="overlapVal">80</span>%
      </div>
    </section>

    <section class="card">
      <div class="row">
        <button id="infer" class="primary" disabled>🚀 开始推理</button>
        <span id="inferStatus" class="status"></span>
      </div>
      <div id="result" class="result hidden">
        <div id="badge" class="badge"></div>
        <div class="sim-bar"><span>声纹匹配分</span> <b id="simScore">-</b></div>
        <h3>转写</h3>
        <div id="transcript" class="transcript">-</div>
        <div id="simsList" class="sims"></div>
        <div id="timing" class="timing"></div>
        <h3>🎧 切出的目标音(模型只听到了这段)</h3>
        <audio id="targetAudio" controls></audio>
        <h3>混音后的 test(对照)</h3>
        <audio id="mixedAudio" controls></audio>
      </div>
    </section>
  </main>

  <script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: 写 app.js**

写 `code/demo_web/static/app.js`：
```javascript
// 录音 + 上传 + 混音 + 推理 + 回放
const $ = (id) => document.getElementById(id);

let enrBlob = null, testBlob = null;
let enrId = null, testId = null;
let mediaRec = null, chunks = [];

function pickMime() {
  const cands = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"];
  for (const c of cands) if (MediaRecorder.isTypeSupported(c)) return c;
  return "";
}

async function record(button, onDone) {
  if (button.dataset.rec === "1") {
    mediaRec && mediaRec.stop();
    return;
  }
  const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, sampleRate: 16000 } });
  chunks = [];
  mediaRec = new MediaRecorder(stream, pickMime() ? { mimeType: pickMime() } : undefined);
  mediaRec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  mediaRec.onstop = () => {
    stream.getTracks().forEach((t) => t.stop());
    button.dataset.rec = "0";
    button.textContent = button.dataset.label || "● 录音";
    button.classList.remove("recording");
    onDone(new Blob(chunks, { type: mediaRec.mimeType }));
  };
  mediaRec.start();
  button.dataset.rec = "1";
  button.textContent = "■ 停止";
  button.classList.add("recording");
}

function setBlob(blob, audioEl, playBtn, setter) {
  const url = URL.createObjectURL(blob);
  audioEl.src = url;
  playBtn.disabled = false;
  playBtn.onclick = () => audioEl.play();
  setter(blob);
}

$("enrRec").onclick = () =>
  record($("enrRec"), (b) => {
    setBlob(b, $("enrAudio"), $("enrPlay"), (x) => (enrBlob = x));
    $("enrStatus").textContent = `已录 ${(b.size / 1024).toFixed(0)} KB, 待上传`;
    maybeEnable();
  });

$("testRec").onclick = () =>
  record($("testRec"), (b) => {
    setBlob(b, $("testAudio"), $("testPlay"), (x) => (testBlob = x));
    $("testStatus").textContent = `已录 ${(b.size / 1024).toFixed(0)} KB, 待上传`;
    maybeEnable();
  });

// 混音控件联动
document.querySelectorAll("input[name=mix]").forEach((r) => r.addEventListener("change", renderMix));
function mixMode() {
  return document.querySelector("input[name=mix]:checked").value;
}
function renderMix() {
  const m = mixMode();
  $("babbleCtrl").classList.toggle("hidden", m !== "babble");
  $("voiceCtrl").classList.toggle("hidden", m !== "voice");
}
$("snr").oninput = (e) => ($("snrVal").textContent = e.target.value);
$("overlap").oninput = (e) => ($("overlapVal").textContent = e.target.value);

// 加载干扰人声列表
fetch("/interferers").then((r) => r.json()).then((list) => {
  const sel = $("interferer");
  sel.innerHTML = "";
  if (!list.length) { sel.innerHTML = "<option value=''>无可用</option>"; return; }
  list.forEach((it) => {
    const o = document.createElement("option");
    o.value = it.id; o.textContent = `${it.name} (${it.duration_s}s)`;
    sel.appendChild(o);
  });
}).catch(() => { $("interferer").innerHTML = "<option value=''>加载失败</option>"; });

function maybeEnable() {
  $("infer").disabled = !(enrBlob && testBlob);
}

async function uploadEnroll() {
  const fd = new FormData();
  fd.append("file", enrBlob, "enroll.webm");
  const r = await fetch("/upload/enroll", { method: "POST", body: fd });
  if (!r.ok) throw new Error("enroll 上传失败: " + (await r.text()));
  const j = await r.json();
  enrId = j.enroll_id;
  $("enrStatus").textContent = `已上传 ${j.duration_s}s`;
}

async function uploadTest() {
  const fd = new FormData();
  fd.append("file", testBlob, "test.webm");
  fd.append("mix_mode", mixMode());
  fd.append("snr_db", $("snr").value);
  fd.append("overlap_ratio", $("overlap").value / 100);
  fd.append("interferer_id", $("interferer").value);
  const r = await fetch("/upload/test", { method: "POST", body: fd });
  if (!r.ok) throw new Error("test 上传失败: " + (await r.text()));
  const j = await r.json();
  testId = j.test_id;
  $("testStatus").textContent = `已上传 ${j.duration_s}s [${j.mix_mode}]`;
  $("mixedAudio").src = j.mixed_url;
  return j;
}

$("infer").onclick = async () => {
  const btn = $("infer"); const st = $("inferStatus");
  btn.disabled = true; st.textContent = "上传 enrollment...";
  $("result").classList.add("hidden");
  try {
    if (!enrId) await uploadEnroll();
    st.textContent = "上传 test + 混音...";
    if (!testId) await uploadTest();
    st.textContent = "推理中(5~15s)...";
    const r = await fetch("/infer", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enroll_id: enrId, test_id: testId }),
    });
    if (!r.ok) throw new Error("推理失败: " + (await r.text()));
    const j = await r.json();
    renderResult(j);
    st.textContent = "完成";
  } catch (e) {
    st.textContent = "❌ " + e.message;
  } finally {
    btn.disabled = false;
  }
};

function renderResult(j) {
  $("result").classList.remove("hidden");
  if (j.error) {
    $("badge").className = "badge err"; $("badge").textContent = "出错: " + j.error;
    $("transcript").textContent = "-";
    return;
  }
  const rej = j.rejected;
  const badge = $("badge");
  badge.className = "badge " + (rej ? "rej" : "ok");
  badge.textContent = rej ? "🚫 拒识: 目标说话人不在场" : "✅ 接受: 已转写目标";
  $("simScore").textContent = j.max_sim.toFixed(3);
  $("transcript").textContent = rej ? "(拒识, 无转写)" : j.transcript;
  const sims = Object.entries(j.sims || {}).map(([k, v]) =>
    `<span class="sim-pill ${k === j.target_speaker ? "tgt" : ""}">${k}: ${v.toFixed(3)}${k === j.target_speaker ? " ★" : ""}</span>`).join("");
  $("simsList").innerHTML = sims;
  $("timing").textContent = `推理 ${j.infer_sec}s | 音频 ${j.duration_s}s | RTF ${j.rtf}`;
  if (j.target_audio_url) $("targetAudio").src = j.target_audio_url;
}
```

- [ ] **Step 3: 写 style.css**

写 `code/demo_web/static/style.css`：
```css
* { box-sizing: border-box; }
body {
  margin: 0; font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
  background: #0f172a; color: #e2e8f0; line-height: 1.5;
}
header { background: #1e293b; padding: 20px; text-align: center; border-bottom: 1px solid #334155; }
header h1 { margin: 0 0 6px; font-size: 22px; }
.hint { color: #94a3b8; font-size: 13px; }
main { max-width: 760px; margin: 0 auto; padding: 16px; }
.card {
  background: #1e293b; border: 1px solid #334155; border-radius: 12px;
  padding: 18px; margin-bottom: 16px;
}
.card h2 { margin: 0 0 8px; font-size: 17px; }
.card h3 { margin: 14px 0 6px; font-size: 14px; color: #cbd5e1; }
.row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
button {
  background: #334155; color: #e2e8f0; border: 0; border-radius: 8px;
  padding: 9px 16px; font-size: 14px; cursor: pointer;
}
button:hover:not(:disabled) { background: #475569; }
button:disabled { opacity: 0.4; cursor: not-allowed; }
button.primary { background: #2563eb; }
button.primary:hover:not(:disabled) { background: #3b82f6; }
button.recording { background: #dc2626; animation: pulse 1s infinite; }
@keyframes pulse { 50% { opacity: 0.6; } }
.status { color: #94a3b8; font-size: 13px; }
audio { width: 100%; margin-top: 8px; }
.mix { display: flex; gap: 14px; flex-wrap: wrap; margin: 12px 0 8px; font-size: 14px; }
.mix label { cursor: pointer; }
.ctrl { margin: 8px 0; font-size: 14px; color: #cbd5e1; }
.ctrl input[type=range] { vertical-align: middle; }
.hidden { display: none; }
.result .badge { display: inline-block; padding: 8px 14px; border-radius: 8px; font-weight: bold; margin-bottom: 10px; }
.badge.ok { background: #16a34a; }
.badge.rej { background: #dc2626; }
.badge.err { background: #b45309; }
.sim-bar { margin: 8px 0; }
.transcript {
  font-size: 22px; font-weight: 600; background: #0f172a; border-radius: 8px;
  padding: 14px; min-height: 28px; border-left: 4px solid #2563eb;
}
.sims { margin-top: 8px; display: flex; gap: 8px; flex-wrap: wrap; }
.sim-pill { background: #334155; padding: 4px 10px; border-radius: 6px; font-size: 13px; }
.sim-pill.tgt { background: #2563eb; }
.timing { color: #94a3b8; font-size: 13px; margin-top: 6px; }
```

- [ ] **Step 4: 端到端手动验证（server 需在跑）**


确保 server 在跑（Task 6 Step 2 的 uvicorn）。浏览器打开 `http://127.0.0.1:7860/`：
- 点「录音」录 1~2s enrollment → 停 → 试听 OK。
- 点「录音」录一句 test → 停。
- 选「混 babble 噪声」SNR=0 → 点推理 → 等 10~20s → 看到转写文本 + 匹配分 + 徽章 + 切出的目标音可播放 + 混音 test 可播放。
- 再选「混第二人声」重叠 80% → 推理 → 看 target 音是否只剩目标的话。
- 换一个人录 test（不同声纹）→ 推理 → 看拒识徽章。

Expected: 全部交互正常，结果区正确渲染。

- [ ] **Step 5: Commit**

```bash
git add code/demo_web/static
git commit -m "feat(demo): 前端单页(录音/上传/混音控件/推理结果/音频回放)"
```

---

## Task 8: run_demo.sh + README.md

**Files:**
- Create: `code/demo_web/run_demo.sh`
- Create: `code/demo_web/README.md`

- [ ] **Step 1: 写 run_demo.sh**

写 `code/demo_web/run_demo.sh`：
```bash
#!/usr/bin/env bash
# demo 网站启动器: source setenv + uvicorn 起在 0.0.0.0:7860
# 公网: 另起 cloudflared quick tunnel(见 README), 把 https://*.trycloudflare.com 发给访客
set -euo pipefail
cd "$(dirname "$0")/.." || { echo "cd code/ 失败"; exit 1; }   # 进 code/
source ../code/setenv.sh
export HF_HUB_OFFLINE=1   # 模型已缓存, 离线跑避免每次校验网络
echo "[demo] 启动 uvicorn 0.0.0.0:7860 (首次加载模型约 40s)..."
exec .venv/Scripts/python.exe -m uvicorn demo_web.server:app --host 0.0.0.0 --port 7860
```

- [ ] **Step 2: 写 README.md**

写 `code/demo_web/README.md`：
```markdown
# 目标说话人 ASR Demo 网站

访客浏览器录音 → 本机常驻 vanilla ASR 引擎推理 → 回显转写/匹配分/拒识/切出的目标音。
支持在线混 babble 噪声 / 第二人声, 现场演示「只转目标、抗干扰」。

## 1. 依赖(已就绪, 列此供复核)

- `code/.venv`: fastapi 0.138 + uvicorn 0.49 + python-multipart(已装; 若缺 `pip install python-multipart`)
- ffmpeg 在 PATH(`C:/ProgramData/ffmpeg-7.1.1-essentials_build/bin`)
- 模型缓存: `E:/hf_cache` 下 whisper-large-v3-turbo / diarizen-wavlm-large-s80-md(setenv 指定)

## 2. 本机启动

```bash
cd E:/midea_target_asr
bash code/demo_web/run_demo.sh
```
日志打印 `[demo-server] ready` 后, 浏览器开 `http://127.0.0.1:7860/`。

## 3. 集成自检(怀疑引擎异常时跑)

```bash
cd E:/midea_target_asr
source code/setenv.sh
cd code
.venv/Scripts/python.exe -m demo_web.selfcheck
```
应打印 `SELFCHECK OK transcript=... max_sim=...`。

## 4. 公网分享(cloudflared quick tunnel)

server 跑起来后, 另开终端:
```bash
cloudflared tunnel --url http://localhost:7860
```
(若没装 cloudflared, 见 exposing-local-server skill; 或 `winget install --id Cloudflare.cloudflared`)
输出里取 `https://<随机>.trycloudflare.com`, 发给访客。访客手机/电脑浏览器打开即可录音推理。

⚠️ 无鉴权, 用完 Ctrl+C 关 cloudflared 和 server。session 音频在 `sessions/`, 清理: `rm code/demo_web/sessions/*.wav`。

## 5. 预期延迟

- 服务启动加载模型 ~40s(一次性)
- 之后每条推理 5~15s(4060, batch=1) + 公网上传延迟, 端到端 ~10~20s
- 单 GPU 同时只跑一条, 多人访问排队

## 6. 演示脚本建议

1. 干净 test → 推理 → 看正常转写。
2. 同一段 test 混 babble SNR=0 → 推理 → 看仍能转 + 点「切出的目标音」听效果。
3. 混第二人声 80% 重叠 → 推理 → 听 target 只剩目标的话。
4. 换人录 test → 拒识徽章。
```

- [ ] **Step 3: chmod + 本机跑一次 run_demo.sh 验证**

Run:
```bash
cd E:/midea_target_asr
bash code/demo_web/run_demo.sh
```
Expected: 同 Task 6 Step 2, 启动到 `ready`。(Ctrl+C 停)

- [ ] **Step 4: Commit**

```bash
git add code/demo_web/run_demo.sh code/demo_web/README.md
git commit -m "feat(demo): run_demo.sh 启动器 + README(本机/自检/公网/延迟说明)"
```

---

## Task 9: 公网端到端验收（cloudflared）

**Files:** 无（验收步骤）

- [ ] **Step 1: 起 server + cloudflared**

终端 1：`bash code/demo_web/run_demo.sh`（等到 `ready`）
终端 2：
```bash
cloudflared tunnel --url http://localhost:7860
```
取输出的 `https://<rand>.trycloudflare.com`。

- [ ] **Step 2: 手机/另一设备实测**

手机连 4G(非本机网络), 浏览器开公网链接, 重复 Task 7 Step 4 的四步演示:
- [ ] enrollment 录音上传 OK
- [ ] test 录音上传 OK
- [ ] 干净/babble/第二人声 三种 mix 各推理一次, 结果正常
- [ ] target 音 + 混音 test 可播放
- [ ] 换人录 test 触发拒识徽章

- [ ] **Step 3: 验收清单回填 spec**

打开 `docs/superpowers/specs/2026-07-12-asr-demo-web-design.md` 第 10 节, 把 7 个 `[ ]` 全勾上, commit:
```bash
git add docs/superpowers/specs/2026-07-12-asr-demo-web-design.md
git commit -m "docs(demo): 验收清单全部通过"
```

---

## 验收标准（对应 spec §10）

- [ ] `bash code/demo_web/run_demo.sh` 启动到 `ready`, 监听 7860
- [ ] `selfcheck.py` 用 kws_0+cmd_0 返回非空 transcript + max_sim>0 + target.wav 存在
- [ ] Chrome + Safari 录音 enrollment + test 上传成功, 可回放
- [ ] mix_mode = none/babble/voice 三种, 混音 test 可回放且与 clean 对比明显
- [ ] `/infer` 返回转写 + 徽章 + target 切片可播放
- [ ] cloudflared 公网链接在手机浏览器可录音 + 推理 + 出结果
- [ ] 换人录 test 触发拒识徽章
