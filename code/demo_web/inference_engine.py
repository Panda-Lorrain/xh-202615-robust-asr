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
