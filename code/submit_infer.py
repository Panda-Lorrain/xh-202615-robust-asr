"""标准化推理脚本: enrollment + recognition -> result.json + timing.json

纯 stdlib subprocess 顶层编排器,零侵入复用 noise_classify / se_denoise /
enroll_infer / llm_reject。详见 spec:
docs/superpowers/specs/2026-07-01-submit-infer-and-deliverables-design.md
"""
import os
import json
import glob
import wave
import contextlib

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


def load_pairs(pairs_json):
    """读 --pairs manifest: [{enrollment, recognition}, ...] -> [(enr, rec), ...]。"""
    rows = json.load(open(pairs_json, encoding="utf-8"))
    return [(r["enrollment"], r["recognition"]) for r in rows]


def expand_inputs(args):
    """把 CLI 输入展开为 [(enrollment, recognition), ...] 统一列表。"""
    if args.pairs:
        return load_pairs(args.pairs)
    recs = sorted(glob.glob(os.path.join(args.recognition_folder, "*.wav")))
    return [(args.enrollment, r) for r in recs]


def decide_reject(max_sim, llm_verdict, strategy, sim_thr, use_llm):
    """融合拒识决策。返回 True=拒识。
    - 无 LLM 或 sim_only: 拒 iff max_sim < sim_thr
    - llm_only:           拒 iff llm != accept
    - llm_or_sim(默认):   拒 iff (llm != accept) AND (max_sim < sim_thr)
    """
    if not use_llm or strategy == "sim_only":
        return max_sim < sim_thr
    if strategy == "llm_only":
        return llm_verdict != "accept"
    return llm_verdict != "accept" and max_sim < sim_thr


def main():
    # 占位,Task 6 实现
    print("[submit_infer] skeleton only — see Task 6 for full pipeline")


if __name__ == "__main__":
    main()
