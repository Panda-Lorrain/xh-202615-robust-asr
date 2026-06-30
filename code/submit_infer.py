"""标准化推理脚本: enrollment + recognition -> result.json + timing.json

纯 stdlib subprocess 顶层编排器,零侵入复用 noise_classify / se_denoise /
enroll_infer / llm_reject。详见 spec:
docs/superpowers/specs/2026-07-01-submit-infer-and-deliverables-design.md
"""
import os
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


def main():
    # 占位,Task 6 实现
    print("[submit_infer] skeleton only — see Task 6 for full pipeline")


if __name__ == "__main__":
    main()
