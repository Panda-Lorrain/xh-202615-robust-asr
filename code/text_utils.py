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
