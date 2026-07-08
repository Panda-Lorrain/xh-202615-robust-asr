"""转写文本工具: 繁简归一 + 数字后处理 + target timeline 切割(纯函数, enroll_infer 与 to_submission 共用)。"""
import re
import numpy as np


def to_simplified(text):
    """繁→简归一(zhconv)。空值直通。转写后归一 + submission 兜底都用。
    zhconv 缺失 graceful 跳过+告警(与 eval_datasetA._norm_zh 失败模式一致, 避免裸崩阻塞整批推理)。"""
    if not text:
        return text
    try:
        import zhconv
    except ImportError:
        import warnings
        warnings.warn(
            "zhconv 未装: to_simplified 跳过(繁体不转简), 官方CER口径下 CER 虚高。"
            "装: uv pip install -r code/requirements.txt",
            RuntimeWarning, stacklevel=2)
        return text
    return zhconv.convert(text, "zh-cn")


def digit_postproc(text):
    """阿拉伯→中文数字后处理, 对齐 ref 中文数字口径(从 MiMo-V2.5-ASR 对比学到的 quick win)。

    实测: vanilla 含数字句 CER 0.739→0.608(-0.13), 全量 0.661→0.632(-0.029), 121/315 条改善。
    稳健: 主办方若归一化数字(25==二十八)则无收益但无害; 若 ref 中文数字直接比则赚 0.029。对冲 caliber-A 风险。
    规则: 百分比 X%→百分之X; 纯数字 1-3 位(0-999, 覆盖温度/时间/百分比/风量)→中文; ≥4 位多为幻觉串(如916213)保留原样。
    依赖: cn2an (uv pip install cn2an)。未装则 graceful 跳过返回原文, 不破坏流程。
    """
    if not text:
        return text
    try:
        import cn2an
    except ImportError:
        # 2026-07-08: 静默跳过→告警。官方CER口径不归一数字, cn2an 缺则阿拉伯数字直进提交,
        # 含数字句 CER 虚高(~0.03 全量, 含数字句更甚)。cn2an 已声明 code/requirements.txt。
        import warnings
        warnings.warn(
            "cn2an 未装: digit_postproc 跳过(阿拉伯数字不转中文), 官方CER口径下 CER 虚高。"
            "装: uv pip install -r code/requirements.txt",
            RuntimeWarning, stacklevel=2)
        return text
    text = re.sub(r"(\d+)%", lambda m: "百分之" + cn2an.an2cn(m.group(1)), text)
    # (?<!\d) 前置: 确保≥4位幻觉串(如916213)整串不匹配, 只转独立的1-3位数字(0-999)
    text = re.sub(r"(?<!\d)\d{1,3}(?!\d)", lambda m: cn2an.an2cn(m.group(0)), text)
    return text


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


# 通用非家居类目词(先验, 非 A 集拟合): 新闻/财经/体育/娱乐/公司 —— 家居指令不会出现
_CONTENT_GATE_NEWS_BLACK = [
    "产业", "资本", "投资", "制度", "政府", "债务", "市场", "调研", "报告", "期货", "股票",
    "基金", "贷款", "住房", "房地产", "报道", "新闻", "记者", "日前", "发布", "价格", "广告", "拍摄",
    # 繁体新闻/财经(转写保留繁体时命中)
    "期貨", "報告", "市場", "調研", "調查", "顯示", "股份", "有限公司", "落戶", "服務",
    # 体育/娱乐/其他
    "四強", "席位", "聚杯", "婚姻", "导演", "考试", "生意", "无法阻挡",
]


def is_valid_command(text, len_thr=20):
    """转写内容有效性校验(content_gate, 2026-07-08): True=像有效家居指令(保留), False=强非指令信号(拒)。

    用途: submit_infer.decide_reject 的独立加拒通道 —— 对 sim≥thr 的 accept 再判转写内容,
    拒掉新闻/英文/乱码非目标干扰(提 RR), pos 侧顺带拒幻觉灾难(降 CER)。Pareto 改进, 不损效率(纯函数)。

    hold-out 泛化验证(code/exp_content_gate_holdout.py, 2026-07-08, 回应过拟合担忧):
    A 集分 train/val, len_thr=20(纯先验零 train 拟合), val ΔTotalScore +0.0134(+1.6 分/80 满分),
    bootstrap CI p5=+0.007 稳赚; L 不敏感(18-30 全正); pos 误拒原 CER mean 0.98(CER≥1 占~89% 反赚)。
    默认 True 保留(宁放过不误拒 pos), 仅强非指令信号才 False。
    无外部依赖(纯中文范围判断), 与 text_utils 模块风格一致。
    """
    if not text or not text.strip():
        return False
    nch = sum(1 for c in text if "一" <= c <= "鿿")
    if nch == 0:                                    # 纯非中文(ok/tooling, 家居指令无此情况)
        return False
    if len(text) >= 3 and nch < len(text) * 0.5:   # 英文为主(productive/i can't go)
        return False
    if any(w in text for w in _CONTENT_GATE_NEWS_BLACK):  # 通用非家居类目词
        return False
    if len(text) > len_thr:                         # 超长叙述(家居指令极少>20 字)
        return False
    return True
