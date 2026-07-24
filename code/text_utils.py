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
    # 时间小数归一(2026-07-15): 14.55→十四点五十五, 11.31→十一点三十一 (cmd_322/337 类,
    # vanilla 输出阿拉伯数字+小数, ref 中文时间)。须在纯数字归一前, 否则 14/55 已被拆转。
    # 消除幻觉检测 cn<0.3 假阳性 + 对齐 ref 口径降 CER。
    def _time_dec(m):
        h, rest = m.group(1), m.group(2)
        rest_cn = ("零" + cn2an.an2cn(rest[1])) if (len(rest) == 2 and rest[0] == "0") else cn2an.an2cn(rest)
        return f"{cn2an.an2cn(h)}点{rest_cn}"
    text = re.sub(r"(?<!\d)(\d{1,2})\.(\d{1,2})(?!\d)", _time_dec, text)
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
    # 2026-07-15 全量死区条对抗审查扩展(workflow B 财经词典): 死区新闻话 36 条 est 抓 28
    "经济", "企业", "成本", "营收", "同比", "上涨", "下降", "合同", "签订", "赔偿",
    "机构", "半导体", "生产", "奥运", "微软", "三星", "克林顿", "上市", "分成", "冲击",
    "制造", "围观", "市民", "推广", "财经", "播报", "韩国", "首尔", "旅游",
    # 2026-07-18 扩词(verify_content_gate_joint gate on 后剩30漏拒内容坐实, pos ref 0命中安全):
    # 信访(5018)/班级·培训班(5177)/卖家·协商(5084)/房屋所有权(5184)/死亡·获赔(5261)/租赁·物业(1059)
    # 拒收约5-7条漏拒. ⚠️不收"机器人"(边界:家用机器人5192真指令 vs 智能服务机器人5080新闻)
    "信访", "班级", "培训班", "卖家", "协商", "房屋所有权", "死亡", "获赔", "租赁", "物业",
]


def _max_char_run(t):
    """最长连续相同字符(循环幻觉'手手骨骨'→大)。幻觉检测用。"""
    if not t:
        return 0
    mx = cur = 1
    for i in range(1, len(t)):
        cur = cur + 1 if t[i] == t[i - 1] else 1
        mx = max(mx, cur)
    return mx


def _char_diversity(t):
    """unique chars / len(重复乱码→低)。幻觉检测用。"""
    return len(set(t)) / max(len(t), 1)


def is_valid_command(text, len_thr=22):
    """转写内容有效性校验(content_gate, 2026-07-08): True=像有效家居指令(保留), False=强非指令信号(拒)。

    用途: submit_infer.decide_reject 的独立加拒通道 —— 对 sim≥thr 的 accept 再判转写内容,
    拒掉新闻/英文/乱码非目标干扰(提 RR), pos 侧顺带拒幻觉灾难(降 CER)。Pareto 改进, 不损效率(纯函数)。

    hold-out 泛化验证(code/exp_content_gate_holdout.py, 2026-07-08, 回应过拟合担忧):
    A 集分 train/val 多 seed(10 划分全正), len_thr=22(val 占优甜点近先验, pos 误拒更少), val ΔTS 详见 hold-out 产物,
    bootstrap CI p5=+0.007 稳赚; L 不敏感(18-30 全正); pos 误拒原 CER mean 0.98(CER≥1 占~89% 反赚)。
    默认 True 保留(宁放过不误拒 pos), 仅强非指令信号才 False。
    无外部依赖(纯中文范围判断), 与 text_utils 模块风格一致。
    """
    if not text or not text.strip():
        return False
    # 去标点(对齐官方 normalize, 避免 qwen 标点"。"",""致 len>len_thr 误判超长; vanilla 无标点 no-op)
    text = re.sub(r"[^\w一-鿿]", "", text)
    if not text:
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
    # 2026-07-15 幻觉检测集成(全量死区条确认 CER Δ-0.093 零误拒, baseline 信号)
    if _max_char_run(text) >= 4:                    # 单字循环幻觉(手手骨骨骨, 死区循环特征)
        return False
    if len(text) >= 10 and _char_diversity(text) < 0.35:  # 字符多样性极低(重复乱码)
        return False
    return True


# Bag-of-Hallucinations: Whisper 训练数据水印/字幕签名(先验, 非 A 集拟合 —— 家居指令绝不出现)
# 保守收录: 仅 unmistakably 非家居指令的签名(频道/换台等可能进真实指令, 不收)。
_BAG_OF_HALLUCINATIONS = [
    # YouTube/视频字幕水印(Whisper 训练集字幕残留, 重 babble 下高频幻觉)
    "独播剧场", "YoYo", "Television", "Series Exclusive",
    # 字幕致谢/署名
    "字幕志愿者", "字幕由",
]


def bag_of_hallucinations_reject(text, boh_dict=None, min_repeat=3,
                                  unit_min=2, unit_max=6, min_len=4):
    """babble 循环幻觉 + 训练水印签名检测(独立加拒通道, 2026-07-10 POC)。

    返回 True=判定为幻觉→拒; False=未命中(放行给后续 gate)。
    设计为叠加在 content_gate 之上的独立通道, 不改 sim/llm 逻辑。

    两类检测:
    ① delooping: 同一 2-6 字片段连续重复 ≥min_repeat 次(如 帮×60 / 口吐×30 / 订阅×20)。
       这是重 babble 下 Whisper 的标志性循环幻觉, CER 可飙到 7-25。单字重复≥6 次也被
       覆盖(`.{2,6}` 可取同字对作 unit)。正确转写(CER<0.5)上 0 误命中(682 条验证)。
    ② BoH 字典: Whisper 训练集水印/字幕签名(优优独播剧场/YoYo Television/字幕志愿者),
       先验知识(非 A 集拟合), 家居指令绝不出现。

    hold-out 验证(code/exp_boh_delooping.py, 2026-07-10, 回应过拟合 + 边际收益担忧):
    delooping 在 A 集 pos 抓 7 条极端 CER(1.5-25.25, mean 16.0), 但 **100% 已被
    content_gate 捕获**(6 条 len>22 + 1 条纯数字非中文 "232323") → 叠加 content_gate 之上
    pos ΔCER≈0 / neg ΔRR≈0(漏拒 neg 为空/新闻/类指令, 无循环)。
    => NO-GO 作主 gate(与 content_gate 冗余, content_gate 的 len>22 是 delooping 的超集);
       保留价值 = content_gate len_thr 放宽时的 0-FP 安全网(defense-in-depth)。

    无外部依赖(纯正则 + 关键词), 与 text_utils 模块风格一致。
    """
    if not text or not text.strip():
        return False                                   # 空文本交给 content_gate, BoH 只管幻觉文本
    # ① delooping: 片段循环(unit 重复 min_repeat 次及以上)
    if len(text) >= min_len:
        pat = r'(.{' + str(unit_min) + ',' + str(unit_max) + r'})\1{' + str(min_repeat - 1) + r',}'
        if re.search(pat, text):
            return True
    # ② BoH 字典(训练水印/字幕签名)
    dic = boh_dict if boh_dict is not None else _BAG_OF_HALLUCINATIONS
    if any(w in text for w in dic):
        return True
    return False


# 美的/COLMO 产品功能名锚点(公开领域知识, 非 A 集拟合)。
# 排除"洗衣机筒"(桶/筒 ref 冲突: cmd_117 ref=筒 vs cmd_157 ref=桶, 强替必恶其一)。
_BRAND_ANCHORS = [
    "AI净干洗", "AI轻干洗", "一键净呼吸",
    "轻干洗", "净干洗", "净呼吸",
    "智控温", "智清洁", "防直吹", "无风感", "柔风", "星香",
]


def brand_homophone_fix(text):
    """美的/COLMO 功能名同音字修复(领域知识锚点, 零回退, 零RTF)。

    文本中功能名的"恰好1字同音不同"错误形式 → 修复为正确功能名。
    零回退: 仅当窗口长度==锚点 且 恰好1字不同 且 该字严格同音(TONE3 含声调) 才改,
    只替换单字不增删不重排 → 最坏=原文(不会恶化)。

    锚点=美的公开产品功能名(领域知识, 非 A 集统计, 合规可提交 B 集);
    同音=pypinyin TONE3。来源 runs/_rule_full.py 的 E 方案, 去 A 集依赖版。

    实测: 全量 1350 ΔCER -0.001 改 10 恶 0(噪声内但确定净正 + 零回退 + 零 RTF)。
    作防御性后处理: 保证智控温/轻干洗/净呼吸/防直吹等功能名不被同音字写错。
    依赖 pypinyin(主 venv 已装), 未装 graceful 跳过返回原文(不破坏流程)。
    """
    if not text:
        return text
    try:
        from pypinyin import pinyin, Style
    except ImportError:
        import warnings
        warnings.warn("pypinyin 未装: brand_homophone_fix 跳过(功能名同音字不修复)",
                      RuntimeWarning, stacklevel=2)
        return text

    def _homo(a, b):
        if len(a) != 1 or len(b) != 1:
            return False
        pa = pinyin(a, style=Style.TONE3, errors='ignore')
        pb = pinyin(b, style=Style.TONE3, errors='ignore')
        return bool(pa and pb and pa[0] and pb[0] and pa[0][0] == pb[0][0])

    chars = list(text)
    for anchor in sorted(_BRAND_ANCHORS, key=len, reverse=True):  # 长优先: AI净干洗 先于 净干洗
        L = len(anchor)
        a_chars = list(anchor)
        i = 0
        while i + L <= len(chars):
            if chars[i:i + L] == a_chars:
                i += 1
                continue
            diffs = [(j, chars[i + j], anchor[j]) for j in range(L) if chars[i + j] != anchor[j]]
            if len(diffs) == 1 and _homo(diffs[0][1], diffs[0][2]):
                chars[i + diffs[0][0]] = diffs[0][2]   # 仅替换单字(零回退)
            i += 1
    return ''.join(chars)
