"""W6 评测脚本：CER（中文字符级）+ RTF + 拒识指标。真实数据来时直接复用。
- cer(hyp, ref): 字符级错误率（中英文通用）
- compute_rtf(infer_s, audio_s): 实时因子
- rejection_metrics(tp,fp,tn,fn): 拒识精确率/召回/F1
- batch_cer(hyp_dir, ref_dir): 批量目录配对算 CER
"""
import os, glob, jiwer


def cer(hyp: str, ref: str) -> float:
    """字符级错误率 CER = (S+I+D)/N，标准 Levenshtein 编辑距离 / ref 字符数。

    口径对齐(2026-07-08 核实, 见 memory official-scoring-spec):
    - 主办方参考 editdistance 库; 本实现用 jiwer.wer, 已验证两库在 7 个边界用例
      (完美/1字替换/全错/空hyp拒识/插入>1/繁体/数字) 逐条 delta=0.00e+00, 算法层完全等价。
    - 与标准公式一致: Hyp 空(拒识)→全删→CER=1.0; 插入多→可>1(不封顶); 分母 N=ref 字符数。
    - ⚠️ 预处理口径(繁简/数字/标点/英文混排)待主办方确认, 比算法对齐影响更大:
        * 繁体 vs 简体 ref 每字算替换(CER 虚高 ~0.5) → 提交侧 to_simplified 已强转简体
        * "26度" vs "二十六度" 3 编辑 CER 0.75 → digit_postproc 把 hyp 阿拉伯转中文对齐 ref
    """
    h = " ".join(list(hyp.replace(" ", "").replace("\n", "").lower()))
    r = " ".join(list(ref.replace(" ", "").replace("\n", "").lower()))
    if len(r.replace(" ", "")) == 0:
        return 0.0 if len(h.replace(" ", "")) == 0 else 1.0
    return jiwer.wer(r, h)


def compute_rtf(infer_time_s: float, audio_dur_s: float) -> float:
    return infer_time_s / audio_dur_s if audio_dur_s > 0 else float("inf")


def rejection_metrics(tp: int, fp: int, tn: int, fn: int) -> dict:
    """拒识指标。
    tp=正确拒非目标, fp=误拒目标(伤转写), tn=正确转目标, fn=漏拒(非目标被转,伤拒识率)。
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    # 拒识率(题面40%): 非目标段中被正确拒的比例 = tp/(tp+fn); 目标转写率 = tn/(tn+fp)
    reject_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {"reject_precision": precision, "reject_recall": recall, "reject_f1": f1,
            "reject_rate(非目标正确拒比例)": reject_rate}


def batch_cer(hyp_dir: str, ref_dir: str) -> tuple:
    """hyp_dir/*.txt 与 ref_dir/*.txt 同名配对算 CER，返回 (每条CER, 平均CER)。"""
    results = {}
    for hp in sorted(glob.glob(os.path.join(hyp_dir, "*.txt"))):
        name = os.path.basename(hp)
        rp = os.path.join(ref_dir, name)
        if not os.path.exists(rp):
            continue
        with open(hp, encoding="utf-8") as f:
            h = f.read().strip()
        with open(rp, encoding="utf-8") as f:
            r = f.read().strip()
        results[name] = cer(h, r)
    avg = sum(results.values()) / len(results) if results else 0.0
    return results, avg


# ===== 主办方官方 CER 口径(2026-07-08 坐实, 见 memory official-scoring-spec) =====
# 来源: 主办方参考脚本(unicodedata.NFKC + lower + 去 P*/空白 + editdistance 累计池)。
# 与上方 cer()(jiwer 逐条, 由调用方算术平均) 的三处差异:
#   1. 归一化 normalize_text: NFKC + lower + 去所有标点(Unicode P*)和空白(含内部空格)
#      —— 旧 cer 仅 replace(" ","")+lower, 不去标点不 NFKC
#   2. 聚合: 累计池 total_errors/total_chars —— 旧 cer 由 eval_datasetA/实验脚本逐条平均
#   3. 不做繁简归一 —— 旧链路 _norm_zh 用 zhconv; 提交侧 to_simplified 已把 pred 转简体对齐 ref
# 实测(1364 条 vanilla, 2026-07-08): overall 两口径差<0.01; 提交侧归一(digit+to_simplified)后
#   vanilla overall 0.664→0.595, dicow 1.248→1.214; vanilla-dicow 优势 Δ 不变(-0.62)。
import unicodedata
import string as _string
try:
    import editdistance as _editdistance
except ImportError:
    _editdistance = None


def normalize_text(text: str) -> str:
    """主办方归一化(照抄主办方脚本): NFKC + lower + strip + 去所有标点(P*)和空白。"""
    if text is None:
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = text.strip()
    out = []
    for ch in text:
        if ch in _string.whitespace or unicodedata.category(ch).startswith("P"):
            continue
        out.append(ch)
    return "".join(out)


class CERMetric:
    """主办方 CER 累计池度量(照抄主办方脚本): total_errors / total_chars 聚合, 非逐条平均。

    边界与主办方一致: target 空 → errors==0 给 0.0 否则 1.0; 非空 → errors/char_cnt(可>1, 不封顶)。
    拒识条(text 空)天然贡献 errors=len(ref), char=len(ref) → CER=1.0, 无需特殊处理。
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_chars = 0
        self.total_errors = 0
        self.per_sample_results = []

    def update(self, preds, targets):
        if isinstance(preds, str):
            preds = [preds]
        if isinstance(targets, str):
            targets = [targets]
        assert len(preds) == len(targets), "preds 和 targets 长度必须一致"
        assert _editdistance is not None, "editdistance 未装: uv pip install editdistance"
        for pred, target in zip(preds, targets):
            norm_pred = normalize_text(pred)
            norm_target = normalize_text(target)
            errors = _editdistance.eval(norm_pred, norm_target)
            char_cnt = len(norm_target)
            if char_cnt == 0:
                cer_value = 0.0 if errors == 0 else 1.0
            else:
                cer_value = errors / char_cnt
            self.total_errors += errors
            self.total_chars += char_cnt
            self.per_sample_results.append({
                "norm_pred": norm_pred, "norm_target": norm_target,
                "errors": errors, "target_chars": char_cnt, "cer": cer_value,
            })

    def compute(self):
        if self.total_chars == 0:
            overall = 0.0 if self.total_errors == 0 else 1.0
        else:
            overall = self.total_errors / self.total_chars
        return {"cer": overall, "total_errors": self.total_errors,
                "total_chars": self.total_chars, "per_sample": self.per_sample_results}


def cer_pool(preds, refs):
    """便捷: 官方累计池 CER(total_errors/total_chars)。preds/refs 为 list[str](单条 str 自动包裹)。"""
    if isinstance(preds, str):
        preds = [preds]
    if isinstance(refs, str):
        refs = [refs]
    m = CERMetric()
    m.update(preds, refs)
    return m.compute()["cer"]


def cer_official(pred: str, ref: str) -> float:
    """官方口径单条 CER = editdistance(norm_pred, norm_ref) / len(norm_ref); 空 ref 同 CERMetric 边界。"""
    if _editdistance is None:
        raise ImportError("editdistance 未装: uv pip install editdistance")
    norm_pred = normalize_text(pred)
    norm_ref = normalize_text(ref)
    if len(norm_ref) == 0:
        return 0.0 if len(norm_pred) == 0 else 1.0
    return _editdistance.eval(norm_pred, norm_ref) / len(norm_ref)


if __name__ == "__main__":
    print("=== W6 评测脚本自测 ===")
    print("\n[CER 中文]")
    print("  完全匹配:", cer("把空调调到二十六度", "把空调调到二十六度"))
    print("  1字错(七→六):", round(cer("把空调调到二十七度", "把空调调到二十六度"), 4))
    print("  全错:", cer("xyz", "把空调调到二十六度"))
    print("  空 hyp:", cer("", "把空调调到二十六度"))
    print("\n[RTF]")
    print("  DiCoW 30s 音频 1.73s 推理 → RTF =", round(compute_rtf(1.73, 30.0), 4), "(实测 0.058)")
    print("\n[拒识指标] tp=10 正确拒非目标, fp=2 误拒目标, tn=20 正确转目标, fn=1 漏拒")
    m = rejection_metrics(tp=10, fp=2, tn=20, fn=1)
    for k, v in m.items():
        print(f"  {k}: {round(v,4)}")
    print("\n[说明] 真实数据来时：hyp=模型转写txt目录, ref=标注txt目录, batch_cer 算平均 CER。")
