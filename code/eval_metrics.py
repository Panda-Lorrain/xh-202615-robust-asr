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
