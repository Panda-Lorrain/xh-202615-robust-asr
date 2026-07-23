"""导出数据集A完整信息到CSV，包含ground truth、模型输出、评分"""
import json
import csv
import os

# 设置路径 - 从 code 目录的上级目录找 datasetA
code_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(code_dir)

# 加载数据集A ground truth
with open(os.path.join(project_dir, "datasetA/pos.jsonl"), encoding="utf-8") as f:
    pos_gt = [json.loads(line) for line in f]
with open(os.path.join(project_dir, "datasetA/neg.jsonl"), encoding="utf-8") as f:
    neg_gt = [json.loads(line) for line in f]

# 加载模型输出（当前最优配置：qwen后端，thr=0.27，sim_only）
# 使用 out_pos_baodi 和 out_neg_baodi 作为最新结果
with open(os.path.join(code_dir, "out_pos_baodi/result.json"), encoding="utf-8") as f:
    pos_result = json.load(f)
with open(os.path.join(code_dir, "out_neg_baodi/result.json"), encoding="utf-8") as f:
    neg_result = json.load(f)

# 建立 utt_id -> result 的映射
pos_result_map = {r["utt_id"]: r for r in pos_result["results"]}
neg_result_map = {r["utt_id"]: r for r in neg_result["results"]}

# 计算CER的辅助函数
def calc_cer(ref, hyp):
    """计算字符错误率"""
    if not ref:
        return 0.0 if not hyp else 1.0
    # 简单的编辑距离计算
    import editdistance
    return editdistance.eval(list(ref), list(hyp)) / len(ref)

# 准备CSV数据
rows = []

# POS集（正样本）
print("处理POS集...")
for idx, item in enumerate(pos_gt):
    # result中的utt_id格式是 utt{序号}_cmd_{id}
    utt_id = f"utt{str(idx).zfill(4)}_cmd_{item['id']}"
    result = pos_result_map.get(utt_id, {})

    ref_text = item.get("识别文本", "")
    hyp_text = result.get("text", "")
    sim = result.get("max_sim", None)
    rejected = result.get("rejected", None)
    llm_verdict = result.get("llm_verdict", None)
    noise_type = result.get("noise_type", "")

    # 计算CER
    cer = calc_cer(ref_text, hyp_text) if ref_text else None

    rows.append({
        "集": "POS",
        "utt_id": utt_id,
        "id": item["id"],
        "唤醒音频": item["唤醒音频"],
        "唤醒文本": item["唤醒文本"],
        "识别音频": item["识别音频"],
        "ground_truth": ref_text,
        "模型输出": hyp_text,
        "sim": sim,
        "是否拒识": rejected,
        "llm_verdict": llm_verdict,
        "噪声类型": noise_type,
        "CER": round(cer, 4) if cer is not None else None,
        "正确": "✓" if cer is not None and cer < 0.5 else "✗" if cer is not None else "",
    })

# NEG集（负样本）
print("处理NEG集...")
for idx, item in enumerate(neg_gt):
    # result中的utt_id格式是 utt{序号}_cmd_{id}
    utt_id = f"utt{str(idx).zfill(4)}_cmd_{item['id']}"
    result = neg_result_map.get(utt_id, {})

    # neg集的ground truth是null（应该拒识）
    ref_text = item.get("识别文本", None)
    hyp_text = result.get("text", "")
    sim = result.get("max_sim", None)
    rejected = result.get("rejected", None)
    llm_verdict = result.get("llm_verdict", None)
    noise_type = result.get("noise_type", "")

    rows.append({
        "集": "NEG",
        "utt_id": utt_id,
        "id": item["id"],
        "唤醒音频": item["唤醒音频"],
        "唤醒文本": item["唤醒文本"],
        "识别音频": item["识别音频"],
        "ground_truth": "(应拒识)" if ref_text is None else ref_text,
        "模型输出": hyp_text if hyp_text else "(空)",
        "sim": sim,
        "是否拒识": rejected,
        "llm_verdict": llm_verdict,
        "噪声类型": noise_type,
        "CER": None,  # neg集不计算CER
        "正确": "✓" if rejected else "✗",  # neg集：拒识=正确
    })

# 写入CSV
output_file = "datasetA_完整分析.csv"
with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"\n已导出到: {output_file}")
print(f"共 {len(rows)} 条 (POS: {len(pos_gt)}, NEG: {len(neg_gt)})")

# 统计
pos_correct = sum(1 for r in rows if r["集"] == "POS" and r["正确"] == "✓")
neg_correct = sum(1 for r in rows if r["集"] == "NEG" and r["正确"] == "✓")
print(f"\n统计:")
print(f"  POS正确率: {pos_correct}/{len(pos_gt)} = {pos_correct/len(pos_gt)*100:.1f}%")
print(f"  NEG正确率(拒识): {neg_correct}/{len(neg_gt)} = {neg_correct/len(neg_gt)*100:.1f}%")
