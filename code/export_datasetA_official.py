"""导出数据集A官方原始数据到CSV"""
import json
import csv
import os

code_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(code_dir)

# 加载数据集A
with open(os.path.join(project_dir, "datasetA/pos.jsonl"), encoding="utf-8") as f:
    pos_gt = [json.loads(line) for line in f]
with open(os.path.join(project_dir, "datasetA/neg.jsonl"), encoding="utf-8") as f:
    neg_gt = [json.loads(line) for line in f]

rows = []

# POS集
for item in pos_gt:
    rows.append({
        "集": "POS",
        "id": item["id"],
        "唤醒音频": item["唤醒音频"],
        "唤醒文本": item["唤醒文本"],
        "识别音频": item["识别音频"],
        "识别文本": item["识别文本"],
    })

# NEG集
for item in neg_gt:
    rows.append({
        "集": "NEG",
        "id": item["id"],
        "唤醒音频": item["唤醒音频"],
        "唤醒文本": item["唤醒文本"],
        "识别音频": item["识别音频"],
        "识别文本": "",  # neg集没有识别文本
    })

# 写入CSV
output_file = os.path.join(code_dir, "datasetA_官方数据.csv")
with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=["集", "id", "唤醒音频", "唤醒文本", "识别音频", "识别文本"])
    writer.writeheader()
    writer.writerows(rows)

print(f"已导出: {output_file}")
print(f"共 {len(rows)} 条 (POS: {len(pos_gt)}, NEG: {len(neg_gt)})")
