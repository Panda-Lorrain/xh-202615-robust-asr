"""生成未满分音频错误分析 CSV (2026-07-09): 全部 CER>0 分档, 供用户人工标难点。

用户决策: 全部分档 + 暂时自己判断(不自动提特征)。CSV 留空难点类别+备注列。

数据源: exp_vanilla_full.json (全量1362 per-sample: uid/ref/max_sim/vanilla_text/vanilla_cer)。
  注: vanilla_cer 是旧口径(jiwer逐条); 官方口径overall 0.5947(recompute_official_cer.json summary)。
  分档用 vanilla_cer per-sample(相对排序; CER=0 筛选与官方口径一致——同一条转写正确)。
时长: wave 标准库读 cmd_X.wav header (recognition 带噪 + enrollment 唤醒词), 无外部依赖。
分档: 1_死区CER>1 / 2_严重0.5-1 / 3_中等0.1-0.5 / 4_轻微0-0.1。档内按 CER 降序。
输出: code/error_analysis_pos_unfull.csv (utf-8-sig Excel 友好; 难点类别+备注留空给用户标)。
"""
import json, csv, wave, os
from collections import Counter

DATA = json.load(open(r"E:\midea_target_asr\code\exp_vanilla_full.json", encoding="utf-8"))
POS_DIR = r"E:\midea_target_asr\datasetA\pos"


def wav_sec(path):
    if not os.path.isfile(path):
        return -1
    try:
        with wave.open(path, "rb") as w:
            return round(w.getnframes() / w.getframerate(), 2)
    except Exception:
        return -1


def bucket(cer):
    if cer > 1:
        return "1_死区CER>1"
    if cer > 0.5:
        return "2_严重0.5-1"
    if cer > 0.1:
        return "3_中等0.1-0.5"
    return "4_轻微0-0.1"


rows = []
for r in DATA:
    cer = float(r.get("vanilla_cer", 0))
    if cer <= 0:  # 满分(转写完全正确)跳过
        continue
    uid = r["uid"]  # cmd_X
    rec = os.path.join(POS_DIR, uid + ".wav")
    enw = os.path.join(POS_DIR, uid.replace("cmd", "kws") + ".wav")
    rows.append({
        "档": bucket(cer),
        "uid": uid,
        "recognition_path": rec,
        "enrollment_path": enw,
        "ref": r.get("ref", ""),
        "vanilla_text": r.get("vanilla_text", ""),
        "vanilla_cer": round(cer, 4),
        "max_sim": round(float(r.get("max_sim", 0)), 4),
        "rec_sec": wav_sec(rec),
        "enroll_sec": wav_sec(enw),
        "难点类别": "",   # 用户标: 音量小/语速快/语速慢/babble强/重叠/英文干扰/静音/设备未说话/其他
        "备注": "",       # 用户标
    })

rows.sort(key=lambda x: (x["档"], -x["vanilla_cer"]))

out = r"E:\midea_target_asr\code\error_analysis_pos_unfull.csv"
with open(out, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=[
        "档", "uid", "recognition_path", "enrollment_path", "ref",
        "vanilla_text", "vanilla_cer", "max_sim", "rec_sec", "enroll_sec",
        "难点类别", "备注"])
    w.writeheader()
    w.writerows(rows)

c = Counter(x["档"] for x in rows)
print(f"未满分(CER>0): {len(rows)} 条 / 全量 {len(DATA)} 条 (满分CER=0: {len(DATA)-len(rows)} 条)")
for k in sorted(c):
    print(f"  {k}: {c[k]} 条")
print(f"CSV -> {out}")
