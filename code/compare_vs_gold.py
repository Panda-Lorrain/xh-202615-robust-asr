"""三方对比 v2: 陈德基(金标准, 仅~21条, 信息主要在自然语言) vs 江星影(50条) vs 罗慧琪(50条)。
正确口径: ①对比限定在陈德基标了的条目 ②难点分块对比, 只在该块金标准标签非空时算命中/漏标/误标
       (陈德基范式=标签精简+自然语言承载细节, 空标签≠无难点, 不能当误标基准)
"""
import csv, os
from collections import Counter

C_PATH = r"D:\annot_陈德基.csv"
J_PATH = r"C:\Users\26875\Desktop\annot_江星影.csv"
L_PATH = r"C:\Users\26875\Desktop\annot_罗慧琪.csv"
OUT_DIR = r"E:\midea_target_asr\code\annot_pack\compare"

HP_REC = {"音量小", "语速快", "语速慢", "babble强", "重叠", "英文干扰", "静音/未说话", "循环幻觉", "其他"}
HP_ENW = {"背景嘈杂", "有其他说话人", "唤醒词不清", "音量小", "唤醒词截断", "多人同说", "静音/无有效语音", "其他"}


def norm_tags(s):
    if not s:
        return set()
    s = s.replace("；", ";").replace("，", ";").replace(",", ";")
    return {t.strip() for t in s.split(";") if t.strip()}


def load(path, enc):
    d = {}
    with open(path, encoding=enc, newline="") as f:
        for r in csv.DictReader(f):
            d[r["uid"]] = {
                "rec": norm_tags(r.get("rec_难点", "")),
                "enw": norm_tags(r.get("enw_难点", "")),
                "rec_note": (r.get("rec_自然语言", "") or "").strip(),
                "enw_note": (r.get("enw_自然语言", "") or "").strip(),
            }
    return d


C, J, L = load(C_PATH, "utf-8-sig"), load(J_PATH, "gb18030"), load(L_PATH, "utf-8-sig")

# 金标准标注范围: 陈德基任一字段非空
GOLD = [u for u in sorted(C) if C[u]["rec"] or C[u]["enw"] or C[u]["rec_note"] or C[u]["enw_note"]]
print(f"金标准(陈德基)实际标注范围: {len(GOLD)} 条")
print(f"  其中 rec难点非空: {sum(1 for u in GOLD if C[u]['rec'])} / rec自然语言非空: {sum(1 for u in GOLD if C[u]['rec_note'])}")
print(f"  其中 enw难点非空: {sum(1 for u in GOLD if C[u]['enw'])} / enw自然语言非空: {sum(1 for u in GOLD if C[u]['enw_note'])}")

# 江/罗 对金标准范围覆盖率
j_cov = sum(1 for u in GOLD if J[u]["rec"] or J[u]["enw"])
l_cov = sum(1 for u in GOLD if L[u]["rec"] or L[u]["enw"])
print(f"  江星影覆盖金标准: {j_cov}/{len(GOLD)} ; 罗慧琪覆盖: {l_cov}/{len(GOLD)}")

# ===== 1. 金标准标签词频 (仅金标准范围) =====
def vocab(d, uids, key):
    c = Counter()
    for u in uids:
        c.update(d[u][key])
    return c

print("\n===== 金标准 rec 标签词频 =====")
cv = vocab(C, GOLD, "rec")
print(f"  {dict(cv)}")
print("===== 金标准 enw 标签词频 =====")
print(f"  {dict(vocab(C, GOLD, 'enw'))}")

# ===== 2. 金标准标签 vs 默认表 =====
print("\n===== 默认标签表 vs 金标准实际 (回答‘默认选项哪里不贴合’) =====")
print(f"rec 默认有但金标准从不用: {HP_REC - set(cv)}")
print(f"enw 默认有但金标准从不用: {HP_ENW - set(vocab(C, GOLD, 'enw'))}")

# ===== 3. 江/罗 vs 金标准: 分块, 只在该块金标准标签非空时算 =====
def eval_block(who, key):
    d = {"J": J, "L": L}[who]
    base = [u for u in GOLD if C[u][key]]  # 只在该块金标准非空的条目上
    hit = miss = extra = 0
    hit_tags, miss_tags, extra_tags = Counter(), Counter(), Counter()
    n_cov = 0  # 队员在该条有标
    for u in base:
        g, p = C[u][key], d[u][key]
        if p:
            n_cov += 1
        hit += len(g & p); miss += len(g - p); extra += len(p - g)
        hit_tags.update(g & p); miss_tags.update(g - g & p)  # 占位, 下方重算
    # 重算 tag 级
    hit_tags, miss_tags, extra_tags = Counter(), Counter(), Counter()
    for u in base:
        g, p = C[u][key], d[u][key]
        hit_tags.update(g & p)
        miss_tags.update(g - p)
        extra_tags.update(p - g)
    return len(base), n_cov, hit, miss, extra, hit_tags, miss_tags, extra_tags

for who, name in [("J", "江星影"), ("L", "罗慧琪")]:
    print(f"\n===== {name} vs 金标准 =====")
    for key, blk in [("rec", "rec"), ("enw", "enw")]:
        nb, ncov, hit, miss, extra, ht, mt, et = eval_block(who, key)
        prec = hit / (hit + extra) if (hit + extra) else 0
        rec = hit / (hit + miss) if (hit + miss) else 0
        print(f"  {blk}: 基准条目(金标准该块非空) {nb} | 队员标了 {ncov} | 命中{hit} 漏标{miss} 误标{extra} | P={prec:.2f} R={rec:.2f}")
        print(f"       命中标签: {ht.most_common()}")
        print(f"       漏标标签: {mt.most_common()}")
        print(f"       误标标签: {et.most_common()}")

# ===== 4. 自然语言详尽度 (金标准范围) =====
print("\n===== 自然语言详尽度 (仅金标准范围 %d 条) =====" % len(GOLD))
for who, name in [("C", "陈德基·金"), ("J", "江星影"), ("L", "罗慧琪")]:
    d = {"C": C, "J": J, "L": L}[who]
    for key in ("rec_note", "enw_note"):
        written = [len(d[u][key]) for u in GOLD if d[u][key]]
        avg = sum(written) / len(written) if written else 0
        print(f"  [{name}] {key}: 写了 {len(written)}/{len(GOLD)}, 平均 {avg:.0f} 字/条, 最长 {max(written) if written else 0} 字")

# ===== 5. 逐条三方对照 (金标准范围) =====
rows = []
for u in GOLD:
    rows.append({
        "uid": u,
        "金_rec": ";".join(sorted(C[u]["rec"])), "江_rec": ";".join(sorted(J[u]["rec"])), "罗_rec": ";".join(sorted(L[u]["rec"])),
        "金_enw": ";".join(sorted(C[u]["enw"])), "江_enw": ";".join(sorted(J[u]["enw"])), "罗_enw": ";".join(sorted(L[u]["enw"])),
        "金_rec_note": C[u]["rec_note"], "金_enw_note": C[u]["enw_note"],
        "江_rec_note": J[u]["rec_note"], "罗_rec_note": L[u]["rec_note"],
    })
os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "three_way_compare.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"\n逐条三方对照(金标准范围) → {OUT_DIR}\\three_way_compare.csv ({len(rows)} 条)")
