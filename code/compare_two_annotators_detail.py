"""2 标注员深度比对: 只看活跃条目(至少一方标了难点或写了自然语言)。
测试阶段仅前 50 余条有标注, 后续空对空条目剔除。
用法: code/.venv/Scripts/python.exe code/compare_two_annotators_detail.py
"""
import csv, os
from collections import Counter

J_PATH = r"C:\Users\26875\Desktop\annot_江星影.csv"
L_PATH = r"C:\Users\26875\Desktop\annot_罗慧琪.csv"
OUT_DIR = r"E:\midea_target_asr\code\annot_pack\compare"
JA, LA = "江星影", "罗慧琪"


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


J = load(J_PATH, "gb18030")
L = load(L_PATH, "utf-8-sig")

# 活跃条目: 至少一方在任一块标了难点 或 写了自然语言
def active(u):
    j, l = J[u], L[u]
    return bool(j["rec"] or j["enw"] or l["rec"] or l["enw"]
                or j["rec_note"] or j["enw_note"] or l["rec_note"] or l["enw_note"])

ACT = [u for u in sorted(set(J) & set(L)) if active(u)]
print(f"总并集 {len(set(J) & set(L))} 条 → 活跃(有实际标注) {len(ACT)} 条, 其余空对空已剔除\n")

# ===== 1. 一致性 (仅活跃) =====
n_rec_eq = n_enw_eq = n_both_eq = n_disp = 0
for u in ACT:
    j, l = J[u], L[u]
    rec_eq = j["rec"] == l["rec"]
    enw_eq = j["enw"] == l["enw"]
    n_rec_eq += rec_eq
    n_enw_eq += enw_eq
    n_both_eq += (rec_eq and enw_eq)
    n_disp += (not (rec_eq and enw_eq))
print("===== 一致性 (仅 %d 条活跃) =====" % len(ACT))
print(f"rec 难点完全一致: {n_rec_eq} ({n_rec_eq/len(ACT)*100:.1f}%)")
print(f"enw 难点完全一致: {n_enw_eq} ({n_enw_eq/len(ACT)*100:.1f}%)")
print(f"两块都一致(共识): {n_both_eq} ({n_both_eq/len(ACT)*100:.1f}%)")
print(f"任一块分歧:       {n_disp} ({n_disp/len(ACT)*100:.1f}%)")

# ===== 2. 各自标签词频 (仅活跃) =====
def vocab(uids, who, key):
    c = Counter()
    for u in uids:
        c.update({JA: J, LA: L}[who][u][key])
    return c

print("\n===== rec 难点标签词频 (活跃条目) =====")
print(f"[{JA}] {vocab(ACT, JA, 'rec')}")
print(f"[{LA}] {vocab(ACT, LA, 'rec')}")
print("\n===== enw 难点标签词频 (活跃条目) =====")
print(f"[{JA}] {vocab(ACT, JA, 'enw')}")
print(f"[{LA}] {vocab(ACT, LA, 'enw')}")

jrv, lrv = set(vocab(ACT, JA, "rec")), set(vocab(ACT, LA, "rec"))
jev, lev = set(vocab(ACT, JA, "enw")), set(vocab(ACT, LA, "enw"))
print("\n===== 标签术语差异 =====")
print(f"rec 仅 {JA} 用: {jrv - lrv} / 仅 {LA} 用: {lrv - jrv}")
print(f"enw 仅 {JA} 用: {jev - lev} / 仅 {LA} 用: {lev - jev}")

# ===== 3. 差异方向 =====
def diff_dir(key):
    jo, lo = Counter(), Counter()
    for u in ACT:
        jo.update(J[u][key] - L[u][key])
        lo.update(L[u][key] - J[u][key])
    return jo, lo

for key, name in [("rec", "recognition"), ("enw", "enrollment")]:
    jo, lo = diff_dir(key)
    print(f"\n===== {name} 差异方向 =====")
    print(f"  {JA} 有 {LA} 无: {jo.most_common()}")
    print(f"  {LA} 有 {JA} 无: {lo.most_common()}")

# ===== 4. 自然语言详尽度 (仅活跃) =====
def note_stat(who):
    d = {JA: J, LA: L}[who]
    rec_empty = sum(1 for u in ACT if not d[u]["rec_note"])
    enw_empty = sum(1 for u in ACT if not d[u]["enw_note"])
    rec_written = [(u, d[u]["rec_note"]) for u in ACT if d[u]["rec_note"]]
    enw_written = [(u, d[u]["enw_note"]) for u in ACT if d[u]["enw_note"]]
    return rec_empty, enw_empty, rec_written, enw_written

print("\n===== 自然语言描述详尽度 (活跃 %d 条) =====" % len(ACT))
for who in (JA, LA):
    re_, ee_, rw, ew = note_stat(who)
    print(f"[{who}] rec 写了 {len(ACT)-re_}/{len(ACT)} ; enw 写了 {len(ACT)-ee_}/{len(ACT)}")
both_rec_empty = sum(1 for u in ACT if not J[u]["rec_note"] and not L[u]["rec_note"])
both_enw_empty = sum(1 for u in ACT if not J[u]["enw_note"] and not L[u]["enw_note"])
print(f"两人 rec 自然语言都空: {both_rec_empty}/{len(ACT)} ; enw 都空: {both_enw_empty}/{len(ACT)}")

print("\n--- 罗慧琪写的 rec 自然语言全量 (看是否=标签复述) ---")
for u, n in note_stat(LA)[2]:
    print(f"  {u}: {n!r}")
print("--- 江星影写的自然语言全量 ---")
for u, n in note_stat(JA)[2]:
    print(f"  {u} rec: {n!r}")
for u, n in note_stat(JA)[3]:
    print(f"  {u} enw: {n!r}")

# ===== 5. 复查清单: 只看难点分歧, 自动判范式冲突 vs 真分歧 =====
# 范式冲突(可批量归一, 不必逐条听): rec 江用{babble强/重叠/循环幻觉/英文干扰} 罗用{其他}兜底
REC_PARADIGM_J = {"babble强", "重叠", "循环幻觉", "英文干扰"}
os.makedirs(OUT_DIR, exist_ok=True)

rows = []
for u in ACT:
    j, l = J[u], L[u]
    rec_d = j["rec"] != l["rec"]
    enw_d = j["enw"] != l["enw"]
    if not (rec_d or enw_d):
        continue
    kind = []
    if rec_d:
        # 江 细分标签 罗仅"其他/音量/语速" → 范式冲突
        j_spec = j["rec"] & REC_PARADIGM_J
        l_trivial = l["rec"] <= {"其他", "音量小", "语速快", "语速慢"}
        kind.append("rec范式冲突(可归一)" if (j_spec and l_trivial) else "rec真分歧(需听)")
    if enw_d:
        # enw 典型范式分裂: 江"多人同说" vs 罗"唤醒词截断/不清"
        j_multi = "多人同说" in j["enw"]
        l_kws = l["enw"] & {"唤醒词截断", "唤醒词不清"}
        kind.append("enw范式冲突(可归一)" if (j_multi and l_kws) else "enw真分歧(需听)")
    rows.append({
        "uid": u, "分类": ";".join(kind),
        "江_rec": ";".join(sorted(j["rec"])), "罗_rec": ";".join(sorted(l["rec"])),
        "江_enw": ";".join(sorted(j["enw"])), "罗_enw": ";".join(sorted(l["enw"])),
        "江_rec_note": j["rec_note"], "罗_rec_note": l["rec_note"],
        "江_enw_note": j["enw_note"], "罗_enw_note": l["enw_note"],
    })

with open(os.path.join(OUT_DIR, "review_pending.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["uid"])
    w.writeheader()
    w.writerows(rows)

# 按优先级分桶
must_listen = [r for r in rows if "真分歧" in r["分类"]]
can_batch = [r for r in rows if "范式冲突" in r["分类"] and "真分歧" not in r["分类"]]
print(f"\n===== 复查清单 → {OUT_DIR}\\review_pending.csv ({len(rows)} 条分歧) =====")
print(f"  🔴 真分歧(必须逐条听音): {len(must_listen)} 条")
for r in must_listen:
    print(f"     {r['uid']} | {r['分类']} | 江={r['江_rec'] or '—'}/{r['江_enw'] or '—'} 罗={r['罗_rec'] or '—'}/{r['罗_enw'] or '—'}")
print(f"  🟡 范式冲突(定术语表可批量归一, 不必逐条听): {len(can_batch)} 条")
