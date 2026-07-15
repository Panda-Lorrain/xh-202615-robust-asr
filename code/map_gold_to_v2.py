"""把陈德基 26 条金标准按 v2 规范做映射验证。
自动判 C 类(循环/英文幻觉), 输出每条 ref+vanilla+用户note, 供人工映射 A/B/D + 层2。
"""
import csv, re

META = r"E:\midea_target_asr\code\error_analysis_pos_unfull.csv"
GOLD_PATH = r"D:\annot_陈德基.csv"

M = {r["uid"]: r for r in csv.DictReader(open(META, encoding="utf-8-sig"))}


def nt(s):
    if not s:
        return set()
    return {t.strip() for t in s.replace("；", ";").replace("，", ";").replace(",", ";").split(";") if t.strip()}


G = []
for r in csv.DictReader(open(GOLD_PATH, encoding="utf-8-sig")):
    if nt(r.get("rec_难点", "")) or nt(r.get("enw_难点", "")) or (r.get("rec_自然语言", "") or "").strip() or (r.get("enw_自然语言", "") or "").strip():
        G.append(r)

def has_loop(t):
    return bool(re.search(r'(.{3,}?)\1', t or ''))

def has_eng(vanilla, ref):
    # vanilla 有≥2连英文字母, 且 ref 基本没有
    return bool(re.search(r'[a-zA-Z]{2,}', vanilla or '')) and not bool(re.search(r'[a-zA-Z]{2,}', ref or ''))

print(f"金标准 {len(G)} 条, 按v2映射:\n")
for r in G:
    u = r["uid"]
    m = M.get(u, {})
    ref = (m.get("ref", "") or "").strip()
    van = (m.get("vanilla_text", "") or "").strip()
    note = (r.get("rec_自然语言", "") or "").strip()
    c1 = has_loop(van)
    c2 = has_eng(van, ref)
    c_tag = []
    if c1: c_tag.append("C1循环")
    if c2: c_tag.append("C2英文")
    print(f"### {u} | CER={r['CER']} sim={r['max_sim']}")
    print(f"  [自动C判] {','.join(c_tag) if c_tag else '无'}")
    print(f"  ref:    {ref[:50]}")
    print(f"  vanilla:{van[:60]}")
    print(f"  用户note: {note[:90]}")
    print()
