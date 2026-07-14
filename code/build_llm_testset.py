"""POC A 测试集构造: 验证 llm_reject (Qwen2.5-3B) 区分家居指令(accept) vs 非家居指令(reject)
的纯文本判别力 —— 新架构"LLM 凭内容识别 target"的根基验证。

⚠️ 防循环论证(POC 结论有效性的关键):
  llm_reject 内部用"家电实体+控制动作+参数合理性+对象合理性"CoT 判断(llm_reject.py:42 SYSTEM_PROMPT)。
  若负例也用"无家电词"筛 → 同义反复(用无家电词筛, 再用有无家电词判, 必对) → precision 虚高, 结论无效。
  破法: 负例含 e 对抗层(含家电词但非指令), 逼 LLM 靠指令性/参数/对象判断而非"有无家电词"。

构造(共 ~400, seed=42 可复现):
  正例 accept ~200: pos.ref 按家电类型分层抽样(真实家居指令)
  负例 reject ~200:
    a 干扰人新闻话 ~60: vanilla_text, 多speaker+无家电无动作+CER>1+len>3 (真实干扰人话)
    b 乱码语气词  ~40: vanilla_text, ≤3字 或 中文占比<0.3 (真实碎片)
    c 英文水印    ~25: dicow_text, 英文为主(vanilla 英文仅6不够, dicow 英文幻觉18.8%)
    d 空转写      ~25: dicow_text 空 (vanilla 空仅2不够)
    e 对抗(含家电词非指令) ~50: 硬编码(参数越界/闲聊/陈述/疑问) —— 防循环论证关键

输出: code/llm_testset_pocA.json  [{text, gold, layer, uid_or_src}]
"""
import json, os, random
from collections import defaultdict

random.seed(42)  # 可复现

_HERE = os.path.dirname(os.path.abspath(__file__))


def load(p):
    with open(os.path.join(_HERE, os.path.basename(p)), encoding="utf-8") as f:
        return json.load(f)


pos = load("code/pos_pairs_datasetA.json")
van = load("code/exp_vanilla_full.json")
slc = load("code/out_pos_slices_full.json")

# ---- uid 对齐 ----
van_by = {r["uid"]: r for r in van}
slc_by = {os.path.splitext(os.path.basename(r["recognition"]))[0]: r for r in slc}

# ---- 家电实体 / 动作词表(用于 a 层筛选 + 正例分层) ----
APPLIANCES = ["空调", "灯光", "窗帘", "音乐", "新风", "洗碗机", "洗衣机", "烤箱",
              "闹钟", "清香烟机", "门", "灯", "净化器", "热水器", "音箱", "风扇",
              "投影", "地暖", "浴霸", "香薰", "加湿器", "净水器"]
APPLIANCES_FULL = APPLIANCES + ["帘", "窗", "电视", "机器人", "扫地", "马桶", "摄像头"]
ACTIONS = ["打开", "开启", "开机", "关闭", "关掉", "关机", "关上", "调到", "调高", "调低",
           "调亮", "调暗", "启动", "暂停", "播放", "停止", "定时", "设置", "小一点",
           "大一点", "升高", "降低", "增加", "减少", "升温", "降温", "加热", "制热",
           "制冷", "送风", "风速", "风量", "模式", "切换", "开", "关"]


def has_app(t):
    return any(w in t for w in APPLIANCES_FULL)


def has_act(t):
    return any(w in t for w in ACTIONS)


def cn_ratio(t):
    cn = sum(1 for c in t if "一" <= c <= "鿿")
    return cn / max(len(t), 1)


def en_ratio(t):
    en = sum(1 for c in t if c.isalpha() and ord(c) < 128)
    return en / max(len(t), 1)


# ============ 正例 accept ~200: pos.ref 家电类型分层抽样 ============
# ⚠️ 先清洗: pos.ref 里 31%(425/1364) 是意图/闲聊/查询等非直接控制指令
# (如"我要做饭了""权志龙专辑叫什么""帮我策划旅游"), gold=accept 有争议。
# POC A 正例只取"含家电词或动作词"的干净直接指令(~939), 保证 gold 无争议。
# 防循环论证不靠此筛选(靠 e 层): 正例和 e 负例都含家电词,
# LLM 必须靠"是否真控制指令"区分; "含词"筛选只用于保证正例 gold 正确性。
# follow-up: 意图/隐式指令(我要做饭/我有点热)的 LLM 识别是 POC A 通过后的扩展。
APPLIANCE_KEYS = ["空调", "灯光", "窗帘", "音乐", "新风", "洗碗机", "洗衣机",
                  "烤箱", "闹钟", "清香烟机", "门"]
pos_clean = [r for r in pos if has_app(r["ref"]) or has_act(r["ref"])]
print(f"[正例池] pos.ref {len(pos)} -> 干净(含家电或动作) {len(pos_clean)}, "
      f"剔除 {len(pos) - len(pos_clean)} 意图/闲聊/查询"
      f"({(len(pos) - len(pos_clean)) / len(pos) * 100:.0f}%)")
groups = defaultdict(list)
for r in pos_clean:
    key = "其他"
    for w in APPLIANCE_KEYS:
        if w in r["ref"]:
            key = w
            break
    groups[key].append(r)

TARGET_POS = 200
total_pos = sum(len(v) for v in groups.values())
positives = []
for k, lst in groups.items():
    n = max(1, round(len(lst) * TARGET_POS / total_pos))
    positives.extend(random.sample(lst, min(n, len(lst))))
random.shuffle(positives)
positives = positives[:TARGET_POS]
pos_rows = [{"text": r["ref"], "gold": "accept", "layer": "pos",
             "uid_or_src": f"cmd_{r['id']}"} for r in positives]

# ============ 负例 a 干扰人新闻话 ~60 ============
# 多speaker + 无家电无动作 + CER>1 + len>3 (真实干扰人话)
TARGET_A = 60
a_pool = []
for uid, v in van_by.items():
    s = slc_by.get(uid, {})
    if len(s.get("speakers", [])) < 2:
        continue
    t = (v.get("vanilla_text", "") or "").strip()
    if len(t) <= 3 or has_app(t) or has_act(t):
        continue
    if v.get("vanilla_cer", 0) <= 1.0:
        continue
    a_pool.append((uid, t))
random.shuffle(a_pool)
used = set()
a_rows = []
for uid, t in a_pool:
    if uid in used:
        continue
    a_rows.append({"text": t, "gold": "reject", "layer": "a_news", "uid_or_src": uid})
    used.add(uid)
    if len(a_rows) >= TARGET_A:
        break

# ============ 负例 b 乱码语气词 ~40 ============
# len≤3 或 中文占比<0.3, 排除已用 uid
TARGET_B = 40
b_pool = []
for uid, v in van_by.items():
    t = (v.get("vanilla_text", "") or "").strip()
    if not t:
        continue
    if len(t) <= 3 or cn_ratio(t) < 0.3:
        b_pool.append((uid, t))
random.shuffle(b_pool)
b_rows = []
for uid, t in b_pool:
    if uid in used:
        continue
    b_rows.append({"text": t, "gold": "reject", "layer": "b_garble", "uid_or_src": uid})
    used.add(uid)
    if len(b_rows) >= TARGET_B:
        break

# ============ 负例 c 英文 ~25 (dicow_text, vanilla 英文不够) ============
TARGET_C = 25
c_pool = []
for r in van:
    t = (r.get("dicow_text", "") or "").strip()
    if t and en_ratio(t) > 0.6:
        c_pool.append((r["uid"], t))
random.shuffle(c_pool)
c_rows = []
for uid, t in c_pool:
    if uid in used:
        continue
    c_rows.append({"text": t, "gold": "reject", "layer": "c_en", "uid_or_src": uid})
    used.add(uid)
    if len(c_rows) >= TARGET_C:
        break

# ============ 负例 d 空转写 ~25 (dicow_text 空) ============
TARGET_D = 25
d_pool = [r["uid"] for r in van if not (r.get("dicow_text", "") or "").strip()]
random.shuffle(d_pool)
d_rows = []
for uid in d_pool:
    if uid in used:
        continue
    d_rows.append({"text": "", "gold": "reject", "layer": "d_empty", "uid_or_src": uid})
    used.add(uid)
    if len(d_rows) >= TARGET_D:
        break

# ============ 负例 e 对抗(含家电词但非指令) ~50 —— 防循环论证核心 ============
# 4 子类: 参数越界 / 闲聊提及家电 / 非控制陈述 / 疑问求助
E_NEGATIVES = [
    # e1 参数越界(不合理参数, 应 reject)
    "空调调到四十度", "空调开到零下十度", "把风速调到百分之两百", "帮我定一个三十小时的闹钟",
    "热水器烧到一百度", "灯光调到百分之两百亮", "空调开到五十度制热", "把温度调到负十度",
    "定一个零分钟的闹钟", "空调调到九十九度", "把亮度调到百分之一千", "风扇开到最高一百档",
    "烤箱设定到五百度",
    # e2 闲聊提及家电(非控制, 应 reject)
    "你家空调什么牌子", "这个电视画质真不错", "灯泡坏了去哪买", "这窗帘颜色挺好看的",
    "扫地机器人贵不贵啊", "新买的音箱效果很好", "空调费电吗", "你们家热水器多大容量",
    "这款净化器好用吗", "这音乐真好听啊", "那个风扇安静吗", "洗碗机值得买吗",
    "净水器哪个牌子好",
    # e3 非控制陈述(应 reject)
    "空调已经在制热了", "窗帘是昨天刚装的", "热水器修好了", "这是我最喜欢的音乐",
    "灯是自己关的", "扫地机器人没电了", "新风系统一直在转", "闹钟还没响呢",
    "烤箱预热好了", "洗衣机还在转", "音乐停了", "空调遥控器找不到了",
    # e4 疑问/求助(非指令, 应 reject)
    "空调怎么拆开", "灯不亮了怎么办", "这个音箱怎么连蓝牙", "净水器滤芯多久换一次",
    "窗帘卡住了怎么修", "扫地机器人不充电了", "闹钟怎么调时间", "热水器漏水怎么办",
    "音乐怎么下载到手机", "风扇摇头怎么关掉", "电视没信号了", "空调不制冷是什么原因",
]
# 校验 e 层确实"含家电词或动作词"(否则失去对抗意义)
_e_check = [t for t in E_NEGATIVES if not (has_app(t) or has_act(t))]
assert not _e_check, f"e 层以下条目不含家电/动作词, 失去对抗意义: {_e_check}"
e_rows = [{"text": t, "gold": "reject", "layer": "e_adv_含家电非指令",
           "uid_or_src": "synthetic"} for t in E_NEGATIVES]

# ============ 汇总 + 落盘 ============
all_rows = pos_rows + a_rows + b_rows + c_rows + d_rows + e_rows
random.shuffle(all_rows)  # 打乱顺序避免 LLM 批量处理的位置偏差(llm_reject 逐条独立, 但保险)

out_path = os.path.join(_HERE, "llm_testset_pocA.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(all_rows, f, ensure_ascii=False, indent=2)

# ---- 统计 ----
from collections import Counter
layer_cnt = Counter(r["layer"] for r in all_rows)
gold_cnt = Counter(r["gold"] for r in all_rows)
print(f"=== POC A 测试集构造完成 ===")
print(f"总数 {len(all_rows)} | gold: {dict(gold_cnt)}")
print(f"分层: {dict(layer_cnt)}")
print(f"  正例 pos(accept): {len(pos_rows)}")
print(f"  a 干扰人新闻话: {len(a_rows)} (池{len(a_pool)})")
print(f"  b 乱码语气词:   {len(b_rows)} (池{len(b_pool)})")
print(f"  c 英文水印:     {len(c_rows)} (池{len(c_pool)})")
print(f"  d 空转写:       {len(d_rows)} (池{len(d_pool)})")
print(f"  e 对抗(含家电非指令): {len(e_rows)}")
print(f"-> {out_path}")
print("\n--- 各层抽检样例(前3) ---")
for layer in ["pos", "a_news", "b_garble", "c_en", "d_empty", "e_adv_含家电非指令"]:
    samples = [r["text"] for r in all_rows if r["layer"] == layer][:3]
    print(f"  [{layer}] {samples}")
print("\n--- e 对抗层全部(防循环论证核心, 核对每条确实'含家电词非指令') ---")
for t in E_NEGATIVES:
    tag = []
    if has_app(t):
        tag.append("家电词")
    if has_act(t):
        tag.append("动作词")
    print(f"  {t}  <- {'+'.join(tag)}")
