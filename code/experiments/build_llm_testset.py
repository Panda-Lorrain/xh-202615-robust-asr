"""构造 LLM 拒识测试集 code/llm_reject_testset.json。

正样本(accept, 合法指令): 取自 test_wav/dataset/raw 的 target 指令 + #1论文
  Type9/10(美的支持/其他品牌支持的合法指令) + 唤醒词(Type0 accept)。
负样本(reject, 应拒识): 取自 raw 的 nontarget 干扰闲聊 + #1论文
  Type5/6/7(闲聊/无实体/不合理参数/ASR乱码)。

gold 标签对齐 #1 论文 13 类 Utterance-Type 拒识 schema:
  accept = 应被设备接受/执行; reject = 应拒识。
"""
import os, json

_HERE = os.path.dirname(os.path.abspath(__file__))
RAW_MANIFEST = os.path.normpath(os.path.join(_HERE, "..", "test_wav", "dataset", "raw", "manifest.json"))
OUT = os.path.join(_HERE, "llm_reject_testset.json")


def from_raw():
    """从 TTS manifest 抽 target(accept)/nontarget(reject) 真实转写文本。"""
    items = []
    if os.path.isfile(RAW_MANIFEST):
        d = json.load(open(RAW_MANIFEST, encoding="utf-8"))
        for it in d["items"]:
            role = it["role"]
            if role == "target_cmd":           # 合法家居指令
                items.append({"text": it["ref"], "gold": "accept",
                              "source": "raw/target", "name": it["name"]})
            elif role == "nontarget_cmd":      # 干扰闲话
                items.append({"text": it["ref"], "gold": "reject",
                              "source": "raw/nontarget", "name": it["name"]})
            # enrollment 不计入(它是参考音频, 非识别输出)
    return items


# ---- 按 #1 论文 reject 类别补充的合成样例(扩充覆盖面) ----
SYNTH_ACCEPT = [
    # Type9 美的支持的合法指令 + Type0 唤醒词 accept
    ("把客厅空调制冷模式打开", "synth/type9"),
    ("小美小美，把电视音量调大一点", "synth/wakeword+type9"),
    ("关闭卧室所有灯光", "synth/type9"),
    ("设个十分钟后提醒我关火的闹钟", "synth/type9"),
    ("让扫地机回充电站", "synth/type9"),
    ("打开客厅的暖风机", "synth/type10_other_brand"),
]

SYNTH_REJECT = [
    # Type1 非法语言
    ("你给我滚开，烦死了", "synth/type1_illegal"),
    # Type4 ASR乱码/无意义短句
    ("这这这", "synth/type4_asrerror"),
    ("嗯啊那个就是", "synth/type4_asrerror"),
    # Type5 非对设备说的闲聊(多人/自言自语) — raw 已有, 再补
    ("妈，晚饭做好了叫我一声", "synth/type5_chat"),
    ("这游戏真好玩再来一局", "synth/type5_chat"),
    # Type6 语义不合理的指令(参数超范围)
    ("把空调调到四十度", "synth/type6_unreasonable"),
    ("热水器水温调到一百度", "synth/type6_unreasonable"),
    # Type7 对设备说的歧义闲聊(无明确实体+动作)
    ("你今天有没有好朋友呀", "synth/type7_ambiguous"),
    ("推荐一下江苏的小吃", "synth/type7_ambiguous_query"),
    # Type3 非人声/拟声(转写为无意义)
    ("汪汪汪", "synth/type3_nonhuman"),
]


def main():
    rows = from_raw()
    n_accept_raw = sum(1 for r in rows if r["gold"] == "accept")
    n_reject_raw = sum(1 for r in rows if r["gold"] == "reject")
    print(f"[raw] accept={n_accept_raw} reject={n_reject_raw}")

    for text, src in SYNTH_ACCEPT:
        rows.append({"text": text, "gold": "accept", "source": src})
    for text, src in SYNTH_REJECT:
        rows.append({"text": text, "gold": "reject", "source": src})

    na = sum(1 for r in rows if r["gold"] == "accept")
    nr = sum(1 for r in rows if r["gold"] == "reject")
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"[done] total={len(rows)} accept={na} reject={nr} -> {OUT}")


if __name__ == "__main__":
    main()
