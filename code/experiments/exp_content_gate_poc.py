"""内容有效性二次校验 PoC（2026-07-08）。

问题: RR 卡在 90.5%(thr0.27), 45 条漏拒 neg 是 sim>=0.27 的高 sim 干扰。
观察: 漏拒 neg 转写多为新闻/英文/乱码, 非有效家居指令。
假设: 在 sim>=thr 的 accept 上加 content_gate(转写内容是否像有效家居指令),
      能拒掉大部分漏拒 neg → 提 RR, 且 pos 侧误拒代价可接受(甚至拒掉幻觉灾难赚 CER)。

本脚本纯分析, 不改 decide_reject。输出:
  1) neg 侧: content_gate 能再拒多少漏拒 neg → 新 RR
  2) pos 侧: 被误拒的 pos 占比 + 原CER 水平(>=1 则拒了反赚)
  3) TotalScore 变化估算(官方口径累计池)

判别规则为初版(长度/中文比例/指令动词白名单/新闻词黑名单), 不依赖唤醒词列表(待主办方要)。
"""
import json, sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from text_utils import to_simplified, digit_postproc
from eval_metrics import normalize_text
import editdistance


def submit_norm(t):
    return digit_postproc(to_simplified(t or ""))


def nfk(t):
    return normalize_text(t or "")


def cer_pair(hyp, ref):
    h, r = nfk(hyp), nfk(submit_norm(ref))  # ref 也过提交归一(全简体全中文数字)
    if not r:
        return 0.0
    return editdistance.eval(h, r) / len(r)


# --- 有效家居指令判别(初版规则) ---
CMD_VERBS = ["打开", "关闭", "开", "关", "调高", "调低", "调", "设置", "设定", "播放",
             "暂停", "停止", "放", "切", "换", "增加", "减小", "升", "降", "把", "帮",
             "要", "想", "启动", "预热", "取消", "确认", "查询", "查", "洗", "煮", "烤",
             "蒸", "扫", "拖", "关掉", "打开"]
NEWS_BLACK = ["产业", "资本", "投资", "制度", "政府", "债务", "市场", "调研", "报告",
              "期货", "股票", "基金", "贷款", "住房", "房地产", "报道", "新闻", "记者",
              "日前", "发布", "导演", "婚姻", "考试", "生意", "布料", "价格", "学生",
              "广告", "拍摄", "无法阻挡",
              # 繁体新闻/财经/公司(转写保留繁体时命中)
              "期貨", "報告", "市場", "調研", "調查", "顯示", "股份", "有限公司",
              "落戶", "服務", "四強", "席位", "聚杯"]
HOME_ENT = ["空调", "温度", "电视", "灯", "音乐", "歌", "门", "窗", "风", "热", "冷",
            "模式", "度", "摄氏", "音量", "频道", "机器", "扫", "扇", "暖", "湿", "水",
            "电", "饭", "器", "冷冻", "冷藏", "洗衣", "净化"]


def is_valid_command(text):
    """保守版(黑名单驱动): True=保留(默认), False=强非指令信号→拒.
    初版白名单(必须命中动词) pos 误拒 41% 太激进 → 改黑名单, 只在高置信非指令时拒."""
    if not text or not text.strip():
        return False
    nch = sum(1 for c in text if "一" <= c <= "鿿")
    # 强非指令信号(高 precision) → 拒
    if nch == 0:                                    # 纯非中文(ok/tooling, 家居指令无此情况)
        return False
    if len(text) >= 3 and nch < len(text) * 0.5:   # 英文为主(tooling/productive/i can't go)
        return False
    if any(w in text for w in NEWS_BLACK):          # 新闻/财经词命中
        return False
    if len(text) > 20:                              # 超长叙述(pos 家居指令极少>20字)
        return False
    return True


THR = 0.27


def get_text(x):
    """pos 样本取转写字段(鲁棒多种命名)."""
    return (x.get("vanilla_text") or x.get("text") or x.get("transcript") or "")


def main():
    # --- pos ---
    pos = json.load(open(os.path.join(_HERE, "exp_vanilla_full.json"), encoding="utf-8"))
    if isinstance(pos, dict):
        pos = pos.get("results") or pos.get("rows") or []
    sample = pos[0] if pos else {}
    print(f"[pos] n={len(pos)} sample_keys={list(sample.keys())[:12]}")
    print(f"[pos] sample_text={get_text(sample)[:40]!r} ref={str(sample.get('ref',''))[:40]!r}")

    pos_accept = [x for x in pos if "max_sim" in x and float(x.get("max_sim", 0) or 0) >= THR]
    pos_gate_rej = [x for x in pos_accept
                    if not is_valid_command(submit_norm(get_text(x)))]
    print(f"\n[pos] accept(sim>={THR})={len(pos_accept)}, content_gate 误拒={len(pos_gate_rej)}"
          f" ({len(pos_gate_rej)/max(1,len(pos_accept)):.3f})")

    if pos_gate_rej:
        rej_cers = [cer_pair(get_text(x), x.get("ref", "")) for x in pos_gate_rej]
        ge1 = sum(1 for c in rej_cers if c >= 1.0)
        print(f"[pos] 被误拒原CER: mean={sum(rej_cers)/len(rej_cers):.3f} "
              f"CER>=1占比={ge1}/{len(rej_cers)}={ge1/len(rej_cers):.2f} "
              f"(这些拒了变1.0反赚; <1的拒了亏)")
        # 净效果: 拒后这些 pos CER=1.0, 拒前是原CER; mean<1 → 拒了亏 mean*(n) 分
        print(f"[pos] 被误拒群体 mean CER {sum(rej_cers)/len(rej_cers):.3f} "
              f"{'<1.0 → 误拒有代价' if sum(rej_cers)/len(rej_cers)<1 else '>=1.0 → 拒了反赚'}")

    # --- neg ---
    neg = json.load(open(os.path.join(_HERE, "out_neg_full", "result.json"), encoding="utf-8"))
    neg_rows = neg.get("results", neg) if isinstance(neg, dict) else neg
    n_neg = len(neg_rows)
    neg_leak = [r for r in neg_rows if float(r.get("max_sim", 0) or 0) >= THR]
    neg_gate_rej = [r for r in neg_leak if not is_valid_command(r.get("text", "") or "")]
    kept_leak = [r for r in neg_leak if is_valid_command(r.get("text", "") or "")]
    new_rr = (n_neg - len(kept_leak)) / n_neg
    print(f"\n[neg] n={n_neg} 原漏拒(sim>={THR})={len(neg_leak)} "
          f"RR {1-len(neg_leak)/n_neg:.3f} → +gate 再拒 {len(neg_gate_rej)} → RR {new_rr:.3f}")
    print(f"[neg] gate 没拒掉的漏拒 neg(像指令, 天花板): {len(kept_leak)} 条")
    for r in sorted(kept_leak, key=lambda x: -float(x.get("max_sim", 0) or 0)):
        print(f"      sim={float(r.get('max_sim',0)):.3f} {(r.get('text','') or '')[:35]!r}")

    # --- TotalScore 估算(官方累计池 pos CER + RR) ---
    # pos 累计池: accept 且过 gate 的按原 CER, 被 gate 拒的按 1.0
    def pool_pos_cer(accept_pool, gate_rej_pool):
        err = ch = 0
        for x in accept_pool:
            h, r = nfk(submit_norm(get_text(x))), nfk(submit_norm(x.get("ref", "")))
            if r:
                err += editdistance.eval(h, r); ch += len(r)
        # gate 拒的 pos: CER=1.0 → errors += len(ref), chars += len(ref)
        for x in gate_rej_pool:
            r = nfk(submit_norm(x.get("ref", "")))
            err += len(r); ch += len(r)
        return err / ch if ch else 0

    pos_cer_now = pool_pos_cer(pos_accept, [])  # 现状(thr0.27 全转写, 不 gate)
    pos_cer_gate = pool_pos_cer([x for x in pos_accept if x not in pos_gate_rej], pos_gate_rej)
    rr_now = 1 - len(neg_leak) / n_neg
    print(f"\n[TotalScore 估算 w1=w2=0.4] "
          f"现状: pos_CER={pos_cer_now:.4f} RR={rr_now:.4f} "
          f"→ {0.4*(1-pos_cer_now)+0.4*rr_now:.4f}")
    print(f"[TotalScore 估算 w1=w2=0.4] +gate: pos_CER={pos_cer_gate:.4f} RR={new_rr:.4f} "
          f"→ {0.4*(1-pos_cer_gate)+0.4*new_rr:.4f}")
    print(f"[结论] gate 净变化: pos_CER {pos_cer_now:.4f}→{pos_cer_gate:.4f} "
          f"({'赚' if pos_cer_gate<pos_cer_now else '亏'}), "
          f"RR {rr_now:.4f}→{new_rr:.4f}")


if __name__ == "__main__":
    main()
