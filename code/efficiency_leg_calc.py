#!/usr/bin/env python3
"""L20/L40 效率腿(20 分)分数换算: 读 submit_infer 产出的 timing.json + result.json,
按若干候选官方映射口径(待主办方确认 w1/w2 + 效率分映射)输出效率腿分数区间。

官方口径(2026-07-08 FAQ 坐实, memory official-scoring-spec):
  - 效率 20%(L20-46G 评测), 拆时间分 + 内存分(各 10, 待确认)
  - RTF 按 batch=1 测(每条独立推理时间 / 音频时间)
  - 排名公式 TotalScore = w1*(1-CER) + w2*RR + w_eff*效率  (w 具体值待确认)

本脚本两个 RTF 口径都报告(诚实):
  - overall_rtf: 端到端 = 总墙钟/总音频, 含 SE + 切timeline + ASR转写 + 模型加载摊薄
                (全量跑才准, 加载被 n_utt 摊薄; 这是评委复现将看到的量级 → 主报告值)
  - duration_infer_rtf: = sum(per-utt infer_sec)/总音频, 纯推理不含模型加载
                ⚠️ qwen/firered 后端 infer_sec 只含切 timeline, 漏 ASR 转写 → 低估真实 RTF

用法:
  python code/efficiency_leg_calc.py code/out_pos_baodi/timing.json
  python code/efficiency_leg_calc.py code/out_pos_baodi/timing.json --result code/out_pos_baodi/result.json
  python code/efficiency_leg_calc.py code/out_pos_baodi/timing.json --rtf 0.095   # 手动外推/调试
"""
import os
import sys
import json
import argparse


def _linear_drop(value, full_at, zero_at, full_score):
    """value ≤ full_at → full_score; full_at..zero_at 线性降到 0; ≥ zero_at → 0。"""
    if value <= full_at:
        return full_score
    if value >= zero_at:
        return 0.0
    return full_score * (zero_at - value) / (zero_at - full_at)


def _step(rtf, full_score):
    """阶梯映射: <0.5 满分 / 0.5-1 70% / 1-2 40% / >2 零分。"""
    if rtf < 0.5:
        return full_score
    if rtf < 1.0:
        return 0.7 * full_score
    if rtf < 2.0:
        return 0.4 * full_score
    return 0.0


# 时间腿候选映射(官方未公布具体公式, 覆盖宽松→严格): name -> fn(rtf, full_score) -> score
TIME_MAPPINGS = [
    ("宽松 RTF<=1满分->3为零", lambda r, m: _linear_drop(r, 1.0, 3.0, m)),
    ("中等 RTF<=0.5满分->2为零", lambda r, m: _linear_drop(r, 0.5, 2.0, m)),
    ("严格 RTF<=0.3满分->1为零", lambda r, m: _linear_drop(r, 0.3, 1.0, m)),
    ("阶梯 <.5/.5-1/1-2/>2",    _step),
    ("极严 RTF<=0.1满分->0.5为零", lambda r, m: _linear_drop(r, 0.1, 0.5, m)),
]

# 内存腿候选: peak ≤ cap*frac 满分, 线性降到 cap 为零(qwen 路线 peak ~5-8GiB, 多数映射近满分)
MEM_FRACS = [
    ("宽松 peak<=50%cap满分", 0.5),
    ("中等 peak<=30%cap满分", 0.3),
    ("严格 peak<=20%cap满分", 0.2),
    ("最宽 不OOM即满分",      1.0),
]


def main():
    ap = argparse.ArgumentParser(description="L20/L40 效率腿(20分)换算: timing.json+result.json -> 分数区间")
    ap.add_argument("timing_json", help="submit_infer 产出的 timing.json")
    ap.add_argument("--result", help="submit_infer 产出的 result.json(取 peak_mem_mib max; 不给则内存腿按满分估)")
    ap.add_argument("--time-leg", type=float, default=10.0, help="时间腿满分(默认 10)")
    ap.add_argument("--mem-leg", type=float, default=10.0, help="内存腿满分(默认 10)")
    ap.add_argument("--mem-cap-gib", type=float, default=46.0, help="评测卡显存上限 GiB(L20=46)")
    ap.add_argument("--rtf", type=float, default=None,
                    help="手动指定 RTF 覆盖 timing.json(外推 L20 / 调试用)")
    ap.add_argument("--use-infer-rtf", action="store_true",
                    help="用 duration_infer_rtf 打分(默认 overall_rtf; qwen 后端此值低估)")
    args = ap.parse_args()

    with open(args.timing_json, encoding="utf-8") as f:
        t = json.load(f)
    overall_rtf = args.rtf if args.rtf is not None else t.get("overall_rtf")
    total_audio = t.get("total_audio_sec")
    total_wall = t.get("total_wall_sec")
    duration_infer = t.get("duration_infer_sec")
    rtf_infer = (duration_infer / total_audio) if (duration_infer and total_audio) else None
    phases = t.get("phases", {})

    print("=" * 70)
    print(f"设备: {t.get('device')} | n_utt: {t.get('n_utt')} | 评测卡显存上限: {args.mem_cap_gib} GiB")
    print(f"总音频: {total_audio}s | 总墙钟: {total_wall}s")
    print(f"overall_rtf(端到端, 含SE+切timeline+ASR转写+模型加载摊薄): {overall_rtf}")
    if rtf_infer is not None:
        print(f"duration_infer_rtf(纯推理 infer_sec 累加 / audio): {rtf_infer:.4f}")
        print(f"  [warn] qwen/firered 后端: infer_sec 只含切timeline 漏 ASR 转写 -> 此值低估真实 RTF")
    if phases:
        print("phases wall_sec: " + ", ".join(f"{k}={v.get('wall_sec')}" for k, v in phases.items()))
    print("-" * 70)

    # ---- 时间腿 ----
    if args.use_infer_rtf and rtf_infer is None:
        print("[error] --use-infer-rtf 但 timing.json 缺 duration_infer_sec/total_audio_sec, 无法算 infer RTF"); sys.exit(1)
    rtf_for_scoring = rtf_infer if args.use_infer_rtf else overall_rtf
    if rtf_for_scoring is None:
        print("[error] 无 RTF(timing.json 缺 overall_rtf 且未给 --rtf)"); sys.exit(1)
    label = "duration_infer_rtf" if args.use_infer_rtf else "overall_rtf"
    print(f"时间腿打分 RTF = {rtf_for_scoring}  ({label}, 满分 {args.time_leg})")
    time_scores = []
    for name, fn in TIME_MAPPINGS:
        s = fn(rtf_for_scoring, args.time_leg)
        time_scores.append(s)
        print(f"  [{name:30s}] -> {s:5.2f}")
    print("-" * 70)

    # ---- 内存腿 ----
    mem_scores = []
    peak_gib = None
    if args.result and os.path.exists(args.result):
        with open(args.result, encoding="utf-8") as f:
            r = json.load(f)
        peaks = [it.get("peak_mem_mib") for it in r.get("results", []) if it.get("peak_mem_mib")]
        if peaks:
            peak_mib = max(peaks)
            peak_gib = peak_mib / 1024.0
            cap_mib = args.mem_cap_gib * 1024
            print(f"峰值显存: {peak_mib} MiB = {peak_gib:.2f} GiB / {args.mem_cap_gib} GiB (满分 {args.mem_leg})")
            for name, frac in MEM_FRACS:
                s = _linear_drop(peak_mib, cap_mib * frac, cap_mib, args.mem_leg)
                mem_scores.append(s)
                print(f"  [{name:30s}] -> {s:5.2f}")
        else:
            print("[warn] result.json 无 peak_mem_mib(enroll_infer 才产), 内存腿按满分估")
    else:
        print(f"[info] 未给 --result, 内存腿按满分 {args.mem_leg} 估(qwen 路线显存充裕, 保守可减 1-2)")

    print("-" * 70)
    # ---- 总分区间 ----
    t_min, t_max = (min(time_scores), max(time_scores)) if time_scores else (0.0, 0.0)
    if mem_scores:
        m_min, m_max = min(mem_scores), max(mem_scores)
    else:
        m_min, m_max = args.mem_leg, args.mem_leg
    print(f"效率腿区间: 时间 [{t_min:.2f}, {t_max:.2f}]/{args.time_leg:.0f}  +  "
          f"内存 [{m_min:.2f}, {m_max:.2f}]/{args.mem_leg:.0f}")
    print(f"  => 效率腿合计 [{t_min + m_min:.2f}, {t_max + m_max:.2f}] / {args.time_leg + args.mem_leg:.0f}")
    print(f"  (官方公布具体公式后取单一值。qwen RTF 0.289@4060, L20 算力约 4-6x → 外推 overall_rtf ~0.05-0.09,")
    print(f"   时间腿多数映射近满分; 内存腿 qwen 1.7B+diar+whisper-fe peak<10GiB 远低于 46GiB cap, 稳满分)")
    print("=" * 70)


if __name__ == "__main__":
    main()
