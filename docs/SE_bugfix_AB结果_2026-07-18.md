# SE orphaned-bug bugfix A/B 结果 (2026-07-18, 对抗审查修正版)

> ⚠️ **本文档 2026-07-18 经 4-agent 对抗审查 + `code/audit_se_bugfix.py` 一手重算修正**。
> 原 doc/commit message 有 3 处错误已修:① "两边 accepted 文本不一致 0 条" 是
> `compare_se_bugfix.py` 字段名 bug(读 `transcript` 实为 `text`)致假象,实测 **109/383**;
> ② "SE 对转写无害甚至略好(accepted-only 0.4606<0.4777)" 是选择性偏差,apples-to-apples
> 交集 SE 反而恶化 **+0.0476**;③ "+0.1049 完全来自 sim mismatch 误拒" 漏了 diar 崩溃(22%)
> 与转写恶化(12%)两机制。决策(关 SE)不变,归因全部重写。

## 背景: 发现并修复 SE 输出从未被使用的潜伏 bug
- **Bug**: `submit_infer.py` 的 `rec_for_enroll` 变量赋值后从未读取(死变量, 8da1e98 初版起),
  `enroll_pairs` 一直写原始音频 `rec_in` 路径, SE 输出 `se_out` 是孤儿目录 → **SE 全程空转
  (~30.6% RTF 白烧)**, 从 vanilla/qwen 后端启用起即如此。此前所有 "--no-se 零 CER 影响 /
  50 条字节一致 / sim 0 差异" 结论均由此 bug 决定(SE 两分支输入完全相同, trivially 相等)。
- **修复** (commit c8c739d): SE 生效时把 `enroll_pairs` 的 recognition 路径重映射到 `se_out`,
  让 enroll_infer 真正读到降噪音频。enrollment 路径不动(保持原始 kws, 符合任务语义)。
  验证: SE/noSE 两臂 max_sim 1362/1364 条发生变化(mean -0.0898), 反证 se_out 确被读入。
  (死变量 `rec_for_enroll` 已删, bugfix 走 `rec_paths` 重映射。)

## A/B 结果 (qwen 后端, thr=0.27, 全量 1364 pos)

| 指标 | SE(bugfixed, 真生效) | noSE | 差异 |
|---|---|---|---|
| overall CER(含拒) | **0.8292** | **0.7243** | **+0.1049 恶化** |
| cer(仅 accepted, ⚠️不可比) | 0.4606 | 0.4777 | 见下"选择性偏差" |
| correct_rate(CER<0.5) | 21.85% | 35.70% | -13.85pp |
| 误拒率(pos 被拒, 伤 CER) | 68.33% (932/1364) | 47.21% (644/1364) | +21.12pp |
| 拒识数 | 932/1364 | 644/1364 | 净增拒识 **288**(翻转 386 = 337 多拒 + 49 少拒) |
| **交集 cer(apples-to-apples, both-accepted 383 条)** | **0.4281** | **0.3805** | **+0.0476 SE 在同集合反而恶化** |
| SE diar_fail 条数 | **207** | 2 | DF3 过衰减→DiariZen ValueError |
| overall RTF | 0.2053 | 0.1420 | **+45%**(SE 占 noSE 管线 30.6%) |
| wall(s) | 718 | 497 | +44% |

> 数据来源: `code/audit_se_bugfix.py`(复用 `eval_metrics.cer` + `_norm_zh` 繁简归一, 与官方
> 评测同口径) → `code/audit_se_bugfix.json`。overall/拒识数与 `result_eval.json` 逐字段对账 0 误差。

## overall CER +0.1049 的三机制分解 (修归因)

原归因"+0.1049 完全来自 sim mismatch 误拒"**错误**。逐条 delta 分桶(总 delta = +0.1049):

| 机制桶 | 条数 | 对 delta 贡献 | 占正贡献 | 说明 |
|---|---|---|---|---|
| **B. sim-drop 误拒** | 246 | +0.0765 | **66%** | SE 改音频→声纹域变→max_sim 降到 thr0.27 下→pos 被误拒(单端 mismatch: recognition 过 SE, enrollment 不过) |
| **A. diar 崩溃** | 91 | +0.0252 | **22%** | DeepFilterNet3 把音频过衰减至近静音(SE-RMS/orig mean 0.004)→DiariZen 找不到 speaker→ValueError→forced 拒。noSE 这 91 条本可转写(cer 0.62) |
| **C. 转写恶化** | 383 | +0.0134 | **12%** | 两边都 accepted 条里, SE 转写 net 更差(见下) |
| D. lucky-accept 抵消 | 49 | -0.0102 | — | SE sim 反升让 49 条 noSE 拒的进入 accepted(部分有益) |
| E. both-rejected | 595 | 0 | — | 两边都拒, 贡献 0 |

**结论**: sim mismatch 是最大单一机制(66%)但**非全部**;DF3 过衰减致 diar 崩溃(22%)与 SE
直接恶化转写(12%)是两个独立机制。SE 不是"仅通过 mismatch 伤分", 而是从三路同时伤分。

## "SE 对转写无害"被推翻: 文本不一致 109/383 + 交集恶化 +0.0476

- **字段名 bug 纠正**: 原 `compare_se_bugfix.py:41` 读 `r.get('transcript','')` 但 result schema
  字段是 `text` → 永远空串 → "0 条不一致"假象。改 `text` 后实测: 两边都 accepted 的 **383 条
  里 109 条(28.5%)文本不同**。其中 SE 改善 21 条、SE 恶化 68 条、中性 20 条 → **SE 净恶化转写**。
- 改善样例(SE 救回): cmd_2701 `播放不。`→`播放巴赫初级钢琴曲。` / cmd_212 `关闭微针,好像。`→`关闭微蒸烤箱。`
- 恶化样例(SE 搞烂): cmd_2468 `关闭显示屏。`→`温度,我感觉到。` / cmd_2989 `调成抽湿模式。`→`聊城抽丝魔卷。`
- **accepted-only 0.4606<0.4777 是选择性偏差**: SE 误拒掉 337 条 borderline 高 CER 条, accepted
  集合(SE 432)比 noSE(720)更小且偏向易条, 跨集合比 accepted-only 不公平。**apples-to-apples
  口径(两边都 accepted 的 383 条交集): SE 0.4281 vs noSE 0.3805, delta +0.0476, SE 同集合反而更差。**

## 决策: ⛔ 保持 --no-se(qwen 主线) — Pareto 最优坐实

1. **分桶无反例**: 按 noSE max_sim 分桶(代理音频质量), SE 在**每个**桶 overall 含拒 CER 全恶化:
   lo[<0.2] Δ+0.001 / mid[0.2,0.4) Δ+0.167 / hi[≥0.4] Δ+0.107。不存在"高 SNR 条 SE 有益"反例。
2. **双端 SE 不值得测**: 即便消除 mismatch(误拒回落~47%), accepted 纳入更难条, CER 收益上界
   <0.017(噪声内, ±0.04 噪声底), 而 SE 仍付 30.6% RTF(伤唯一有头空间的效率腿)。enrollment
   是 ~1.8s 干净 kws, DF3 对干净音频过压制风险高, 可能反向再降 sim。net 负, 不测。
3. **效率双赢**: 关 SE 省 30.6% RTF(RTF 0.205→0.142) + 避免 overall CER +0.1049 恶化。
4. ⚠️ **仅 qwen 后端做过 A/B**; vanilla/dicow 后端 SE 效果未测(run_baodi 默认 BACKEND=vanilla,
   其 SE 开关行为见 `run_baodi.sh` 注释)。若 vanilla 主线也要关 SE, 需补 vanilla+SE vs noSE A/B。

## 评审/答辩精确表述(避免露怯)
- ❌ 不要说 "SE 恶化 CER / SE 不好"(答不出机制 = 技术不扎实)。
- ✅ 要说: "SE bugfix 后真生效, 全量 A/B 显示单端应用(recognition 过 SE + enrollment 不过)
  从三路同时伤分——声纹域 mismatch 致 21pp 误拒(66%)、DF3 过衰减致 diar 崩溃(22%)、转写净
  恶化(12%, 交集 +0.0476)。分桶证明每个音频质量档 SE 都无益, 双端 SE 收益在噪声内且仍付 30%
  RTF。故 qwen 主线关 SE, Pareto 最优。"

## 说明
- 本文修正版由 `code/audit_se_bugfix.py` 一手重算(不抄审查 agent 数字; 实测中审查维度1称
  "翻转 644"系误算, 实为翻转 386 = 337+49, 本文以实测为准)。
- 对抗审查 4-agent 报告见 session 记录; `compare_se_bugfix.py` / `se_bugfix_record.py` 字段名
  bug 已修(`transcript`→`text`)。
- 原始评测: `code/out_pos_SE_bugfixed_1364/result_eval.json` + `code/out_pos_noSE_1364/result_eval.json`。
