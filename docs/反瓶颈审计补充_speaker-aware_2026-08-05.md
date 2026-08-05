# 反瓶颈审计补充：speaker-aware 探查的正确归类（2026-08-05）

`docs/speaker_aware探查_2026-08-05.md` 已完成一轮 `max_sim + stno` 融合探查，必须纳入本次审计：

- 通用区分 AUC 从 `0.862` 提到 `0.910`，但这不是提交收益；
- 已拒 pos vs 正确拒 neg 的 operational recovery 最高 precision 约 `0.75`，低于当前评分 break-even `0.763`；
- in-sample net `+0.0088` 在 5-fold hold-out 变为 `+0.0003 ± 0.0019`，低于噪声地板，因此结论是 **当前无训练 speaker-aware 融合 implementation-NO-GO**。

这条结果进一步收窄了“低成本拒识”空间，但没有封死以下尚未验证路线：

1. Qwen decoder 置信度、token entropy/no-speech 等 ASR 内部信号；
2. 面向真实/同环境非 A 数据的拒识校准器或 target-conditioned verifier；
3. target embedding 进入 ASR encoder、target activity 与 ASR loss 联合训练；
4. 真实录音域 TSE / REAL-PS4 / AISHELL-4 或自录家居数据训练。

因此，不要重复扫 `max_sim + stno`，也不要把它的 NO-GO 改写成“所有 speaker-aware 方法无解”。它应归类为 `implementation-NO-GO`，而不是 `direction-NO-GO`。
