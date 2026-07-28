# multi-voice 证伪验收包索引 (verify_mv_fail)

> 客观摆样本, 不预设结论. 用户亲自听/看 3 类证伪依据是否成立.

- 数据源: code/runs/_multivoice_full + _deadzone_selector
- 每条样本目录: enrollment.wav / recognition_original.wav / sep_sourceA.wav / sep_sourceB.wav / argmax_target_slice.wav + summary.json
- argmax_target_slice.wav 来自 code/stability_matrix/_slices/cmd_<N>.wav (主线 enroll_infer qwen 切的 target timeline)

## A 类 — 分离失败 (SepFormer 两路都没拿到 target, oracle_cer>0.8)

共 5 条. 共同特征: SepFormer 把目标说话人话分到了**两路都不像**的位置, 两路转写都偏离 ref → 选路再准也救不回 (oracle 也 >0.8). 这是**分离本身失败**, 不是 heuristic 选路问题.

### cmd_181  (ref: `把窗帘打开`)
- sim(argmax target) = `0.583`  audio_sec = `1.76`
- 主线 argmax qwen: `把窗帘打开。`  CER = `0.0`
- SepFormer sep_sourceA.wav: `嗯。`  CER=`1.0`  score=`-1.0`  gate_pass=`True`
- SepFormer sep_sourceB.wav: `所以。`  CER=`1.0`  score=`-1.0`  gate_pass=`True`
- oracle 选 src0 (CER=1.0)  heuristic 选 src0 (CER=1.0, reason=tie_both_valid_shorter)
- **该听哪个**: 听 enrollment.wav(目标说话人声纹) → recognition_original.wav(完整混音) → sep_sourceA/B.wav(SepFormer 两路). 判断: SepFormer 两路里有任何一路能听清 ref 吗? 若都听不出 → 分离失败坐实.

### cmd_2533  (ref: `开启防直吹模式`)
- sim(argmax target) = `0.543`  audio_sec = `2.88`
- 主线 argmax qwen: `开启王之春。`  CER = `0.7142857142857143`
- SepFormer sep_sourceA.wav: `他在收集。`  CER=`1.0`  score=`0.5`  gate_pass=`True`
- SepFormer sep_sourceB.wav: `收集可能是新模式。`  CER=`0.8571`  score=`2.0`  gate_pass=`True`
- oracle 选 src1 (CER=0.8571)  heuristic 选 src1 (CER=0.8571, reason=score_0.50_vs_2.00)
- **该听哪个**: 听 enrollment.wav(目标说话人声纹) → recognition_original.wav(完整混音) → sep_sourceA/B.wav(SepFormer 两路). 判断: SepFormer 两路里有任何一路能听清 ref 吗? 若都听不出 → 分离失败坐实.

### cmd_2786  (ref: `开启防直吹`)
- sim(argmax target) = `0.400`  audio_sec = `2.78`
- 主线 argmax qwen: `开启纺织锤。`  CER = `0.6`
- SepFormer sep_sourceA.wav: `防止疾病。`  CER=`1.0`  score=`0.5`  gate_pass=`True`
- SepFormer sep_sourceB.wav: `垂七放之垂。`  CER=`1.0`  score=`2.5`  gate_pass=`True`
- oracle 选 src0 (CER=1.0)  heuristic 选 src1 (CER=1.0, reason=score_0.50_vs_2.50)
- **该听哪个**: 听 enrollment.wav(目标说话人声纹) → recognition_original.wav(完整混音) → sep_sourceA/B.wav(SepFormer 两路). 判断: SepFormer 两路里有任何一路能听清 ref 吗? 若都听不出 → 分离失败坐实.

### cmd_286  (ref: `预约下午四点`)
- sim(argmax target) = `0.455`  audio_sec = `1.68`
- 主线 argmax qwen: `毕业下午四点。`  CER = `0.3333333333333333`
- SepFormer sep_sourceA.wav: `今天下午是女士。`  CER=`0.8333`  score=`0.5`  gate_pass=`True`
- SepFormer sep_sourceB.wav: `天下盛世。`  CER=`0.8333`  score=`0.5`  gate_pass=`True`
- oracle 选 src0 (CER=0.8333)  heuristic 选 src1 (CER=0.8333, reason=tie_both_valid_shorter)
- **该听哪个**: 听 enrollment.wav(目标说话人声纹) → recognition_original.wav(完整混音) → sep_sourceA/B.wav(SepFormer 两路). 判断: SepFormer 两路里有任何一路能听清 ref 吗? 若都听不出 → 分离失败坐实.

### cmd_2997  (ref: `腹泻适宜吃什么`)
- sim(argmax target) = `0.436`  audio_sec = `2.64`
- 主线 argmax qwen: `路线指引是什么？`  CER = `0.7142857142857143`
- SepFormer sep_sourceA.wav: `路线，事业支持我们。`  CER=`1.1429`  score=`0.5`  gate_pass=`True`
- SepFormer sep_sourceB.wav: `是的。`  CER=`1.0`  score=`-1.0`  gate_pass=`True`
- oracle 选 src1 (CER=1.0)  heuristic 选 src0 (CER=1.1429, reason=score_0.50_vs_-1.00)
- **该听哪个**: 听 enrollment.wav(目标说话人声纹) → recognition_original.wav(完整混音) → sep_sourceA/B.wav(SepFormer 两路). 判断: SepFormer 两路里有任何一路能听清 ref 吗? 若都听不出 → 分离失败坐实.


## B 类 — mel 被破坏 (主战场 argmax 好, SepFormer 后变差)

共 5 条. 筛选: sim≥0.4 (主战场, argmax 选对 target) & 主线 argmax CER<0.2 (原本好转写), 但 SepFormer 两路 oracle 都 >0.5. 显示 SepFormer 重分离破坏了原本能转好的 target mel (artifact/失真).

### cmd_146  (ref: `打开观影模式`)
- sim = `0.500`  argmax CER = `0.0` (好转写)  →  SepFormer oracle CER = `0.6667` (变差, Δ=+0.667)
- 主线 argmax 转写: `打开观影模式。`
- SepFormer sep_sourceA.wav: `打开关机。`  CER=`0.6667`  score=`2.5`
- SepFormer sep_sourceB.wav: `啊。`  CER=`1.0`  score=`-1.0`
- oracle 选 src0 (CER=0.6667)  heuristic 选 src0 (CER=0.6667)
- **该听哪个**: 听 argmax_target_slice.wav (主线切的, qwen 能转好) → sep_sourceA/B.wav (SepFormer 两路). 判断: 主线切片里 ref 听得清吗? SepFormer 两路是不是反而糊了/有 artifact?

### cmd_2007  (ref: `风速调最小`)
- sim = `0.517`  argmax CER = `0.0` (好转写)  →  SepFormer oracle CER = `0.6` (变差, Δ=+0.6)
- 主线 argmax 转写: `风速调最小。`
- SepFormer sep_sourceA.wav: `是这样做的。`  CER=`1.0`  score=`0.5`
- SepFormer sep_sourceB.wav: `分度条最小。`  CER=`0.6`  score=`0.5`
- oracle 选 src1 (CER=0.6)  heuristic 选 src0 (CER=1.0)
- **该听哪个**: 听 argmax_target_slice.wav (主线切的, qwen 能转好) → sep_sourceA/B.wav (SepFormer 两路). 判断: 主线切片里 ref 听得清吗? SepFormer 两路是不是反而糊了/有 artifact?

### cmd_2129  (ref: `开启自动模式`)
- sim = `0.583`  argmax CER = `0.0` (好转写)  →  SepFormer oracle CER = `0.6667` (变差, Δ=+0.667)
- 主线 argmax 转写: `开启自动模式。`
- SepFormer sep_sourceA.wav: `未来。`  CER=`1.0`  score=`-1.0`
- SepFormer sep_sourceB.wav: `开启这段文字。`  CER=`0.6667`  score=`2.5`
- oracle 选 src1 (CER=0.6667)  heuristic 选 src1 (CER=0.6667)
- **该听哪个**: 听 argmax_target_slice.wav (主线切的, qwen 能转好) → sep_sourceA/B.wav (SepFormer 两路). 判断: 主线切片里 ref 听得清吗? SepFormer 两路是不是反而糊了/有 artifact?

### cmd_277  (ref: `羊毛衫怎么洗`)
- sim = `0.421`  argmax CER = `0.0` (好转写)  →  SepFormer oracle CER = `0.6667` (变差, Δ=+0.667)
- 主线 argmax 转写: `羊毛衫怎么洗？`
- SepFormer sep_sourceA.wav: `看清楚。`  CER=`1.0`  score=`0.5`
- SepFormer sep_sourceB.wav: `杨浩然怎么写？`  CER=`0.6667`  score=`2.5`
- oracle 选 src1 (CER=0.6667)  heuristic 选 src1 (CER=0.6667)
- **该听哪个**: 听 argmax_target_slice.wav (主线切的, qwen 能转好) → sep_sourceA/B.wav (SepFormer 两路). 判断: 主线切片里 ref 听得清吗? SepFormer 两路是不是反而糊了/有 artifact?

### cmd_73  (ref: `关掉油烟机`)
- sim = `0.424`  argmax CER = `0.0` (好转写)  →  SepFormer oracle CER = `0.6` (变差, Δ=+0.6)
- 主线 argmax 转写: `关掉油烟机。`
- SepFormer sep_sourceA.wav: `关掉原音。`  CER=`0.6`  score=`2.5`
- SepFormer sep_sourceB.wav: `对。`  CER=`1.0`  score=`-1.0`
- oracle 选 src0 (CER=0.6)  heuristic 选 src0 (CER=0.6)
- **该听哪个**: 听 argmax_target_slice.wav (主线切的, qwen 能转好) → sep_sourceA/B.wav (SepFormer 两路). 判断: 主线切片里 ref 听得清吗? SepFormer 两路是不是反而糊了/有 artifact?


## C 类 — ★heuristic 双 accept (两路都过 content_gate) — 用户重点核查

共 8 条 (从 deadzone_selector 200 条 TRAP_both_valid 集里抽, 该集占 184/200=92%). 筛选: 两路都过 `is_valid_command` (即 TRAP_both_valid). 用户质疑: 两路真的都像家居指令吗? 还是 content_gate 门槛太低把非指令也放过了?

**核查要点**: `content_gate_pass=True` 只代表 "没命中明显非指令信号(英文主导/news黑名单词/超长/字循环/低多样)" — 它是为**主流程拒绝新闻/英文幻觉**设计的宽松 gate, **不是判断 "真家居指令" 的严格判别器**. 真正选路的是 `cmd_score` (设备词+动作词+品牌锚点).

**所以 92% 双accept 的解读**:
  - 字面: 92% 的样本两路都过了宽松 content_gate (True)
  - 实质: 大部分情况**一路是真指令, 另一路是 gate 没拦住的天气/点歌/财经/幼教等短句** → cmd_score 仍能正确区分
  - 用户该判: 看下面每条两路原文, 真两路都像指令的有几条? gate 放过非指令的有几条?

### C 类两路转写对照表 (用户肉眼判断)

| uid | ref | src0 (sep_sourceA) | src1 (sep_sourceB) | src0 score | src1 score | oracle | heuristic选 | 解读 |
|---|---|---|---|---|---|---|---|---|
| cmd_2627 | `关掉空调` | `新疆地区天气。` | `关掉空调。` | 0.5 | 5.5 | src1 | src1 | **单边像指令** (另一路 gate 放过非指令) / heuristic选对 |
| cmd_2488 | `把空调关上` | `点一首刘德华的《冰雨》。` | `把空调关上。` | 0.5 | 5.5 | src1 | src1 | **单边像指令** (另一路 gate 放过非指令) / heuristic选对 |
| cmd_2983 | `关机空调` | `另外一所幼儿园。` | `开空调。` | 0.5 | 2.5 | src1 | src1 | **单边像指令** (另一路 gate 放过非指令) / heuristic选对 |
| cmd_2541 | `关屏显` | `关屏显。` | `合约的持续进行。` | 0.5 | 0.5 | src0 | src0 | **两路都不像** (gate 双放过短/弱文本) / heuristic选对 |
| cmd_2896 | `开启ECO` | `您的设置。` | `开启一搜。` | 2.5 | 2.5 | src1 | src0 | **真双指令** (两路 score≥2) / heuristic选错 |
| cmd_2452 | `给我放下一个路口` | `开始播放。` | `给我换，都很柔弱的。` | 2.5 | 0.5 | src1 | src0 | **单边像指令** (另一路 gate 放过非指令) / heuristic选错 |
| cmd_2102 | `关掉智控温` | `嗯。` | `欢迎诸位朋友。` | -1.0 | 0.5 | src0 | src1 | **两路都不像** (gate 双放过短/弱文本) / heuristic选错 |
| cmd_2605 | `防直吹模式` | `纺织吹模式。` | `养殖模式。` | 2.0 | 2.0 | src0 | src1 | **真双指令** (两路 score≥2) / heuristic选错 |


### C 类逐条详情 (含 heuristic 评分拆解)

#### cmd_2627  (ref: `关掉空调`  sim=0.288)
- **sep_sourceA.wav**: `新疆地区天气。`
  - CER vs ref = `1.5`  content_gate_pass = `True`  cmd_score = `0.5`  L(去标点) = `6`
  - 评分信号: 无强信号 (仅长度±0.5)
- **sep_sourceB.wav**: `关掉空调。`
  - CER vs ref = `0.0`  content_gate_pass = `True`  cmd_score = `5.5`  L(去标点) = `4`
  - 评分信号: 设备词['空调'](+2) / 动作词['关掉'](+2)
- oracle 选 src1 (CER=0.0)  |  heuristic 选 src1 (CER=0.0, reason=`score_0.50_vs_5.50`)
- **本条判读**: 单边像指令 (真指令=src1, 另一路 src0 gate 放过非指令) → heuristic 选对 (cmd_score 看设备/动作词区分, 真指令=src1)
- **该听哪个**: 听 sep_sourceA.wav vs sep_sourceB.wav. 肉眼/耳判: 真指令那路听得清吗? 非指令那路真的是天气/点歌/财经/幼教等 gate 漏放的吗?

#### cmd_2488  (ref: `把空调关上`  sim=0.105)
- **sep_sourceA.wav**: `点一首刘德华的《冰雨》。`
  - CER vs ref = `1.8`  content_gate_pass = `True`  cmd_score = `0.5`  L(去标点) = `9`
  - 评分信号: 无强信号 (仅长度±0.5)
- **sep_sourceB.wav**: `把空调关上。`
  - CER vs ref = `0.0`  content_gate_pass = `True`  cmd_score = `5.5`  L(去标点) = `5`
  - 评分信号: 设备词['空调'](+2) / 动作词['关上'](+2)
- oracle 选 src1 (CER=0.0)  |  heuristic 选 src1 (CER=0.0, reason=`score_0.50_vs_5.50`)
- **本条判读**: 单边像指令 (真指令=src1, 另一路 src0 gate 放过非指令) → heuristic 选对 (cmd_score 看设备/动作词区分, 真指令=src1)
- **该听哪个**: 听 sep_sourceA.wav vs sep_sourceB.wav. 肉眼/耳判: 真指令那路听得清吗? 非指令那路真的是天气/点歌/财经/幼教等 gate 漏放的吗?

#### cmd_2983  (ref: `关机空调`  sim=0.104)
- **sep_sourceA.wav**: `另外一所幼儿园。`
  - CER vs ref = `1.75`  content_gate_pass = `True`  cmd_score = `0.5`  L(去标点) = `7`
  - 评分信号: 无强信号 (仅长度±0.5)
- **sep_sourceB.wav**: `开空调。`
  - CER vs ref = `0.5`  content_gate_pass = `True`  cmd_score = `2.5`  L(去标点) = `3`
  - 评分信号: 设备词['空调'](+2)
- oracle 选 src1 (CER=0.5)  |  heuristic 选 src1 (CER=0.5, reason=`score_0.50_vs_2.50`)
- **本条判读**: 单边像指令 (真指令=src1, 另一路 src0 gate 放过非指令) → heuristic 选对 (cmd_score 看设备/动作词区分, 真指令=src1)
- **该听哪个**: 听 sep_sourceA.wav vs sep_sourceB.wav. 肉眼/耳判: 真指令那路听得清吗? 非指令那路真的是天气/点歌/财经/幼教等 gate 漏放的吗?

#### cmd_2541  (ref: `关屏显`  sim=0.237)
- **sep_sourceA.wav**: `关屏显。`
  - CER vs ref = `0.0`  content_gate_pass = `True`  cmd_score = `0.5`  L(去标点) = `3`
  - 评分信号: 无强信号 (仅长度±0.5)
- **sep_sourceB.wav**: `合约的持续进行。`
  - CER vs ref = `2.3333`  content_gate_pass = `True`  cmd_score = `0.5`  L(去标点) = `7`
  - 评分信号: 无强信号 (仅长度±0.5)
- oracle 选 src0 (CER=0.0)  |  heuristic 选 src0 (CER=0.0, reason=`tie_both_valid_shorter`)
- **本条判读**: 两路都不像 (gate 双放过弱/短文本, 无设备/动作词) → heuristic 选对 (平局 tiebreak 短句, 撞对)
- **该听哪个**: 听 sep_sourceA.wav vs sep_sourceB.wav. 肉眼/耳判: 真指令那路听得清吗? 非指令那路真的是天气/点歌/财经/幼教等 gate 漏放的吗?

#### cmd_2896  (ref: `开启ECO`  sim=0.334)
- **sep_sourceA.wav**: `您的设置。`
  - CER vs ref = `1.0`  content_gate_pass = `True`  cmd_score = `2.5`  L(去标点) = `4`
  - 评分信号: 动作词['设置'](+2)
- **sep_sourceB.wav**: `开启一搜。`
  - CER vs ref = `0.6`  content_gate_pass = `True`  cmd_score = `2.5`  L(去标点) = `4`
  - 评分信号: 动作词['开启'](+2)
- oracle 选 src1 (CER=0.6)  |  heuristic 选 src0 (CER=1.0, reason=`tie_both_valid_shorter`)
- **本条判读**: 真双指令 (两路都命中设备/动作/功能词) → heuristic 选错 (平局 tiebreak 更短, 撞错)
- **该听哪个**: 听 sep_sourceA.wav vs sep_sourceB.wav. 肉眼/耳判: 真指令那路听得清吗? 非指令那路真的是天气/点歌/财经/幼教等 gate 漏放的吗?

#### cmd_2452  (ref: `给我放下一个路口`  sim=0.216)
- **sep_sourceA.wav**: `开始播放。`
  - CER vs ref = `1.0`  content_gate_pass = `True`  cmd_score = `2.5`  L(去标点) = `4`
  - 评分信号: 动作词['播放', '放'](+2)
- **sep_sourceB.wav**: `给我换，都很柔弱的。`
  - CER vs ref = `0.75`  content_gate_pass = `True`  cmd_score = `0.5`  L(去标点) = `8`
  - 评分信号: 无强信号 (仅长度±0.5)
- oracle 选 src1 (CER=0.75)  |  heuristic 选 src0 (CER=1.0, reason=`score_2.50_vs_0.50`)
- **本条判读**: 单边像指令 (真指令=src0, 另一路 src1 gate 放过非指令) → heuristic 选错 (cmd_score 给 src0 更高分因 动作词['播放', '放'], 但 oracle 指向另一路)
- **该听哪个**: 听 sep_sourceA.wav vs sep_sourceB.wav. 肉眼/耳判: 真指令那路听得清吗? 非指令那路真的是天气/点歌/财经/幼教等 gate 漏放的吗?

#### cmd_2102  (ref: `关掉智控温`  sim=0.039)
- **sep_sourceA.wav**: `嗯。`
  - CER vs ref = `1.0`  content_gate_pass = `True`  cmd_score = `-1.0`  L(去标点) = `1`
  - 评分信号: 无强信号 (仅长度±0.5)
- **sep_sourceB.wav**: `欢迎诸位朋友。`
  - CER vs ref = `1.2`  content_gate_pass = `True`  cmd_score = `0.5`  L(去标点) = `6`
  - 评分信号: 无强信号 (仅长度±0.5)
- oracle 选 src0 (CER=1.0)  |  heuristic 选 src1 (CER=1.2, reason=`score_-1.00_vs_0.50`)
- **本条判读**: 两路都不像 (gate 双放过弱/短文本, 无设备/动作词) → heuristic 选错 (cmd_score 给 src1 更高分因 弱信号, 但 oracle 指向另一路)
- **该听哪个**: 听 sep_sourceA.wav vs sep_sourceB.wav. 肉眼/耳判: 真指令那路听得清吗? 非指令那路真的是天气/点歌/财经/幼教等 gate 漏放的吗?

#### cmd_2605  (ref: `防直吹模式`  sim=0.321)
- **sep_sourceA.wav**: `纺织吹模式。`
  - CER vs ref = `0.4`  content_gate_pass = `True`  cmd_score = `2.0`  L(去标点) = `5`
  - 评分信号: 功能词['模式'](+1.5)
- **sep_sourceB.wav**: `养殖模式。`
  - CER vs ref = `0.6`  content_gate_pass = `True`  cmd_score = `2.0`  L(去标点) = `4`
  - 评分信号: 功能词['模式'](+1.5)
- oracle 选 src0 (CER=0.4)  |  heuristic 选 src1 (CER=0.6, reason=`tie_both_valid_shorter`)
- **本条判读**: 真双指令 (两路都命中设备/动作/功能词) → heuristic 选错 (平局 tiebreak 更短, 撞错)
- **该听哪个**: 听 sep_sourceA.wav vs sep_sourceB.wav. 肉眼/耳判: 真指令那路听得清吗? 非指令那路真的是天气/点歌/财经/幼教等 gate 漏放的吗?


## 汇总 — 用户判断后填

- A 类 (分离失败): ___ 条坐实 SepFormer 没分出 target / ___ 条其实有一路能听清 (分离OK只是选路问题)
- B 类 (mel 破坏): ___ 条 argmax 切片明显比 SepFormer 两路清晰 / ___ 条差不多 (SepFormer 没明显破坏)
- C 类 (双 accept): ___ 条真双家居指令 / ___ 条单边像指令 gate 放过非指令 / ___ 条两路都不像

## 结论 (不下判断, 用户填)

- multi-voice 整体 NO-GO 的 3 条依据是否成立:
  - A 分离失败: 成立 / 部分成立 / 不成立
  - B mel 破坏: 成立 / 部分成立 / 不成立
  - C 双accept 92%: 数字准确但解读需修正 (gate 宽松 vs 真双指令) / 数字本身有问题