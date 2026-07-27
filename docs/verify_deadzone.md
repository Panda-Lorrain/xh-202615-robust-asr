# 死区验收包索引(听音核实 — 用户必答)

> 生成: 2026-07-27 13:15 by exp_deadzone_diag.py --phase verify

> 每样本目录 `code/runs/_verify_deadzone_<uid>/`, 8 工位文件清单见下表.


## 🔑 核心问题(每条样本必答)

死区到底多少是**真地板(人耳不可辨)**、多少是**机器能力不够(人耳可辨机器做不到=可修)**?
机器无法判定, **请你用人耳听每条样本的 `recognition.wav`, 对照下面的 ref 文本, 回答: target 说的话, 人耳能否听清?**

- ✅ **人耳能听清 ref** → 即便机器 CER 很高, 也是机器能力不够 = **可修(训练/微调/增强能攻)**
- ❌ **人耳听不清 ref**(纯噪音/完全淹没) → **真地板**(任何模型都救不了)

这是判定死区天花板的**唯一可靠依据**(2637/2475 都是用户耳朵先于数据定位的)。


## 8 工位文件清单

| 工位 | 文件 | 听什么 |
|---|---|---|
| 1 | `enrollment.wav` | 目标说话人参考音(原 wav, 锁定 target 音色) |
| **2** | **`recognition.wav`** | **识别音频原 wav(双人重叠+噪声) — 核心验收, 听 target 说的话能否听清** |
| 3 | `enr_spk{i}.wav` | enrollment 是否被 diar 拆多 speaker(若有则 enrollment 污染) |
| 4 | `rec_spk{i}_full.wav` | diar 切出的各 speaker 全 timeline 段(含重叠区) |
| 5 | `rec_spk{i}_excl_raw.wav` | diar 切出的各 speaker 独占帧(避重叠, 抽声纹用) |
| 6 | `target_slice.wav` | argmax 选 target 切出的 timeline 切片(喂 ASR 的实际音频) |
| 7 | `假如选spk{i}_当target.wav` | 假如选其他 speaker 当 target(对照, 听是否另一人才是对的) |
| 8 | `postprocess_steps.json` + `summary.json` | sims / argmax 选谁 / 重叠率 / qwen 转写 / ref / CER |

## 样本列表

| uid | sim | bucket | qwen CER | n_spk | overlap | argmax target | **ref(请听 recognition 对照)** | 机器初判故障 |
|---|---|---|---|---|---|---|---|---|
| cmd_0 | 0.254 | [0.2,0.4) 主战场 | 0.0 | 1 | 0% | spk0 | **空调开到制热调到二十五度风量调到百分之三十** | ✅ 机器成功(参考: 不用重点听) |
| cmd_18 | 0.058 | <0.2 死区 | 1.0 | 2 | 14% | spk0 | **关闭灯光** | 🔥 机器转写与 ref 完全不沾边 — 待听 target 说的 ref 人耳能否辨(真地板 vs 可修) |
| cmd_2000 | 0.096 | <0.2 死区 | 1.5 | 2 | 56% | spk0 | **播放盖世音雄原创音乐** | 🔥 重叠率 56% + 机器高 CER — 待听重叠区 target 能否辨 |
| cmd_2041 | 0.139 | <0.2 死区 | 0.5714285714285714 | 2 | 22% | spk1 | **开启防直吹模式** | 机器转写接近但有错(可纠正类) |
| cmd_2098 | 0.019 | <0.2 死区 | 1.0 | 2 | 34% | spk0 | **调到二十八度** | 🔥 重叠率 34% + 机器高 CER — 待听重叠区 target 能否辨 |

## 用户听完后请回填

| uid | 人耳能否听清 target 说的 ref? (能/部分/不能) | 真地板 or 可修 | 备注 |
|---|---|---|---|
| cmd_0 | _待填_ | _待填_ | |
| cmd_18 | _待填_ | _待填_ | |
| cmd_2000 | _待填_ | _待填_ | |
| cmd_2041 | _待填_ | _待填_ | |
| cmd_2098 | _待填_ | _待填_ | |