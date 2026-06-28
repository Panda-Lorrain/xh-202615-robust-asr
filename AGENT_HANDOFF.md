# Agent 交接文档（XH-202615 美的目标说话人 ASR）

> **下个 agent 第一时间读此文件 → CLAUDE.md → PROGRESS.md → RESULTS.md → 边缘部署规划.md → 项目阶段盘点.md**
> 更新：2026-06-29（T18）。T17 后**新会话接手**，用户选「多线并行铺开」→ Workflow 三线 de-risk 全 READY + 线A 一锤定音验证瓶颈。**先读下方🆕 T18 速览**，完整数字见 RESULTS.md T18。

---

## 🆕 T18 速览（2026-06-29 接手后多线并行铺开）

**三线 de-risk 全 READY**（Workflow 3 agent，各建独立 venv：`code/.venv_se` / `code/.venv_campp` / `.venv_llm`[项目根]）：

- **线C W5-LLM 拒识** ✅强阳：Qwen2.5-3B 零样本 34 条 **F1=0.878 / Recall=1.0**（最难 case 全对；5 误拒是合法复杂指令→prompt 调优）。拒识 40% 核心层已验证。脚本 `code/llm_reject.py`。
- **线A SE增强 一锤定音** ✅部分阳：DeepFilterNet3 降噪后 overall CER **4.27→3.65(Δ−0.62)**；**babble(人声,贴真实)Δ−4.20 巨大** / pink(稳态)+2.20 反伤；**diar-fail 33→0**（diarization 完全稳定）；CER 绝对值仍高(3.65)→瓶颈多元。**验证"瓶颈部分在音频质量"诊断**。脚本 `code/se_denoise.py`+`eval_se_cer.py`。
- **线B CAM++** ❌证伪(已定论)：per-speaker 公平对照(CAM++ 0.191 vs wespeaker 0.218, 正确率 0.00<0.04)→ **不值得替代 wespeaker, 主线维持 wespeaker**; sherpa-onnx 留边缘部署备用。脚本 `code/enroll_infer_campp.py`。
- **线A SE 条件化(新, 最优)** ✅：按 noise_type 分流(babble/white=0 全力, pink=6 温和) overall CER **2.82(−34%)**, 优于单一 =0(3.65)/=6(3.95); diar-fail 33→0。脚本 `code/merge_se_conditional.py`。

**新 P0 优先级**：① SE 前置条件化落地(快赢：babble/低SNR 启用，pink/white 用 `--atten-lim-db=6`) ② CAM++ per-speaker 公平对照(定论去留) ③ 中文家居微调(重，攻 Whisper 中文) ④ SE-DiCoW 接入(攻 100% 重叠死区)。

**数据增强暂缓**(用户定)：MUSAN/DEMAND+RIR+AISHELL 方案已评估存档(RESULTS T18)，等真实数据/通道数确认再定。

**坑**：HF 下载用 `curl -sSL 经代理+hf-mirror 直链`(snapshot_download 失败)；csukuangfj 仓 401 改 hf-mirror 无代理。

---

## 0. 项目一句话状态（T17 历史）
组合主线（STNO + DiCoW + 声纹enrollment + LLM拒识）的 **enrollment→target 锁定 + STNO 拒识链路已跑通验证**（干净场景完美），但 **带噪题目分布是当前瓶颈**：wespeaker 声纹在中文+噪声退化严重，导致 450 条全集 **87% 误拒、仅 4% 正确**。下一步核心是**攻带噪鲁棒性**（CAM++/阈值扫/enrollment增强/SE增强）。

---

## 1. 最新进展（T17，2026-06-28~29，已 commit）
| 产出 | 文件 | 状态 |
|---|---|---|
| **Part1** enrollment→wespeaker 锁定唯一 target | `code/enroll_infer.py` | ✅ 干净场景验证成功；带噪诊断完成 |
| **TTS 数据集**（趁 mimo-tts 限时免费）| `code/tts_dataset_gen.py` + `build_dataset.py` | ✅ 21 条 raw + 450 条矩阵 |
| **Part2** fork patch 固化 | `code/DiCoW-inference/repro/` | ✅ apply_patches.sh 幂等 EXIT=0 |
| **Part3** 项目阶段盘点 | `项目阶段盘点.md` | ✅ |
| **边缘部署规划** | `边缘部署规划.md` + memory | ✅ 战略级，待确认硬件 |
| **能力画像** | `code/eval_enrollment.py` + `enroll_wespeaker_full.json` | ✅ 450条：87%拒识/4%正确 |
| 阈值扫 + enrollment 加噪增强 | `code/enroll_infer.py --always-generate --enroll-augment` | 🔄 进行中（`enroll_aug_full.json`）|

---

## 2. 当前能力画像（诚实，答辩核心）
- **干净场景**：enrollment→锁定 target→转写「请把客厅的空调温度调到二十六度」(CER=0, sim=0.816)；不同人→兜底拒识空输出(sim=0.035)。**链路成立**。
- **题目分布 450 条**（带噪 −5~5dB + 重叠 0-100%）：**87% 拒识 / 4% 正确**。sim 随 overlap 单调降(0.32→0.12)，SNR 越低越差。
- **根因**：① wespeaker(VoxCeleb英文)声纹中文+噪声退化（噪声是主因，重叠次要）；② 阈值 0.5 在题目分布太严；③ 少数不拒识条出 Whisper 英文幻觉。
- **CAM++ 对比受阻**：modelscope 1.37.1 装上但 `from modelscope.pipelines import pipeline` import 挂起（无 proxy 也卡）。

---

## 3. 后续任务清单（优先级排序，直接可做）

### 🔴 P0 — 攻带噪鲁棒性（瓶颈已诊断：转移至 Whisper 转写质量）
1. ✅ **阈值扫 + enrollment 加噪增强（本 agent 已完成）**
   - **增强有效**：enrollment 加噪 emb → 均 sim 0.218→0.348(+59%)；同阈值0.5 拒识率 0.87→0.72，正确率 0.04→0.11
   - 阈值扫：拒识率 0.12(0.1)–0.72(0.5) 可调，但**正确率天花板~14%**
   - **关键结论——瓶颈转移**：阈值旋钮解不了根本，**真正瓶颈从"声纹拒识"转到"Whisper 带噪+重叠转写质量"** → 下一步必须 frontend SE 增强（下条升最高优先）
   - 数据：`code/enroll_aug_full.json`；最佳工作点 threshold~0.2
2. **解 CAM++ 集成**（验证 wespeaker→CAM++ 改进，原生中文更鲁棒假设）
   - modelscope 本环境 import 卡 → **方案A**：独立干净 venv（`uv venv`）装 modelscope 跑 CAM++ 抽 emb 存文件，主 venv 读；**方案B**：sherpa-onnx 的 campplus/ERes2Net ONNX（隔离，不碰 transformers，推荐先试）
   - 脚本 `code/campp_vs_wespeaker.py` 已就绪，只待加载方式解
3. **frontend SE 增强**（RASTAR/VSAEC，论文 #6/#7）再抽声纹/diarization — 攻噪声根源

### 🟡 P1 — 拒识/效率/微调
4. **W5 LLM 拒识**（Qwen-2.5-3B 语义拒识）— 拒识 40% 当前只有 STNO 这一层机制，缺语义层
5. **D2 中文家居微调**（DiCoW 在家居指令微调）— CER 40% 提升
6. **D5 效率/W8 边缘部署**（待确认目标硬件 MCU/网关/服务器 → 向主办方确认）

### 🟢 P2 — 待外部输入
7. **确认测试集通道数**（决定 DSENet/KWS 多通道能否用，待主办方或看 A 集数据）
8. **真实比赛数据**来时：用 `eval_metrics.py` + `final/final_manifest.json` ground truth 跑全面 CER/拒识率

---

## 4. 关键命令速查
```bash
# 环境（每次新终端先 source）
source code/setenv.sh && export HF_HUB_OFFLINE=1

# enrollment→target 推理（方案B，独立脚本）
code/.venv/Scripts/python.exe code/enroll_infer.py \
  --enrollment E:/midea_target_asr/test_wav/dataset/raw/enrollment/target_long_01.wav \
  --recognition <wav或folder> --always-generate --enroll-augment --out-json <out.json>

# 能力评估 + 扫阈值
code/.venv/Scripts/python.exe code/eval_enrollment.py --enroll-json <out.json> --threshold 0.2 --label X

# 完整 DiCoW pipeline（T14 验证过，PIPELINE_EXIT=0）
code/.venv/Scripts/python.exe code/DiCoW-inference/inference.py \
  --dicow-model E:/hf_cache/DiCoW_v3_2 --diarization-model E:/hf_cache/diarizen-wavlm-large-s80-md \
  --input-folder <wav> --output-folder <out> --device cuda --verbose

# TTS 合成数据（WSL 跑，mimo-tts 限时免费窗口注意时效）
wsl.exe -d Ubuntu-22.04 bash -lc 'python3 /mnt/e/midea_target_asr/code/tts_dataset_gen.py'
# 组装矩阵（Windows）
code/.venv/Scripts/python.exe code/build_dataset.py

# fork patch 固化校验
bash code/DiCoW-inference/repro/apply_patches.sh
```

---

## 5. 已知坑（务必避开）
- **modelscope 1.37 在本环境 import 挂起**（`pipelines` 卡，datasets 正常）→ CAM++ 待独立 venv/sherpa-onnx
- **ESC-50 真实环境音下载失败**（GitHub HTTP/2 中断）→ 数据集用程序噪声(white/pink/babble)；真实环境音放 `test_wav/dataset/env_noise/` 重跑 build_dataset 自动加入
- **git user = midea-overnight-loop**（非用户本人 Panda_Lorrain；本 overnight loop 项目用此身份是设计，手动 commit 注意）
- **DiCoW-inference/ 整棵 gitignore**，但 `repro/` 已 force add 入 git；改动 fork 代码后 re-run apply_patches.sh 重新导出
- **wespeaker 拼写 VoxCeleb 带 x**（曾拼错 voceleb 404）
- **setenv.sh 必须 source**（设 HF_HOME=E:/hf_cache 等，否则权重找不到）
- **uv 禁裸 pip**；PyPI SSL 失败用清华源 `-i https://pypi.tuna.tsinghua.edu.cn/simple`
- **某些恶劣音频（ov100/snr-5）触发 DiariZen reconstruct `negative dimensions` crash** — enroll_infer 已加 try/except 容错跳过

---

## 6. 环境 & 资产位置
- **GPU**：RTX 4060 Laptop 8GB / torch 2.5.1+cu124 / Python 3.12 / `code/.venv`
- **权重**（全 E 盘）：`E:/hf_cache/DiCoW_v3_2`（0.89G）、`diarizen-wavlm-large-s80-md`、`hub/models--pyannote--wespeaker-voxceleb-resnet34-LM`（6.6M）
- **数据集**：`test_wav/dataset/raw/`（21条TTS种子，入git）/`final/`（450条矩阵，ignore，可 build_dataset 重建）/`final/final_manifest.json`（ground truth，入git）
- **文档**：`00_技术路线总纲与行动地图.md`（W1-W7 主路线）、`01-03` 模块/答辩、`papers/`（19 PDF）、`边缘部署规划.md`、`项目阶段盘点.md`
- **WSL**（Ubuntu-22.04）：mimo-tts 在此跑，key 在 `~/.hermes/.env`

---

## 7. 设计/决策记录（避免重复踩坑）
- **W3/W4 重新定义**：原"Personal VAD + CAM++"证伪（Personal VAD 无开源实现），改为 enrollment→wespeaker 锁定 target（复用已跑通组件）。设计文档 `docs/superpowers/specs/2026-06-28-enrollment-target-and-patch-fixation-design.md`
- **Part1 用方案B（独立脚本）非方案A（改pipeline.py）**：不碰跑通的 pipeline.py/inference.py，向后兼容天然
- **CAM++ 从"沉没成本"修正为"带噪鲁棒性数据驱动备选"**：wespeaker 中文+噪声退化实测后，CAM++ 原生中文有了引入理由
- **wespeaker 复用 diar._embedding**（不独立加载）：`diar._embedding(waveform[None,None])` → (batch,256)，零额外加载
- **STNO 数据结构**：`[N_speakers, 4, T@50Hz]` float32，4 行=(sil/target/nontarget/overlap) 每帧 one-hot
