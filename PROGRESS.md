# 自主 Loop 进度跟踪（XH-202615 overnight）

> **这是自主 loop 的"大脑"。每次 wakeup 第一件事：读本文件，确认目标、看到哪了、下一步、遇阻什么。**
> 用户授权 overnight 自主推进，明早看初步效果。
> 创建：2026-06-27。

---

## 🎯 总目标（不可跑偏的锚点）

在**无真实数据 + 通道数未确认**的前提下，推进**通道无关**的可执行项，把方案从纸面推向可验证：

- **W1（最高优先）**：跑通 SE-DiCoW 开源 baseline → 解除"零实测"答辩红线，产出第一批真实 CER/RTF 数据
- **W2**：数据仿真 pipeline 骨架（中文多人 + SNR −5~5dB + 重叠 0-100%，不依赖真实数据）
- **W6**：评测脚本（CER + 拒识率 + RTF 测量）
- 全程 **git 提交**每个里程碑；**下载/缓存全部落 E 盘**（禁 C 盘）

## ⛔ 硬约束（违反即跑偏）
1. **下载/缓存禁落 C 盘**：权重、HF cache、uv cache、pip cache、torch hub 全部重定向到 E 盘（HF_HOME / UV_CACHE_DIR / PIP_CACHE_DIR / TORCH_HOME）
2. **不碰 C 盘大文件**；不删记忆里的禁删项（`E:\python` 5.6G、`E:\python期末大作业`）
3. **Python 用 uv**（uv run / uv add），禁裸 pip install；新工具落 `E:\Tools\`
4. **GitHub/HF 走代理** 127.0.0.1:7897（或 hf-mirror 镜像 fallback）
5. **给 Windows exe 传路径用 E:\ 或 E:/**，勿用 /e/
6. **只做 W1/W2/W6 通道无关项**；遇阻记录后跳下一可做项，不强求、不卡死
7. **每次 wakeup 先读本文件**；每个里程碑 git commit + 更新本文件

## 🔍 环境探查结果（首轮填）
- GPU：✅ NVIDIA RTX 4060 Laptop, **8GB VRAM**, 驱动 572.83（**可跑推理！全力 W1**；显存偏紧，长音频需分块）
- CUDA/驱动：572.83（CUDA 12.x 兼容）；torch 待装 CUDA 版
- uv/python：✅ uv 0.11.24 / Python 3.12.13 / conda 25.7（用 uv，不用 conda）
- torch：❌ 未装 → uv 装 CUDA 12.x 版
- 磁盘：E 自由 ~119G（够，权重落此），C 自由 ~112G（**禁用**）
- 代理：✅ HTTPS/HTTP_PROXY=127.0.0.1:7897 已设

## 📋 Loop 步骤清单（勾选推进）
- [ ] 0. git init + 首次提交文档体系（00/01/02/03 + 精读 + paper_index + CLAUDE.md + pdf + _txt）
- [ ] 1. 配置 C 盘重定向：HF_HOME / UV_CACHE_DIR / TORCH_HOME → E 盘；HF_ENDPOINT=hf-mirror
- [ ] 2. clone SE-DiCoW（github.com/BUTSpeechFIT/TS-ASR-Whisper）→ `E:\midea_target_asr\code\TS-ASR-Whisper\`
- [ ] 3. uv 建项目环境 + 装依赖（torch、transformers、datasets 等，落 E 盘）
- [ ] 4. 下 HF 权重 `BUT-FIT/SE_DiCoW`（落 E:\hf_cache）
- [ ] 5. 跑 SE-DiCoW 官方 demo（验证环境通）
- [ ] 6. W6 评测脚本：RTF 测量（L20 之外的本机 GPU 基线）+ CER 工具
- [ ] 7. W2 数据仿真 pipeline 骨架（AISHELL/WenetSpeech + MUSAN + 重叠混合）
- [ ] 8. 用仿真数据测 SE-DiCoW 初步中文效果 → git 提交初步结果
- [ ] 9. 若余时：W3 PVAD / W4 声纹 接口骨架

## 📝 进度日志（append-only）
- 2026-06-27 T0：创建 .gitignore + PROGRESS.md（loop 大脑）；启动环境探查
- 2026-06-27 T1：✅ 环境探查→**有 RTX4060 8GB**，全力 W1；✅ git init + 首次提交 `c1cb258`(54 文件：文档+19 PDF+16 txt)；✅ setenv.sh(C盘重定向 HF/uv/torch→E 盘)；🔄 clone SE-DiCoW(bg bzuw3i1ez)
- 2026-06-27 T2：✅ clone TS-ASR-Whisper(训练仓库)完成；⚠️ **决策：它是训练框架(Hydra/lhotse/SLURM/DiariZen)，纯推理 baseline 改用专门推理仓库 `BUTSpeechFIT/DiCoW`**；🔄 clone DiCoW 推理仓库(bg) + 🔄 uv 建 .venv 装 torch CUDA(bg)
- 2026-06-27 T3：✅ clone DiCoW 推理仓库完成；分析推理流程=inference.py(DiCoWPipeline: DiariZen diarization→STNO→DiCoW 目标转写)，默认模型 `BUT-FIT/DiCoW_v3_2` + `BUT-FIT/diarizen-wavlm-large-s80-md`；⚠️ 坑：① requirements 有 `-e git+DiariZen#egg=pyannote.audio` + 要 submodule ② 推理仓库要 torch 2.5.1(我装 2.7.0)；**策略：优先装全跑完整 pipeline，坑则退最小推理(模型加载+generate 测 RTF)**；🔄 torch 装(bg bxfs8du58)
- 2026-06-27 T4：⚠️ **torch 装成 CPU 版**(2.7.0+cpu, cuda False——清华源默认 CPU)；✅ git-lfs 3.7.1 / ffmpeg 7.1.1 已装；⚠️ DiariZen submodule 空(--depth1 未拉)；🔄 **重装 torch 2.5.1+cu124 CUDA**(pytorch.org 官方 wheel) + 🔄 拉 DiariZen submodule
- 2026-06-27 T5：✅ **torch 2.5.1+cu124 CUDA 可用**(`cuda_avail True`, RTX4060 识别)；✅ DiariZen submodule 完整(diarizen/+dscore/+pyannote-audio/)；🔄 装 transformers 4.55 + 最小推理依赖(numpy/librosa/soundfile) + 🔄 DiCoW_v3_2 权重下载(bg bbjukzw0k, hf-mirror)
- 2026-06-27 T6：✅ transformers 4.55+numpy 1.26.4 装好；✅ **DiCoW_v3_2 权重完整**(model.safetensors 3.6G lfs 拉全 + trust_remote_code 全套代码)；✅ config 印证 FDDT(`fddt_init=disparagement`抑制式/diagonal/四类)+ turbo(decoder 4 层)；✅ DiCoW_v3_2=非 enrollment；✅ 测试音频 EN2002a_30s.wav；✅ 最小推理脚本 minimal_infer.py 就绪；▶ 跑最小推理(验证模型+RTF)
- 2026-06-27 T6b：🔧 trust_remote_code 踩坑连环解：① modules 缓存缺 SCBs/coattention/decoding/utils(手动补全)② transformers 4.55 缺 WHISPER_ATTENTION_CLASSES(降级 4.42.4)③ 缺 pandas(补装)；均解。
- 2026-06-27 T7：🎉 **W1 最小推理跑通！** DiCoW_v3_2 加载 2.6s, **params 0.89G**(印证 turbo), 30s 音频推理 1.73s, **RTF=0.058**, 峰值显存 **2.13GB**(8GB 够)；**解除"零实测"答辩红线**；转 EN2002a 英文会议通顺；⚠️ 全-target STNO(非真 target-speaker)；结果见 RESULTS.md
- 2026-06-27 T8：✅ diarizen 权重下完(531M)；🔧 装完整 pipeline 依赖(numpy<2/hf-hub<1.0/gradio/pyannote/lhotle/einops/setuptools<81 等连环冲突，均解)；⛔ **完整 pipeline 止损**：DiariZen/pyannote.audio fork 在 Python3.12+现代setuptools 深层不兼容(namespace 冲突→patch pkgutil→circular import)，~10 轮 debug 无底洞。**完整 pipeline 留作后续**(用 DiariZen 官方 conda Python3.11 环境，或独立 diarization 进程产 STNO 再喂 DiCoW)。**minimal 推理成果已 secure(git c2e5b8f)**。▶ 转 W6 评测/STNO 实验，ScheduleWakeup 接管
- 2026-06-27 T9：🎯 **STNO 控制实验成功**(答辩黄金素材)：A全target转398字 / B前半target只转前半 / C后半target只转后半 / **D全non-target→0字拒识**。验证 FDDT/STNO 机制可控制(target转/silence跳/non-target拒识)，印证 `fddt_init=disparagement` 抑制式初始化；组合主线 STNO→转写/拒识 链路证实，是完整 pipeline 的机制替代验证。结果见 RESULTS.md。git 提交。
- 2026-06-27 T10：⚠️ 中文测试受阻：edge-tts(MS TTS)在系统级 Clash 全局代理下 SSL 失败(speech.platform.bing.com 证书 mismatch)，unset 环境变量无效。**W2 TTS 合成需关系统代理**或换本地 TTS。非阻塞，等真实中文数据或关代理后补。
- 2026-06-27 T11：✅ **W6 评测脚本完成**(`code/eval_metrics.py`)：CER(中文字符级,jiwer)/RTF/拒识指标(precision/recall/f1/reject_rate)/batch_cer 批量。自测通过(CER 0.0/0.111/1.0, RTF 0.058, 拒识 F1 0.87)。真实数据来时直接复用。git 提交。
- 2026-06-27 T12：✅ **W2 数据仿真 pipeline 完成**(`code/simulate_pipeline.py`)：add_noise(按SNR)/mix_overlap(重叠率)/simulate/build_set 批量。自测通过(重叠功率2x正确，写出 sim_test wav)。真实单人中文音频来时造重叠带噪集。git 提交。
- 2026-06-27 T13：🔄 **完整 pipeline 第二次尝试**(ScheduleWakeup)：✅ 依赖/namespace/circular **全部解决**(diarizen import OK！补 pyannote __init__ pkgutil namespace patch + audio __init__ __version__ 硬编码 + matplotlib + pytorch-metric-learning)。⛔ 但 diarization 权重加载受阻：from_pretrained 需额外 **wespeaker embedding 模型**(pyannote/wespeaker-voceleb-resnet34-LM)，hf-mirror clone 超时(网络不稳)+ snapshot_download 连不上 Hub。**最终止损**，留作后续。
- 2026-06-28 **项目改名** `midea_papers` → `midea_target_asr`（已从资料收集演进为参赛工程，旧名名不副实）。**阶段A 文件内容已全改**（8文件替换 + CLAUDE.md标题改贴切 + 3脚本项目内路径相对化 `__file__` + setenv注释通用化，py_compile通过）；**阶段B 目录物理改名 + projects目录迁移待用户会话外执行**（Windows不能重命名cwd；memory随projects目录搬走，勿漏）。**下一个agent先读 `RENAME_HANDOFF.md` 核查改名状态并 commit 这批改动。**

## 🚧 完整 pipeline 解决方案（供后续 turn/人，已 90% 通）
**已解决的 patch（在 fork 代码，DiCoW-inference/ 不入 git）**：
1. `DiariZen/pyannote-audio/pyannote/__init__.py`：`declare_namespace` → `pkgutil.extend_path`（解 namespace 冲突）
2. `DiariZen/pyannote-audio/pyannote/audio/__init__.py`：顶部加 `__version__="0.0.0-fork-patch"`（解 circular import）
3. 依赖补齐：toml / setuptools<81(pkg_resources) / einops / semver / matplotlib / pytorch-metric-learning；transformers 4.42.4 + numpy<2 + hf-hub<1.0
**待解决（下次）**：
- 预下 wespeaker：`git clone https://hf-mirror.com/pyannote/wespeaker-voceleb-resnet34-LM E:/hf_cache/wespeaker-voceleb-resnet34-LM`（网络稳时）
- patch `DiariZen/diarizen/pipelines/inference.py` from_pretrained 用本地路径（diarizen_hub=E:/hf_cache/diarizen-wavlm-large-s80-md, embedding_model=E:/hf_cache/wespeaker-voceleb-resnet34-LM/pytorch_model.bin）
- PYTHONPATH=`DiCoW-inference;DiCoW-inference/DiariZen;DiCoW-inference/DiariZen/pyannote-audio`
- 跑 `inference.py --input-folder <wav> --dicow-model E:/hf_cache/DiCoW_v3_2 --diarization-model <any>`（from_pretrained patch 后忽略 diarization-model）
- **建议环境**：DiariZen 官方 conda Python 3.11（避开 Python 3.12 兼容坑）+ 稳定网络预下三权重

## 📊 overnight 阶段小结（T1-T10，git 4 提交）
- ✅ **W1 minimal 推理跑通**：DiCoW_v3_2，RTF=0.058，params 0.89G，GPU 峰值 2.13GB（解除零实测红线）
- ✅ **STNO 机制验证**：target→转/silence→跳/non-target→拒识（答辩黄金素材，FDDT 内建拒识证实）
- ✅ **环境完备**：torch 2.5.1+cu124 / transformers 4.42.4 / 依赖齐 / 缓存落 E 盘
- ⛔ 完整 pipeline 止损（pyannote namespace/circular，用官方 conda 环境）
- ⛔ 中文 TTS 受阻（系统代理 SSL，关代理后补）
- ▶ **下一步（ScheduleWakeup 04:50 接管）**：W6 评测脚本(jiwer CER+RTF) / W2 数据仿真(关代理后 TTS 或下 AISHELL) / 完整 pipeline(官方 conda) / STNO 深入实验

## 🚧 遇阻 / 决策记录
- **TS-ASR-Whisper 是训练仓库不是推理**：README 明确"inference-only 用 BUTSpeechFIT/DiCoW"。训练仓库留作后续微调用，baseline 跑通用推理仓库。
- **显存 8GB 偏紧**：Whisper-large-v3-turbo + SE-DiCoW 推理需分块/限 batch，RTF 测试时监控显存。

## 🚧 遇阻 / 决策记录
- _（首轮填）_

## ➡️ 下一步
- 看环境探查结果：有 GPU→全力 W1；无 GPU→转 W2/W6 骨架（CPU 可做）+ 记录 GPU 缺失为风险
