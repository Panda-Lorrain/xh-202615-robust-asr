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
- [ ] 2. clone SE-DiCoW（github.com/BUTSpeechFIT/TS-ASR-Whisper）→ `E:\midea_papers\code\TS-ASR-Whisper\`
- [ ] 3. uv 建项目环境 + 装依赖（torch、transformers、datasets 等，落 E 盘）
- [ ] 4. 下 HF 权重 `BUT-FIT/SE_DiCoW`（落 E:\hf_cache）
- [ ] 5. 跑 SE-DiCoW 官方 demo（验证环境通）
- [ ] 6. W6 评测脚本：RTF 测量（L20 之外的本机 GPU 基线）+ CER 工具
- [ ] 7. W2 数据仿真 pipeline 骨架（AISHELL/WenetSpeech + MUSAN + 重叠混合）
- [ ] 8. 用仿真数据测 SE-DiCoW 初步中文效果 → git 提交初步结果
- [ ] 9. 若余时：W3 PVAD / W4 声纹 接口骨架

## 📝 进度日志（append-only）
- 2026-06-27 T0：创建 .gitignore + PROGRESS.md（loop 大脑）；启动环境探查

## 🚧 遇阻 / 决策记录
- _（首轮填）_

## ➡️ 下一步
- 看环境探查结果：有 GPU→全力 W1；无 GPU→转 W2/W6 骨架（CPU 可做）+ 记录 GPU 缺失为风险
