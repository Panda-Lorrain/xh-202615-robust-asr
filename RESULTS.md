# W1 实测结果（XH-202615 overnight loop）

> 首个真实推理数据，**解除"零实测"答辩红线**。明早看此文件 + PROGRESS.md。
> 生成：2026-06-27。

## 🎉 里程碑：DiCoW baseline 最小推理跑通

| 指标 | 值 | 意义 |
|---|---|---|
| 模型 | BUT-FIT/DiCoW_v3_2（Whisper-large-v3-turbo + FDDT） | TS-ASR 开源 baseline |
| **参数量** | **0.89G** | ✅ 印证 turbo ≈809M（**非 1.5B**，03 答辩红线 7 正确） |
| 加载时间 | 2.6s | |
| 推理（30s 音频） | 1.73s | |
| **RTF** | **0.058** | 远快于实时；本机 4060 已如此，L20（48GB）会更优。**20% 效率分前景极佳** |
| **峰值显存** | **2.13GB** | 8GB 4060 绰绰有余，L20 无压力 |
| 测试音频 | EN2002a_30s.wav（英文会议 30s） | DiariZen 自带样例 |

### 转写样例（全-target STNO，即转写整段所有人）
> "yeah yeah but i do not know about you but usually in windows right click does not do anything there is a it opens a menu..."（通顺，质量好）

### FDDT 配置印证（来自 config.json，答辩素材）
- `use_fddt: true` / `use_initial_fddt: true`
- `fddt_init: "disparagement"`（= 论文的**抑制式初始化**）
- `fddt_is_diagonal: true`（对角约束）
- `fddt_use_target/non_target/overlap/silence: true`（全四类 STNO）
- `encoder_layers: 32` + `decoder_layers: 4`（**turbo 配置**，印证 4 层 decoder）

### 说明（诚实）
- 这是**最小推理**：构造"全 target"STNO（整段当目标说话人），验证模型加载 + forward + generate + RTF，**非真 target-speaker 转写**。
- 真 target-speaker 需 DiariZen diarization 生成真实 STNO（完整 pipeline，下一步补）。

### 环境信息
- torch 2.5.1+cu124 / transformers 4.42.4 / python 3.12 / RTX 4060 Laptop 8GB
- 权重 `E:/hf_cache/DiCoW_v3_2`（model.safetensors 3.6G）
- 缓存全落 E 盘（HF_HOME/UV_CACHE_DIR，禁 C 盘）✅
- 脚本 `code/minimal_infer.py`

---

## 下一步（overnight 续做）
1. **完整 pipeline**（DiariZen diarization → 真 STNO → 真 target-speaker）：装 pyannote/DiariZen，跑 inference.py
2. **中文音频测试**：验证 DiCoW 中文能力（题目是中文）
3. **W6 评测脚本**：CER 计算 + RTF 批量
4. **W2 数据仿真 pipeline**：中文多人 + SNR −5~5dB + 重叠 0-100%
