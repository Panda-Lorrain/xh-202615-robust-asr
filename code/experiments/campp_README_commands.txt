# 线B产出: CAM++(sherpa-onnx)集成 + 声纹鲁棒性对比 —— GPU/CPU 实验命令清单
# =====================================================================
# 环境定型: 方案B sherpa-onnx(已解 modelscope 挂起)。
#   venv = E:/midea_target_asr/code/.venv_campp (Python3.12, sherpa-onnx 1.13.3 + onnx 1.22 + ort 1.27 + librosa + numpy/soundfile)
#   模型 = E:/hf_cache/campplus/campplus.onnx (Wespeaker/wespeaker-voxceleb-campplus, voxceleb_CAM++.onnx 29MB, 512d, 已加 sherpa meta framework=wespeaker)
# =====================================================================
# 已完成的 CPU 验证(本 agent 跑通, 无需重跑):
#   - test_campp_load.py: import 不挂起, dim=512, 3条音频抽emb成功(范数77/145/101)
#   - campp_margin_diag.py: 8对样本 CAM++ margin=+0.128 (target均0.539 vs nontarget均0.411) → 模型在TTS可区分
#   - campp_enroll_full.py: 450矩阵整段均sim=0.121 (低于wespeaker 0.218), 100%<0.5阈值
# =====================================================================

# === 命令1 [CPU, 快, 主agent可跑]: wespeaker 4样本对照侧 ===
# 产出 wespeaker_quick_result.json, 与 campp_quick_result.json(CAM++侧已跑) 对照看带噪衰减
source code/setenv.sh && export HF_HUB_OFFLINE=1
code/.venv/Scripts/python.exe code/campp_vs_wespeaker_wespeaker.py

# === 命令2 [CPU, 快]: 对比两模型 4样本干净/带噪/异人 sim 表 ===
# 手工对照 campp_quick_result.json vs wespeaker_quick_result.json
# 关注: +5dB带噪 sim 衰减(谁更鲁棒)、同人-异人 margin(谁区分更好)
code/.venv_campp/Scripts/python.exe -c "import json; print('CAM++:', json.load(open('code/campp_quick_result.json')))"

# === 命令3 [GPU, 重, 主线对照基准]: wespeaker 450矩阵 per-speaker sim(已有 enroll_wespeaker_full.json) ===
# 这是不公平对照基线: wespeaker 走 diarization 分离后 per-speaker 抽 emb, CAM++(命令5)同理才公平
source code/setenv.sh && export HF_HUB_OFFLINE=1
code/.venv/Scripts/python.exe code/eval_enrollment.py --enroll-json code/enroll_wespeaker_full.json --label wespeaker-perspk

# === 命令4 [CPU, 已跑]: CAM++ 整段(无diar)450矩阵, 对照 wespeaker-perspk 看混音污染 ===
code/.venv_campp/Scripts/python.exe code/eval_enrollment.py --enroll-json code/campp_enroll_full.json --label CAMpp-integral

# === 命令5 [关键·决定CAM++是否有价值]: CAM++ per-speaker 公平对照 ===
# 需复用 DiariZen diarization 分段(主.venv+GPU), 但声纹换 CAM++(.venv_campp sherpa-onnx)。
# 两 venv 不能同进程加载, 方案: 在主.venv 跑 diar 存分段→CAM++ 读分段抽emb。
# → 主agent: 复制 enroll_infer.py 的 diar+collect_clean_audio 逻辑, 把 get_emb() 换成
#   subprocess 调 .venv_campp 的 campp 抽emb, 或用 sherpa-onnx OfflineSpeakerDiarization
#   (自带 campplus+segmentation, 纯ONNX, GPU可选, 见命令6)。
#   推荐: 直接用命令6(sherpa-onnx 自带 diar), 一条龙纯 CAM++ pipeline。

# === 命令6 [GPU可选/纯CPU, 推荐·一条龙CAM++ diar]: sherpa-onnx OfflineSpeakerDiarization ===
# sherpa-onnx 自带 OfflineSpeakerDiarization (segmentation_pyannote + campplus emb + clustering)
# 全 ONNX, 与 transformers 零冲突, 可纯CPU或GPU。这是 CAM++ 公平 per-speaker 对比的最干净路径。
# 需补: 下载 segmentation_pyannote 模型 + clustering。
# 参考: https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/index.html
#   .venv_campp/Scripts/python.exe -c "import sherpa_onnx; help(sherpa_onnx.OfflineSpeakerDiarization)"
# (本 agent 未跑此步——属进一步深挖, 见 verdict 聚焦建议)
