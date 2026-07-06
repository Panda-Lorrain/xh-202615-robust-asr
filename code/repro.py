"""可复现性公共模块: 全局种子固定 + 模型路径 resolve + GPU 峰值显存计量。

横切关注点归一(spec docs/superpowers/specs/2026-07-06-reproducibility-hardening-design.md §4.2)。
5 脚本(submit_infer/enroll_infer/se_denoise/llm_reject/noise_classify)共用。三 venv
(.venv/.venv_se/.venv_llm)都能 import —— repro.py 放 code/, 脚本所在目录自动入 sys.path。
延迟 import torch/numpy → submit_infer 主进程 stdlib-only 也能 import 不崩。

模型路径: 代码 default 走 HF repo id(主办方核查环境联网 HF 自动下载, C11);
本地开发 setenv.sh 设 MODEL_* env 指现有缓存 override, 免重下。
"""
import os
import random

# 4 个 HF 模型 repo id(公开可下载, Plan agent 2026-07-06 核实)
REPO_IDS = {
    "VANILLA": "openai/whisper-large-v3-turbo",
    "DICOW":   "BUT-FIT/DiCoW_v3_2",
    "DIAR":    "BUT-FIT/diarizen-wavlm-large-s80-md",
    "QWEN":    "Qwen/Qwen2.5-3B-Instruct",
}
# env key 映射: 本地 setenv.sh 设 MODEL_* 指缓存 override, 主办方不设则走 repo id
_ENV_KEYS = {"VANILLA": "MODEL_VANILLA", "DICOW": "MODEL_DICOW",
             "DIAR": "MODEL_DIAR", "QWEN": "MODEL_QWEN"}


def set_global_seed(seed=42):
    """固定 random + numpy + torch + cuda + cudnn, 保证可复现(FAQ 核查硬要求 2)。

    延迟 import(torch/numpy 在函数体内 try/except), 兼容 submit_infer 主进程无 torch。
    cudnn.deterministic=True / benchmark=False 牺牲少量速度换卷积确定性
    (初赛效率不优先 C4, 可接受)。
    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def resolve_model(name):
    """name ∈ REPO_IDS → env override(本地缓存) 或 HF repo id(主办方自动下载)。

    argparse default 求值时调(setenv.sh 在 python 启动前 source, env 已设)。
    """
    return os.environ.get(_ENV_KEYS[name], REPO_IDS[name])


def resolve_df_base_dir(fallback):
    """DF3 例外: 不走 HF repo id(init_df 接目录路径), env override 或 fallback 本地目录。

    原始权重来自 GitHub Rikorose/DeepFilterNet release(非 HF), README 说明获取方式。
    """
    return os.environ.get("DF_MODEL_BASE_DIR", fallback)


def reset_peak_gpu():
    """重置峰值显存计数器(每条推理循环开始调, 记每条整条峰值, 不被二次 generate 重置)。无 CUDA 静默。"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def peak_gpu_mib():
    """返回峰值显存(MiB)。无 CUDA/无 torch 返回 None(不崩)。"""
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
    except ImportError:
        pass
    return None
