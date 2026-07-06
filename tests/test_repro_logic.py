"""repro.py 单测: 种子确定性 + 模型路径 env override + 显存计量(CPU 安全)。

spec 2026-07-06-reproducibility-hardening-design §7。自跑兼容(os.environ 直接操作,
不依赖 pytest monkeypatch fixture)。
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
import numpy as np
from repro import (set_global_seed, resolve_model, resolve_df_base_dir,
                   reset_peak_gpu, peak_gpu_mib, REPO_IDS)


def test_set_global_seed_deterministic():
    """同 seed 两次 set_global_seed 后, torch.rand/np.random.rand 各自两次一致。"""
    import torch
    set_global_seed(42)
    t1 = torch.rand(5).clone()
    n1 = np.random.rand(5).copy()
    set_global_seed(42)
    t2 = torch.rand(5).clone()
    n2 = np.random.rand(5).copy()
    assert torch.equal(t1, t2), f"torch.rand 不确定: {t1} vs {t2}"
    assert np.array_equal(n1, n2), f"np.random.rand 不确定: {n1} vs {n2}"
    print("test_set_global_seed_deterministic OK")


def test_resolve_model_env_override():
    """env MODEL_VANILLA 设→override; 不设→HF repo id(4 个 key 值与 spec §4.2 一致)。"""
    os.environ["MODEL_VANILLA"] = "/local/cache/whisper"
    try:
        assert resolve_model("VANILLA") == "/local/cache/whisper"
    finally:
        del os.environ["MODEL_VANILLA"]
    assert resolve_model("VANILLA") == "openai/whisper-large-v3-turbo"
    assert resolve_model("DICOW") == "BUT-FIT/DiCoW_v3_2"
    assert resolve_model("DIAR") == "BUT-FIT/diarizen-wavlm-large-s80-md"
    assert resolve_model("QWEN") == "Qwen/Qwen2.5-3B-Instruct"
    print("test_resolve_model_env_override OK")


def test_resolve_df_base_dir():
    """env DF_MODEL_BASE_DIR 设→override; 不设→fallback 本地目录。"""
    os.environ["DF_MODEL_BASE_DIR"] = "/local/df"
    try:
        assert resolve_df_base_dir("fallback") == "/local/df"
    finally:
        del os.environ["DF_MODEL_BASE_DIR"]
    assert resolve_df_base_dir("fallback") == "fallback"
    print("test_resolve_df_base_dir OK")


def test_peak_gpu_mib_no_crash():
    """reset_peak_gpu/peak_gpu_mib 在无 CUDA 机器不崩; 返回 None 或数值。"""
    reset_peak_gpu()  # 无 CUDA 时内部 try/except 静默
    m = peak_gpu_mib()
    assert m is None or isinstance(m, (int, float)), f"peak 类型异常: {type(m)}"
    print(f"test_peak_gpu_mib_no_crash OK (peak={m})")


def test_repo_ids_keys():
    """REPO_IDS 含 4 个 key。"""
    assert set(REPO_IDS.keys()) == {"VANILLA", "DICOW", "DIAR", "QWEN"}
    print("test_repo_ids_keys OK")


if __name__ == "__main__":
    test_set_global_seed_deterministic()
    test_resolve_model_env_override()
    test_resolve_df_base_dir()
    test_peak_gpu_mib_no_crash()
    test_repo_ids_keys()
    print("ALL PASS")
