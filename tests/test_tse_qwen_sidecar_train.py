import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from tse_qwen_sidecar_train import _validate_rows, _validate_split


def _row(uid: str, speaker: str, ref: str) -> dict:
    return {
        "id": uid,
        "recognition_audio": f"synthetic/{uid}.wav",
        "target_src": f"E:/midea_datasets/data_aishell/wav/train/{speaker}/{uid}.wav",
        "target_spk": speaker,
        "ref": ref,
        "enrollment_embedding": f"cache/{uid}.npy",
        "target_activity": f"activity/{uid}.npy",
    }


def test_manifest_guard_rejects_dataset_a():
    row = _row("x", "S0001", "测试")
    row["recognition_audio"] = "E:/midea_target_asr/datasetA/pos/x.wav"
    try:
        _validate_rows([row], require_scale=False)
    except ValueError as error:
        assert "Dataset A leakage" in str(error)
    else:
        raise AssertionError("Dataset A path was not rejected")


def test_split_guard_rejects_same_speaker():
    try:
        _validate_split(
            [_row("train", "S0001", "训练")],
            [_row("val", "S0001", "验证")],
        )
    except ValueError as error:
        assert "speakers overlap" in str(error)
    else:
        raise AssertionError("speaker overlap was not rejected")
