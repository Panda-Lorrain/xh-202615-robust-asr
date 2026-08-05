from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from unittest.mock import patch


EXPERIMENTS = Path(__file__).resolve().parents[1] / "code" / "experiments"
FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "runs"
    / "dacf_scale_falsification_test_fixture"
)
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from build_dacf_counterfactual import build_dacf_counterfactual  # noqa: E402
from build_dacf_scale_falsification import (  # noqa: E402
    ACTIVITY_AUC_GATE,
    BUILD_SEED,
    MATCHER_UPDATES,
    N_FINAL_MIXTURES,
    N_TRAIN_MIXTURES,
    N_VAL_MIXTURES,
    SOURCE_SPLITS,
    build_scale_falsification,
    read_excluded_speakers,
)


def _write_prior_manifest(path: Path) -> None:
    row = {
        "dataset_a_used": False,
        "query_speaker_id": "old_C",
        "enrollment_spk": "old_C",
        "mixture_speakers": {"A": "old_A", "B": "old_B"},
        "interferer_spks": ["old_A", "old_B"],
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_prior_manifest_excludes_every_named_speaker(tmp_path: Path) -> None:
    manifest = tmp_path / "prior.jsonl"
    _write_prior_manifest(manifest)
    speakers, hashes = read_excluded_speakers([manifest])
    assert speakers == {"old_A", "old_B", "old_C"}
    assert list(hashes) == [manifest.resolve().as_posix()]
    assert len(next(iter(hashes.values()))) == 64


def test_scale_wrapper_has_one_fixed_configuration(tmp_path: Path) -> None:
    manifest = tmp_path / "prior.jsonl"
    _write_prior_manifest(manifest)
    aishell_root = tmp_path / "data_aishell"
    aishell_root.mkdir()
    source_items = []
    for index in range(243):
        speaker = "old_A" if index == 0 else f"new_{index:03d}"
        for utterance in range(2):
            source_items.append(
                {
                    "wav": str(aishell_root / f"{speaker}_{utterance}.wav"),
                    "spk": speaker,
                    "utt": f"{speaker}_{utterance}",
                    "ref": "read speech",
                    "split": "train",
                    "source_corpus": "AISHELL-1",
                }
            )

    captured = {}

    def fake_build(items, out_dir, **kwargs):
        captured["items"] = list(items)
        captured["out_dir"] = Path(out_dir)
        captured["kwargs"] = dict(kwargs)
        return {
            "dataset_a_used": False,
            "total_mixtures": kwargs["n_train_mixtures"]
            + kwargs["n_val_mixtures"]
            + kwargs["n_final_mixtures"],
        }

    output = tmp_path / "scale"
    with patch(
        "build_dacf_scale_falsification.load_aishell_items",
        return_value=source_items,
    ), patch(
        "build_dacf_scale_falsification.build_dacf_counterfactual",
        side_effect=fake_build,
    ):
        result = build_scale_falsification(
            aishell_root, output, exclude_manifests=[manifest]
        )

    assert all(item["spk"] != "old_A" for item in captured["items"])
    assert captured["kwargs"]["n_train_mixtures"] == N_TRAIN_MIXTURES
    assert captured["kwargs"]["n_val_mixtures"] == N_VAL_MIXTURES
    assert captured["kwargs"]["n_final_mixtures"] == N_FINAL_MIXTURES
    assert captured["kwargs"]["seed"] == BUILD_SEED
    assert captured["kwargs"]["max_mixtures"] == 80
    prereg = json.loads((output / "PREREGISTRATION.json").read_text(encoding="utf-8"))
    assert prereg["source_splits"] == list(SOURCE_SPLITS)
    assert prereg["matcher"]["updates"] == MATCHER_UPDATES
    assert prereg["fixed_gate"]["activity_auc"] == ACTIVITY_AUC_GATE
    assert prereg["selection_policy"].startswith("one scale-only run")
    assert prereg["build"]["final_mixtures"] == N_FINAL_MIXTURES
    assert result["preregistered"] is True
    assert (output / "build_report.json").is_file()


def test_scale_output_must_be_new_or_empty(tmp_path: Path) -> None:
    manifest = tmp_path / "prior.jsonl"
    _write_prior_manifest(manifest)
    aishell_root = tmp_path / "data_aishell"
    aishell_root.mkdir()
    output = tmp_path / "scale"
    output.mkdir()
    (output / "stale.txt").write_text("stale", encoding="utf-8")
    try:
        build_scale_falsification(
            aishell_root, output, exclude_manifests=[manifest]
        )
    except FileExistsError as exc:
        assert "new or empty" in str(exc)
    else:
        raise AssertionError("non-empty output directory must be rejected")


def test_core_default_cap_cannot_be_raised_past_audited_scale_limit() -> None:
    try:
        build_dacf_counterfactual(
            [],
            "unused",
            n_train_mixtures=1,
            n_val_mixtures=0,
            max_mixtures=81,
        )
    except ValueError as exc:
        assert "audited research cap" in str(exc)
    else:
        raise AssertionError("max_mixtures above the audited cap must be rejected")


if __name__ == "__main__":
    root = FIXTURE_ROOT.resolve(strict=False)
    runs_root = (Path(__file__).resolve().parents[1] / "code" / "runs").resolve()
    if runs_root not in root.parents:
        raise RuntimeError(f"unsafe fixture root: {root}")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    for index, test in enumerate(
        (
            test_prior_manifest_excludes_every_named_speaker,
            test_scale_wrapper_has_one_fixed_configuration,
            test_scale_output_must_be_new_or_empty,
        )
    ):
        fixture = root / f"case_{index}"
        fixture.mkdir()
        test(fixture)
    test_core_default_cap_cannot_be_raised_past_audited_scale_limit()
    print("ALL PASS: DACF scale falsification contract")
