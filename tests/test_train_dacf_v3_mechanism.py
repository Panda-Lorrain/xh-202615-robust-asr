"""Small synthetic contract tests for train_dacf_v3_mechanism."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "code" / "experiments"
FIXTURE_ROOT = ROOT / "code" / "runs" / "dacf_v3_mechanism_test_fixture"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from dacf_v3_ecst import DACFV3ECSTOutput  # noqa: E402
from train_dacf_v3_mechanism import (  # noqa: E402
    MechanismContractError,
    MechanismTrainingConfig,
    TRAINING_PREREGISTRATION,
    cache_payload_sha256,
    load_feature_cache,
    run_training,
    sha256_array,
    validate_training_preregistration,
    write_training_preregistration,
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _scalar(value: str) -> np.ndarray:
    return np.asarray(value)


class FakeMechanismModel(nn.Module):
    """A tiny query-only model used to exercise the trainer plumbing."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, mixture: torch.Tensor, enrollment: torch.Tensor) -> DACFV3ECSTOutput:
        del mixture
        score = self.scale * enrollment[:, 0] * 5.0
        frames = score[:, None].expand(-1, 4)
        latent = frames[:, :, None].expand(-1, 4, 128)
        return DACFV3ECSTOutput(
            activity_logits=frames,
            presence_logits=score,
            activity_probability=torch.sigmoid(frames),
            query_conditioned_frames=latent,
        )


class TrainDacfV3MechanismTest(unittest.TestCase):
    def setUp(self) -> None:
        if FIXTURE_ROOT.exists():
            shutil.rmtree(FIXTURE_ROOT)
        self.cache_root = FIXTURE_ROOT / "cache"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self._build_cache()
        self.protocol_preregistration = FIXTURE_ROOT / "DATA_PROTOCOL_PREREGISTRATION.json"
        self.protocol_preregistration.write_text(
            json.dumps(
                {
                    "schema": "dacf-v3-official-aishell-protocol-v0.2",
                    "experiment_id": "synthetic-contract-test",
                    "dataset_a_used": False,
                    "source_corpus": "AISHELL-1",
                    "fixed_gate": {
                        "presence_threshold": 0.5,
                        "presence_auc": 0.85,
                        "activity_auc": 0.80,
                        "present_recall": 0.80,
                        "absent_rr": 0.95,
                        "query_response_mean": 0.20,
                        "query_response_group_bootstrap_ci_lower": 0.05,
                        "query_permutation_auc_drop_min": 0.15,
                    },
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        self.protocol_build_report = FIXTURE_ROOT / "DATA_PROTOCOL_BUILD_REPORT.json"
        self.protocol_build_report.write_text(
            json.dumps({"schema": "synthetic-build-report"}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        audit_code = EXPERIMENTS / "build_dacf_v3_protocol.py"
        self.protocol_audit_report = FIXTURE_ROOT / "DATA_PROTOCOL_REAUDIT.json"
        self.protocol_audit_report.write_text(
            json.dumps(
                {
                    "schema": "dacf-v3-official-aishell-reaudit-v0.1",
                    "protocol_schema": "dacf-v3-official-aishell-protocol-v0.2",
                    "dataset_a_used": False,
                    "source_corpus": "AISHELL-1",
                    "hard_negative_verified_count": 0,
                    "protocol_preregistration": self.protocol_preregistration.as_posix(),
                    "protocol_preregistration_sha256": _sha256_file(
                        self.protocol_preregistration
                    ),
                    "source_build_report": self.protocol_build_report.as_posix(),
                    "source_build_report_sha256": _sha256_file(
                        self.protocol_build_report
                    ),
                    "audit_code": audit_code.as_posix(),
                    "audit_code_sha256": _sha256_file(audit_code),
                    "manifest_sha256": {
                        split: {
                            path.as_posix(): _sha256_file(path)
                            for path in paths
                        }
                        for split, paths in self.protocol_manifest_paths.items()
                    },
                    "audit": {"all_cross_split_overlaps_zero": True},
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        if FIXTURE_ROOT.exists():
            shutil.rmtree(FIXTURE_ROOT)

    def _build_cache(self) -> None:
        mixture_dir = self.cache_root / "mixture"
        query_dir = self.cache_root / "query"
        mixture_dir.mkdir(parents=True, exist_ok=True)
        query_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict] = []
        for split, group_count in (("train", 2), ("dev", 2)):
            for group_index in range(group_count):
                group_id = f"{split}_mix_{group_index:04d}"
                mixture_speakers = {
                    "A": f"{split}_spk_{group_index}_a",
                    "B": f"{split}_spk_{group_index}_b",
                }
                features = np.zeros((128, 4), dtype=np.float32)
                features[0] = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
                mixture_path = mixture_dir / f"{split}_{group_index}.npz"
                np.savez_compressed(
                    mixture_path,
                    input_features=features,
                    feature_attention_mask=np.ones(4, dtype=np.int8),
                    mixture_sha256=np.asarray(f"audio-{split}-{group_index}"),
                    qwen_config_sha256=np.asarray("qwen-config"),
                    qwen_feature_spec_sha256=np.asarray("qwen-spec"),
                    feature_array_sha256=np.asarray(sha256_array(features)),
                )
                mixture_file_sha = _sha256_file(mixture_path)
                feature_sha = sha256_array(features)
                for role_index, (role, speaker, present) in enumerate(
                    (
                        ("present_A", f"{split}_spk_{group_index}_a", True),
                        ("present_B", f"{split}_spk_{group_index}_b", True),
                        ("absent_C", f"{split}_spk_{group_index}_c", False),
                    )
                ):
                    row_id = f"{group_id}__{role}"
                    embedding = np.zeros(512, dtype=np.float32)
                    embedding[0] = 1.0 if present else -1.0
                    embedding_view2 = embedding.copy()
                    activity = np.ones(4, dtype=np.float32) if present else np.zeros(4, dtype=np.float32)
                    query_path = query_dir / f"{split}__{row_id}.npz"
                    np.savez_compressed(
                        query_path,
                        enrollment_embedding=embedding,
                        enrollment_embedding_view2=embedding_view2,
                        target_activity=activity,
                        row_id=_scalar(row_id),
                        split=_scalar(split),
                        base_mixture_id=_scalar(group_id),
                        query_role=_scalar(role),
                        query_speaker_id=_scalar(speaker),
                        target_present=np.asarray(present),
                        source_corpus=_scalar("AISHELL-1"),
                        mixture_feature_sha256=_scalar(mixture_file_sha),
                        target_activity_sha256=_scalar("activity-source-sha"),
                        target_activity_array_sha256=_scalar(sha256_array(activity)),
                        enrollment_embedding_sha256=_scalar(sha256_array(embedding)),
                        enrollment_embedding_view2_sha256=_scalar(sha256_array(embedding_view2)),
                    )
                    rows.append(
                        {
                            "cache_schema": "dacf-v3-feature-cache-v0.1",
                            "id": row_id,
                            "split": split,
                            "base_mixture_id": group_id,
                            "query_role": role,
                            "query_role_id": role_index,
                            "query_role_id_used_as_model_input": False,
                            "query_speaker_id": speaker,
                            "mixture_speakers": mixture_speakers,
                            "target_present": present,
                            "source_corpus": "AISHELL-1",
                            "dataset_a_used": False,
                            "mixture_sha256": f"audio-{split}-{group_index}",
                            "mixture_feature": f"mixture/{mixture_path.name}",
                            "mixture_feature_sha256": mixture_file_sha,
                            "mixture_input_features_sha256": feature_sha,
                            "query_feature": f"query/{query_path.name}",
                            "query_npz_sha256": _sha256_file(query_path),
                            "target_activity_sha256": "activity-source-sha",
                            "target_activity_array_sha256": sha256_array(activity),
                            "speaker_ids": [speaker],
                            "resolved_audio_paths": {},
                            "resolved_source_paths": {},
                        }
                    )
        manifest = self.cache_root / "features_manifest.jsonl"
        _write_jsonl(manifest, rows)
        self.protocol_manifest_paths: dict[str, list[Path]] = {}
        for split in ("train", "dev"):
            protocol_manifest = FIXTURE_ROOT / "protocol_manifests" / f"{split}.jsonl"
            _write_jsonl(protocol_manifest, [{"split": split, "synthetic": True}])
            self.protocol_manifest_paths[split] = [protocol_manifest]
        self.protocol_manifest_paths["final"] = []
        report = {
            "schema": "dacf-v3-feature-cache-report-v0.1",
            "cache_schema": "dacf-v3-feature-cache-v0.1",
            "builder_code": (EXPERIMENTS / "build_dacf_v3_features.py").as_posix(),
            "builder_code_sha256": _sha256_file(
                EXPERIMENTS / "build_dacf_v3_features.py"
            ),
            "source_corpus": "AISHELL-1",
            "dataset_a_used": False,
            "qwen_config_sha256": "qwen-config",
            "feature_extractor_spec_sha256": "qwen-spec",
            "split_contract": {
                "splits": ["train", "dev"],
                "final_gate_split": None,
                "final_deferred": True,
            },
            "counts": {
                "rows": len(rows),
                "split_counts": {
                    "train": {"rows": 6, "groups": 2},
                    "dev": {"rows": 6, "groups": 2},
                },
            },
            "input_manifests": {
                split: [
                    {"path": path.as_posix(), "sha256": _sha256_file(path)}
                    for path in paths
                ]
                for split, paths in self.protocol_manifest_paths.items()
                if split in {"train", "dev"}
            },
            "manifest_sha256": _sha256_file(manifest),
            "cache_sha256": cache_payload_sha256(self.cache_root),
        }
        (self.cache_root / "cache_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _rewrite_report_hashes(self) -> None:
        manifest = self.cache_root / "features_manifest.jsonl"
        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in rows:
            query_path = self.cache_root / row["query_feature"]
            mixture_path = self.cache_root / row["mixture_feature"]
            row["query_npz_sha256"] = _sha256_file(query_path)
            row["mixture_feature_sha256"] = _sha256_file(mixture_path)
        _write_jsonl(manifest, rows)
        report = json.loads((self.cache_root / "cache_report.json").read_text(encoding="utf-8"))
        report["manifest_sha256"] = _sha256_file(manifest)
        report["cache_sha256"] = cache_payload_sha256(self.cache_root)
        (self.cache_root / "cache_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def test_validates_train_dev_only_and_group_contract(self) -> None:
        cache = load_feature_cache(self.cache_root)
        self.assertEqual([len(cache.groups[split]) for split in ("train", "dev")], [2, 2])
        self.assertEqual([row.role for row in cache.groups["train"][0].rows], ["present_A", "present_B", "absent_C"])
        self.assertEqual(cache.same_text_status["status"], "deferred")

    def test_default_training_contract_is_preregistered(self) -> None:
        config = MechanismTrainingConfig()
        self.assertEqual(config.seed, 2026080606)
        self.assertEqual(config.epochs, 20)
        self.assertEqual(config.groups_per_epoch, 96)
        self.assertEqual(config.fixed_updates, 1920)
        self.assertEqual(config.learning_rate, 3.0e-4)
        self.assertEqual(config.weight_decay, 1.0e-4)
        self.assertEqual(config.grad_clip_norm, 5.0)
        self.assertEqual(config.counterfactual_margin, 0.20)
        self.assertEqual(config.bootstrap_replicates, 2000)
        self.assertEqual(TRAINING_PREREGISTRATION["updates"], 1920)
        self.assertEqual(TRAINING_PREREGISTRATION["gradient_clip_norm"], 5.0)
        self.assertEqual(TRAINING_PREREGISTRATION["bootstrap_replicates"], 2000)
        self.assertEqual(TRAINING_PREREGISTRATION["cublas_workspace_config"], ":4096:8")

    def test_rejects_final_contract_before_loading_model_inputs(self) -> None:
        report_path = self.cache_root / "cache_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["split_contract"]["final_deferred"] = False
        report_path.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(MechanismContractError, "final_deferred"):
            load_feature_cache(self.cache_root)

    def test_rejects_forbidden_query_role_id_even_when_hashes_are_updated(self) -> None:
        manifest_rows = [
            json.loads(line)
            for line in (self.cache_root / "features_manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        query_path = self.cache_root / manifest_rows[0]["query_feature"]
        with np.load(query_path, allow_pickle=False) as data:
            values = {key: data[key] for key in data.files}
        values["query_role_id"] = np.asarray(0, dtype=np.int64)
        np.savez_compressed(query_path, **values)
        self._rewrite_report_hashes()
        with self.assertRaisesRegex(MechanismContractError, "forbidden query_role_id"):
            load_feature_cache(self.cache_root)

    def test_rejects_payload_sha_tamper(self) -> None:
        mixture_path = self.cache_root / "mixture" / "train_0.npz"
        mixture_path.write_bytes(mixture_path.read_bytes() + b"tamper")
        with self.assertRaisesRegex(MechanismContractError, "mixture_feature_sha256 mismatch"):
            load_feature_cache(self.cache_root)

    def test_rejects_padded_frames_and_role_id_disagreement(self) -> None:
        mixture_path = self.cache_root / "mixture" / "train_0.npz"
        with np.load(mixture_path, allow_pickle=False) as data:
            values = {key: data[key] for key in data.files}
        values["feature_attention_mask"] = np.asarray([1, 1, 1, 0], dtype=np.int8)
        np.savez_compressed(mixture_path, **values)
        self._rewrite_report_hashes()
        with self.assertRaisesRegex(MechanismContractError, "frames must all be valid"):
            load_feature_cache(self.cache_root)

        self.setUp()
        manifest_path = self.cache_root / "features_manifest.jsonl"
        rows = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rows[0]["query_role_id"] = 2
        _write_jsonl(manifest_path, rows)
        report_path = self.cache_root / "cache_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["manifest_sha256"] = _sha256_file(manifest_path)
        report["cache_sha256"] = cache_payload_sha256(self.cache_root)
        report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
        with self.assertRaisesRegex(MechanismContractError, "disagrees with query_role"):
            load_feature_cache(self.cache_root)

    def test_rejects_query_mixture_speaker_membership_mismatch(self) -> None:
        manifest_path = self.cache_root / "features_manifest.jsonl"
        rows = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        foreign = next(
            row["query_speaker_id"]
            for row in rows
            if row["split"] == "train"
            and row["base_mixture_id"] == "train_mix_0001"
            and row["query_role"] == "absent_C"
        )
        for row in rows:
            if row["split"] == "train" and row["base_mixture_id"] == "train_mix_0000":
                row["mixture_speakers"]["A"] = foreign
        _write_jsonl(manifest_path, rows)
        self._rewrite_report_hashes()
        with self.assertRaisesRegex(
            MechanismContractError, "query/mixture speaker membership mismatch"
        ):
            load_feature_cache(self.cache_root)

    def test_external_preregistration_binds_cache_protocol_and_source_hashes(self) -> None:
        prereg_path = FIXTURE_ROOT / "TRAINING_PREREGISTRATION.json"
        payload = write_training_preregistration(
            self.cache_root,
            self.protocol_preregistration,
            self.protocol_audit_report,
            prereg_path,
            allow_synthetic_cache=True,
        )
        cache = load_feature_cache(self.cache_root)
        validated = validate_training_preregistration(prereg_path, cache)
        self.assertEqual(validated["training_contract"], TRAINING_PREREGISTRATION)
        self.assertTrue(validated["feature_cache"]["final_deferred"])
        self.assertTrue(validated["protocol_reaudit"]["all_cross_split_overlaps_zero"])
        self.assertIn("code/experiments/train_dacf_v3_mechanism.py", payload["source_sha256"])

        payload["training_contract"]["seed"] += 1
        prereg_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        with self.assertRaisesRegex(MechanismContractError, "settings changed"):
            validate_training_preregistration(prereg_path, cache)

    def test_formal_training_refuses_to_start_without_external_preregistration(self) -> None:
        with self.assertRaisesRegex(MechanismContractError, "independently written"):
            run_training(self.cache_root, FIXTURE_ROOT / "formal_without_prereg")

    def test_trains_fake_model_emits_checkpoint_and_scoped_verdict(self) -> None:
        output = FIXTURE_ROOT / "output"
        report = run_training(
            self.cache_root,
            output,
            config=MechanismTrainingConfig(
                epochs=1,
                groups_per_epoch=2,
                bootstrap_replicates=20,
                device="cpu",
            ),
            model_factory=FakeMechanismModel,
            allow_prereg_override=True,
        )
        self.assertTrue((output / "final_checkpoint.pt").is_file())
        self.assertTrue((output / "mechanism_report.json").is_file())
        self.assertIn(report["verdict"], {"conditional-GO", "implementation-NO-GO"})
        self.assertEqual(report["verdict"], "implementation-NO-GO")
        self.assertFalse(report["verdict_scope"]["final_opened"])
        self.assertEqual(report["verdict_scope"]["cer"], "not measured")
        self.assertEqual(report["dev_evaluation_count"], 1)
        self.assertTrue(report["checkpoint_frozen_before_dev"])
        self.assertEqual(report["threshold"], 0.5)
        self.assertEqual(report["protocol_version"], "dacf-v3-mechanism-protocol-v0.2")
        self.assertEqual(report["same_text_metric"]["status"], "deferred")
        self.assertEqual(report["same_text_metric"]["state"], "not_constructed")
        self.assertFalse(report["same_text_metric"]["current_mechanism_gate"])
        self.assertEqual(report["same_text_metric"]["threshold_before_qwen"], 0.75)
        self.assertEqual(report["checkpoint"]["sha256_before_dev"], report["checkpoint"]["sha256_after_dev"])
        self.assertEqual(
            report["training"]["checkpoint_policy"],
            "fixed_updates_final_checkpoint_then_one_dev_evaluation",
        )
        self.assertEqual(report["training_preregistration"]["actual"]["updates"], 2)
        self.assertFalse(report["training_preregistration"]["matches_default_contract"])
        actual_prereg = report["training_preregistration"]["actual"]
        self.assertEqual(actual_prereg["gradient_clip_norm"], 5.0)
        self.assertEqual(actual_prereg["precision"], "FP32")
        self.assertEqual(actual_prereg["cublas_workspace_config"], ":4096:8")
        self.assertTrue(actual_prereg["torch_deterministic_algorithms"])
        self.assertTrue(actual_prereg["cudnn_deterministic"])
        self.assertFalse(actual_prereg["cudnn_benchmark"])
        self.assertEqual(actual_prereg["scheduler"], "none")
        self.assertFalse(actual_prereg["early_stop"])
        self.assertFalse(actual_prereg["hyperparameter_scan"])
        self.assertIn("presence_auc", report["dev"]["metrics"])
        self.assertIsNone(report["dev"]["metrics"]["activity_frame_auc"])
        self.assertEqual(
            report["dev"]["metrics"]["activity_frame_auc_scope"],
            "present rows only; absent all-zero frames excluded",
        )
        self.assertIn("cyclic +1", report["dev"]["query_permutation_policy"])
        self.assertIn("speaker", report["dev"]["cluster_bootstrap_ci_95"])
        self.assertNotIn(
            "same_text_different_speaker_auc",
            report["mechanism_gate"]["fixed_gate"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
