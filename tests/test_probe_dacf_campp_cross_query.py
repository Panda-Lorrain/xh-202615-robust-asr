from __future__ import annotations

import hashlib
import inspect
import json
import shutil
import sys
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "code" / "experiments"
FIXTURE_ROOT = ROOT / "code" / "runs" / "dacf_campp_cross_query_probe_20260806" / "_test_fixture"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from dacf_campp_cross_query import DACFCAMPPQueryMatcher  # noqa: E402
from probe_dacf_campp_cross_query import (  # noqa: E402
    FIXED_PRESENCE_THRESHOLD,
    GATE_THRESHOLDS,
    _conditional_gate,
    _forward_group,
    _two_view_gate,
    _validate_and_load_cache,
    compute_group_loss,
    run_probe,
)


class ProbeDACFCAMPPCrossQueryTests(unittest.TestCase):
    @contextmanager
    def temporary_directory(self, prefix: str):
        root = FIXTURE_ROOT.resolve(strict=False)
        runs_root = (ROOT / "code" / "runs").resolve()
        if runs_root not in root.parents:
            raise RuntimeError(f"unsafe fixture root: {root}")
        path = root / prefix.strip("_")
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
        yield path

    @staticmethod
    def _write_cache(
        root: Path,
        *,
        train_groups: int = 1,
        val_groups: int = 1,
        final_groups: int = 0,
    ) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, object]] = []
        rng = np.random.default_rng(23)
        for split, count in (
            ("train", train_groups),
            ("val", val_groups),
            ("final", final_groups),
        ):
            for index in range(count):
                group_id = f"{split}_group_{index}"
                tokens = rng.normal(size=(8, 512)).astype(np.float16)
                mixture_path = root / "mixture" / f"{group_id}.npz"
                mixture_path.parent.mkdir(parents=True, exist_ok=True)
                mixture_sha = hashlib.sha256(f"{split}:{index}".encode()).hexdigest()
                np.savez_compressed(
                    mixture_path,
                    tokens=tokens,
                    mixture_sha256=np.asarray(mixture_sha),
                )
                for role_index, role in enumerate(
                    ("present_A", "present_B", "absent_C")
                ):
                    query_path = root / "query" / f"{group_id}_{role}.npz"
                    query_path.parent.mkdir(parents=True, exist_ok=True)
                    embedding = rng.normal(size=512).astype(np.float32)
                    embedding_view2 = (
                        embedding + 0.01 * rng.normal(size=512)
                    ).astype(np.float32)
                    np.savez_compressed(
                        query_path,
                        embedding=embedding,
                        embedding_view2=embedding_view2,
                    )
                    activity_path = root / "activity" / f"{group_id}_{role}.npy"
                    activity_path.parent.mkdir(parents=True, exist_ok=True)
                    activity = np.zeros(16, dtype=np.float32)
                    if role != "absent_C":
                        activity[3:9] = 1.0
                    np.save(activity_path, activity)
                    rows.append(
                        {
                            "split": split,
                            "base_mixture_id": group_id,
                            "id": f"{group_id}__{role}",
                            "query_role": role,
                            "query_role_id": role_index,
                            "target_present": role != "absent_C",
                            "mixture_feature": f"mixture/{mixture_path.name}",
                            "query_feature": f"query/{query_path.name}",
                            "target_activity": f"activity/{activity_path.name}",
                            "mixture_sha256": mixture_sha,
                            "query_speaker_id": f"{split}_speaker_{index}_{role_index}",
                            "dataset_a_used": False,
                        }
                    )
        manifest = root / "features_manifest.jsonl"
        manifest.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        return manifest

    def test_schema_and_shared_mixture_tensor(self):
        with self.temporary_directory("_probe_schema_") as tmp:
            manifest = self._write_cache(Path(tmp), train_groups=1, val_groups=1)
            loaded = _validate_and_load_cache(
                Path(tmp), manifest=manifest, strict_counts=False
            )
            self.assertEqual(len(loaded.train_groups), 1)
            group = loaded.train_groups[0]
            self.assertEqual(set(group.queries), {"present_A", "present_B", "absent_C"})
            self.assertEqual(group.mixture_tokens.shape, (8, 512))
            self.assertFalse(loaded.audit["dataset_a_used"])
            self.assertEqual(loaded.audit["train_rows"], 3)
            self.assertEqual(loaded.audit["val_rows"], 3)
            # The group has one object, not three independently loaded A/B/C
            # tensors; both views are available for every query row.
            self.assertIs(group.mixture_tokens, group.mixture_tokens)
            for query in group.queries.values():
                self.assertEqual(tuple(query.embedding.shape), (512,))
                self.assertEqual(tuple(query.embedding_view2.shape), (512,))

    def test_cross_split_mixture_sha_and_path_overlap_are_rejected(self):
        with self.temporary_directory("_probe_split_overlap_") as tmp:
            root = Path(tmp)
            manifest = self._write_cache(root, train_groups=1, val_groups=1)
            rows = [
                json.loads(line)
                for line in manifest.read_text(encoding="utf-8").splitlines()
                if line
            ]
            train = next(row for row in rows if row["split"] == "train")
            for row in rows:
                if row["split"] == "val":
                    row["mixture_feature"] = train["mixture_feature"]
                    row["mixture_sha256"] = train["mixture_sha256"]
            manifest.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "artifact overlap"):
                _validate_and_load_cache(root, manifest=manifest, strict_counts=False)

    def test_query_role_id_is_audit_only_and_not_a_model_argument(self):
        signature = inspect.signature(DACFCAMPPQueryMatcher.forward)
        self.assertEqual(
            list(signature.parameters),
            ["self", "mixture_tokens", "enrollment_embedding"],
        )
        source = inspect.getsource(_forward_group)
        self.assertNotIn("query_role_id", source)

    def test_fixed_gate_and_threshold_are_not_tunable(self):
        self.assertEqual(FIXED_PRESENCE_THRESHOLD, 0.5)
        passing = {
            "auc": GATE_THRESHOLDS["auc"],
            "present_recall": GATE_THRESHOLDS["present_recall"],
            "absent_rr": GATE_THRESHOLDS["absent_rr"],
            "query_response_mean": GATE_THRESHOLDS["query_response_mean"],
            "activity_auc": GATE_THRESHOLDS["activity_auc"],
        }
        passed, checks = _conditional_gate(passing)
        self.assertTrue(passed)
        self.assertTrue(all(checks.values()))
        for key in passing:
            failed = dict(passing)
            failed[key] -= 0.001
            self.assertFalse(_conditional_gate(failed)[0], key)

    def test_small_synthetic_group_forwards_both_views_and_backpropagates(self):
        with self.temporary_directory("_probe_backward_") as tmp:
            manifest = self._write_cache(Path(tmp), train_groups=1, val_groups=1)
            loaded = _validate_and_load_cache(
                Path(tmp), manifest=manifest, strict_counts=False
            )
            model = DACFCAMPPQueryMatcher(
                feature_dim=512, query_dim=32, logit_scale=10.0, top_fraction=0.25
            )
            total, components, outputs = compute_group_loss(
                model, loaded.train_groups[0], device=torch.device("cpu")
            )
            self.assertTrue(torch.isfinite(total))
            self.assertEqual(set(components), {
                "presence_bce",
                "activity_bce",
                "abc_margin",
                "view_logits_consistency",
                "total",
            })
            self.assertEqual(outputs[0].frame_logits.shape, outputs[1].frame_logits.shape)
            total.backward()
            self.assertGreater(
                float(model.mixture_projection.weight.grad.abs().sum()), 0.0
            )
            self.assertGreater(
                float(model.enrollment_projection.weight.grad.abs().sum()), 0.0
            )

    def test_synthetic_trainer_reports_both_views_and_fixed_audit_flags(self):
        with self.temporary_directory("_probe_run_") as tmp:
            cache_root = Path(tmp)
            manifest = self._write_cache(cache_root, train_groups=1, val_groups=1)
            result = run_probe(
                cache_root,
                manifest=manifest,
                updates=2,
                seed=20260806,
                device="cpu",
                strict_counts=False,
                output_json=cache_root / "matcher_result.json",
                checkpoint=cache_root / "matcher_checkpoint.pt",
            )
            self.assertIn(result["verdict"], {"conditional-GO", "implementation-NO-GO"})
            self.assertEqual(set(result["train"]), {"main", "view2"})
            self.assertEqual(set(result["val"]), {"main", "view2"})
            self.assertFalse(result["dataset_a_used"])
            self.assertFalse(result["cer_measured"])
            self.assertFalse(result["hard_negative_verified"])
            self.assertFalse(result["threshold_tuned"])
            self.assertEqual(result["config"]["fixed_presence_threshold"], 0.5)
            self.assertLess(result["model"]["trainable_parameter_count"], 100_000)
            self.assertTrue(Path(result["artifacts"]["result_json"]).exists())
            self.assertTrue(Path(result["artifacts"]["checkpoint"]).exists())

    def test_final_holdout_alone_decides_scale_gate(self):
        with self.temporary_directory("_probe_final_") as tmp:
            cache_root = Path(tmp)
            manifest = self._write_cache(
                cache_root, train_groups=1, val_groups=1, final_groups=0
            )
            base = _validate_and_load_cache(
                cache_root, manifest=manifest, strict_counts=False
            )
            cache_with_final = replace(base, final_groups=base.val_groups)
            with patch(
                "probe_dacf_campp_cross_query._validate_and_load_cache",
                return_value=cache_with_final,
            ):
                result = run_probe(
                    cache_root,
                    manifest=manifest,
                    updates=1,
                    seed=20260806,
                    device="cpu",
                    strict_counts=False,
                    output_json=cache_root / "matcher_result.json",
                    checkpoint=cache_root / "matcher_checkpoint.pt",
                )
            self.assertEqual(result["decision_split"], "final")
            self.assertIn("final", result)
            self.assertFalse(result["val_used_for_selection"])
            self.assertFalse(result["final_used_for_selection"])
            self.assertEqual(result["gate"], _two_view_gate(result["final"])[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
