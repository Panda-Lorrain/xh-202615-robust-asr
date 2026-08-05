"""Lightweight static/JSON contract tests for the D8 real-Qwen smoke.

These tests intentionally do not import Qwen, load a checkpoint, touch CUDA,
or run the smoke.  The real command is an explicit separate gate.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "experiments" / "smoke_dacf_qwen_real.py"
RESULT = ROOT / "code" / "runs" / "dacf_qwen_real_smoke_20260806" / "result.json"


class D8RealSmokeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SCRIPT.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source, filename=str(SCRIPT))

    def test_script_parses_and_exposes_only_contract_entrypoints(self):
        names = {
            node.name
            for node in self.tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertIn("_run", names)
        self.assertIn("_write_result", names)
        self.assertIn("main", names)

    def test_static_guards_for_offline_pure_asr_and_no_optimizer(self):
        lowered = self.source.lower()
        self.assertIn("local_files_only", lowered)
        self.assertIn("hf_hub_offline", lowered)
        self.assertIn("dataseta", lowered)
        self.assertIn("dataset_a", lowered)
        self.assertIn("torch.no_grad", lowered)
        self.assertIn(".backward()", lowered)
        self.assertNotIn("torch.optim", lowered)
        self.assertNotIn(".step()", lowered)

    def test_dataset_a_guard_rejects_path_without_loading_qwen(self):
        spec = importlib.util.spec_from_file_location("d8_real_smoke_contract", SCRIPT)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        with self.assertRaises(RuntimeError):
            module._assert_no_dataset_a(r"E:\data\DatasetA\sample.wav", "test")
        module._assert_no_dataset_a(r"E:\midea_datasets\data_aishell\wav\train\S0001\x.wav", "test")

    def test_result_json_schema_if_real_smoke_has_run(self):
        if not RESULT.is_file():
            self.skipTest("D8 real smoke has not produced result.json yet")
        result = json.loads(RESULT.read_text(encoding="utf-8"))
        for key in (
            "schema_version",
            "role",
            "verdict",
            "direction",
            "status",
            "contract",
            "paths",
            "runtime",
            "audio_layer",
            "checks",
            "gradients",
            "error",
        ):
            self.assertIn(key, result)
        self.assertEqual(result["role"], "D8")
        self.assertEqual(result["direction"], "direction-unresolved")
        self.assertIn(result["verdict"], {"conditional-GO", "implementation-NO-GO"})
        self.assertFalse(result["contract"]["optimizer_used"])
        self.assertFalse(result["contract"]["dataset_a_read"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
