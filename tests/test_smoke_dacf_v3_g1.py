import hashlib
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "code" / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from smoke_dacf_v3_g1 import _read_rows  # noqa: E402


class DACFV3G1ContractTests(unittest.TestCase):
    def setUp(self):
        self.fixture = ROOT / "code" / "runs" / "dacf_v3_g1_contract_fixture"
        self.fixture.mkdir(parents=True, exist_ok=True)
        self.manifest = self.fixture / "features_manifest.jsonl"

    def _row(self, role, present, mixture_sha="a" * 64):
        suffix = role[-1]
        return {
            "base_mixture_id": "g0",
            "query_role": role,
            "target_present": present,
            "mixture_sha256": mixture_sha,
            "recognition_audio": str(self.fixture / "mixture.wav"),
            "enrollment_audio": str(self.fixture / f"e{suffix}.wav"),
            "enrollment_audio_view2": str(self.fixture / f"e{suffix}b.wav"),
            "source_corpus": "AISHELL-1",
            "dataset_a_used": False,
        }

    def _write(self, rows):
        self.manifest.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    def test_accepts_exact_abc_contract_without_using_role_as_model_input(self):
        rows = [
            self._row("present_A", True),
            self._row("present_B", True),
            self._row("absent_C", False),
        ]
        self._write(rows)
        loaded = _read_rows(self.manifest, "g0")
        self.assertEqual({row["query_role"] for row in loaded}, {"present_A", "present_B", "absent_C"})

    def test_rejects_nonidentical_mixture_and_dataset_a(self):
        rows = [
            self._row("present_A", True),
            self._row("present_B", True, "b" * 64),
            self._row("absent_C", False),
        ]
        self._write(rows)
        with self.assertRaisesRegex(ValueError, "byte-identical"):
            _read_rows(self.manifest, "g0")
        rows[1]["mixture_sha256"] = "a" * 64
        rows[2]["dataset_a_used"] = True
        self._write(rows)
        with self.assertRaisesRegex(ValueError, "Dataset-A"):
            _read_rows(self.manifest, "g0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
