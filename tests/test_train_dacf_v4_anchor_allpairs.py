from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "code" / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from train_dacf_v4_anchor_allpairs import (  # noqa: E402
    PREREGISTRATION_SCHEMA,
    PROTOCOL_VERSION,
    REPORT_SCHEMA,
    _source_hashes,
)


class AnchorAllPairsTrainerTests(unittest.TestCase):
    def test_protocol_names_are_distinct_from_learned_projection_run(self) -> None:
        self.assertIn("anchor", PROTOCOL_VERSION)
        self.assertIn("anchor", PREREGISTRATION_SCHEMA)
        self.assertIn("anchor", REPORT_SCHEMA)

    def test_source_hashes_bind_every_imported_research_component(self) -> None:
        hashes = _source_hashes()
        self.assertEqual(
            set(hashes),
            {
                "code/experiments/dacf_v4_relation.py",
                "code/experiments/dacf_v4_anchor_relation.py",
                "code/experiments/dacf_v4_objective.py",
                "code/experiments/train_dacf_v3_mechanism.py",
                "code/experiments/train_dacf_v4_allpairs.py",
                "code/experiments/train_dacf_v4_anchor_allpairs.py",
            },
        )
        self.assertTrue(all(len(value) == 64 for value in hashes.values()))


if __name__ == "__main__":
    unittest.main()
