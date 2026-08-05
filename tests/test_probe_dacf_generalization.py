"""Lightweight contract tests for the DACF mini-G2 probe."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


EXPERIMENTS = Path(__file__).resolve().parents[1] / "code" / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from probe_dacf_generalization import (  # noqa: E402
    _conditional_gate,
    _roc_auc,
)


class ProbeDACFGeneralizationTests(unittest.TestCase):
    def test_auc_is_tie_aware(self):
        self.assertEqual(_roc_auc([0.9, 0.8], [0.2, 0.1]), 1.0)
        self.assertEqual(_roc_auc([0.5], [0.5]), 0.5)
        self.assertEqual(_roc_auc([0.1], [0.9]), 0.0)

    def test_gate_requires_every_preregistered_condition(self):
        passing = {
            "roc_auc": 0.80,
            "present_recall": 0.75,
            "absent_rr": 0.75,
            "query_permutation_response_mean": 0.20,
        }
        passed, checks = _conditional_gate(passing)
        self.assertTrue(passed)
        self.assertTrue(all(checks.values()))

        for key in passing:
            failed = dict(passing)
            failed[key] -= 0.01
            passed, checks = _conditional_gate(failed)
            self.assertFalse(passed, key)
            self.assertIn(False, checks.values())

    def test_auc_rejects_missing_class(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            _roc_auc([], [0.1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
