from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "code" / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from build_dacf_v4b_audio import (  # noqa: E402
    AUDIO_SCHEMA,
    MANIFEST_SCHEMA_VERSION,
)


class V4BAudioContractTests(unittest.TestCase):
    def test_v4b_schemas_are_not_v3_aliases(self) -> None:
        self.assertIn("v4b", AUDIO_SCHEMA)
        self.assertIn("v4b", MANIFEST_SCHEMA_VERSION)

    def test_audio_builder_source_declares_final_deferred(self) -> None:
        source = (EXPERIMENTS / "build_dacf_v4b_audio.py").read_text(encoding="utf-8")
        self.assertIn('"official_test_loaded": False', source)
        self.assertIn('"final_deferred": True', source)
        self.assertNotIn('load_aishell_items', source)

    def test_old_a_only_speed_transform_is_explicitly_disabled(self) -> None:
        source = (EXPERIMENTS / "build_dacf_v4b_audio.py").read_text(encoding="utf-8")
        self.assertIn('params["target_speed_rate"] = 1.0', source)
        self.assertIn('"enrollment_src_view2"', source)


if __name__ == "__main__":
    unittest.main()
