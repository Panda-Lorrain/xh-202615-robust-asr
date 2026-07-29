#!/usr/bin/env python3
"""Build a Dataset-A *evaluation-only* manifest for paired Sidecar testing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pos-jsonl", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--slice-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.dataset_root)
    slice_dir = Path(args.slice_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(args.pos_jsonl, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            source = json.loads(line)
            uid = f"cmd_{source['id']}"
            enrollment = root / source["唤醒音频"]
            recognition = slice_dir / f"{uid}.wav"
            if not enrollment.exists() or not recognition.exists():
                continue
            rows.append(
                {
                    "id": uid,
                    "enrollment_audio": str(enrollment),
                    "recognition_audio": str(recognition),
                    "ref": source["识别文本"],
                    "dataset_role": "A_TEST_ONLY",
                }
            )
    if not rows:
        raise ValueError("no complete Dataset-A evaluation pairs")
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[A eval manifest] rows={len(rows)} -> {output}")


if __name__ == "__main__":
    main()
