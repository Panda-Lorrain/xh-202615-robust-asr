#!/usr/bin/env python3
"""Paired official-CER comparison for raw and TSE-enhanced Qwen outputs."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_metrics import CERMetric  # noqa: E402
from text_utils import digit_postproc, to_simplified  # noqa: E402


def submit_norm(text: str) -> str:
    return digit_postproc(to_simplified(text or ""))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--raw-json", required=True)
    parser.add_argument("--enhanced-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--route-manifest",
        help="optional overlap-fallback manifest with per-segment features",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260729)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in open(args.manifest, encoding="utf-8")
        if line.strip()
    ]
    raw = json.load(open(args.raw_json, encoding="utf-8"))
    enhanced = json.load(open(args.enhanced_json, encoding="utf-8"))
    routes_by_id = {}
    if args.route_manifest:
        routes_by_id = {
            str(row["id"]): row
            for row in (
                json.loads(line)
                for line in open(args.route_manifest, encoding="utf-8")
                if line.strip()
            )
        }
    refs = [row["ref"] for row in rows]
    raw_hyps = [submit_norm(raw.get(row["id"], "")) for row in rows]
    enhanced_hyps = [
        submit_norm(enhanced.get(row["id"], "")) for row in rows
    ]

    raw_metric = CERMetric()
    raw_metric.update(raw_hyps, refs)
    raw_result = raw_metric.compute()
    enhanced_metric = CERMetric()
    enhanced_metric.update(enhanced_hyps, refs)
    enhanced_result = enhanced_metric.compute()
    raw_errors = np.asarray(
        [row["errors"] for row in raw_result["per_sample"]]
    )
    enhanced_errors = np.asarray(
        [row["errors"] for row in enhanced_result["per_sample"]]
    )
    target_chars = np.asarray(
        [row["target_chars"] for row in raw_result["per_sample"]]
    )

    rng = np.random.default_rng(args.seed)
    bootstrap_delta = []
    for _ in range(args.bootstrap_samples):
        indices = rng.integers(0, len(rows), len(rows))
        bootstrap_delta.append(
            float(
                (
                    enhanced_errors[indices].sum()
                    - raw_errors[indices].sum()
                )
                / max(int(target_chars[indices].sum()), 1)
            )
        )

    oracle_hyps = [
        enhanced_hyps[index]
        if enhanced_errors[index] < raw_errors[index]
        else raw_hyps[index]
        for index in range(len(rows))
    ]
    oracle_metric = CERMetric()
    oracle_metric.update(oracle_hyps, refs)
    per_sample = []
    for index, row in enumerate(rows):
        uid = str(row["id"])
        route_row = routes_by_id.get(uid, {})
        per_sample.append(
            {
                "id": uid,
                "ref": refs[index],
                "raw_hyp": raw_hyps[index],
                "enhanced_hyp": enhanced_hyps[index],
                "raw_errors": int(raw_errors[index]),
                "enhanced_errors": int(enhanced_errors[index]),
                "delta_errors": int(
                    enhanced_errors[index] - raw_errors[index]
                ),
                "segments": route_row.get("segments", []),
            }
        )
    result = {
        "n": len(rows),
        "raw_cer": raw_result["cer"],
        "enhanced_cer": enhanced_result["cer"],
        "delta_enhanced_minus_raw": (
            enhanced_result["cer"] - raw_result["cer"]
        ),
        "delta_bootstrap_ci95": [
            float(np.percentile(bootstrap_delta, 2.5)),
            float(np.percentile(bootstrap_delta, 97.5)),
        ],
        "better": int(np.sum(enhanced_errors < raw_errors)),
        "same": int(np.sum(enhanced_errors == raw_errors)),
        "worse": int(np.sum(enhanced_errors > raw_errors)),
        "oracle_fallback_cer": oracle_metric.compute()["cer"],
        "per_sample": per_sample,
    }
    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in result.items()
             if key != "per_sample"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
