"""Build the DACF-v4b role-rotation and global source-WAV schedule.

This stage writes no generated audio.  It proves that the intended
multi-positive protocol is feasible before an audio builder is allowed to
run.  Train uses 48 official AISHELL-train speakers and dev uses 12 official
AISHELL-dev speakers.  Six rounds give every speaker exactly 2xA, 2xB and
2xC.  Every source WAV path and source-byte SHA is consumed exactly once.

The official AISHELL test split is deliberately never loaded here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from build_dacf_counterfactual import (
    _assert_not_dataset_a,
    load_aishell_items,
    normalize_source_items,
)


SCHEDULE_SCHEMA = "dacf-v4b-role-rotation-schedule-v0.1"
AUDIT_SCHEMA = "dacf-v4b-role-rotation-audit-v0.1"
SOURCE_CORPUS = "AISHELL-1"
ROLES = ("A", "B", "C")
ROUNDS = 6
SOURCES_PER_SPEAKER = 16
SPLIT_SPEAKER_COUNTS: Mapping[str, int] = {"train": 48, "dev": 12}
PROTOCOL_TO_SOURCE_SPLIT: Mapping[str, str] = {"train": "train", "dev": "dev"}
DEFAULT_SEEDS: Mapping[str, int] = {"train": 2026080648, "dev": 2026080612}
MAX_PAIR_USE = 2
PAIR_SEARCH_ATTEMPTS = 5000


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _stable_seed(base_seed: int, text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return base_seed ^ int.from_bytes(digest[:8], "big")


def _pair(left: str, right: str) -> tuple[str, str]:
    if left == right:
        raise ValueError("a speaker cannot be paired with itself")
    return tuple(sorted((left, right)))


def _role_for(speaker_index: int, round_index: int) -> str:
    return ROLES[(speaker_index + round_index) % len(ROLES)]


def build_role_rotation(
    speaker_ids: Sequence[str],
    *,
    seed: int,
    rounds: int = ROUNDS,
    pair_search_attempts: int = PAIR_SEARCH_ATTEMPTS,
) -> list[dict[str, Any]]:
    """Return deterministic groups with exact roles and bounded pair reuse."""

    speakers = tuple(str(value) for value in speaker_ids)
    if len(speakers) < 3 or len(speakers) % 3:
        raise ValueError("speaker count must be a positive multiple of three")
    if len(set(speakers)) != len(speakers):
        raise ValueError("speaker ids must be unique")
    if rounds < 3 or rounds % 3:
        raise ValueError("round count must be a positive multiple of three")
    if pair_search_attempts < 1:
        raise ValueError("pair_search_attempts must be positive")

    pair_counts: Counter[tuple[str, str]] = Counter()
    used_ab_pairs: set[tuple[str, str]] = set()
    used_triples: set[tuple[str, str, str]] = set()
    emitted: list[dict[str, Any]] = []
    groups_per_round = len(speakers) // 3

    for round_index in range(rounds):
        role_pools = {
            role: [
                speaker
                for speaker_index, speaker in enumerate(speakers)
                if _role_for(speaker_index, round_index) == role
            ]
            for role in ROLES
        }
        if any(len(values) != groups_per_round for values in role_pools.values()):
            raise AssertionError("role pools are not exactly balanced")

        best: tuple[tuple[int, int, tuple[tuple[str, str, str], ...]], list[tuple[str, str, str]]] | None = None
        for attempt in range(pair_search_attempts):
            rng = random.Random(seed + round_index * 1_000_003 + attempt * 97)
            pools = {role: list(values) for role, values in role_pools.items()}
            for values in pools.values():
                rng.shuffle(values)
            candidate = list(zip(pools["A"], pools["B"], pools["C"]))
            triples = [tuple(group) for group in candidate]
            if any(len(set(group)) != 3 for group in candidate):
                continue
            if any(_pair(a, b) in used_ab_pairs for a, b, _ in candidate):
                continue
            if any(tuple(sorted(group)) in used_triples for group in candidate):
                continue
            increments = Counter(
                pair
                for a, b, c in candidate
                for pair in (_pair(a, b), _pair(a, c), _pair(b, c))
            )
            if any(pair_counts[pair] + count > MAX_PAIR_USE for pair, count in increments.items()):
                continue
            reused = sum(
                count
                for pair, count in increments.items()
                if pair_counts[pair] > 0
            )
            max_after = max(
                pair_counts[pair] + count for pair, count in increments.items()
            )
            score = (reused, max_after, tuple(sorted(triples)))
            if best is None or score < best[0]:
                best = (score, candidate)
                if reused == 0 and max_after == 1:
                    break
        if best is None:
            raise RuntimeError(
                f"could not construct round {round_index} under pair constraints"
            )

        for round_group_index, (speaker_a, speaker_b, speaker_c) in enumerate(best[1]):
            group_id = f"r{round_index:02d}_g{round_group_index:02d}"
            emitted.append(
                {
                    "round_index": round_index,
                    "round_group_index": round_group_index,
                    "group_id": group_id,
                    "speakers": {"A": speaker_a, "B": speaker_b, "C": speaker_c},
                }
            )
            used_ab_pairs.add(_pair(speaker_a, speaker_b))
            used_triples.add(tuple(sorted((speaker_a, speaker_b, speaker_c))))
            pair_counts.update(
                (_pair(speaker_a, speaker_b), _pair(speaker_a, speaker_c), _pair(speaker_b, speaker_c))
            )

    expected_groups = rounds * groups_per_round
    if len(emitted) != expected_groups:
        raise AssertionError("role rotation group accounting failed")
    audit_role_rotation(emitted, speakers=speakers, rounds=rounds)
    return emitted


def audit_role_rotation(
    groups: Sequence[Mapping[str, Any]],
    *,
    speakers: Sequence[str],
    rounds: int = ROUNDS,
) -> Mapping[str, Any]:
    speaker_set = set(speakers)
    role_counts: dict[str, Counter[str]] = defaultdict(Counter)
    round_counts: dict[int, Counter[str]] = defaultdict(Counter)
    pair_counts: Counter[tuple[str, str]] = Counter()
    ab_pairs: list[tuple[str, str]] = []
    group_ids: set[str] = set()
    for group in groups:
        group_id = str(group["group_id"])
        if group_id in group_ids:
            raise ValueError("duplicate role-rotation group id")
        group_ids.add(group_id)
        round_index = int(group["round_index"])
        role_map = group["speakers"]
        if not isinstance(role_map, Mapping) or set(role_map) != set(ROLES):
            raise ValueError("every group must expose exactly A/B/C speakers")
        values = [str(role_map[role]) for role in ROLES]
        if len(set(values)) != 3 or not set(values).issubset(speaker_set):
            raise ValueError("group speaker identities are invalid")
        for role, speaker in zip(ROLES, values):
            role_counts[speaker][role] += 1
            round_counts[round_index][speaker] += 1
        a, b, c = values
        ab_pairs.append(_pair(a, b))
        pair_counts.update((_pair(a, b), _pair(a, c), _pair(b, c)))

    expected_per_role = rounds // 3
    expected_roles = Counter({role: expected_per_role for role in ROLES})
    for speaker in speaker_set:
        if role_counts[speaker] != expected_roles:
            raise ValueError(
                f"speaker {speaker} role counts are {dict(role_counts[speaker])}, "
                f"expected {dict(expected_roles)}"
            )
    for round_index in range(rounds):
        if set(round_counts[round_index]) != speaker_set:
            raise ValueError(f"round {round_index} does not contain every speaker")
        if any(count != 1 for count in round_counts[round_index].values()):
            raise ValueError(f"round {round_index} repeats a speaker")
    if len(ab_pairs) != len(set(ab_pairs)):
        raise ValueError("an A/B recognition pair is repeated")
    if pair_counts and max(pair_counts.values()) > MAX_PAIR_USE:
        raise ValueError("a speaker pair exceeds the maximum use count")
    return {
        "speakers": len(speaker_set),
        "rounds": rounds,
        "groups": len(groups),
        "groups_per_round": len(speaker_set) // 3,
        "role_count_per_speaker": dict(expected_roles),
        "present_count_per_speaker": 2 * expected_per_role,
        "absent_count_per_speaker": expected_per_role,
        "unique_ab_pairs": len(set(ab_pairs)),
        "pair_use_max": max(pair_counts.values()) if pair_counts else 0,
        "pair_use_histogram": {
            str(count): sum(1 for value in pair_counts.values() if value == count)
            for count in sorted(set(pair_counts.values()))
        },
    }


def _group_items_by_speaker(
    items: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in normalize_source_items(items):
        if str(row.get("source_corpus", SOURCE_CORPUS)) != SOURCE_CORPUS:
            raise ValueError("v4b source corpus must be AISHELL-1")
        _assert_not_dataset_a(row["wav"])
        grouped[str(row["spk"])].append(dict(row))
    return dict(grouped)


def select_speakers(
    items: Sequence[Mapping[str, Any]],
    *,
    count: int,
    seed: int,
) -> tuple[tuple[str, ...], dict[str, list[dict[str, Any]]]]:
    grouped = _group_items_by_speaker(items)
    eligible = sorted(
        speaker
        for speaker, rows in grouped.items()
        if len({str(row["wav"]) for row in rows}) >= SOURCES_PER_SPEAKER
    )
    rng = random.Random(seed)
    rng.shuffle(eligible)
    selected = tuple(eligible[:count])
    if len(selected) != count:
        raise ValueError(
            f"need {count} speakers with >= {SOURCES_PER_SPEAKER} unique WAVs"
        )
    return selected, {speaker: grouped[speaker] for speaker in selected}


def allocate_source_wavs(
    groups: Sequence[Mapping[str, Any]],
    *,
    speaker_items: Mapping[str, Sequence[Mapping[str, Any]]],
    seed: int,
    sha_resolver: Callable[[Path], str] | None = None,
    path_resolver: Callable[[str], Path] | None = None,
) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
    """Attach distinct rec/view1/view2 WAVs and return the global ledger."""

    sha_resolver = sha_resolver or _sha256_file
    path_resolver = path_resolver or (lambda value: Path(value).resolve(strict=True))
    queues: dict[str, list[dict[str, Any]]] = {}
    for speaker, rows in speaker_items.items():
        unique: dict[str, dict[str, Any]] = {}
        for raw in normalize_source_items(rows):
            unique.setdefault(str(path_resolver(str(raw["wav"]))), dict(raw))
        ordered = list(unique.values())
        random.Random(_stable_seed(seed, speaker)).shuffle(ordered)
        if len(ordered) < SOURCES_PER_SPEAKER:
            raise ValueError(f"speaker {speaker} has fewer than 16 unique WAVs")
        queues[speaker] = ordered

    cursors: Counter[str] = Counter()
    path_uses: Counter[str] = Counter()
    sha_uses: Counter[str] = Counter()
    source_by_path: dict[str, Mapping[str, Any]] = {}
    emitted: list[dict[str, Any]] = []

    def take(speaker: str) -> Mapping[str, Any]:
        if speaker not in queues:
            raise ValueError(f"schedule references unselected speaker {speaker}")
        index = cursors[speaker]
        if index >= len(queues[speaker]):
            raise ValueError(f"speaker {speaker} exhausted its source WAV queue")
        cursors[speaker] += 1
        item = dict(queues[speaker][index])
        path = path_resolver(str(item["wav"]))
        path_text = path.as_posix()
        digest = sha_resolver(path)
        if len(digest) != 64:
            raise ValueError("source SHA resolver did not return SHA256 hex")
        path_uses[path_text] += 1
        sha_uses[digest] += 1
        source = {
            "path": path_text,
            "sha256": digest,
            "speaker_id": str(item["spk"]),
            "utterance_id": str(item["utt"]),
            "transcript": str(item.get("ref", "")),
            "source_corpus": str(item.get("source_corpus", SOURCE_CORPUS)),
        }
        source_by_path[path_text] = source
        return source

    for group in groups:
        role_sources: dict[str, Any] = {}
        for role in ROLES:
            speaker = str(group["speakers"][role])
            sources: dict[str, Any] = {}
            if role in {"A", "B"}:
                sources["recognition"] = take(speaker)
            sources["enrollment_view1"] = take(speaker)
            sources["enrollment_view2"] = take(speaker)
            role_sources[role] = {
                "speaker_id": speaker,
                "sources": sources,
            }
        emitted.append({**dict(group), "roles": role_sources})

    expected_uses = {
        speaker: SOURCES_PER_SPEAKER for speaker in speaker_items
    }
    if dict(cursors) != expected_uses:
        raise ValueError(
            f"source usage per speaker disagrees: {dict(cursors)} != {expected_uses}"
        )
    if any(count != 1 for count in path_uses.values()):
        raise ValueError("a source WAV path was reused")
    if any(count != 1 for count in sha_uses.values()):
        raise ValueError("a source WAV byte SHA was reused")
    ledger = {
        "unique_source_paths": len(path_uses),
        "unique_source_sha256": len(sha_uses),
        "source_path_use_count_min": min(path_uses.values()),
        "source_path_use_count_max": max(path_uses.values()),
        "source_sha_use_count_min": min(sha_uses.values()),
        "source_sha_use_count_max": max(sha_uses.values()),
        "source_uses_per_speaker": dict(sorted(cursors.items())),
        "sources": [source_by_path[path] for path in sorted(source_by_path)],
    }
    return emitted, ledger


def _canonical_aishell_root(value: str | Path) -> Path:
    _assert_not_dataset_a(value)
    root = Path(value).resolve(strict=True)
    if not root.is_dir() or not (root / "wav" / "train").is_dir() or not (root / "wav" / "dev").is_dir():
        raise ValueError("AISHELL root must contain official wav/train and wav/dev")
    return root


def _audit_source_routes(
    schedule: Sequence[Mapping[str, Any]],
    *,
    aishell_root: Path,
    source_split: str,
) -> None:
    allowed = (aishell_root / "wav" / source_split).resolve(strict=True)
    for group in schedule:
        for role in ROLES:
            role_row = group["roles"][role]
            speaker = str(role_row["speaker_id"])
            for source in role_row["sources"].values():
                path = Path(source["path"]).resolve(strict=True)
                if allowed not in path.parents:
                    raise ValueError(f"source escapes official wav/{source_split}: {path}")
                if path.parent.name != speaker:
                    raise ValueError("source parent speaker directory disagrees")
                if str(source["speaker_id"]) != speaker:
                    raise ValueError("source metadata speaker disagrees")


def build_schedule(
    aishell_root: str | Path,
    output_dir: str | Path,
) -> Mapping[str, Any]:
    root = _canonical_aishell_root(aishell_root)
    _assert_not_dataset_a(output_dir)
    output = Path(output_dir).resolve(strict=False)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    schedules: dict[str, list[dict[str, Any]]] = {}
    ledgers: dict[str, Mapping[str, Any]] = {}
    selected_speakers: dict[str, tuple[str, ...]] = {}
    rotation_audits: dict[str, Mapping[str, Any]] = {}
    for protocol_split in ("train", "dev"):
        source_split = PROTOCOL_TO_SOURCE_SPLIT[protocol_split]
        # The loop is intentionally restricted to train/dev.  AISHELL test is
        # neither enumerated nor passed to load_aishell_items.
        items = load_aishell_items(root, (source_split,))
        selected, selected_items = select_speakers(
            items,
            count=SPLIT_SPEAKER_COUNTS[protocol_split],
            seed=DEFAULT_SEEDS[protocol_split],
        )
        bare = build_role_rotation(
            selected,
            seed=DEFAULT_SEEDS[protocol_split],
        )
        rotation_audits[protocol_split] = audit_role_rotation(
            bare, speakers=selected
        )
        attached, ledger = allocate_source_wavs(
            bare,
            speaker_items=selected_items,
            seed=DEFAULT_SEEDS[protocol_split],
        )
        for global_index, group in enumerate(attached):
            group["protocol_split"] = protocol_split
            group["source_split"] = source_split
            group["base_mixture_id"] = f"{protocol_split}_r{group['round_index']:02d}_g{group['round_group_index']:02d}"
            group["global_group_index"] = global_index
        _audit_source_routes(attached, aishell_root=root, source_split=source_split)
        schedules[protocol_split] = attached
        ledgers[protocol_split] = ledger
        selected_speakers[protocol_split] = selected

    speaker_overlap = sorted(
        set(selected_speakers["train"]) & set(selected_speakers["dev"])
    )
    path_overlap = sorted(
        {value["path"] for value in ledgers["train"]["sources"]}
        & {value["path"] for value in ledgers["dev"]["sources"]}
    )
    sha_overlap = sorted(
        {value["sha256"] for value in ledgers["train"]["sources"]}
        & {value["sha256"] for value in ledgers["dev"]["sources"]}
    )
    if speaker_overlap or path_overlap or sha_overlap:
        raise ValueError("train/dev schedule overlap audit failed")

    schedule_payload: Mapping[str, Any] = {
        "schema": SCHEDULE_SCHEMA,
        "dataset_a_used": False,
        "source_corpus": SOURCE_CORPUS,
        "aishell_root": root.as_posix(),
        "loaded_source_splits": ["train", "dev"],
        "final_deferred": True,
        "official_test_loaded": False,
        "rounds": ROUNDS,
        "sources_per_speaker": SOURCES_PER_SPEAKER,
        "seeds": dict(DEFAULT_SEEDS),
        "splits": schedules,
    }
    schedule_path = output / "schedule.json"
    _write_json(schedule_path, schedule_payload)
    audit_payload: Mapping[str, Any] = {
        "schema": AUDIT_SCHEMA,
        "dataset_a_used": False,
        "source_corpus": SOURCE_CORPUS,
        "schedule_path": schedule_path.as_posix(),
        "schedule_sha256": _sha256_file(schedule_path),
        "loaded_source_splits": ["train", "dev"],
        "official_test_loaded": False,
        "final_deferred": True,
        "rotation": rotation_audits,
        "source_ledgers": ledgers,
        "cross_split_overlap": {
            "speaker": speaker_overlap,
            "source_path": path_overlap,
            "source_sha256": sha_overlap,
        },
        "hard_negative_verified_count": 0,
        "limitations": [
            "This is a source schedule only; no generated audio, feature, model, CER, RR, or RTF claim is made.",
            "AISHELL read speech is not a verified same-command home hard-negative set.",
            "The official AISHELL test split is deferred and was not loaded.",
        ],
    }
    _write_json(output / "audit_report.json", audit_payload)
    return audit_payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aishell-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_schedule(args.aishell_root, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


__all__ = [
    "AUDIT_SCHEMA",
    "DEFAULT_SEEDS",
    "MAX_PAIR_USE",
    "ROLES",
    "ROUNDS",
    "SCHEDULE_SCHEMA",
    "SOURCES_PER_SPEAKER",
    "allocate_source_wavs",
    "audit_role_rotation",
    "build_role_rotation",
    "build_schedule",
    "select_speakers",
]


if __name__ == "__main__":
    raise SystemExit(main())
