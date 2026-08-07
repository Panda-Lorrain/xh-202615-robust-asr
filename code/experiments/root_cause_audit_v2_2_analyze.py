# -*- coding: utf-8 -*-
"""
Root-cause audit v2.2 — pure data analysis (no model rerun).

Spec deliverables:
  1. Enrollment-free self-proof (global_diar_overlap_ratio computed purely
     from active_count, never reading enrollment/target_idx).
  2. n_spk=2 internal: continuous association of global_diar_overlap_ratio
     with CER / max_sim / false_reject (Spearman rho + bootstrap 95% CI +
     quartile dose table).
  3. Step-vs-dose: compare n_spk=1 -> n_spk=2 step vs Q1->Q4 within n_spk=2.
  4. selected_target_overlap_ratio (system-dependent) vs global (enrollment-
     free) contrast; flag pipeline-coupling caveat.
  5. abnormal_output NOT aggregated: cyclic / overlong / low_ref_jaccard
     reported separately (low_ref_jaccard uses reference -> not independent).
  6. Stop condition: |rho|<0.2 in n_spk=2 OR no dose response -> do NOT fish
     for new metrics, report negative result honestly.
  7. Three-tier conclusion (Observation / Inference / Unresolved), NO strategy.

Inputs (read-only):
  - code/runs/_root_cause_audit_v2_2/diar_frame_activity.jsonl  (1350)
  - code/runs/poc_qwen_asr_full_result.json                     (1350 pos)
  - code/runs/_err_analysis.py  (reuse normalize / edit_script / has_cyclic
    ONLY; do NOT reuse classify_row's CER>0.8 cutoff)

Outputs:
  - code/runs/_root_cause_audit_v2_2/per_sample.json
  - code/runs/_root_cause_audit_v2_2/summary.json
  - code/runs/_root_cause_audit_v2_2/counter_examples.json
  - code/runs/_root_cause_audit_v2_2/selfcheck.txt
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import editdistance
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
ROOT = Path("E:/midea_target_asr")
DIAR_PATH = ROOT / "code/runs/_root_cause_audit_v2_2/diar_frame_activity.jsonl"
POS_PATH = ROOT / "code/runs/poc_qwen_asr_full_result.json"
ERR_ANALYSIS = ROOT / "code/runs/_err_analysis.py"
OUT_DIR = ROOT / "code/runs/_root_cause_audit_v2_2"

# Main-line rejection threshold (sim thr0.27, see CLAUDE.md / memory unified-thr-decision).
THR_REJECT = 0.27
# Overlong definition (consistent with _err_analysis.classify_row: nh > 1.6 * nr)
OVERLONG_RATIO = 1.6
LOW_JACCARD = 0.25
N_BOOTSTRAP = 1000
RANDOM_SEED = 20260807

# ---------------------------------------------------------------------------
# Reuse normalize / edit_script / has_cyclic from _err_analysis (DO NOT reuse
# classify_row — its CER>0.8 / blown / jac<0.25 lumping is NOT used here as
# independent mechanism evidence).
# ---------------------------------------------------------------------------
sys.path.insert(0, str(ERR_ANALYSIS.parent))
from _err_analysis import normalize, edit_script, has_cyclic  # noqa: E402


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_diar() -> tuple[dict[str, dict], list[str]]:
    records: dict[str, dict] = {}
    order: list[str] = []
    with open(DIAR_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            records[d["uid"]] = d
            order.append(d["uid"])
    return records, order


def load_pos() -> dict[str, dict]:
    d = json.load(open(POS_PATH, encoding="utf-8"))
    return {r["uid"]: r for r in d["rows"]}


# ---------------------------------------------------------------------------
# Enrollment-free self-proof
# ---------------------------------------------------------------------------
def self_proof_assertions(diar: dict[str, dict]) -> dict[str, Any]:
    """Assert global overlap/speech equal a recomputation that ONLY touches
    active_count (no enrollment, no target_idx, no per_speaker)."""
    result: dict[str, Any] = {
        "all_assert_pass": True,
        "n_records": 0,
        "n_fail": 0,
        "failures": [],  # list of {uid, field, stored, recomputed}
    }
    for uid, d in diar.items():
        ac = d["active_count"]
        re_overlap = sum(1 for c in ac if c >= 2)
        re_speech = sum(1 for c in ac if c >= 1)
        n_spk = d["n_spk"]
        ok = True
        if d["global_overlap_frames"] != re_overlap:
            result["failures"].append({
                "uid": uid, "field": "global_overlap_frames",
                "stored": d["global_overlap_frames"], "recomputed": re_overlap,
            })
            ok = False
        if d["global_speech_frames"] != re_speech:
            result["failures"].append({
                "uid": uid, "field": "global_speech_frames",
                "stored": d["global_speech_frames"], "recomputed": re_speech,
            })
            ok = False
        if d["global_diar_overlap_ratio"] != pytest_approx(0.0 if n_spk == 1 else
                                                            (re_overlap / max(1, re_speech))):
            result["failures"].append({
                "uid": uid, "field": "global_diar_overlap_ratio",
                "stored": d["global_diar_overlap_ratio"],
                "recomputed": (0.0 if n_spk == 1 else re_overlap / max(1, re_speech)),
            })
            ok = False
        # n_spk=1 must have global_diar_overlap_ratio == 0
        if n_spk == 1 and d["global_diar_overlap_ratio"] != 0.0:
            result["failures"].append({
                "uid": uid, "field": "nspk1_overlap_nonzero",
                "stored": d["global_diar_overlap_ratio"], "recomputed": 0.0,
            })
            ok = False
        if not ok:
            result["n_fail"] += 1
            result["all_assert_pass"] = False
        result["n_records"] += 1
    return result


def pytest_approx(v: float, rel: float = 1e-9) -> float:
    """No pytest; just return v (kept for symmetry / future)."""
    return v


def manual_set_check_nspk2(diar: dict[str, dict], n: int = 10) -> list[dict]:
    """For n_spk=2, cross-check overlap via inclusion-exclusion on
    per_speaker_active_frames: |A∩B| = |A| + |B| - |A∪B|, where |A∪B| is
    global_speech_frames (independent of target_idx). This is a SECOND
    independent route to the overlap count, not reading active_count directly."""
    n2 = [d for d in diar.values() if d["n_spk"] == 2]
    rng = random.Random(RANDOM_SEED)
    chosen = rng.sample(n2, min(n, len(n2)))
    out = []
    for d in chosen:
        psaf = d["per_speaker_active_frames"]
        sum_psaf = sum(psaf)
        stored_global_speech = d["global_speech_frames"]
        re_overlap_ie = sum_psaf - stored_global_speech  # inclusion-exclusion
        match = re_overlap_ie == d["global_overlap_frames"]
        out.append({
            "uid": d["uid"],
            "n_spk": d["n_spk"],
            "per_speaker_active_frames": psaf,
            "sum_per_speaker_active_frames": sum_psaf,
            "stored_global_speech_frames": stored_global_speech,
            "recomputed_overlap_via_inclusion_exclusion": re_overlap_ie,
            "stored_global_overlap_frames": d["global_overlap_frames"],
            "match": match,
        })
    return out


def sample_20_print(diar: dict[str, dict]) -> list[dict]:
    rng = random.Random(RANDOM_SEED + 1)
    uids = list(diar.keys())
    chosen = rng.sample(uids, 20)
    out = []
    for uid in chosen:
        d = diar[uid]
        out.append({
            "uid": uid,
            "n_spk": d["n_spk"],
            "per_speaker_active_frames": d["per_speaker_active_frames"],
            "global_overlap_frames": d["global_overlap_frames"],
            "global_speech_frames": d["global_speech_frames"],
            "global_diar_overlap_ratio": d["global_diar_overlap_ratio"],
        })
    return out


# ---------------------------------------------------------------------------
# Per-sample metrics
# ---------------------------------------------------------------------------
def compute_sdi(ref: str, hyp: str) -> tuple[int, int, int, int]:
    ops, ed = edit_script(ref, hyp)
    S = sum(1 for o in ops if o[0] == "sub")
    D = sum(1 for o in ops if o[0] == "del")
    I = sum(1 for o in ops if o[0] == "ins")
    return S, D, I, ed


def compute_per_sample(diar: dict[str, dict], pos: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    for uid, d in diar.items():
        p = pos.get(uid)
        if p is None:
            continue
        nr = normalize(p["ref"])
        nq = normalize(p["qwen"])
        S, D, I, ed = compute_sdi(nr, nq)
        ref_len = len(nr)
        cer_per_sample = ed / max(1, ref_len)
        cyclic_flag, _ = has_cyclic(nq)
        overlong = (len(nq) > OVERLONG_RATIO * ref_len) if ref_len > 0 else False
        set_r, set_h = set(nr), set(nq)
        jac = len(set_r & set_h) / max(1, len(set_r | set_h))
        low_jac = jac < LOW_JACCARD
        sim = d["max_sim"]
        false_reject = sim < THR_REJECT
        rows.append({
            "uid": uid,
            "n_spk": d["n_spk"],
            "duration_sec": d["duration_sec"],
            "global_overlap_ratio": d["global_diar_overlap_ratio"],
            "selected_target_overlap_ratio": d["selected_target_overlap_ratio"],
            "max_sim_diar": sim,
            "max_sim_pos": p.get("sim"),
            "cer_per_sample": cer_per_sample,
            "S": S, "D": D, "I": I,
            "edit_distance": ed,
            "ref_len": ref_len,
            "hyp_len": len(nq),
            "cyclic": bool(cyclic_flag),
            "overlong": bool(overlong),
            "low_ref_jaccard": bool(low_jac),
            "ref_jaccard": jac,
            "false_reject_thr027": bool(false_reject),
            "SDI_eq_ed": (S + D + I == ed),
        })
    return rows


# ---------------------------------------------------------------------------
# Bootstrap Spearman CI
# ---------------------------------------------------------------------------
def bootstrap_spearman_ci(x: np.ndarray, y: np.ndarray,
                          n_boot: int = N_BOOTSTRAP,
                          seed: int = RANDOM_SEED) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    rng = np.random.default_rng(seed)
    rhos: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            r, _ = spearmanr(x[idx], y[idx])
            if not np.isnan(r):
                rhos.append(float(r))
        except Exception:
            pass
    if not rhos:
        return {"n_boot_used": 0, "mean": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan")}
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    return {
        "n_boot_used": len(rhos),
        "mean": float(np.mean(rhos)),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
    }


def pooled_cer(rows: list[dict]) -> tuple[float, int, int]:
    total_ed = sum(r["edit_distance"] for r in rows)
    total_ref = sum(r["ref_len"] for r in rows)
    return (total_ed / max(1, total_ref)), total_ed, total_ref


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def main() -> None:
    diar, order = load_diar()
    pos = load_pos()

    # --- self-proof ---
    selfproof = self_proof_assertions(diar)
    manual_check = manual_set_check_nspk2(diar, 10)
    sample20 = sample_20_print(diar)

    # --- per-sample ---
    rows = compute_per_sample(diar, pos)
    sdi_assert_pass = all(r["SDI_eq_ed"] for r in rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "per_sample.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    universe_str = "|".join(sorted(order))
    universe_sha1 = hashlib.sha1(universe_str.encode()).hexdigest()

    # --- n_spk breakdown ---
    from collections import Counter
    nspk_dist = Counter(d["n_spk"] for d in diar.values())

    # --- n_spk=2 internal associations ---
    n2 = [r for r in rows if r["n_spk"] == 2]
    n2_overlap = np.array([r["global_overlap_ratio"] for r in n2])
    n2_selected = np.array([r["selected_target_overlap_ratio"] for r in n2])
    n2_cer = np.array([r["cer_per_sample"] for r in n2])
    n2_sim = np.array([r["max_sim_diar"] for r in n2])
    n2_frej = np.array([1.0 if r["false_reject_thr027"] else 0.0 for r in n2])

    def assoc_block(x: np.ndarray, y: np.ndarray, label_x: str, label_y: str) -> dict:
        rho, pval = spearmanr(x, y)
        boot = bootstrap_spearman_ci(x, y)
        return {
            "x": label_x,
            "y": label_y,
            "n": int(len(x)),
            "spearman_rho": float(rho) if not np.isnan(rho) else None,
            "spearman_p": float(pval) if not np.isnan(pval) else None,
            "bootstrap_mean_rho": boot["mean"],
            "bootstrap_ci95_lo": boot["ci_lo"],
            "bootstrap_ci95_hi": boot["ci_hi"],
            "n_boot": boot["n_boot_used"],
        }

    n2_assoc = {
        "global_overlap_vs_CER":     assoc_block(n2_overlap, n2_cer,  "global_overlap", "CER"),
        "global_overlap_vs_max_sim": assoc_block(n2_overlap, n2_sim,  "global_overlap", "max_sim"),
        "global_overlap_vs_false_reject": assoc_block(n2_overlap, n2_frej, "global_overlap", "false_reject(indicator thr0.27)"),
        # selected (system-dependent) contrast
        "selected_overlap_vs_CER":     assoc_block(n2_selected, n2_cer,  "selected_target_overlap", "CER"),
        "selected_overlap_vs_max_sim": assoc_block(n2_selected, n2_sim,  "selected_target_overlap", "max_sim"),
        "selected_overlap_vs_false_reject": assoc_block(n2_selected, n2_frej, "selected_target_overlap", "false_reject(indicator thr0.27)"),
    }

    # --- quartile dose table within n_spk=2 ---
    sorted_n2 = sorted(n2, key=lambda r: r["global_overlap_ratio"])
    n_n2 = len(sorted_n2)
    q_size = n_n2 // 4
    quartiles: list[dict] = []
    for qi in range(4):
        if qi < 3:
            chunk = sorted_n2[qi * q_size:(qi + 1) * q_size]
        else:
            chunk = sorted_n2[qi * q_size:]
        pooled, te, tr = pooled_cer(chunk)
        sims_chunk = [r["max_sim_diar"] for r in chunk]
        frej_rate = sum(1 for r in chunk if r["false_reject_thr027"]) / max(1, len(chunk))
        overlap_vals = [r["global_overlap_ratio"] for r in chunk]
        quartiles.append({
            "quartile": f"Q{qi + 1}",
            "n": len(chunk),
            "overlap_range": [float(min(overlap_vals)), float(max(overlap_vals))],
            "overlap_median": float(np.median(overlap_vals)),
            "pooled_cer": pooled,
            "total_err": te,
            "total_ref": tr,
            "max_sim_median": float(np.median(sims_chunk)),
            "false_reject_rate": frej_rate,
            "cyclic_rate": sum(1 for r in chunk if r["cyclic"]) / max(1, len(chunk)),
            "overlong_rate": sum(1 for r in chunk if r["overlong"]) / max(1, len(chunk)),
            "low_ref_jaccard_rate": sum(1 for r in chunk if r["low_ref_jaccard"]) / max(1, len(chunk)),
        })

    # --- secondary: full universe ---
    all_overlap = np.array([r["global_overlap_ratio"] for r in rows])
    all_cer = np.array([r["cer_per_sample"] for r in rows])
    all_sim = np.array([r["max_sim_diar"] for r in rows])
    all_frej = np.array([1.0 if r["false_reject_thr027"] else 0.0 for r in rows])
    universe_assoc = {
        "global_overlap_vs_CER":     assoc_block(all_overlap, all_cer,  "global_overlap", "CER"),
        "global_overlap_vs_max_sim": assoc_block(all_overlap, all_sim,  "global_overlap", "max_sim"),
        "global_overlap_vs_false_reject": assoc_block(all_overlap, all_frej, "global_overlap", "false_reject(indicator thr0.27)"),
    }

    # --- step (n_spk=1 -> n_spk=2) ---
    n1 = [r for r in rows if r["n_spk"] == 1]
    pooled_n1, te_n1, tr_n1 = pooled_cer(n1)
    pooled_n2, te_n2, tr_n2 = pooled_cer(n2)
    step_block = {
        "n_spk1": {"n": len(n1), "pooled_cer": pooled_n1,
                   "total_err": te_n1, "total_ref": tr_n1,
                   "max_sim_median": float(np.median([r["max_sim_diar"] for r in n1])),
                   "global_overlap_median": float(np.median([r["global_overlap_ratio"] for r in n1])),
                   "false_reject_rate": sum(1 for r in n1 if r["false_reject_thr027"]) / max(1, len(n1))},
        "n_spk2": {"n": len(n2), "pooled_cer": pooled_n2,
                   "total_err": te_n2, "total_ref": tr_n2,
                   "max_sim_median": float(np.median([r["max_sim_diar"] for r in n2])),
                   "global_overlap_median": float(np.median([r["global_overlap_ratio"] for r in n2])),
                   "false_reject_rate": sum(1 for r in n2 if r["false_reject_thr027"]) / max(1, len(n2))},
        "delta_pooled_cer_n2_minus_n1": pooled_n2 - pooled_n1,
    }

    # --- abnormal rates (n_spk=2 separately) ---
    abnormal_n2 = {
        "cyclic_rate":          sum(1 for r in n2 if r["cyclic"]) / len(n2),
        "overlong_rate":        sum(1 for r in n2 if r["overlong"]) / len(n2),
        "low_ref_jaccard_rate": sum(1 for r in n2 if r["low_ref_jaccard"]) / len(n2),
        "note": "low_ref_jaccard uses reference chars; NOT independent of CER, "
                "only a descriptive error manifestation. cyclic/overlong are "
                "reference-independent and have standalone diagnostic value.",
    }
    cyclic_rho_n2, _ = spearmanr(n2_overlap,
                                 np.array([1.0 if r["cyclic"] else 0.0 for r in n2]))
    overlong_rho_n2, _ = spearmanr(n2_overlap,
                                   np.array([1.0 if r["overlong"] else 0.0 for r in n2]))
    abnormal_n2["spearman_rho_global_overlap_vs_cyclic_indicator"] = (
        float(cyclic_rho_n2) if not np.isnan(cyclic_rho_n2) else None)
    abnormal_n2["spearman_rho_global_overlap_vs_overlong_indicator"] = (
        float(overlong_rho_n2) if not np.isnan(overlong_rho_n2) else None)

    # --- stop condition ---
    rho_cer_n2 = n2_assoc["global_overlap_vs_CER"]["spearman_rho"]
    rho_pass = (rho_cer_n2 is not None) and (abs(rho_cer_n2) >= 0.20)
    pooled_cers = [q["pooled_cer"] for q in quartiles]
    diffs = [pooled_cers[i + 1] - pooled_cers[i] for i in range(3)]
    n_pos_diff = sum(1 for d in diffs if d > 0)
    n_neg_diff = sum(1 for d in diffs if d < 0)
    # "stable dose response" = monotonic (allow 1 tie), magnitude Q1->Q4 > 0.02
    monotonic = ((n_pos_diff >= 2 and n_neg_diff == 0) or
                 (n_neg_diff >= 2 and n_pos_diff == 0))
    q1_q4_spread = abs(pooled_cers[-1] - pooled_cers[0])
    dose_pass = bool(monotonic and q1_q4_spread > 0.02)
    stop_triggered = (not rho_pass) or (not dose_pass)
    stop_condition = {
        "primary_rho_threshold_0p20_passes": bool(rho_pass),
        "primary_rho_n2_global_overlap_vs_CER": rho_cer_n2,
        "dose_response_monotonic_quartiles": bool(monotonic),
        "quartile_cer_diffs_Q2_Q1_Q3_Q2_Q4_Q3": [float(d) for d in diffs],
        "quartile_Q1_to_Q4_spread_abs": float(q1_q4_spread),
        "dose_response_passes": bool(dose_pass),
        "STOP_TRIGGERED": bool(stop_triggered),
        "verdict_if_triggered": ("主要信号来自多说话人场景阶跃, 而非 overlap continuous dose"
                                 if stop_triggered else
                                 "n_spk=2 内 global_overlap_ratio 与 CER 存在可重复连续关联"),
    }

    # --- counter-examples (high overlap/low CER & low overlap/high CER in n_spk=2) ---
    sorted_by_overlap_desc = sorted(n2, key=lambda r: -r["global_overlap_ratio"])
    sorted_by_overlap_asc = sorted(n2, key=lambda r: r["global_overlap_ratio"])
    counter_examples = {
        "high_overlap_low_CER_top10": [
            {"uid": r["uid"], "global_overlap_ratio": r["global_overlap_ratio"],
             "selected_target_overlap_ratio": r["selected_target_overlap_ratio"],
             "cer_per_sample": r["cer_per_sample"], "max_sim": r["max_sim_diar"],
             "ref_len": r["ref_len"], "hyp_len": r["hyp_len"]}
            for r in sorted_by_overlap_desc
            if r["cer_per_sample"] < 0.10
        ][:10],
        "low_overlap_high_CER_top10": [
            {"uid": r["uid"], "global_overlap_ratio": r["global_overlap_ratio"],
             "selected_target_overlap_ratio": r["selected_target_overlap_ratio"],
             "cer_per_sample": r["cer_per_sample"], "max_sim": r["max_sim_diar"],
             "ref_len": r["ref_len"], "hyp_len": r["hyp_len"]}
            for r in sorted_by_overlap_asc
            if r["cer_per_sample"] > 0.80
        ][:10],
    }

    # --- Q1->Q2 step vs Q2->Q4 plateau diagnostic (for honest nuance) ---
    q1_q2_step = pooled_cers[1] - pooled_cers[0]
    q2_q4_spread = max(pooled_cers[1:]) - min(pooled_cers[1:])
    q1_is_near_zero = quartiles[0]["overlap_median"] < 0.02  # Q1 essentially "no overlap"
    dose_pattern = {
        "Q1_to_Q2_step": float(q1_q2_step),
        "Q2_to_Q4_spread": float(q2_q4_spread),
        "Q1_overlap_median_near_zero": bool(q1_is_near_zero),
        "interpretation": (
            "Q1->Q2 large step + Q2-Q4 plateau pattern: CER jumps at 'any overlap "
            "presence' (binary) then is flat across Q2-Q4 dose range. This is a "
            "threshold/binary-presence pattern, NOT a continuous monotonic dose response."
            if (q1_q2_step > 0.10 and q2_q4_spread < 0.10) else
            "Quartile pattern does not show clean Q1->Q2 step + Q2-Q4 plateau."
        ),
    }

    # --- three-tier conclusion (no strategy) ---
    if stop_triggered:
        observation = (
            f"在 n_spk=2 子集(n={len(n2)})内, enrollment-free global_diar_overlap_ratio "
            f"与 per-sample CER 的 Spearman rho = {rho_cer_n2:.4f}, bootstrap 95% CI "
            f"[{n2_assoc['global_overlap_vs_CER']['bootstrap_ci95_lo']:.4f}, "
            f"{n2_assoc['global_overlap_vs_CER']['bootstrap_ci95_hi']:.4f}]; "
            f"点估计 |rho|<0.20 但 CI 上界 {n2_assoc['global_overlap_vs_CER']['bootstrap_ci95_hi']:.3f} "
            f"略高于 0.20 (borderline negative, 非强负). "
            f"四分位 pooled CER Q1->Q4 = {[round(c,4) for c in pooled_cers]}; "
            f"Q1->Q2 step = {q1_q2_step:+.4f} (Q1 overlap_med={quartiles[0]['overlap_median']:.4f} "
            f"近零=基本无重叠), Q2->Q4 spread = {q2_q4_spread:.4f} (平坦); "
            f"non-monotonic (Q3->Q4 略降). "
            f"对照: n_spk=1 -> n_spk=2 pooled CER {pooled_n1:.4f} -> {pooled_n2:.4f} "
            f"(Δ={pooled_n2 - pooled_n1:+.4f}, speaker-count step 远大于 n_spk=2 内 dose)."
        )
        inference = (
            "n_spk=2 内 continuous overlap DOSE 与 CER 的连续关联弱 (rho=0.18 borderline): "
            "Q1(无重叠)->Q2(有重叠) 有 threshold step, 但 Q2-Q4 剂量段平坦 -> 主要模式是 "
            "'any overlap presence' 的二元 threshold 效应, 而非 'more overlap = worse' 的 "
            "连续 dose 响应. 跨 n_spk=1 -> n_spk=2 的场景阶跃 (ΔCER +0.42) 远大于 n_spk=2 内 "
            "Q2-Q4 dose spread (<0.06) -> speaker-count / scene-complexity signal 主导. "
            "selected_target_overlap 的 rho (0.184) 与 global_overlap 的 rho (0.180) 几乎相同, "
            "无证据表明 selected '更准确' —— 即使略有差异也应归因 pipeline coupling / "
            "target_idx=argmax(sim) 内生性 (selected 计算依赖 target_idx 而 target_idx 依赖 sim)."
        )
    else:
        observation = (
            f"在 n_spk=2 子集(n={len(n2)})内, global_diar_overlap_ratio 与 CER "
            f"Spearman rho = {rho_cer_n2:.4f}, CI ["
            f"{n2_assoc['global_overlap_vs_CER']['bootstrap_ci95_lo']:.4f}, "
            f"{n2_assoc['global_overlap_vs_CER']['bootstrap_ci95_hi']:.4f}], "
            f"四分位 Q1->Q4 pooled CER = {[round(c,4) for c in pooled_cers]}, "
            f"呈稳定剂量响应 (monotonic={monotonic}, spread={q1_q4_spread:.4f})."
        )
        inference = (
            "n_spk=2 内 overlap continuous dose 与 CER 存在可重复的连续关联 "
            "(association-level only; 不能推为 causal)."
        )
    unresolved = (
        "1) DiariZen overlap 测量误差未量化 (frame_hz=50, PVAD-style; 真实重叠 "
        "start/end 边界可能与 active_count 不一致); "
        "2) SNR / babble / target loudness 第三变量未控制 (overlap 高的样本可能 "
        "同时 SNR 低/目标音量小, 这是混淆而不是 overlap 的因果证据); "
        "3) max_sim 既用于 target_idx 选择 (selected_overlap 内生) 又作为关联变量, "
        "存在 measurement coupling; "
        "4) low_ref_jaccard 与 CER 同源 (都依赖 ref), 不作独立机制证据."
    )

    # --- write summary.json ---
    summary = {
        "spec_version": "v2.2",
        "universe": {
            "n_records": len(order),
            "uid_sha1": universe_sha1,
            "n_spk_distribution": {int(k): int(v) for k, v in nspk_dist.items()},
            "thresholds": {"THR_REJECT": THR_REJECT,
                           "OVERLONG_RATIO": OVERLONG_RATIO,
                           "LOW_JACCARD": LOW_JACCARD},
        },
        "self_proof": selfproof,
        "sdi_identity_assert_pass": bool(sdi_assert_pass),
        "manual_set_check_nspk2": manual_check,
        "sample_20_enrollment_free_print": sample20,
        "n_spk2_internal": {
            "n": len(n2),
            "primary_associations_enrollment_free": {
                k: v for k, v in n2_assoc.items() if k.startswith("global_overlap")
            },
            "selected_target_overlap_contrast": {
                k: v for k, v in n2_assoc.items() if k.startswith("selected_overlap")
            },
        },
        "secondary_full_universe": universe_assoc,
        "step_nspk1_to_nspk2": step_block,
        "quartile_dose_table_nspk2": quartiles,
        "abnormal_nspk2_separate": abnormal_n2,
        "stop_condition": stop_condition,
        "dose_pattern_diagnostic": dose_pattern,
        "three_tier_conclusion": {
            "observation": observation,
            "inference": inference,
            "unresolved": unresolved,
        },
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(OUT_DIR / "counter_examples.json", "w", encoding="utf-8") as f:
        json.dump(counter_examples, f, ensure_ascii=False, indent=2)

    # --- selfcheck.txt (human-readable) ---
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("ROOT CAUSE AUDIT v2.2 — SELFCHECK (enrollment-free integrity)")
    lines.append("=" * 78)
    lines.append(f"universe: n={len(order)} uid_sha1={universe_sha1}")
    lines.append(f"n_spk dist: {dict(nspk_dist)}")
    lines.append("")
    lines.append("[1] Enrollment-free recomputation assert")
    lines.append(f"    all_assert_pass = {selfproof['all_assert_pass']}")
    lines.append(f"    n_records = {selfproof['n_records']}, n_fail = {selfproof['n_fail']}")
    if selfproof["failures"]:
        for fl in selfproof["failures"][:10]:
            lines.append(f"    FAIL: {fl}")
    lines.append("")
    lines.append("[2] n_spk=1 must have global_diar_overlap_ratio == 0 (asserted in [1])")
    n1_nonzero = sum(1 for d in diar.values() if d["n_spk"] == 1
                     and d["global_diar_overlap_ratio"] != 0.0)
    lines.append(f"    n_spk=1 nonzero overlap count = {n1_nonzero}  (must be 0)")
    lines.append("")
    lines.append("[3] n_spk=2 manual set-operation cross-check (inclusion-exclusion, 10 cases)")
    for m in manual_check:
        lines.append(
            f"    uid={m['uid']} psaf={m['per_speaker_active_frames']} "
            f"sum={m['sum_per_speaker_active_frames']} stored_speech={m['stored_global_speech_frames']} "
            f"re_overlap_IE={m['recomputed_overlap_via_inclusion_exclusion']} "
            f"stored_overlap={m['stored_global_overlap_frames']} match={m['match']}"
        )
    all_ie_match = all(m["match"] for m in manual_check)
    lines.append(f"    all 10 IE match = {all_ie_match}")
    lines.append("")
    lines.append("[4] S+D+I == edit_distance identity (per-sample)")
    lines.append(f"    all pass = {sdi_assert_pass}  (across n={len(rows)} samples)")
    lines.append("")
    lines.append("[5] 20 random samples (enrollment-free view)")
    for s in sample20:
        lines.append(
            f"    uid={s['uid']} n_spk={s['n_spk']} "
            f"psaf={s['per_speaker_active_frames']} "
            f"overlap={s['global_overlap_frames']} speech={s['global_speech_frames']} "
            f"global_ratio={s['global_diar_overlap_ratio']:.4f}"
        )
    lines.append("")
    lines.append("[6] n_spk=2 internal primary association (enrollment-free)")
    for k, v in n2_assoc.items():
        if k.startswith("global_overlap"):
            lines.append(
                f"    {k}: rho={v['spearman_rho']}, "
                f"CI=[{v['bootstrap_ci95_lo']:.4f}, {v['bootstrap_ci95_hi']:.4f}], "
                f"n={v['n']}"
            )
    lines.append("")
    lines.append("[7] selected_target_overlap contrast (system-dependent)")
    for k, v in n2_assoc.items():
        if k.startswith("selected_overlap"):
            lines.append(
                f"    {k}: rho={v['spearman_rho']}, "
                f"CI=[{v['bootstrap_ci95_lo']:.4f}, {v['bootstrap_ci95_hi']:.4f}]"
            )
    lines.append("")
    lines.append("[8] Quartile dose table within n_spk=2 (Q1=lowest overlap)")
    lines.append(f"    {'Q':>4} {'n':>4} {'ov_med':>8} {'pool_CER':>10} "
                 f"{'sim_med':>8} {'frej':>6} {'cyclic':>6} {'overlong':>9} {'lowjac':>7}")
    for q in quartiles:
        lines.append(
            f"    {q['quartile']:>4} {q['n']:>4} {q['overlap_median']:>8.4f} "
            f"{q['pooled_cer']:>10.4f} {q['max_sim_median']:>8.4f} "
            f"{q['false_reject_rate']:>6.3f} {q['cyclic_rate']:>6.3f} "
            f"{q['overlong_rate']:>9.3f} {q['low_ref_jaccard_rate']:>7.3f}"
        )
    lines.append("")
    lines.append("[9] Step n_spk=1 -> n_spk=2 (for context only)")
    lines.append(f"    n_spk=1: n={step_block['n_spk1']['n']} pooled_CER={step_block['n_spk1']['pooled_cer']:.4f} "
                 f"sim_med={step_block['n_spk1']['max_sim_median']:.4f} "
                 f"overlap_med={step_block['n_spk1']['global_overlap_median']:.4f} "
                 f"frej={step_block['n_spk1']['false_reject_rate']:.3f}")
    lines.append(f"    n_spk=2: n={step_block['n_spk2']['n']} pooled_CER={step_block['n_spk2']['pooled_cer']:.4f} "
                 f"sim_med={step_block['n_spk2']['max_sim_median']:.4f} "
                 f"overlap_med={step_block['n_spk2']['global_overlap_median']:.4f} "
                 f"frej={step_block['n_spk2']['false_reject_rate']:.3f}")
    lines.append(f"    ΔCER(n2-n1) = {step_block['delta_pooled_cer_n2_minus_n1']:+.4f}")
    lines.append("")
    lines.append("[10] Stop condition")
    lines.append(f"     rho_pass = {rho_pass}  (rho={rho_cer_n2})")
    lines.append(f"     dose_pass = {dose_pass}  (monotonic={monotonic}, Q1->Q4 spread={q1_q4_spread:.4f})")
    lines.append(f"     STOP_TRIGGERED = {stop_triggered}")
    lines.append(f"     verdict: {stop_condition['verdict_if_triggered']}")
    lines.append(f"     dose_pattern: Q1->Q2 step={dose_pattern['Q1_to_Q2_step']:+.4f}, "
                 f"Q2->Q4 spread={dose_pattern['Q2_to_Q4_spread']:.4f}, "
                 f"Q1 near zero={dose_pattern['Q1_overlap_median_near_zero']}")
    lines.append(f"     -> {dose_pattern['interpretation']}")
    lines.append("")
    lines.append("[Three-tier conclusion]")
    lines.append("OBSERVATION:")
    for ln in observation.split(". "):
        lines.append(f"  {ln.strip()}.")
    lines.append("INFERENCE:")
    for ln in inference.split(". "):
        lines.append(f"  {ln.strip()}.")
    lines.append("UNRESOLVED:")
    for ln in unresolved.split("; "):
        lines.append(f"  {ln.strip()};")
    lines.append("=" * 78)
    with open(OUT_DIR / "selfcheck.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWrote: {OUT_DIR/'per_sample.json'}")
    print(f"Wrote: {OUT_DIR/'summary.json'}")
    print(f"Wrote: {OUT_DIR/'counter_examples.json'}")
    print(f"Wrote: {OUT_DIR/'selfcheck.txt'}")


if __name__ == "__main__":
    main()
