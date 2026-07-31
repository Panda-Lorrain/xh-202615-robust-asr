#!/usr/bin/env python
"""可选 scene-route 后端：双人样本 SepFormer 两路分离 + Qwen 内容选路。

输入是 enroll_infer 已完成的 JSON。单人样本和任何失败样本均保持主线文本；
仅 ``len(speakers) == 2`` 的样本进入固定路由。该脚本不做阈值选择，也不读取
Dataset-A 标签或参考文本。
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time


HERE = os.path.dirname(os.path.abspath(__file__))


def _venv_python(venv_dir):
    return (
        os.path.join(venv_dir, "Scripts", "python.exe")
        if os.name == "nt"
        else os.path.join(venv_dir, "bin", "python")
    )


PY_QWEN = os.environ.get("PY_QWEN") or _venv_python(
    os.path.join(HERE, ".venv_qwen")
)
DEFAULT_SEPFORMER_DIR = (
    r"E:/hf_cache/sepformer-whamr16k"
    if os.name == "nt"
    else "/root/hf_cache/sepformer-whamr16k"
)


def uid_from_path(path):
    return os.path.splitext(os.path.basename(path))[0]


def eligible_scene_uids(rows):
    """返回需要 scene route 的 uid；只接受恰好两个 diar speaker 的成功样本。"""
    return [
        uid_from_path(row.get("recognition", ""))
        for row in rows
        if not row.get("error")
        and len(row.get("speakers") or []) == 2
        and row.get("recognition")
    ]


def choose_scene_text(texts):
    """按冻结的内容启发式从两路文本中选一路。"""
    from exp_multivoice_route import cmd_score, route_heuristic

    if len(texts) != 2:
        raise ValueError(f"scene route requires exactly 2 texts, got {len(texts)}")
    idx, reason = route_heuristic([{"text": text} for text in texts])
    return idx, reason, [cmd_score(text) for text in texts]


def _normalize_text(text):
    from text_utils import brand_homophone_fix, digit_postproc, to_simplified

    return brand_homophone_fix(digit_postproc(to_simplified(text or "")))


def main():
    ap = argparse.ArgumentParser(
        description="双人样本 SepFormer 两路分离 + Qwen heuristic 选路"
    )
    ap.add_argument("--enroll-json", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--qwen-batch-size", type=int, default=1)
    ap.add_argument(
        "--sepformer-dir",
        default=os.environ.get("MODEL_SEPFORMER", DEFAULT_SEPFORMER_DIR),
        help="SepFormer 本地缓存目录；MODEL_SEPFORMER 可覆盖",
    )
    ap.add_argument(
        "--work-dir",
        default=None,
        help="临时两路 WAV 目录的父目录；默认使用系统临时目录",
    )
    args = ap.parse_args()

    if args.qwen_batch_size != 1:
        ap.error("提交候选固定 Qwen batch size=1")

    from repro import set_global_seed

    set_global_seed(args.seed)
    with open(args.enroll_json, encoding="utf-8") as f:
        rows = json.load(f)

    eligible = set(eligible_scene_uids(rows))
    routes = {}
    if not eligible:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "routes": routes,
                    "n_eligible": 0,
                    "n_routed": 0,
                    "n_fallback": 0,
                    "wall_sec": 0.0,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print("[scene-route] 无双人样本，保持主线")
        return
    if not os.path.isdir(args.sepformer_dir):
        ap.error(
            f"SepFormer 缓存不存在：{args.sepformer_dir}。"
            "请先确认本机已有缓存并用 MODEL_SEPFORMER 指向它；本脚本不会自动下载。"
        )

    # 延迟导入大依赖，确保无双人样本时不会加载 SepFormer/torch。
    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    from exp_sepformer_qwen import load_sepformer, separate

    t0 = time.perf_counter()
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    tmp_root = tempfile.mkdtemp(prefix="scene_route_", dir=args.work_dir)
    slice_dir = os.path.join(tmp_root, "slices")
    os.makedirs(slice_dir, exist_ok=True)
    uid2slice_uids = {}

    try:
        print(
            f"[scene-route] load SepFormer → {args.sepformer_dir}; "
            f"eligible={len(eligible)}"
        )
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        sep_model = load_sepformer(device, args.sepformer_dir)

        for index, row in enumerate(rows, 1):
            uid = uid_from_path(row.get("recognition", ""))
            if uid not in eligible:
                continue
            try:
                audio, sr = librosa.load(row["recognition"], sr=16000)
                sources = separate(audio, sep_model)
                if sources.shape[0] != 2:
                    raise RuntimeError(f"SepFormer n_src={sources.shape[0]} 非 2")
                slice_uids = []
                for src_idx in range(2):
                    slice_uid = f"{uid}__src{src_idx}"
                    sf.write(
                        os.path.join(slice_dir, slice_uid + ".wav"),
                        np.ascontiguousarray(sources[src_idx].astype(np.float32)),
                        sr,
                    )
                    slice_uids.append(slice_uid)
                uid2slice_uids[uid] = slice_uids
                routes[uid] = {"status": "separated", "slice_uids": slice_uids}
            except Exception as exc:
                routes[uid] = {
                    "status": "fallback_mainline",
                    "reason": f"{type(exc).__name__}: {str(exc)[:160]}",
                }
            if index % 50 == 0:
                print(f"  [scene-route] scanned {index}/{len(rows)}")

        # Qwen 子进程启动前释放 SepFormer 显存，避免两套模型同时驻留。
        del sep_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if uid2slice_uids:
            qwen_out = os.path.join(tmp_root, "uid2text.json")
            cmd = [
                PY_QWEN,
                os.path.join(HERE, "qwen_asr_backend.py"),
                "--slice-dir",
                slice_dir,
                "--out",
                qwen_out,
                "--seed",
                str(args.seed),
                "--batch-size",
                "1",
            ]
            subprocess.run(cmd, check=True)
            with open(qwen_out, encoding="utf-8") as f:
                uid2text = json.load(f)

            for uid, slice_uids in uid2slice_uids.items():
                texts = [_normalize_text(uid2text.get(suid, "")) for suid in slice_uids]
                try:
                    picked, reason, scores = choose_scene_text(texts)
                    routes[uid] = {
                        "status": "routed",
                        "picked_src": picked,
                        "reason": reason,
                        "scores": scores,
                        "texts": texts,
                        "transcript": texts[picked],
                    }
                except Exception as exc:
                    routes[uid] = {
                        "status": "fallback_mainline",
                        "reason": f"{type(exc).__name__}: {str(exc)[:160]}",
                        "texts": texts,
                    }
    finally:
        # 仅清理由本次调用创建的临时可重生成 WAV，不触碰用户输入或历史产物。
        shutil.rmtree(tmp_root, ignore_errors=True)

    n_routed = sum(r.get("status") == "routed" for r in routes.values())
    n_fallback = sum(
        r.get("status") == "fallback_mainline" for r in routes.values()
    )
    payload = {
        "routes": routes,
        "n_eligible": len(eligible),
        "n_routed": n_routed,
        "n_fallback": n_fallback,
        "qwen_batch_size": 1,
        "wall_sec": round(time.perf_counter() - t0, 3),
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(
        f"[scene-route] routed={n_routed}/{len(eligible)}, "
        f"fallback={n_fallback}, wall={payload['wall_sec']}s"
    )


if __name__ == "__main__":
    main()
