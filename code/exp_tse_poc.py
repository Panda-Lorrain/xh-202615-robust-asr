#!/usr/bin/env python
"""TSELM 单条 POC：enrollment + 混合 audio → 提取 target → 保存 wav。

TSELM (Tang et al., 2024) target speaker extraction:
  mix (含 target + 干扰) + regi (enrollment 参考) → 提取 target 干净 audio
  架构: WavLM 离散化 + cross-attention (enroll 条件) + Conformer LM + HiFiGAN 解码

POC 验证三个翻车点 (主线 CER 都~1.0):
  - cmd_2637 重叠区 (sim 0.585 主战场, TSE 对症)
  - cmd_18/2098 死区 (sim<0.2 物理地板, TSE 期望救不回 → 验证 memory 主战区分界)
  - cmd_2251/2687/2630 额外重叠组高 sim 高 CER

依赖: code/.venv_tse (torch+transformers+speechbrain+hyperpyyaml)
权重: E:/hf_cache/tselm/{tselm_l.pth, kmeans/, hifigan/, wavlm/}
"""
import os, sys, json, argparse, time, traceback
import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml

# 把 TSELM 源码加到 path
TSELM_ROOT = r"E:/midea_target_asr/code/TSELM"
sys.path.insert(0, TSELM_ROOT)


def load_tselm_model(config_path: str, ckpt_path: str, device: str = "cuda:0"):
    """按 TSELM 官方 inference.py 流程加载模型。"""
    print(f"[load] config={config_path}")
    with open(config_path, "r") as f:
        config = load_hyperpyyaml(f)
    model = config["model"]
    print(f"[load] ckpt={ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.cuda(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    print("[load] done")
    return model


def extract_one(model, mix_wav: str, regi_wav: str, device: str = "cuda:0",
                out_path: str = None, sr_out: int = 16000):
    """对单条 (mix, enroll) 提取 target，保存 16kHz wav。返回 (target_audio [T], sr)。"""
    mix, sr1 = torchaudio.load(mix_wav)  # [1, T]
    regi, sr2 = torchaudio.load(regi_wav)
    # TSELM 训练 16kHz; 数据已是 16kHz, 不重采样
    if sr1 != 16000:
        mix = torchaudio.functional.resample(mix, sr1, 16000)
    if sr2 != 16000:
        regi = torchaudio.functional.resample(regi, sr2, 16000)
    mix = mix.mean(dim=0, keepdim=False).to(device)  # [T] mono
    regi = regi.mean(dim=0, keepdim=False).to(device)
    # 数据 float32
    mix = mix.float()
    regi = regi.float()
    # Model.inference 内部会 split_audio 切 mix + truc_wav pad regi 到 64080
    with torch.no_grad():
        recon, length = model.inference(mix.unsqueeze(0), regi.unsqueeze(0))  # [1,T'], int
    recon = recon.detach().cpu().float()
    if out_path:
        torchaudio.save(out_path, recon, sr_out)
    return recon, length


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=r"E:/midea_target_asr/code/TSELM/config/tselm_l_poc.yaml")
    ap.add_argument("--ckpt", default=r"E:/hf_cache/tselm/tselm_l.pth")
    ap.add_argument("--out-dir", default=r"E:/midea_target_asr/code/runs/_tse_poc")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--samples", nargs="+",
                    default=["cmd_2637", "cmd_18", "cmd_2098",
                             "cmd_2251", "cmd_2687", "cmd_2630"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    model = load_tselm_model(args.config, args.ckpt, device=args.device)

    # 样本路径配置
    pos_dir = r"E:/midea_target_asr/datasetA/pos"
    results = {}
    for uid in args.samples:
        mix_wav = os.path.join(pos_dir, f"{uid}.wav")
        regi_wav = os.path.join(pos_dir, f"kws_{uid.split('_')[1]}.wav")
        out_wav = os.path.join(args.out_dir, f"{uid}_tse.wav")
        if not os.path.exists(mix_wav):
            print(f"  [skip] {uid}: mix 不存在 {mix_wav}")
            continue
        if not os.path.exists(regi_wav):
            print(f"  [skip] {uid}: enroll 不存在 {regi_wav}")
            continue
        t0 = time.time()
        try:
            recon, length = extract_one(model, mix_wav, regi_wav,
                                        device=args.device, out_path=out_wav)
            dt = time.time() - t0
            # 输入/输出时长
            info_mix = torchaudio.info(mix_wav)
            in_dur = info_mix.num_frames / info_mix.sample_rate
            out_dur = length / 16000
            print(f"  {uid}: in={in_dur:.2f}s out={out_dur:.2f}s "
                  f"RTF={dt/out_dur:.2f} ({dt:.1f}s) -> {out_wav}")
            results[uid] = {
                "mix_wav": mix_wav,
                "enroll_wav": regi_wav,
                "out_wav": out_wav,
                "in_dur_s": round(in_dur, 3),
                "out_dur_s": round(out_dur, 3),
                "extract_seconds": round(dt, 2),
                "rtf_extract": round(dt / out_dur, 3),
                "status": "ok",
            }
        except Exception as e:
            print(f"  {uid} FAIL {type(e).__name__}: {str(e)[:120]}")
            traceback.print_exc()
            results[uid] = {"status": "fail", "err": f"{type(e).__name__}: {str(e)[:200]}"}

    out_json = os.path.join(args.out_dir, "tse_extract_summary.json")
    json.dump(results, open(out_json, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n=== summary → {out_json} ===")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
