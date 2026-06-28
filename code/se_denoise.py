"""线A — SE(语音增强) de-risk: DeepFilterNet3 批量降噪。

诊断背景(RESULTS.md:179): 阈值旋钮解不了根本,真正瓶颈从"声纹拒识"转到
"Whisper 带噪+重叠转写质量"。本脚本用现成轻量 SE(DeepFilterNet3, 8.7MB 权重,
纯CPU可跑)对 test_wav/dataset/final/ 下带噪 wav 批量降噪,输出到
test_wav/dataset/se_denoised/(保持扁平结构, 与 final/ 同名, 便于直接喂 enroll_infer.py
的 --recognition-folder)。

设计要点:
- DeepFilterNet3 原生 48kHz full-band。本数据集 16kHz。
  策略: 16k -> 48k (torchaudio resample) -> DF3 降噪 -> 48k -> 16k (还原 DiCoW 输入采样率)。
  DF3 的增强滤波器对带宽有限制(48k 全频带), 16k 上采样后高频段无内容、DF3 不会臆造,
  低频(<=8kHz)语音仍被有效增强, 故该上下采样对 16k 数据是合理且无损可用域的。
- 权重缓存强制落 E 盘: monkeypatch df.utils.get_cache_dir -> E:/df_cache/...
  (appdirs 在 Windows 用 CSIDL 注册表, 不认 LOCALAPPDATA env, 必须 patch)。
- 默认 att_limit=200ms / atten_lim_db=0..(无上界) 由 DF3 自适应估计, 仅用 post_filter 去残余音乐噪。

用法:
  source code/setenv.sh   # 设 E 盘缓存/代理
  code/.venv_se/Scripts/python.exe code/se_denoise.py \\
      --in-dir  E:/midea_target_asr/test_wav/dataset/final \\
      --out-dir E:/midea_target_asr/test_wav/dataset/se_denoised \\
      [--limit N] [--device cpu]

CPU 验证(本 agent): --limit 2 跑 2 条看能量/波形变化即可。
GPU 全量(交主 agent): 不加 --limit, 降噪后音频转交给 DiCoW 转写。
"""
import os
import sys
import argparse
import warnings

warnings.filterwarnings("ignore")


def _patch_df_cache_to_e():
    """强制 DeepFilterNet 权重缓存目录落 E 盘(避免 C 盘 AppData)。"""
    e_cache = os.environ.get("DF_CACHE_DIR", "E:/df_cache/DeepFilterNet/Cache")
    try:
        import df.utils as dfu

        dfu.get_cache_dir = lambda: e_cache
    except Exception as e:
        print(f"[warn] patch df cache failed: {e}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="DeepFilterNet3 批量降噪(16k<->48k)")
    ap.add_argument("--in-dir", required=True, help="输入带噪 wav 目录")
    ap.add_argument("--out-dir", required=True, help="输出降噪 wav 目录")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条(0=全部, CPU 验证用)")
    ap.add_argument("--device", default="cpu", help="cpu / cuda (DF3 显存极小)")
    ap.add_argument("--model", default="DeepFilterNet3", choices=["DeepFilterNet2", "DeepFilterNet3"])
    ap.add_argument("--model-base-dir", default="E:/df_cache/DeepFilterNet/Cache/DeepFilterNet3",
                    help="DeepFilterNet3 权重目录(默认 E 盘缓存, 避免落 C 盘 AppData)")
    ap.add_argument("--post-filter", action="store_true", default=True, help="残余音乐噪后滤波")
    ap.add_argument("--atten-lim-db", type=int, default=0,
                    help="最大衰减 dB(0=不限制, DF3 自适应); 设 6/12 可减少过消除")
    args = ap.parse_args()

    _patch_df_cache_to_e()

    import torch
    import torchaudio
    import glob as _glob
    from df.enhance import init_df, enhance

    in_sr = 16000  # 数据集采样率(与 final_manifest.json sr 一致)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[se] init {args.model} on {args.device} ...")
    model, df_state, _ = init_df(model_base_dir=args.model_base_dir,
                                 default_model=args.model,
                                 post_filter=args.post_filter,
                                 log_file=None)
    df_sr = int(df_state.sr())  # 48000
    # override device
    model = model.to(args.device)
    # atten_lim_db 传给 enhance(0 表示不限制 -> DF3 自适应估计); >0 时限制最大衰减
    enh_atten = args.atten_lim_db if args.atten_lim_db > 0 else None

    wavs = sorted(_glob.glob(os.path.join(args.in_dir, "*.wav")))
    if args.limit > 0:
        wavs = wavs[: args.limit]
    print(f"[se] {len(wavs)} files | in_sr={in_sr} df_sr={df_sr} -> {args.out_dir}")

    up = torchaudio.transforms.Resample(in_sr, df_sr)
    down = torchaudio.transforms.Resample(df_sr, in_sr)

    n_done = 0
    for wp in wavs:
        try:
            x, sr = torchaudio.load(wp)  # [C, T], sr 应为 16000
            if sr != in_sr:
                # 万一不是 16k, 先到 16k 再上采样到 df_sr
                x = torchaudio.transforms.Resample(sr, in_sr)(x)
            # mono
            if x.shape[0] > 1:
                x = x.mean(dim=0, keepdim=True)
            # 16k -> 48k
            x48 = up(x)
            # DF3 enhance: [C, T]; atten_lim_db=None -> DF3 自适应, 6/12 限制过消除
            with torch.no_grad():
                enh = enhance(model, df_state, x48.to(args.device), atten_lim_db=enh_atten)
            # 48k -> 16k 还原(DiCoW 期望 16k)
            enh16 = down(enh.cpu())
            out_path = os.path.join(args.out_dir, os.path.basename(wp))
            torchaudio.save(out_path, enh16, in_sr)

            # 简单能量对比(降噪生效与否的客观指标)
            e_in = float(x48.pow(2).mean().sqrt())   # RMS 48k 段
            e_out = float(enh.cpu().pow(2).mean().sqrt())
            n_done += 1
            if n_done <= 5 or n_done % 50 == 0:
                print(f"  [{n_done}/{len(wavs)}] {os.path.basename(wp)} | "
                      f"rms in={e_in:.4f} out={e_out:.4f} (ratio={e_out/max(e_in,1e-9):.3f})")
        except Exception as e:
            print(f"[err] {os.path.basename(wp)}: {e}", file=sys.stderr)

    print(f"[se] DONE {n_done}/{len(wavs)} -> {args.out_dir}")


if __name__ == "__main__":
    main()
