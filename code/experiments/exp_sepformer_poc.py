"""exp_sepformer_poc.py — SepFormer 源分离攻死区(sim<0.2)CER 的 POC(go/no-go)。

【背景】(P1 code/exp_spk_oracle.py 证伪声纹强化后) 死区 pos(max_sim<0.2, 29%, 408条,
vanilla CER 0.841) 经 oracle 验证 = babble 摧毁 mel 物理极限(非选错 target):
  - argmax 选对率 67%(多数切对了), oracle_CER 仍 0.607(作弊完美选 target 也不及格)
  - 单 spk 控制组(target 唯一零歧义, n=18) CER 0.436 → 纯音频摧毁
  → 声纹强化(CAM++/帧选择/US-PVAD)收益封闭(五连受挫), 唯一未证伪 = 音频前端源分离

【POC 问】SepFormer 盲分离(2 源)能否把 target 从 babble 里拎出干净 mel, 让 vanilla Whisper
转写 CER 显著降? vs 当前 vanilla 基线(死区 0.841)。
  GO=是: SepFormer_CER << 0.841(如 <0.6) → 音频前端有效, 投源分离方向
  GO=否: 持平/更差 → 死区是物理极限, 接受 vanilla 0.595 是天花板, 答辩诚实归因

【方法】对死区抽样:
  1. SepFormer(whamr16k, 16kHz 噪声+混响 2 源分离) 分离整条 recognition → 2 流 s0/s1
  2. 选 target 流: wespeaker 抽每流声纹, 与 enrollment 余弦最大 = target(不依赖 diar)
  3. vanilla Whisper-large-v3-turbo 转写 target 流
  4. CER vs ref(官方口径 normalize_text + editdistance 累计池, 复用 eval_metrics.cer_official)
  对比同 UID 的 vanilla_full.vanilla_cer(diar→切 timeline→转写 基线)。

【关键诚实预期】SepFormer 在英文 WHAMR! 训练(中文域失配) + 死区是 babble 最重桶(连 SOTA
都未必救得回)。POC 先证机制, 不盲目投大工程。单 spk 控制组(target 唯一) CER 是最硬试金石:
SepFormer 后仍 >0.4 → 物理极限坐实, no-go。

【环境注意】
  - speechbrain 1.1.0 装到 code/.venv(非 .venv_sep; 因 POC 需 diar+whisper+eval 全 pipeline,
    单独 .venv_sep 要重复装 torch ~5GB 不经济; speechbrain 本身仅 2.2MB, 已有 asteroid_filterbands/torch_audiomentations 依赖)
  - Windows symlink 权限不足 → local_strategy=LocalStrategy.COPY
  - speechbrain LazyModule.ensure_module 的 inspect-guard 用 endswith('/inspect.py') 在
    Windows(\) 失效 → 启动时 monkeypatch 路径分隔符, 否则分离时日志触发 ImportError 崩溃

用法:
  source code/setenv.sh
  code/.venv/Scripts/python.exe code/exp_sepformer_poc.py [--n-sample 40]
产物: code/exp_sepformer_poc.json + stdout 明确 go/no-go 判定。
"""
import os, sys, json, time, argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

# ---- speechbrain Windows 兼容(必须在 import speechbrain.* 前) ----
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import speechbrain.utils.importutils as _sb_iu
import inspect as _inspect, importlib as _importlib, warnings as _warnings


def _patched_ensure_module(self, stacklevel):
    """修正 SB 1.1.0 LazyModule 的 inspect-guard 在 Windows 路径分隔符失效的 bug。

    原代码 endswith('/inspect.py') 在 Windows(\inspect.py) 永远 False → 日志层的
    inspect.getmodule 遍历 sys.modules 触发 LazyModule.__getattr__('__file__') →
    ensure_module → 再触发 getframeinfo → 死循环/ImportError。POC 注释清楚来源。
    """
    fr = None
    try:
        fr = _inspect.getframeinfo(sys._getframe(stacklevel + 1))
    except AttributeError:
        pass
    if fr is not None and fr.filename.replace("\\", "/").endswith("/inspect.py"):
        raise AttributeError()
    if self.lazy_module is None:
        try:
            self.lazy_module = (_importlib.import_module(self.target) if self.package is None
                                else _importlib.import_module("." + self.target, self.package))
        except Exception as e:
            raise ImportError("Lazy import failed (patched ensure_module)") from e
    return self.lazy_module


_sb_iu.LazyModule.ensure_module = _patched_ensure_module
# 标记补丁已装(防双次执行)
_SB_PATCHED = True

import numpy as np
import torch
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoTokenizer, AutoFeatureExtractor

from enroll_infer import get_diarization_mask, collect_clean_audio
from text_utils import to_simplified, digit_postproc
from eval_metrics import cer_official
from repro import set_global_seed, resolve_model


def load_sepformer(device, savedir):
    """SepFormer whamr16k(16kHz 噪声+混响 2 源分离). 返回 model 对象(已在 device)。"""
    from speechbrain.inference.separation import SepformerSeparation as separator
    from speechbrain.utils.fetching import LocalStrategy
    src = "speechbrain/sepformer-whamr16k"
    model = separator.from_hparams(
        source=src, savedir=savedir,
        local_strategy=LocalStrategy.COPY,
        run_opts={"device": str(device)})
    return model


def separate(mix_np, model):
    """分离 mixture → [n_src, T] np.ndarray(16kHz)。mix_np: 1D float32。"""
    mix = torch.from_numpy(np.ascontiguousarray(mix_np.astype(np.float32))).to(model.device)[None, :]
    est = model.separate_batch(mix)  # [1, T, n_src]
    est = est.squeeze(0).detach().cpu().numpy()  # [T, n_src]
    return est.T  # [n_src, T]


def load_whisper(device, vanilla_model, dtype):
    print(f"[load] vanilla Whisper {vanilla_model} on {device}")
    m = AutoModelForSpeechSeq2Seq.from_pretrained(vanilla_model, torch_dtype=dtype).to(device).eval()
    tok = AutoTokenizer.from_pretrained(vanilla_model)
    fe = AutoFeatureExtractor.from_pretrained(vanilla_model)
    return m, tok, fe


def load_diar(diar_model, device):
    print(f"[load] DiariZen {diar_model}")
    from diarizen.pipelines.inference import DiariZenPipeline
    diar = DiariZenPipeline.from_pretrained(diar_model).to(device)
    return diar


def get_emb_factory(diar, device):
    """复用 diar._embedding(wespeaker) 抽声纹(与 enroll_infer.get_emb 一致)。"""
    def get_emb(wav_np):
        w = torch.from_numpy(np.ascontiguousarray(wav_np.astype(np.float32))).to(device)
        if w.dim() == 1:
            w = w[None, None]
        elif w.dim() == 2:
            w = w[None]
        with torch.no_grad():
            emb = diar._embedding(w)
        emb = torch.as_tensor(emb, device=device, dtype=torch.float32)
        return torch.nn.functional.normalize(emb, dim=-1).squeeze(0)
    return get_emb


def transcribe(asr_model, tok, fe, audio_np, device, dtype, language="zh"):
    ifp = fe(audio_np, sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype)
    am = torch.ones(1, ifp.shape[-1], dtype=torch.bool, device=device)
    with torch.no_grad():
        out = asr_model.generate(input_features=ifp, attention_mask=am,
                                 language=language, task="transcribe", max_new_tokens=200)
    seqs = out["sequences"] if isinstance(out, dict) else out
    text = tok.batch_decode(seqs, skip_special_tokens=True)[0].strip()
    text = to_simplified(text)
    text = digit_postproc(text)
    return text


def main():
    ap = argparse.ArgumentParser(description="SepFormer 源分离攻死区 CER POC")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--vanilla-model", default=resolve_model("VANILLA"))
    ap.add_argument("--diarization-model", default=resolve_model("DIAR"))
    ap.add_argument("--sepformer-dir", default="E:/hf_cache/sepformer-whamr16k")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-sample", type=int, default=40, help="死区抽样条数")
    ap.add_argument("--sim-max", type=float, default=0.2, help="死区上界 max_sim<sim_max")
    ap.add_argument("--vanilla-full", default=os.path.join(_HERE, "exp_vanilla_full.json"))
    ap.add_argument("--pairs", default=os.path.join(_HERE, "pos_pairs_datasetA.json"))
    ap.add_argument("--out-json", default=os.path.join(_HERE, "exp_sepformer_poc.json"))
    args = ap.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    # ---- 1. 载入 vanilla_full + pairs, 筛死区, 抽样 ----
    full = json.load(open(args.vanilla_full, encoding="utf-8"))
    pairs = json.load(open(args.pairs, encoding="utf-8"))
    uid2pair = {os.path.splitext(os.path.basename(p["recognition"]))[0]: p for p in pairs}
    deadzone = [d for d in full
                if d.get("max_sim") is not None and d["max_sim"] < args.sim_max
                and d.get("ref") and d["uid"] in uid2pair]
    print(f"[data] vanilla_full={len(full)} 死区(max_sim<{args.sim_max}, 有ref)={len(deadzone)}")
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(deadzone))[:args.n_sample]
    samples = [deadzone[i] for i in sorted(idx)]
    print(f"[data] 抽样 {len(samples)} 条(seed={args.seed})")

    # ---- 2. 加载模型(SepFormer + DiariZen[声纹] + Whisper) ----
    print(f"[load] SepFormer whamr16k → {args.sepformer_dir}")
    sep_model = load_sepformer(device, args.sepformer_dir)
    diar = load_diar(args.diarization_model, device)
    get_emb = get_emb_factory(diar, device)
    asr_model, tok, fe = load_whisper(device, args.vanilla_model, dtype)
    print(f"[load] 全部就绪, peak mem {torch.cuda.max_memory_allocated()/1e6:.0f}MB")

    # ---- 3. 主循环 ----
    results = []
    t_start = time.time()
    for n, d in enumerate(samples):
        pair = uid2pair[d["uid"]]
        enr, rec = pair["enrollment"], pair["recognition"]
        ref = d["ref"]
        uid = d["uid"]
        t0 = time.time()

        try:
            w_enr, _ = librosa.load(enr, sr=16000)
            enroll_emb = get_emb(w_enr)
            audio, sr = librosa.load(rec, sr=16000)

            # SepFormer 分离
            sources = separate(audio, sep_model)  # [n_src, T]
            n_src = sources.shape[0]

            # 选 target 流(声纹余弦最大)
            stream_embs = []
            for i in range(n_src):
                seg = sources[i]
                # wespeaker 需 ≥1s, 短流补齐
                if len(seg) < sr:
                    seg = np.tile(seg, sr // len(seg) + 1)[:sr]
                stream_embs.append(get_emb(seg))
            sims = torch.stack([torch.dot(enroll_emb, e) for e in stream_embs])
            target_idx = int(torch.argmax(sims))
            target_sim = float(sims[target_idx])
            target_audio = sources[target_idx]

            # 转写 target 流
            text = transcribe(asr_model, tok, fe, target_audio, device, dtype)
            sep_cer = float(cer_official(text, ref))

            # 同时记录次优流(诊断: target 是否真在 SepFormer 输出里)
            other_idx = 1 - target_idx if n_src == 2 else -1
            other_cer = None
            if other_idx >= 0:
                other_text = transcribe(asr_model, tok, fe, sources[other_idx], device, dtype)
                other_cer = float(cer_official(other_text, ref))

            dt = time.time() - t0
            v_cer = d.get("vanilla_cer")
            delta = (sep_cer - v_cer) if v_cer is not None else float("nan")
            mark = ("✓更优" if (v_cer is not None and sep_cer < v_cer - 0.01)
                    else ("✗更差" if (v_cer is not None and sep_cer > v_cer + 0.01) else "≈持平"))
            print(f"  [{n+1}/{len(samples)}] {uid} n_src={n_src} sep_sim={target_sim:.3f} "
                  f"sep_CER={sep_cer:.3f} vanilla_CER={v_cer if v_cer is not None else float('nan'):.3f} "
                  f"Δ={delta:+.3f} {mark} ({dt:.1f}s)")
            print(f"      sep:\"{text[:35]}\" other_CER={other_cer if other_cer is not None else 'NA'} "
                  f"ref:\"{ref[:30]}\"")

            results.append({
                "uid": uid, "ref": ref,
                "max_sim_vanilla": d["max_sim"],
                "n_src": n_src,
                "stream_sims": [round(float(s), 4) for s in sims],
                "target_idx": target_idx, "target_sim": round(target_sim, 4),
                "sep_text": text, "sep_cer": round(sep_cer, 4),
                "other_idx": other_idx,
                "other_cer": round(other_cer, 4) if other_cer is not None else None,
                "vanilla_cer_baseline": v_cer,  # diar→切 timeline→转写 基线
                "vanilla_text_baseline": d.get("vanilla_text"),
                "delta_vs_vanilla": round(delta, 4) if v_cer is not None else None,
                "rec_sec": round(dt, 2),
            })
        except Exception as e:
            print(f"  [{n+1}/{len(samples)}] {uid} FAIL {type(e).__name__}: {str(e)[:120]}")
            results.append({"uid": uid, "error": f"{type(e).__name__}: {str(e)[:200]}"})

    total_dt = time.time() - t_start

    # ---- 4. 统计 + go/no-go ----
    valid = [r for r in results if "error" not in r]
    n_valid = len(valid)
    print(f"\n{'='*70}\n[SepFormer POC 结论] 有效 {n_valid}/{len(samples)} 条, 总耗时 {total_dt/60:.1f}min")
    if n_valid == 0:
        print("无有效样本, 无法判定。")
        return

    sep_cers = np.array([r["sep_cer"] for r in valid])
    # 基线对齐(只取有 vanilla_cer_baseline 的)
    paired = [r for r in valid if r.get("vanilla_cer_baseline") is not None]
    n_paired = len(paired)
    v_cers = np.array([r["vanilla_cer_baseline"] for r in paired])
    sep_cers_paired = np.array([r["sep_cer"] for r in paired])

    sep_mean = float(np.mean(sep_cers))
    sep_correct = float(np.mean(sep_cers < 0.5))  # CER<0.5 占比
    sep_near = float(np.mean(sep_cers < 0.1))
    v_mean = float(np.mean(v_cers)) if n_paired else float("nan")
    v_correct = float(np.mean(v_cers < 0.5)) if n_paired else float("nan")

    # oracle(取两流中 CER 更低) → SepFormer 分离质量上限(选 target 流选错时校准)
    both = [r for r in valid if r.get("other_cer") is not None]
    oracle_sep_cers = np.array([min(r["sep_cer"], r["other_cer"]) for r in both])
    oracle_sep_mean = float(np.mean(oracle_sep_cers)) if both else float("nan")
    # 选对率(选的流 CER ≤ 另一流)
    pick_correct = float(np.mean([r["sep_cer"] <= (r["other_cer"] if r["other_cer"] is not None else 9)
                                  for r in both])) if both else float("nan")

    # 逐条对比
    better = sum(1 for r in paired if r["sep_cer"] < r["vanilla_cer_baseline"] - 0.01)
    worse = sum(1 for r in paired if r["sep_cer"] > r["vanilla_cer_baseline"] + 0.01)
    tie = n_paired - better - worse
    delta_paired_mean = float(np.mean(sep_cers_paired - v_cers)) if n_paired else float("nan")

    print(f"\n[核心指标](死区 max_sim<{args.sim_max})")
    print(f"  SepFormer CER 均值:              {sep_mean:.3f}  (correct<0.5: {sep_correct*100:.0f}%, near<0.1: {sep_near*100:.0f}%)")
    print(f"  Vanilla 基线 CER 均值(同UID):    {v_mean:.3f}  (correct<0.5: {v_correct*100:.0f}%)")
    print(f"  Δ(SepFormer - Vanilla):          {delta_paired_mean:+.3f}")
    print(f"  逐条: SepFormer 更优 {better} / 更差 {worse} / 持平 {tie} (n={n_paired})")
    print(f"\n[SepFormer 分离质量诊断]")
    print(f"  SepFormer oracle CER(两流取最优): {oracle_sep_mean:.3f}  (n={len(both)})")
    print(f"  target 流选对率(选流≤另一流):     {pick_correct*100:.0f}%")
    print(f"    ↳ oracle 仍高 = 分离本身没救(物理极限); oracle 低但 argmax 高 = 选流问题")

    # 单 source 情况(n_src 实际=2, 但识别仅 1 有效 speaker)
    target_sims = np.array([r["target_sim"] for r in valid])
    print(f"\n[target 流声纹相似度]")
    print(f"  sep_sim 均值: {float(np.mean(target_sims)):.3f}  (vs vanilla_full max_sim 均值 "
          f"{float(np.mean([r['max_sim_vanilla'] for r in valid])):.3f})")

    # ---- go/no-go 判定 ----
    print(f"\n{'='*70}\n[go/no-go 判定]")
    # 判据: SepFormer CER 显著低于 vanilla 基线(Δ<-0.10 且 correct 率提升) → GO
    #       持平/更差 → NO-GO(物理极限)
    #       oracle 也高 → 分离本身无效; oracle 低但 argmax 高 → 改进选流能救
    significant = (delta_paired_mean < -0.10) and (sep_correct > v_correct + 0.05)
    oracle_recoverable = (oracle_sep_mean < v_mean - 0.10) and not significant

    if significant:
        verdict = "GO=是"
        reason = (f"SepFormer 死区 CER {sep_mean:.3f} 显著低于 vanilla 基线 {v_mean:.3f} "
                  f"(Δ{delta_paired_mean:+.3f}, correct {sep_correct*100:.0f}% vs {v_correct*100:.0f}%)。"
                  f"音频前端源分离有效, 投 SepFormer 集成(或更强 TSE 模型)方向。")
    elif oracle_recoverable:
        verdict = "GO=偏是(改选流)"
        reason = (f"SepFormer argmax CER {sep_mean:.3f} 未胜基线, 但 oracle(两流取优){oracle_sep_mean:.3f} "
                  f"显著低于 vanilla {v_mean:.3f} → 分离拎出了干净 target, 只是声纹选流选错。"
                  f"改进 target 流选择(如 enrollment 更强声纹/diar 配合)能救。")
    else:
        if oracle_sep_mean >= v_mean - 0.05:
            verdict = "GO=否"
            reason = (f"SepFormer 即便 oracle(两流取优)CER {oracle_sep_mean:.3f} 也不显著低于 vanilla 基线 "
                      f"{v_mean:.3f} → 分离本身没拎出更干净 target mel, 死区是 babble 摧毁的物理极限, "
                      f"SepFormer(英文 WHAMR! 域失配 + 死区最重桶)也救不动。"
                      f"→ 接受 vanilla 0.595 是组合主线天花板, 答辩诚实归因。")
        else:
            verdict = "GO=否(偏)"
            reason = (f"SepFormer argmax CER {sep_mean:.3f}, oracle {oracle_sep_mean:.3f} 略低于基线 "
                      f"{v_mean:.3f} 但收益边际(Δ<{0.10}), 投大工程不划算。死区仍以物理极限为主。")

    print(f"  判定: {verdict}")
    print(f"  理由: {reason}")

    summary = {
        "verdict": verdict, "reason": reason,
        "n_sample": len(samples), "n_valid": n_valid, "n_paired": n_paired,
        "sep_cer_mean": round(sep_mean, 4),
        "sep_correct_rate": round(sep_correct, 4),
        "sep_near_perfect_rate": round(sep_near, 4),
        "vanilla_cer_mean_paired": round(v_mean, 4) if n_paired else None,
        "vanilla_correct_rate_paired": round(v_correct, 4) if n_paired else None,
        "delta_mean": round(delta_paired_mean, 4) if n_paired else None,
        "n_sepformer_better": better, "n_sepformer_worse": worse, "n_tie": tie,
        "sepformer_oracle_cer_mean": round(oracle_sep_mean, 4) if both else None,
        "target_stream_pick_correct_rate": round(pick_correct, 4) if both else None,
        "sep_sim_mean": round(float(np.mean(target_sims)), 4),
        "model": "speechbrain/sepformer-whamr16k",
        "total_min": round(total_dt / 60, 2),
    }
    out = {"summary": summary, "results": results}
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[done] {args.out_json} (总耗时 {total_dt/60:.1f}min)")


if __name__ == "__main__":
    main()
