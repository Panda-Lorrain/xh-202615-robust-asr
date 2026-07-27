#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen3-ASR LoRA 1k 对 POC 微调 + A 集 hold-out 评测 (ΔCER 趋势看是否扩 1 万)

⚠️ 任务背景(2026-07-27): 外部训练解禁, 死区 60% 失败是「ASR 错+接近地板」(幻觉/小声/被盖)。
   微调 Qwen3-ASR-1.7B 攻这 60%。主线 qwen CER=0.3436 (官方池, 1350 条 pos 切片)。
   1k 对 POC 只看 ΔCER 趋势,不求收敛。

数据合成 (复用 data_aug_recipe 配方, 预加载池加速; 原 build_pairs 每条 reload 干扰/噪声
        致 4h+, 本脚本预加载一次池 → ~15min/1k 对):
  5 步链: 小声化(-8~-3dB) + 快语速(1.1-1.4x) + 重叠(0-100%, 偏中高) + 加噪(SNR -5~5dB, 偏低)
        + 短 enroll(1.5-2.5s, 30% 加噪污染)
  → recognition_audio (target+interferer mix+noise) + ref (target 原始 transcript)
  Qwen3-ASR 只吃 recognition_audio → ref, 不吃 enrollment (Qwen3-ASR 是纯 ASR 非声纹 aware)

LoRA 微调 (4060 8GB 紧):
  - bf16 + gradient_checkpointing + paged_adamw_8bit + batch=1 + grad_accum=8
  - LoraConfig(r=16, alpha=32, target=q_proj+v_proj, lr=1e-4, dropout=0.05), 3 epoch
  - 自写 transformers Trainer + 数据 collator (qwen_asr 无训练接口)

A 集评测 (铁律: A 集绝不进训练):
  - target_slices_full/cmd_*.wav 1350 条 (复用主线 qwen 测过的同集合)
  - 用 base+adapter 转写, 算 CER (eval_metrics.CERMetric 官方口径: NFKC + 去 P* + 累计池)
  - 复刻 recompute_qwen_official.submit_norm (to_simplified + digit_postproc) 提交侧归一
  - 对比主线 qwen 0.3436, 分桶死区 sim<0.4 vs 主战场 sim≥0.4

判定:
  - ΔCER ≤ -0.04 (超噪声地板) + 死区改善 → 有效, 扩 1 万对
  - ΔCER ±0.04 噪声内 → 不足, 调参/扩数据
  - ΔCER > 0 退化 → 换方向 (LoRA 容量/数据域)

用法:
  code/.venv_qwen/Scripts/python.exe code/exp_qwen_lora_poc.py [--skip-synth] [--skip-train] [--skip-eval]
输出:
  code/runs/_qwen_lora_poc/{aug_manifest.jsonl, adapter/, eval_result.json, train_log.json}
"""
from __future__ import annotations

import os, sys, json, time, glob, argparse, random, math
from dataclasses import asdict
from typing import List, Dict, Optional

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# 复用项目内配方 + 仿真 + 评测
from data_aug_recipe import (synthesize_one, sample_aug_params, SR,
                              DEFAULT_OVERLAP_BUCKETS, DEFAULT_OVERLAP_WEIGHTS,
                              DEFAULT_SNR_BUCKETS, DEFAULT_SNR_WEIGHTS,
                              DEFAULT_NOISE_TYPES, DEFAULT_NOISE_WEIGHTS,
                              QUIET_PROB, QUIET_DB_RANGE, FAST_PROB, FAST_RATE_RANGE)
import librosa, soundfile as sf
from eval_metrics import CERMetric
from text_utils import to_simplified, digit_postproc

RUN_DIR = os.path.join(_HERE, "runs", "_qwen_lora_poc")
AUG_DIR = os.path.join(RUN_DIR, "aug")
ADAPTER_DIR = os.path.join(RUN_DIR, "adapter")
EVAL_OUT = os.path.join(RUN_DIR, "eval_result.json")
TRAIN_LOG = os.path.join(RUN_DIR, "train_log.json")
AUG_MANIFEST = os.path.join(AUG_DIR, "manifest.jsonl")

QWEN_MODEL_PATH = r"E:/hf_cache/Qwen3-ASR-1.7B"
TARGET_SLICES_DIR = r"E:/target_slices_full"
QWEN_BASELINE_RESULT = os.path.join(_HERE, "runs", "poc_qwen_asr_full_result.json")
SEED = 42


# ============================================================================
# 步骤1: 快速合成 1k 训练对 (预加载池加速)
# ============================================================================
def _submit_norm(text: str) -> str:
    """复刻 recompute_qwen_official.submit_norm 提交链路归一。"""
    return digit_postproc(to_simplified(text or ""))


def load_jsonl(path: str) -> List[Dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def preload_pool(items: List[Dict], sr: int = SR, tag: str = "") -> List[np.ndarray]:
    """预加载所有 wav 到内存 (合成期反复随机抽, 避免每次 librosa.load)。"""
    pool = []
    t0 = time.time()
    for i, it in enumerate(items):
        try:
            w, _ = librosa.load(it["wav"], sr=sr)
            pool.append(w.astype(np.float32))
        except Exception as e:
            print(f"  [preload_{tag}] skip {os.path.basename(it['wav'])}: {e}")
    print(f"  [preload_{tag}] {len(pool)}/{len(items)} loaded in {time.time()-t0:.1f}s "
          f"(mem ~{sum(x.nbytes for x in pool)/1e9:.2f} GB)")
    return pool


def synthesize_batch_fast(
    target_items: List[Dict],
    interferer_pool: List[np.ndarray],
    noise_pool: List[np.ndarray],
    out_dir: str,
    n_pairs: int,
    seed: int = 42,
) -> str:
    """快速合成 n_pairs 训练对 → out_dir/{enrollment,recognition}/*.wav + manifest.jsonl。

    与 data_aug_recipe.build_pairs 等价, 但 interferer/noise 池预加载一次,
    跳过 enroll 阶段(Qwen3-ASR 只吃 recognition 不吃 enroll, 节省 IO)。
    """
    os.makedirs(os.path.join(out_dir, "recognition"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "enrollment"), exist_ok=True)
    rng = random.Random(seed)
    manifest_path = os.path.join(out_dir, "manifest.jsonl")
    # 加载所有 target wav 到内存 (1000 × ~5s × 32kbps ≈ 几百 MB, 可控)
    print(f"[synth] 加载 {len(target_items)} target wav 到内存...")
    t_wavs = []
    for i, ti in enumerate(target_items):
        try:
            w, _ = librosa.load(ti["wav"], sr=SR)
            t_wavs.append((ti, w.astype(np.float32)))
        except Exception:
            pass
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(target_items)}]")
    print(f"[synth] {len(t_wavs)} target wav 加载完成")

    t0 = time.time()
    cnt = 0
    fallback_noise = np.zeros(SR * 2, dtype=np.float32)
    with open(manifest_path, "w", encoding="utf-8") as fout:
        for k in range(n_pairs):
            ti, t_wav = t_wavs[k % len(t_wavs)]
            params = sample_aug_params(rng)
            interferer_wav = interferer_pool[rng.randrange(len(interferer_pool))]
            noise_wav = noise_pool[rng.randrange(len(noise_pool))] if noise_pool else None
            enroll, recog, aug = synthesize_one(
                t_wav, interferer_wav, noise_wav,
                nontarget_pool=interferer_pool, rng=rng, **params,
            )
            uid = f"pair_{k:04d}"
            rec_path = os.path.join(out_dir, "recognition", uid + ".wav")
            enr_path = os.path.join(out_dir, "enrollment", uid + ".wav")
            sf.write(rec_path, recog, SR)
            sf.write(enr_path, enroll, SR)
            rec_line = {
                "id": uid,
                "enrollment_audio": enr_path,
                "recognition_audio": rec_path,
                "ref": ti["ref"],
                "target_src": ti["wav"],
                **asdict(aug),
            }
            fout.write(json.dumps(rec_line, ensure_ascii=False) + "\n")
            cnt += 1
            if (k + 1) % 100 == 0:
                elapsed = time.time() - t0
                print(f"  [synth] {k+1}/{n_pairs} ({(k+1)/elapsed:.1f} pair/s)")
    print(f"[synth] 共 {cnt} 训练对 → {out_dir} (耗时 {time.time()-t0:.0f}s)")
    return manifest_path


# ============================================================================
# 步骤2: LoRA 微调 (transformers Trainer + 自写 collator)
# ============================================================================
def _build_train_prompt_text(processor, force_language: str = "Chinese") -> str:
    """构造训练 prompt: chat template (system 空 + user audio) + generation prompt +
    `language Chinese<asr_text>` (强制纯文本输出, 同 qwen3_asr._build_text_prompt)。
    """
    msgs = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    base = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    base = base + f"language {force_language}{'<asr_text>'}"
    return base


class _Dataset(torch.utils.data.Dataset):
    """每条: recognition_audio path + ref text → {input_ids, input_features, labels}。
    labels = input_ids 复制 + 把 <asr_text> 之前的 prompt 部分(含 audio 占位 token)置 -100。
    """
    def __init__(self, manifest_path: str, processor, tokenizer, language: str = "Chinese"):
        self.items = load_jsonl(manifest_path)
        self.processor = processor
        self.tokenizer = tokenizer
        self.language = language
        # 训练 prompt 模板 (逐条一样, 但 audio_token 数随 mel 长度变, processor 自动处理)
        # 这里只 cache 文本 prompt 骨架 (含单个 audio_token 占位), 实际 __getitem__ 时
        # 用 processor 把 audio_token 替换成正确数量的占位符
        self.prompt_skeleton = _build_train_prompt_text(processor, language)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        wav, _ = librosa.load(it["recognition_audio"], sr=SR)
        ref = it["ref"] or ""

        # 构造完整 prompt + ref + eos (作为 target)
        # processor.__call__ 会: ① 把 audio_token 替换成 N 个 audio_token (N=mel chunks)
        # ② tokenize 文本得到 input_ids
        # 训练时完整序列 = prompt + ref + eos; labels 把 prompt 部分 mask
        # 直接用 processor 处理完整字符串(prompt 骨架 + ref), 它会展开 audio_token
        eos_tok = self.tokenizer.eos_token or "<|endoftext|>"
        full_text = self.prompt_skeleton + ref + eos_tok
        inputs = self.processor(
            text=full_text, audio=wav, return_tensors="pt", padding=False,
        )
        # inputs 是 BatchFeature (batch 维 1), squeeze 掉
        item = {}
        for k, v in inputs.items():
            item[k] = v.squeeze(0) if hasattr(v, "squeeze") else v

        # 构建 labels: 1) 找到 "<asr_text>" 在 input_ids 中的位置, 之前全 -100, 之后保留
        # 2) 注意: pad 部分也 -100 (本 batch=1 无 pad)
        input_ids = item["input_ids"]
        labels = input_ids.clone()
        # 找 <asr_text> token id
        asr_text_tok = self.tokenizer.convert_tokens_to_ids("<asr_text>")
        if asr_text_tok is None or asr_text_tok < 0:
            # fallback: tokenizer 没注册时退化为 loss on 全部 (差但不阻塞)
            print("[warn] <asr_text> not in tokenizer, fallback labels all valid")
        else:
            # 找到 asr_text_tok 的位置, mask 它和之前的所有 token
            mask_idx = (input_ids == asr_text_tok).nonzero(as_tuple=False)
            if len(mask_idx) > 0:
                cutoff = mask_idx[0].item() + 1  # 包含 <asr_text> 自身也 mask
                labels[:cutoff] = -100
            else:
                # 没找到 (异常) → 全 mask 跳过这条
                labels[:] = -100
        # mask attention padding (feature_attention_mask 与 input_ids 同长, 应已对齐)
        item["labels"] = labels
        return item


class _Collator:
    """pad input_ids / labels / attention_mask / feature_attention_mask 到 batch 内最长。
    featureAttentionMask 与 input_features 时间维 pad; input_ids/labels 右侧 pad。
    """
    def __init__(self, tokenizer, pad_token_id: Optional[int] = None):
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id if pad_token_id is not None else tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        # 1) 文本侧: input_ids / attention_mask / labels 右 pad
        max_len = max(b["input_ids"].shape[0] for b in batch)
        input_ids, attn, labels = [], [], []
        for b in batch:
            n = b["input_ids"].shape[0]
            pad = max_len - n
            input_ids.append(torch.cat([b["input_ids"],
                                        torch.full((pad,), self.pad_token_id, dtype=b["input_ids"].dtype)]))
            attn.append(torch.cat([b["attention_mask"].to(torch.long),
                                   torch.zeros((pad,), dtype=torch.long)]))
            lab = b["labels"]
            labels.append(torch.cat([lab, torch.full((pad,), -100, dtype=lab.dtype)]))
        input_ids = torch.stack(input_ids)
        attention_mask = torch.stack(attn)
        labels = torch.stack(labels)

        # 2) 音频侧: input_features 形状 (mel_bins=128, T_feat) 每条;
        # feature_attention_mask 形状 (T_feat,); 两者时间维 = dim=-1 / dim=0 对齐
        Ts = [b["input_features"].shape[-1] for b in batch]  # 时间维 = 最后一维
        max_T = max(Ts)
        mel_bins = batch[0]["input_features"].shape[0]  # 128 mel bins (Qwen3-ASR)
        feats = torch.zeros((len(batch), mel_bins, max_T), dtype=batch[0]["input_features"].dtype)
        fam = torch.zeros((len(batch), max_T), dtype=torch.long)
        for i, b in enumerate(batch):
            t = b["input_features"].shape[-1]
            feats[i, :, :t] = b["input_features"]      # mel 全保留, 时间右侧 pad
            fam[i, :t] = b["feature_attention_mask"].to(torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "input_features": feats,
            "feature_attention_mask": fam,
        }


def train_lora(manifest_path: str, model_path: str, out_dir: str,
               epochs: int = 3, lr: float = 1e-4, lora_r: int = 16, lora_alpha: int = 32,
               grad_accum: int = 8, max_steps: int = 0, log_every: int = 5):
    """LoRA 微调 Qwen3-ASR-1.7B → out_dir/adapter。

    显存控制: bf16 + gradient_checkpointing + paged_adamw_8bit + batch=1 + grad_accum。
    qwen_asr.Qwen3ASRForConditionalGeneration.forward 接受 input_ids, input_features,
    attention_mask, feature_attention_mask, labels; loss 由 loss_function(logits,labels,vocab)。
    """
    os.makedirs(out_dir, exist_ok=True)
    from qwen_asr import Qwen3ASRModel  # 触发 AutoModel.register
    from qwen_asr.core.transformers_backend import Qwen3ASRForConditionalGeneration, Qwen3ASRProcessor
    from transformers import AutoProcessor
    from peft import LoraConfig, get_peft_model, TaskType

    print(f"[train] 加载模型 bf16: {model_path}")
    # ⚠️ Qwen3ASRForConditionalGeneration 顶层只 override generate() 没有 forward();
    # 真正的 forward 在 thinker (Qwen3ASRThinkerForConditionalGeneration) 里。
    # 所以 PEFT 必须包 thinker, 否则 forward 走 nn.Module._forward_unimplemented。
    base = Qwen3ASRForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cuda:0",
    )
    thinker = base.thinker  # Qwen3ASRThinkerForConditionalGeneration (有 forward + get_input_embeddings)
    thinker.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    processor = AutoProcessor.from_pretrained(model_path, fix_mistral_regex=True)
    tokenizer = processor.tokenizer

    # 确认 pad token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA: target_modules = text model attention 的 q_proj/v_proj
    # (Qwen3ASRThinkerTextModel.layers.*.self_attn.{q_proj,k_proj,v_proj,o_proj})
    target_modules = ["q_proj", "v_proj"]
    lora_cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05, bias="none",
        task_type=TaskType.CAUSAL_LM, target_modules=target_modules,
    )
    model = get_peft_model(thinker, lora_cfg)
    model.print_trainable_parameters()

    # 数据集 + collator
    ds = _Dataset(manifest_path, processor, tokenizer, language="Chinese")
    print(f"[train] 数据集 {len(ds)} 条")
    collator = _Collator(tokenizer, pad_token_id=tokenizer.pad_token_id)

    # 测试一条数据 (诊断)
    sample = ds[0]
    print(f"[train] sample keys: {list(sample.keys())}, "
          f"input_ids shape {sample['input_ids'].shape}, "
          f"input_features shape {sample['input_features'].shape}, "
          f"labels valid {(sample['labels']!=-100).sum().item()}")

    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=True, collate_fn=collator, num_workers=0,
    )

    # 8bit Adam (显存省 4x vs fp32 Adam)
    import bitsandbytes as bnb
    optimizer = bnb.optim.PagedAdamW8bit(
        model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01,
    )

    model.train()
    device = next(model.parameters()).device
    step = 0
    accum = 0
    losses = []
    t0 = time.time()
    stop = False
    for epoch in range(epochs):
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            # bf16 前向
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(**batch)
                loss = out.loss / grad_accum
            loss.backward()
            accum += 1
            losses.append(out.loss.item())
            if accum >= grad_accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accum = 0
                step += 1
                if step % log_every == 0 or step == 1:
                    recent = losses[-log_every*grad_accum:] if len(losses) >= log_every*grad_accum else losses
                    avg = sum(recent) / len(recent)
                    elapsed = time.time() - t0
                    print(f"  [step {step}] loss={avg:.4f} ({elapsed:.0f}s, {step/(elapsed+1e-6):.2f} step/s)")
            if max_steps and step >= max_steps:
                stop = True
                break
        if stop:
            break

    # flush remaining grads
    if accum > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

    # 保存 adapter
    adapter_path = os.path.join(out_dir, "adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"[train] LoRA adapter 保存: {adapter_path}")
    log = {
        "n_pairs": len(ds), "epochs": epochs, "lr": lr, "lora_r": lora_r,
        "lora_alpha": lora_alpha, "grad_accum": grad_accum, "steps": step,
        "target_modules": target_modules, "elapsed_sec": time.time() - t0,
        "loss_first": losses[0] if losses else None,
        "loss_last_avg10": sum(losses[-10:]) / max(1, len(losses[-10:])),
        "loss_min": min(losses) if losses else None,
        "loss_trajectory": losses[::max(1, len(losses)//50)][:50],  # 降采样
    }
    json.dump(log, open(TRAIN_LOG, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[train] train log → {TRAIN_LOG}")
    return adapter_path


# ============================================================================
# 步骤3: A 集 hold-out 评测 ΔCER
# ============================================================================
def eval_on_a_set(adapter_path: str, model_path: str, slices_dir: str,
                  baseline_json: str, limit: int = 0):
    """用 base+adapter 转写 1350 条 cmd_*.wav, 算官方口径 CER, 对比主线 qwen 0.3436。

    分桶: sim<0.4 死区 vs sim≥0.4 主战场, 看死区改善 vs 主战场退化。
    """
    from qwen_asr import Qwen3ASRModel
    from peft import PeftModel
    from qwen_asr.core.transformers_backend import Qwen3ASRForConditionalGeneration

    print(f"[eval] 加载 base+adapter...")
    base = Qwen3ASRForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cuda:0",
    )
    # adapter 是对 base.thinker 的 PEFT 包装
    base.thinker = PeftModel.from_pretrained(base.thinker, adapter_path)
    base.thinker.eval()
    model = base  # 用顶层 generate (会调 thinker.generate → thinker.forward via PEFT)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_path, fix_mistral_regex=True)

    # 加载 baseline 主线 qwen 结果 (1350 条 sim+ref+qwen)
    baseline = json.load(open(baseline_json, encoding="utf-8"))
    rows = baseline["rows"]
    uid2base = {r["uid"]: r for r in rows}
    sims = {r["uid"]: r["sim"] for r in rows}
    refs = {r["uid"]: r["ref"] for r in rows}

    slices = sorted(glob.glob(os.path.join(slices_dir, "*.wav")))
    if limit:
        slices = slices[:limit]
    print(f"[eval] {len(slices)} 切片, batch 推理...")

    uid2text = {}
    t0 = time.time()
    BS = 8  # 显存允许下尽量大
    n_batches = (len(slices) + BS - 1) // BS

    def _build_prompt(force_language="Chinese"):
        msgs = [
            {"role": "system", "content": ""},
            {"role": "user", "content": [{"type": "audio", "audio": ""}]},
        ]
        base = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return base + f"language {force_language}{'<asr_text>'}"

    prompt_tpl = _build_prompt("Chinese")

    for bi in range(0, len(slices), BS):
        batch_paths = slices[bi:bi+BS]
        batch_uids = [os.path.splitext(os.path.basename(p))[0] for p in batch_paths]
        # 加载音频
        wavs = []
        for p in batch_paths:
            w, _ = librosa.load(p, sr=SR)
            wavs.append(w)
        prompts = [prompt_tpl] * len(wavs)
        try:
            inputs = processor(text=prompts, audio=wavs, return_tensors="pt", padding=True)
            inputs = inputs.to(base.device).to(base.dtype)
            with torch.no_grad():
                gen_ids = model.generate(**inputs, max_new_tokens=256)
            decoded = processor.batch_decode(
                gen_ids.sequences[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True, clean_up_tokenization_spaces=False,
            )
            for uid, txt in zip(batch_uids, decoded):
                # 去掉可能残留的 "language Chinese" 前缀
                txt = txt.strip()
                if txt.startswith("language"):
                    txt = txt[len("language"):].strip()
                    # 也可能形如 "Chinese实际文本"
                    for ln in ["Chinese", "chinese", "ZH", "zh"]:
                        if txt.startswith(ln):
                            txt = txt[len(ln):].strip()
                            break
                uid2text[uid] = txt
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  batch {bi//BS+1}/{n_batches} OOM, fallback 逐条")
            for uid, w in zip(batch_uids, wavs):
                try:
                    inp = processor(text=[prompt_tpl], audio=[w], return_tensors="pt", padding=True)
                    inp = inp.to(base.device).to(base.dtype)
                    with torch.no_grad():
                        g = model.generate(**inp, max_new_tokens=256)
                    txt = processor.batch_decode(
                        g.sequences[:, inp["input_ids"].shape[1]:],
                        skip_special_tokens=True, clean_up_tokenization_spaces=False,
                    )[0].strip()
                    uid2text[uid] = txt
                except Exception as e:
                    print(f"    {uid} FAIL {type(e).__name__}: {str(e)[:60]}")
                    uid2text[uid] = ""
        except Exception as e:
            print(f"  batch {bi//BS+1}/{n_batches} FAIL {type(e).__name__}: {str(e)[:80]}")
            for uid in batch_uids:
                uid2text[uid] = ""
        done = min(bi+BS, len(slices))
        if (bi // BS + 1) % 10 == 0 or done == len(slices):
            print(f"  [{done}/{len(slices)}] batch {bi//BS+1}/{n_batches} "
                  f"({done/(time.time()-t0):.1f}/s)")

    # 算 CER 官方口径 (复刻 recompute_qwen_official)
    # 1) 主线 qwen baseline (从 baseline json 拿, 已是同集合 1350 条)
    base_hyps = [_submit_norm(uid2base[uid]["qwen"]) for uid in uid2text]
    # 2) 本次 LoRA 微调后
    lora_hyps = [_submit_norm(uid2text[uid]) for uid in uid2text]
    refs_norm = [_submit_norm(refs[uid]) for uid in uid2text]  # ref 也归一(数字/繁简)
    sim_arr = [sims[uid] for uid in uid2text]

    m_base = CERMetric(); m_base.update(base_hyps, refs_norm)
    base_cer = m_base.compute()["cer"]
    m_lora = CERMetric(); m_lora.update(lora_hyps, refs_norm)
    lora_cer = m_lora.compute()["cer"]
    delta = lora_cer - base_cer

    # 分桶
    def _bucket(lo, hi):
        idx = [i for i, s in enumerate(sim_arr) if lo <= s < hi]
        if not idx:
            return None
        mb = CERMetric(); mb.update([base_hyps[i] for i in idx], [refs_norm[i] for i in idx])
        ml = CERMetric(); ml.update([lora_hyps[i] for i in idx], [refs_norm[i] for i in idx])
        return {"n": len(idx),
                "base_cer": round(mb.compute()["cer"], 4),
                "lora_cer": round(ml.compute()["cer"], 4),
                "delta": round(ml.compute()["cer"] - mb.compute()["cer"], 4)}

    dead = _bucket(0.0, 0.4)   # 死区
    main = _bucket(0.4, 1.01)  # 主战场
    # 更细的死区分桶
    sub_dead = {f"[{lo},{hi})": _bucket(lo, hi) for lo, hi in
                [(0, 0.2), (0.2, 0.3), (0.3, 0.4)]}

    # 抽样对比 5 条 (诊断)
    samples = []
    for uid in list(uid2text.keys())[:5]:
        samples.append({
            "uid": uid, "sim": round(sims[uid], 3), "ref": refs[uid][:40],
            "base_qwen": uid2base[uid]["qwen"][:40],
            "lora": uid2text[uid][:40],
        })

    out = {
        "n": len(uid2text),
        "baseline_source": baseline_json,
        "baseline_qwen_overall_cer": round(base_cer, 4),
        "lora_overall_cer": round(lora_cer, 4),
        "delta_cer_lora_minus_baseline": round(delta, 4),
        "deadzone_sim_lt_0.4": dead,
        "mainfield_sim_ge_0.4": main,
        "deadzone_fine": sub_dead,
        "samples": samples,
        "elapsed_sec": time.time() - t0,
    }
    json.dump(out, open(EVAL_OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[eval] ===== RESULT (n={len(uid2text)}) =====")
    print(f"  baseline qwen overall CER: {base_cer:.4f}")
    print(f"  LoRA     overall CER      : {lora_cer:.4f}")
    print(f"  ΔCER (lora - baseline)    : {delta:+.4f}")
    if dead:
        print(f"  死区 sim<0.4 (n={dead['n']}): base {dead['base_cer']:.4f} → lora {dead['lora_cer']:.4f} (Δ {dead['delta']:+.4f})")
    if main:
        print(f"  主战场 sim≥0.4 (n={main['n']}): base {main['base_cer']:.4f} → lora {main['lora_cer']:.4f} (Δ {main['delta']:+.4f})")
    print(f"  [eval] → {EVAL_OUT}")
    return out


# ============================================================================
# main
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-manifest", default=r"E:/midea_target_asr/code/_aug_manifests_poc1k/target.jsonl")
    ap.add_argument("--interferer-manifest", default=r"E:/midea_target_asr/code/_aug_manifests_poc1k/interferer.jsonl")
    ap.add_argument("--noise-manifest", default=r"E:/midea_target_asr/code/_aug_manifests_poc1k/noise.jsonl")
    ap.add_argument("--n-pairs", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=0, help="0=不限, 跑完 epochs")
    ap.add_argument("--eval-limit", type=int, default=0, help="0=全部 1350")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-synth", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    os.makedirs(RUN_DIR, exist_ok=True)

    # 步骤1: 合成
    if not args.skip_synth:
        print("=" * 70)
        print("[STEP 1] 合成 1k 训练对")
        print("=" * 70)
        if os.path.exists(AUG_MANIFEST):
            n_existing = sum(1 for _ in open(AUG_MANIFEST, encoding="utf-8"))
            if n_existing >= args.n_pairs:
                print(f"[synth] 已存在 {n_existing} 对 ({AUG_MANIFEST}), skip. 删 directory 可重建.")
            else:
                print(f"[synth] 已存在 {n_existing} 对 < {args.n_pairs}, 重建...")
                interf_items = load_jsonl(args.interferer_manifest)
                noise_items = load_jsonl(args.noise_manifest)
                target_items = load_jsonl(args.target_manifest)
                interf_pool = preload_pool(interf_items, tag="interferer")
                noise_pool = preload_pool(noise_items, tag="noise")
                synthesize_batch_fast(target_items, interf_pool, noise_pool,
                                      AUG_DIR, args.n_pairs, seed=args.seed)
        else:
            interf_items = load_jsonl(args.interferer_manifest)
            noise_items = load_jsonl(args.noise_manifest)
            target_items = load_jsonl(args.target_manifest)
            interf_pool = preload_pool(interf_items, tag="interferer")
            noise_pool = preload_pool(noise_items, tag="noise")
            synthesize_batch_fast(target_items, interf_pool, noise_pool,
                                  AUG_DIR, args.n_pairs, seed=args.seed)

    # 步骤2: LoRA 微调
    adapter_path = os.path.join(ADAPTER_DIR, "adapter")
    if not args.skip_train:
        print("\n" + "=" * 70)
        print("[STEP 2] LoRA 微调 Qwen3-ASR-1.7B")
        print("=" * 70)
        train_lora(AUG_MANIFEST, QWEN_MODEL_PATH, ADAPTER_DIR,
                   epochs=args.epochs, lr=args.lr, lora_r=args.lora_r,
                   lora_alpha=args.lora_alpha, grad_accum=args.grad_accum,
                   max_steps=args.max_steps)

    # 步骤3: A 集评测
    if not args.skip_eval:
        print("\n" + "=" * 70)
        print("[STEP 3] A 集 hold-out 评测 ΔCER")
        print("=" * 70)
        eval_on_a_set(adapter_path, QWEN_MODEL_PATH, TARGET_SLICES_DIR,
                      QWEN_BASELINE_RESULT, limit=args.eval_limit)


if __name__ == "__main__":
    main()
