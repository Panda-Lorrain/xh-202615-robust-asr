"""MiMo-V2.5-ASR 云端 API 实测：作为 cascaded 后端转写器，对比 vanilla Whisper CER

【定位】(2026-07-07 源码核实)：MiMo-V2.5-ASR 不支持 target speaker enrollment
  —— 官方 asr_sft(audio, audio_tag="") 只有 2 参数，HF demo 也无 enrollment 输入框。
  故 MiMo 是【通用/多说话人 ASR】，不是 TS-ASR，只能当 cascaded 后端转写器候选
  (替换当前 vanilla Whisper-large-v3-turbo, CER 0.664)，不能单独做美的题目(缺身份层)。

【对照设计】复用 enroll_infer 的 diar+声纹选 target (与 exp_vanilla_vs_dicow.py 完全一致)，
  唯一变量 = 转写后端：vanilla Whisper vs MiMo-V2.5-ASR 云端 API。公平对照。
  vanilla 基准从 exp_vanilla_full.json 按 uid 取 (已全量算好，无需重跑)。

【合规】仅研究验证，不进 submit_infer：云端 API 不满足 L20 效率分(本地 RTF) +
  数据安全(测试集上传) + 可复现性(黑盒/版本可变) 三红线。

用法：
  # key 二选一：写 code/.mimo_apikey (已 gitignore) 或设环境变量 MIMO_API_KEY
  source code/setenv.sh
  code/.venv/Scripts/python.exe code/exp_mimo_asr.py \\
    --pairs code/pos_pairs_datasetA.json --limit 50 \\
    --vanilla-result code/exp_vanilla_full.json
"""
import os, sys, json, time, argparse, base64, tempfile, re
import numpy as np
import librosa
import soundfile as sf
import requests
import torch
import pyarrow  # 预热：避免后续 import pyannote 时扫描 sys.path 的 DiariZen 目录触发 WinError 6714

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

# 复用 enroll_infer 的工具函数（diar_mask / collect_clean_audio）
from enroll_infer import get_diarization_mask, collect_clean_audio

# diar sys.path（DiariZen + pyannote）
# 注：diarizen 包当前位于 DiCoW-inference.bak 备份目录（活路径 DiCoW-inference/DiariZen/ 为空，
#     疑似近期重组/备份所致），故同时注入活路径与 .bak 两个 base，sys.path 命中即可。
DICOW_INF = os.path.join(_ROOT, "code", "DiCoW-inference")
_DIARIZEN_BASES = [
    os.path.join(DICOW_INF, "DiariZen"),                                  # 活路径（预期位置）
    os.path.join(DICOW_INF, "DiCoW-inference.bak", "DiariZen"),           # .bak 备份（当前实际位置）
]
for _base in _DIARIZEN_BASES:
    for _p in (_base, os.path.join(_base, "pyannote-audio")):
        if os.path.isdir(_p):
            sys.path.insert(0, _p)

DIAR_MODEL = "E:/hf_cache/diarizen-wavlm-large-s80-md"
MIMO_API_URL = "https://token-plan-cn.xiaomimimo.com/v1/chat/completions"  # Token Plan 专用 base URL（tp- key）；按量 sk- 用 api.xiaomimimo.com
KEY_FILE = os.path.join(_HERE, ".mimo_apikey")


def load_key():
    k = os.environ.get("MIMO_API_KEY")
    if k:
        return k.strip()
    if os.path.exists(KEY_FILE):
        return open(KEY_FILE, encoding="utf-8").read().strip()
    raise SystemExit("[fatal] 未找到 MiMo API key：设环境变量 MIMO_API_KEY，或写入 code/.mimo_apikey（已 gitignore）")


# 增强归一化：繁→简 + 去中英文标点。
# 动机：eval_datasetA._norm_zh 只做繁简，不去标点；MiMo 输出原生标点(。，！)会被算进 CER，
# 而 vanilla Whisper 基本无标点 → 对 MiMo 不公平。去标点消除该污染。数字暂不归一化（主办方口径未定，
# 在汇总里分"含数字句/纯文字句"单独看，以剥离中文数字口径红利）。
_PUNCT = re.compile(r"[，。！？、；：“”‘’（）【】《》—…·,.!?;:\"'()\[\]<>]")
def norm_cer_text(t):
    from eval_datasetA import _norm_zh
    t = _norm_zh(t or "")
    return _PUNCT.sub("", t)


def cer_of(text, ref):
    from eval_metrics import cer
    t = norm_cer_text(text)
    r = norm_cer_text(ref)
    if not t:
        return 1.0
    return cer(t, r)


def mimo_transcribe(wav_np, sr, api_key, language="zh", timeout=60):
    """16k mono wav → MiMo 云端 API → 转写文本。返回 (text, 耗时s, err)"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    try:
        sf.write(tmp, wav_np.astype(np.float32), sr, subtype="PCM_16")
        with open(tmp, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    data_url = f"data:audio/wav;base64,{b64}"
    payload = {
        "model": "mimo-v2.5-asr",
        "messages": [{"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": data_url}}
        ]}],
        "asr_options": {"language": language},
    }
    headers = {"api-key": api_key, "Content-Type": "application/json"}
    t0 = time.time()
    try:
        r = requests.post(MIMO_API_URL, headers=headers, json=payload, timeout=timeout)
    except Exception as e:
        return "", time.time() - t0, f"req-err: {type(e).__name__}: {str(e)[:120]}"
    dt = time.time() - t0
    if r.status_code != 200:
        return "", dt, f"HTTP {r.status_code}: {r.text[:200]}"
    try:
        text = r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return "", dt, f"parse-err: {str(e)[:120]} resp={r.text[:200]}"
    return text, dt, None


def main():
    ap = argparse.ArgumentParser(description="MiMo-V2.5-ASR 云端 API 实测 vs vanilla Whisper")
    ap.add_argument("--pairs", required=True, help="pos_pairs_datasetA.json")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--vanilla-result", default=os.path.join(_HERE, "exp_vanilla_full.json"),
                    help="vanilla 基准（按 uid 取 vanilla_cer/vanilla_text）")
    ap.add_argument("--out-json", default=os.path.join(_HERE, "exp_mimo_asr_result.json"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--diar-model", default=DIAR_MODEL)
    ap.add_argument("--language", default="zh", help="MiMo asr_options.language: zh/en/auto")
    args = ap.parse_args()

    api_key = load_key()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"[load] DiariZen {args.diar_model}")
    from diarizen.pipelines.inference import DiariZenPipeline
    diar = DiariZenPipeline.from_pretrained(args.diar_model).to(device)

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

    # vanilla 基准按 uid 索引
    van = {}
    if os.path.exists(args.vanilla_result):
        data = json.load(open(args.vanilla_result, encoding="utf-8"))
        for r in data:
            van[r["uid"]] = r
        print(f"[load] vanilla 基准 {len(van)} 条 ← {args.vanilla_result}")

    pair_rows = json.load(open(args.pairs, encoding="utf-8"))[:args.limit]
    results = []
    for row in pair_rows:
        enr, rec = row["enrollment"], row["recognition"]
        uid = os.path.splitext(os.path.basename(rec))[0]
        ref = row.get("ref", "") or ""
        t0 = time.time()

        # ---- 前置：diar + 声纹选 target（与 vanilla 实验完全一致）----
        w, _ = librosa.load(enr, sr=16000)
        enroll_emb = get_emb(w)

        audio, sr = librosa.load(rec, sr=16000)
        dur = len(audio) / sr
        try:
            diar_out = diar(rec)
        except Exception as e:
            print(f"  [diar-fail] {uid}: {type(e).__name__} {str(e)[:80]}")
            results.append({"uid": uid, "ref": ref, "error": "diar: " + str(e)[:120]})
            continue
        speakers = list(diar_out.labels())
        per_spk = [diar_out.label_timeline(s) for s in speakers]

        audio_len = len(audio) // 320  # 50Hz 帧率
        diar_mask = get_diarization_mask(per_spk, audio_len)
        spk_embs = []
        for i in range(len(speakers)):
            seg = collect_clean_audio(audio, diar_mask, i, sr)
            if seg is None or len(seg) < sr * 0.3:
                segs = [audio[int(s * sr):int(e * sr)] for s, e in per_spk[i]]
                seg = np.concatenate(segs) if segs else np.zeros(sr, dtype=np.float32)
            min_len = sr
            if len(seg) < min_len:
                seg = np.tile(seg, min_len // len(seg) + 1)[:min_len]
            spk_embs.append(get_emb(seg))
        sims = torch.stack([torch.dot(enroll_emb, e) for e in spk_embs])
        target_idx = int(torch.argmax(sims))
        max_sim = float(sims[target_idx])

        # 切 target timeline（含重叠区）拼接 —— 与 vanilla 实验完全一致
        target_segs = sorted([(float(s), float(e)) for s, e in per_spk[target_idx]])
        if target_segs:
            target_audio = np.concatenate([audio[int(s * sr):int(e * sr)] for s, e in target_segs])
        else:
            target_audio = audio
        if len(target_audio) < sr * 0.3:
            target_audio = audio  # target 太短退化整条
        target_dur = len(target_audio) / sr

        # ---- 转写后端：MiMo 云端 API ----
        text, api_dt, err = mimo_transcribe(target_audio, sr, api_key, language=args.language)
        if err:
            print(f"  [mimo-fail] {uid}: {err}")
            results.append({"uid": uid, "ref": ref, "max_sim": max_sim,
                            "target_dur": target_dur, "error": err})
            continue

        m_cer = cer_of(text, ref)
        v_text = van.get(uid, {}).get("vanilla_text", "")
        v_cer = cer_of(v_text, ref) if v_text else None  # 同一增强归一化重算 vanilla，公平对比
        flag = ""
        if v_cer is not None:
            if m_cer < v_cer - 0.01:
                flag = "✓mimo更优"
            elif m_cer > v_cer + 0.01:
                flag = "✗vanilla更优"
            else:
                flag = "≈持平"
        chunk_warn = " ⚠>30s分块" if target_dur > 30 else ""
        total_dt = time.time() - t0
        print(f"[{uid}] sim={max_sim:.3f} ({dur:.1f}s→{target_dur:.1f}s{chunk_warn}, api={api_dt:.1f}s, tot={total_dt:.1f}s) {flag}")
        print(f"  mimo   ({m_cer:.2f}): {text[:55]}")
        if v_cer is not None:
            print(f"  vanilla({v_cer:.2f}): {v_text[:55]}")
        print(f"  ref        : {ref[:55]}")
        results.append({"uid": uid, "ref": ref, "max_sim": max_sim,
                        "mimo_text": text, "vanilla_text": v_text,
                        "mimo_cer": m_cer, "vanilla_cer": v_cer,
                        "target_dur": target_dur, "api_time": api_dt})

    # ---- 汇总对比 ----
    valid = [r for r in results if "mimo_cer" in r]
    print(f"\n===== MiMo vs vanilla CER 对比（{len(valid)} 条有效）=====")
    if valid:
        for label, key in [("mimo", "mimo_cer"), ("vanilla", "vanilla_cer")]:
            cs = [r[key] for r in valid if r.get(key) is not None]
            if not cs:
                continue
            overall = sum(cs) / len(cs)
            correct = sum(1 for c in cs if c < 0.5) / len(cs)
            near = sum(1 for c in cs if c < 0.1) / len(cs)
            print(f"  {label:8}: overall={overall:.4f}  correct(<0.5)={correct:.2%}  near_perfect(<0.1)={near:.2%}")
        both = [r for r in valid if r.get("vanilla_cer") is not None]
        better = sum(1 for r in both if r["mimo_cer"] < r["vanilla_cer"] - 0.01)
        worse = sum(1 for r in both if r["mimo_cer"] > r["vanilla_cer"] + 0.01)
        tie = len(both) - better - worse
        print(f"  逐条: mimo 更优 {better} / 更差 {worse} / 持平 {tie}（共 {len(both)}）")
        over30 = sum(1 for r in valid if r["target_dur"] > 30)
        print(f"  target 超 30s（MiMo 内部按 30s 分块编码）: {over30} 条")
        eng_hallu = sum(1 for r in valid if _has_english(r["mimo_text"]))
        print(f"  mimo 英文幻觉(含 a-z 词): {eng_hallu} 条")
        # 数字口径分离：ref 含数字(中文/阿拉伯)的句，vanilla 易因输出阿拉伯数字失分(MiMo 输出中文数字对齐 ref)。
        # 分组看能剥离"数字口径红利"，观察纯转写能力差异。
        DIGIT = re.compile(r"[0-9一二三四五六七八九十百千万两零]")
        for subset, pred in [("含数字句", lambda r: bool(DIGIT.search(r.get("ref", "")))),
                             ("纯文字句", lambda r: not bool(DIGIT.search(r.get("ref", ""))))]:
            sv = [r for r in both if pred(r)]
            if not sv:
                continue
            m_avg = sum(r["mimo_cer"] for r in sv) / len(sv)
            v_avg = sum(r["vanilla_cer"] for r in sv) / len(sv)
            print(f"  [{subset}] {len(sv)}条: mimo={m_avg:.4f} vs vanilla={v_avg:.4f}")

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[done] → {args.out_json}")


def _has_english(text):
    """粗判转写是否混入英文（家居指令本应纯中文）"""
    import re
    return bool(re.search(r"[A-Za-z]{2,}", text or ""))


if __name__ == "__main__":
    main()
