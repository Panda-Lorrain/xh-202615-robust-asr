"""中文重叠场景完整 pipeline 诊断验证（XH-202615 答辩素材）。

题目核心场景: 带噪(SNR −5~5dB) + ≤2人重叠(0~100%), 只转 target 指令、拒识 non-target。
本脚本分两步(pipeline 由 DiCoW-inference/inference.py 单独跑):
  step1 build : 造 3 档重叠率诊断集(0%顺序 / 50%部分 / 100%完全), 无加噪聚焦分离能力
  step2 eval  : 读 pipeline 输出, 解析分说话人转写, 算 target CER + 分离情况

用法:
  python overlap_experiment.py build
  # 跑 pipeline(见 PROGRESS.md「完整 pipeline 已跑通」):
  #   --input-folder test_wav/overlap --output-folder code/pipeline_overlap_out
  python overlap_experiment.py eval

⚠️ 答辩两条路线(本脚本 eval 验证 diarization 路线):
  - Diarization 路线: pipeline 分离所有说话人各自转写, CER 最低者识别为 target;
    non-target 也会作为独立 speaker 被转出。验证"重叠分离 + target 转写精度"。
  - 拒识路线(只输出 target): 由 STNO non-target mask 实现(non-target→0字), 见 RESULTS.md。
预期结论: 0%重叠→分离成功; 100%重叠+短音频→分离失败/幻觉(单通道死区, 论证多通道/enrollment必要性)。
"""
import os, sys, json, re
import numpy as np
import librosa
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from simulate_pipeline import mix_overlap, add_noise
from eval_metrics import cer

_ROOT = os.path.dirname(_HERE)
TEST_WAV = os.path.join(_ROOT, "test_wav")
OVERLAP_DIR = os.path.join(TEST_WAV, "overlap")
PIPE_OUT = os.path.join(_ROOT, "code", "pipeline_overlap_out")
EVAL_OUT = os.path.join(_ROOT, "code", "overlap_eval_result.json")

# ground truth(mimo-tts 合成输入文本, 见 mimo_tts_zh.py JOBS)
TARGET_TEXT = {
    "zh_target_01": "请把客厅的空调温度调到二十六度",
    "zh_target_02": "小美小美，打开卧室的灯",
    "zh_target_03": "把电视的声音关小一点",
    "zh_target_04": "帮我定一个明天早上七点的闹钟",
}
NONTARGET_TEXT = {
    "zh_nontarget_01": "今天天气真不错，我们出去走走吧",
    "zh_nontarget_02": "这个电影我觉得挺好看的",
}


def _gen_white_noise(n, sr=16000):
    return (np.random.randn(n) * 0.1).astype(np.float32)


def _mix_sequential(target, interferer, gap_s=0.3, sr=16000):
    """顺序拼接: target + 静音间隔 + interferer(0% 重叠, diarization 易分离)。"""
    gap = np.zeros(int(gap_s * sr), dtype=np.float32)
    return np.concatenate([target, gap, interferer]).astype(np.float32)


def build():
    """造 3 档重叠率诊断集(0%顺序 / 50%部分 / 100%完全), 无加噪聚焦 diarization 分离。"""
    np.random.seed(42)
    os.makedirs(OVERLAP_DIR, exist_ok=True)
    sr = 16000
    # (target, nontarget, mix_type, param, tag)
    #   seq=顺序0%重叠(可分离基线), partial=前段50%重叠, full=100%完全重叠(题面极端)
    configs = [
        ("zh_target_01", "zh_nontarget_01", "seq",     None, "seq_clean"),
        ("zh_target_01", "zh_nontarget_01", "partial", 0.5,  "partial50_clean"),
        ("zh_target_01", "zh_nontarget_01", "full",    1.0,  "full_clean"),
    ]
    manifest = []
    for tgt, non, mtype, param, tag in configs:
        t, _ = librosa.load(os.path.join(TEST_WAV, tgt + ".wav"), sr=sr)
        n, _ = librosa.load(os.path.join(TEST_WAV, non + ".wav"), sr=sr)
        if mtype == "seq":
            mixed = _mix_sequential(t, n, 0.3, sr)
            ov_desc = "0%(顺序,可分离)"
        elif mtype == "partial":
            mixed = mix_overlap(t, n, param)
            ov_desc = "50%(前段重叠)"
        else:  # full
            mixed = mix_overlap(t, n, param)
            ov_desc = "100%(完全重叠)"
        out = os.path.join(OVERLAP_DIR, f"{tag}.wav")
        sf.write(out, mixed, sr)
        manifest.append({
            "tag": tag, "wav": out, "dur_s": round(len(mixed) / sr, 2),
            "mix_type": mtype, "overlap_desc": ov_desc,
            "target_src": tgt, "target_ref": TARGET_TEXT[tgt],
            "nontarget_src": non, "nontarget_ref": NONTARGET_TEXT[non],
        })
        print(f"[build] {tag}.wav  dur={len(mixed)/sr:.1f}s  {ov_desc}")
    with open(os.path.join(OVERLAP_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[build] {len(manifest)} 条诊断音频 + manifest → {OVERLAP_DIR}")


_SPEAKER_RE = re.compile(r"🗣️\s*Speaker\s*(\d+)\s*:\s*(.*?)(?=🗣️\s*Speaker\s*\d+\s*:|$)", re.S)


def _parse_pipeline_txt(path):
    """解析 inference.py 输出 → {speaker_idx: 合并文本(去时间戳/去空白)}。"""
    with open(path, encoding="utf-8") as f:
        content = f.read()
    speakers = {}
    for m in _SPEAKER_RE.finditer(content):
        body = m.group(2)
        text = re.sub(r"<\|[^|]*\|>", "", body)      # 去 <|0.00|> 时间戳
        text = re.sub(r"\s+", "", text)              # 去空白
        speakers[int(m.group(1))] = text
    return speakers


def eval():
    """读 pipeline 输出, 每条: CER 最低者=target, 算 target CER + non-target 分离情况。"""
    manifest_path = os.path.join(OVERLAP_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        raise SystemExit(f"[fatal] 先跑 build: 缺 {manifest_path}")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    results = []
    for item in manifest:
        tag = item["tag"]
        txt_path = os.path.join(PIPE_OUT, tag + ".txt")
        if not os.path.exists(txt_path):
            print(f"[eval][skip] {tag}: pipeline 输出不存在(可能 crash) → {txt_path}")
            results.append({"tag": tag, "overlap_desc": item["overlap_desc"],
                            "status": "pipeline_crash_or_missing", "target_ref": item["target_ref"]})
            continue
        speakers = _parse_pipeline_txt(txt_path)
        if not speakers:
            print(f"[eval][warn] {tag}: 未解析出 speaker 文本")
            continue
        target_ref = item["target_ref"]
        spk_cer = {idx: cer(txt, target_ref) for idx, txt in speakers.items()}
        target_spk = min(spk_cer, key=spk_cer.get)
        other_spks = [i for i in speakers if i != target_spk]
        nontarget_transcribed = any(len(speakers[i]) > 0 for i in other_spks)
        results.append({
            "tag": tag, "overlap_desc": item["overlap_desc"],
            "n_speakers": len(speakers),
            "target_speaker_idx": target_spk,
            "target_CER": round(spk_cer[target_spk], 4),
            "target_transcript": speakers[target_spk],
            "target_ref": target_ref,
            "nontarget_speakers": {str(i): speakers[i] for i in other_spks},
            "nontarget_ref": item["nontarget_ref"],
            "nontarget_transcribed": nontarget_transcribed,
            "status": "ok",
        })
        print(f"[eval] {tag} [{item['overlap_desc']}]: {len(speakers)}说话人 | "
              f"target=Speaker{target_spk} CER={spk_cer[target_spk]:.3f} | "
              f"non-target分离转写={'是' if nontarget_transcribed else '否'}")

    with open(EVAL_OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    ok = [r for r in results if r.get("status") == "ok"]
    if ok:
        avg = sum(r["target_CER"] for r in ok) / len(ok)
        print(f"\n[eval] 平均 target CER(成功条) = {avg:.4f}  ({len(ok)}/{len(results)} 条成功)")
    print(f"[eval] 详细 JSON → {EVAL_OUT}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        build()
    elif cmd == "eval":
        eval()
    else:
        print("usage: python overlap_experiment.py [build|eval]")
