"""生成中文 TTS 数据集(小批量验证版) — XH-202615。

趁 MiMo-V2.5-TTS 限时免费,合成 target(冰糖女声) 的 enrollment(长/短) + 指令,
以及 nontarget(苏打男声) 的干扰语音,供后续 build_dataset.py 组装重叠/加噪矩阵,
并供 Part1 enrollment→wespeaker 匹配验证。

数据集结构(本脚本只产"单人 raw wav" + manifest,组装在 build_dataset.py):
  test_wav/dataset/raw/
    enrollment/  target_long_01.wav  (冰糖, ~10-15s, 自述+多指令, 常规 enrollment)
                 target_short_01.wav (冰糖, ~1-2s, 唤醒词, 超短 enrollment 差异化)
                 target_short_02.wav (冰糖, ~1s,   纯唤醒词)
    target/      t_01..t_10.wav      (冰糖, 10 条家居指令, 识别音频里的目标语音)
    nontarget/   n_01..n_08.wav      (苏打, 8 条闲聊, 重叠干扰素材)
    manifest.json                     (每条: name/text/voice/role/ref, 供 eval+组装)

WSL 运行(同 mimo_tts_zh.py):
  wsl.exe -d Ubuntu-22.04 bash -lc 'python3 /mnt/e/midea_target_asr/code/tts_dataset_gen.py'
依赖: openai(WSL 已装); key 在 ~/.hermes/.env 的 XIAOMI_API_KEY/XIAOMI_BASE_URL。
"""
import base64, os, time, json
from openai import OpenAI


def _load_env(path):
    """极简 dotenv 解析(同 mimo_tts_zh.py)。"""
    cfg = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
                v = v[1:-1]
            cfg[k.strip()] = v
    return cfg


env = _load_env(os.path.expanduser("~/.hermes/.env"))
api_key = env.get("XIAOMI_API_KEY")
base_url = env.get("XIAOMI_BASE_URL")
if not api_key or not base_url:
    raise SystemExit("[fatal] ~/.hermes/.env 缺 XIAOMI_API_KEY/XIAOMI_BASE_URL")

client = OpenAI(api_key=api_key, base_url=base_url)
ROOT_OUT = "/mnt/e/midea_target_asr/test_wav/dataset/raw"
for sub in ("enrollment", "target", "nontarget"):
    os.makedirs(os.path.join(ROOT_OUT, sub), exist_ok=True)

TARGET_VOICE = "冰糖"        # 目标说话人(女)
NONTARGET_VOICE = "苏打"     # 干扰说话人(男)
STYLE_CMD = "平静自然的语气，语速适中"          # 指令类
STYLE_CHAT = "随意闲聊的语气"                    # 干扰闲聊类

# --- 语料清单: (name, text) ---
ENROLL_LONG = [
    ("target_long_01",
     "你好，我是这个家的主人。请把客厅的空调温度调到二十六度，"
     "再把卧室的灯打开，最后帮我定一个明天早上七点的闹钟。"),
]
ENROLL_SHORT = [
    ("target_short_01", "小美小美，我在这里。"),
    ("target_short_02", "小美小美。"),
]
TARGET_CMDS = [
    ("t_01", "请把客厅的空调温度调到二十六度"),
    ("t_02", "小美小美，打开卧室的灯"),
    ("t_03", "把电视的声音关小一点"),
    ("t_04", "帮我定一个明天早上七点的闹钟"),
    ("t_05", "打开扫地机器人开始清扫客厅"),
    ("t_06", "把窗帘拉上一半"),
    ("t_07", "启动空气净化器调成睡眠模式"),
    ("t_08", "播放一首轻音乐"),
    ("t_09", "打开热水器把水温调到四十五度"),
    ("t_10", "把客厅的灯调成暖色最低亮度"),
]
NONTARGET_CMDS = [
    ("n_01", "今天天气真不错，我们出去走走吧"),
    ("n_02", "这个电影我觉得挺好看的，推荐你也看看"),
    ("n_03", "你晚上想吃什么，我来做"),
    ("n_04", "明天的会议几点开始，我差点忘了"),
    ("n_05", "我刚看了一个很有趣的新闻"),
    ("n_06", "这个周末有什么安排吗"),
    ("n_07", "帮我倒杯水好吗，谢谢"),
    ("n_08", "那本书我看完了，写得真不错"),
]


def _synth_one(text, voice, style, retries=4):
    """单次合成,指数退避重试(429/瞬时错误)。返回 wav bytes。"""
    for attempt in range(retries):
        try:
            comp = client.chat.completions.create(
                model="mimo-v2.5-tts",
                messages=[
                    {"role": "user", "content": style},
                    {"role": "assistant", "content": text},  # 文本必须在 assistant(pitfall 1)
                ],
                audio={"format": "wav", "voice": voice},
            )
            if not comp.choices or not comp.choices[0].message.audio:
                raise RuntimeError("API 未返回音频(可能内容策略拦截/降级纯文本)")
            return base64.b64decode(comp.choices[0].message.audio.data)
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"  [retry {attempt + 1}/{retries}] {type(e).__name__}: {str(e)[:80]} ... 等{wait}s")
                time.sleep(wait)
            else:
                raise


def synth_and_save(name, text, voice, style, role, subdir, ref=None):
    """合成一条并落盘,返回 manifest 条目。ref=参考文本(target 指令用于 CER 评测)。"""
    out_path = os.path.join(ROOT_OUT, subdir, f"{name}.wav")
    try:
        wav = _synth_one(text, voice, style)
        with open(out_path, "wb") as f:
            f.write(wav)
        print(f"[ok] {subdir}/{name}.wav  {len(wav) // 1024}KB  voice={voice}  | {text}")
        return {"name": name, "path": out_path, "text": text, "voice": voice,
                "role": role, "ref": ref if ref is not None else text}
    except Exception as e:
        print(f"[FAIL] {subdir}/{name}  {type(e).__name__}: {str(e)[:100]}")
        return None


manifest = []
ok = fail = 0

print("=== 长 enrollment (target 冰糖) ===")
for name, text in ENROLL_LONG:
    m = synth_and_save(name, text, TARGET_VOICE, STYLE_CMD, "enrollment_long", "enrollment", ref=text)
    if m: manifest.append(m); ok += 1
    else: fail += 1

print("=== 短 enrollment (target 冰糖, 唤醒词) ===")
for name, text in ENROLL_SHORT:
    m = synth_and_save(name, text, TARGET_VOICE, "清脆响亮的语气", "enrollment_short", "enrollment", ref=text)
    if m: manifest.append(m); ok += 1
    else: fail += 1

print("=== target 指令 (冰糖) ===")
for name, text in TARGET_CMDS:
    m = synth_and_save(name, text, TARGET_VOICE, STYLE_CMD, "target_cmd", "target", ref=text)
    if m: manifest.append(m); ok += 1
    else: fail += 1

print("=== nontarget 干扰 (苏打) ===")
for name, text in NONTARGET_CMDS:
    m = synth_and_save(name, text, NONTARGET_VOICE, STYLE_CHAT, "nontarget_cmd", "nontarget", ref=text)
    if m: manifest.append(m); ok += 1
    else: fail += 1

manifest_path = os.path.join(ROOT_OUT, "manifest.json")
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump({"voices": {"target": TARGET_VOICE, "nontarget": NONTARGET_VOICE},
               "count": len(manifest), "items": manifest}, f, ensure_ascii=False, indent=2)

print(f"\ndone -> {ROOT_OUT}  ({ok} ok / {fail} fail)")
print(f"manifest -> {manifest_path}  ({len(manifest)} 条)")
print("[下一步] 跑 build_dataset.py 把 raw 组装成 重叠×SNR×噪声 矩阵")
