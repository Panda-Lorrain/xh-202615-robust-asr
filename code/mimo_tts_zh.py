"""用小米 MiMo-V2.5-TTS 合成中文智能家居指令测试音频(替代 edge-tts SSL 受阻)。
用途: 验证 DiCoW 中文转写能力(题目是中文),并为后续重叠/拒识实验准备素材。
在 WSL 运行:  wsl.exe -d Ubuntu-22.04 bash -lc 'python3 /mnt/e/midea_target_asr/code/mimo_tts_zh.py'
依赖: openai(WSL 已装 2.36.0); key 在 ~/.hermes/.env 的 XIAOMI_API_KEY/XIAOMI_BASE_URL。
输出: /mnt/e/midea_target_asr/test_wav/zh_*.wav (Windows: E:/midea_target_asr/test_wav/)
"""
import base64, os, time
from openai import OpenAI


def _load_env(path):
    """极简 dotenv 解析:跳过注释/空行,去 surrounding 引号。返回 dict。"""
    cfg = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):  # 去 "..." / '...'
                v = v[1:-1]
            cfg[k.strip()] = v
    return cfg


# --- 读 ~/.hermes/.env(不依赖 shell 环境变量,见 mimo-tts pitfall 8/13) ---
env = _load_env(os.path.expanduser("~/.hermes/.env"))
api_key = env.get("XIAOMI_API_KEY")
base_url = env.get("XIAOMI_BASE_URL")
if not api_key or not base_url:  # 不用 assert(python -O 下会失效)
    raise SystemExit(
        f"[fatal] ~/.hermes/.env 缺 XIAOMI_API_KEY/XIAOMI_BASE_URL "
        f"(key={'有' if api_key else '无'} base_url={'有' if base_url else '无'})"
    )

client = OpenAI(api_key=api_key, base_url=base_url)
OUT = "/mnt/e/midea_target_asr/test_wav"
os.makedirs(OUT, exist_ok=True)

# --- 合成清单: (文件名, 文本, 音色, user风格) ---
# target = 目标说话人(冰糖女声), non-target = 干扰人(苏打男声), 为重叠/拒识留素材
JOBS = [
    ("zh_target_01", "请把客厅的空调温度调到二十六度", "冰糖", "平静自然的语气，语速适中"),
    ("zh_target_02", "小美小美，打开卧室的灯", "冰糖", "平静自然的语气，语速适中"),
    ("zh_target_03", "把电视的声音关小一点", "冰糖", "平静自然的语气，语速适中"),
    ("zh_target_04", "帮我定一个明天早上七点的闹钟", "冰糖", "平静自然的语气，语速适中"),
    ("zh_nontarget_01", "今天天气真不错，我们出去走走吧", "苏打", "随意闲聊的语气"),
    ("zh_nontarget_02", "这个电影我觉得挺好看的", "苏打", "随意闲聊的语气"),
]


def _synth_one(text, voice, style, retries=3):
    """单次合成,带 429/瞬时错误指数退避重试。返回 wav bytes。"""
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
                wait = 2 ** attempt  # 1s/2s/4s 退避
                print(f"  [retry {attempt + 1}/{retries}] {type(e).__name__}: {str(e)[:80]} ... 等{wait}s")
                time.sleep(wait)
            else:
                raise


ok, fail = 0, 0
for name, text, voice, style in JOBS:
    try:
        wav = _synth_one(text, voice, style)
        with open(f"{OUT}/{name}.wav", "wb") as f:
            f.write(wav)
        print(f"[ok] {name}.wav  {len(wav) // 1024}KB  voice={voice}  | {text}")
        ok += 1
    except Exception as e:
        # 只报异常类型+截断消息,避免 SDK repr 泄露 base_url/header
        print(f"[FAIL] {name}  {type(e).__name__}: {str(e)[:100]}")
        fail += 1

print(f"\ndone -> {OUT}  ({ok} ok / {fail} fail)")
