"""目标说话人 ASR demo web server (FastAPI)。

启动(在 code/ 下, 让 demo_web 作包 import):
  source code/setenv.sh
  cd code && .venv/Scripts/python.exe -m uvicorn demo_web.server:app --host 0.0.0.0 --port 7860
"""
import os, sys, uuid, asyncio, shutil, glob
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

_HERE = os.path.dirname(os.path.abspath(__file__))   # demo_web/
_CODE = os.path.dirname(_HERE)                        # code/
_ROOT = os.path.dirname(_CODE)                        # 项目根
for _p in (_HERE, _CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import audio_utils
from inference_engine import InferenceEngine

SESSIONS = os.path.join(_HERE, "sessions")
os.makedirs(SESSIONS, exist_ok=True)
STATIC = os.path.join(_HERE, "static")
os.makedirs(STATIC, exist_ok=True)

app = FastAPI(title="目标说话人 ASR Demo")
engine = InferenceEngine()


@app.on_event("startup")
async def _startup():
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg 不在 PATH。装到 E:\\Tools\\ffmpeg 并加入 PATH, 或确认现有 ffmpeg 在 PATH。")
    print("[demo-server] startup: 加载模型(约 40s)...")
    await asyncio.get_event_loop().run_in_executor(None, engine.load_models)
    print("[demo-server] ready, 监听 0.0.0.0:7860", flush=True)


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.post("/upload/enroll")
async def upload_enroll(file: UploadFile = File(...)):
    eid = "enroll_" + uuid.uuid4().hex[:12]
    dst = os.path.join(SESSIONS, eid + ".wav")
    tmp = dst + ".in"
    raw = await file.read()
    with open(tmp, "wb") as f:
        f.write(raw)
    try:
        audio_utils.to_wav_16k_mono(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    d = audio_utils.duration_s(dst)
    if d < 0.3:
        raise HTTPException(400, f"enrollment 过短 ({d:.2f}s < 0.3s)")
    return {"enroll_id": eid, "duration_s": round(d, 2), "audio_url": f"/audio/{eid}.wav"}


@app.post("/upload/test")
async def upload_test(file: UploadFile = File(...),
                      mix_mode: str = Form("none"),
                      snr_db: float = Form(0.0),
                      interferer_id: str = Form(""),
                      overlap_ratio: float = Form(0.8)):
    tid = "test_" + uuid.uuid4().hex[:12]
    clean = os.path.join(SESSIONS, tid + ".wav")
    mixed = os.path.join(SESSIONS, tid + "_mixed.wav")
    tmp = clean + ".in"
    raw = await file.read()
    with open(tmp, "wb") as f:
        f.write(raw)
    try:
        audio_utils.to_wav_16k_mono(tmp, clean)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    d = audio_utils.duration_s(clean)
    if d < 0.3:
        raise HTTPException(400, f"test 过短 ({d:.2f}s < 0.3s)")
    if mix_mode == "babble":
        pool = os.environ.get("DEMO_BABBLE_DIR", os.path.join(_ROOT, "datasetA", "neg"))
        audio_utils.mix_babble(clean, mixed, snr_db, babble_pool=pool)
    elif mix_mode == "voice":
        # interferer_id 来自 /interferers(= datasetA/neg 文件名 id), 不在 sessions/
        neg_dir = os.path.join(_ROOT, "datasetA", "neg")
        intf = os.path.join(neg_dir, interferer_id + ".wav") if interferer_id else ""
        if not (intf and os.path.exists(intf)):
            raise HTTPException(400, "voice 模式需有效 interferer_id(用 /interferers 取)")
        audio_utils.mix_voice(clean, intf, mixed, overlap_ratio)
    else:  # none
        shutil.copy(clean, mixed)
    return {"test_id": tid, "clean_url": f"/audio/{tid}.wav",
            "mixed_url": f"/audio/{tid}_mixed.wav",
            "duration_s": round(d, 2), "mix_mode": mix_mode}


@app.get("/interferers")
async def interferers():
    neg_dir = os.path.join(_ROOT, "datasetA", "neg")
    out = []
    for w in sorted(glob.glob(os.path.join(neg_dir, "*.wav")))[:50]:
        out.append({"id": os.path.splitext(os.path.basename(w))[0],
                    "name": os.path.basename(w),
                    "duration_s": round(audio_utils.duration_s(w), 2)})
    return out


@app.post("/infer")
async def do_infer(body: dict):
    enroll_id = body.get("enroll_id", "")
    test_id = body.get("test_id", "")
    enr = os.path.join(SESSIONS, enroll_id + ".wav")
    rec = os.path.join(SESSIONS, test_id + "_mixed.wav")
    if not os.path.exists(enr):
        raise HTTPException(404, f"enrollment 不存在: {enroll_id}")
    if not os.path.exists(rec):
        raise HTTPException(404, f"test 不存在: {test_id}")
    infer_id = "inf_" + uuid.uuid4().hex[:12]
    target_out = os.path.join(SESSIONS, infer_id + "_target.wav")
    try:
        result = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None, lambda: engine.infer(enr, rec, target_out_path=target_out)),
            timeout=180)
    except asyncio.TimeoutError:
        raise HTTPException(504, "推理超时 (180s, 可能排队或模型慢)")
    result["target_audio_url"] = f"/audio/{infer_id}_target.wav" if result.get("target_audio_path") else None
    return result


@app.get("/audio/{name}")
async def get_audio(name: str):
    name = os.path.basename(name)  # 防路径穿越
    p = os.path.join(SESSIONS, name)
    if not os.path.exists(p):
        raise HTTPException(404, "audio not found")
    return FileResponse(p, media_type="audio/wav")
