# 目标说话人 ASR Demo 网站

访客浏览器录音 → 本机常驻 vanilla ASR 引擎推理 → 回显转写/匹配分/拒识/切出的目标音。
支持在线混 babble 噪声 / 第二人声, 现场演示「只转目标、抗干扰」。

## 1. 依赖(已就绪, 列此供复核)

- `code/.venv`: fastapi 0.138 + uvicorn 0.49 + python-multipart(已装)
- ffmpeg 在 PATH(`C:/ProgramData/ffmpeg-7.1.1-essentials_build/bin`)
- 模型缓存: `E:/hf_cache` 下 whisper-large-v3-turbo / diarizen-wavlm-large-s80-md(setenv 指定)

## 2. 本机启动

```bash
cd E:/midea_target_asr
bash code/demo_web/run_demo.sh
```
日志打印 `[demo-server] ready` 后, 浏览器开 `http://127.0.0.1:7860/`。

## 3. 集成自检(怀疑引擎异常时跑)

```bash
cd E:/midea_target_asr
source code/setenv.sh
cd code
.venv/Scripts/python.exe -m demo_web.selfcheck
```
应打印 `SELFCHECK OK transcript=... max_sim=...`。
(用 datasetA/pos kws_0+cmd_0; thr=0 强制不拒验证 generate 产出)

## 4. 公网分享(cloudflared quick tunnel)

server 跑起来后, 另开终端:
```bash
cloudflared tunnel --url http://localhost:7860
```
(若没装 cloudflared: `winget install --id Cloudflare.cloudflared`)
输出里取 `https://<随机>.trycloudflare.com`, 发给访客。访客手机/电脑浏览器打开即可录音推理。

⚠️ 无鉴权, 用完 Ctrl+C 关 cloudflared 和 server。session 音频在 `sessions/`, 清理: `rm code/demo_web/sessions/*.wav`。

## 5. 预期延迟

- 服务启动加载模型 ~40s(一次性)
- 之后每条推理 5~15s(4060, batch=1) + 公网上传延迟, 端到端 ~10~20s
- 单 GPU 同时只跑一条, 多人访问排队(engine 内 threading.Lock)

## 6. 演示脚本建议(答辩现场)

1. **干净 test** → 推理 → 看正常转写 + 高匹配分。
2. **同一段 test 混 babble SNR=0** → 推理 → 对比转写变化 + 点「切出的目标音」听模型抽出的目标。
3. **混第二人声 80% 重叠** → 推理 → 听 target 只剩目标说话人的话。
4. **换一个人录 test**(与 enrollment 不同人) → 推理 → 触发 🚫 拒识徽章(目标不在场)。

## 7. 架构(简)

```
浏览器(MediaRecorder 录 webm) → FastAPI(7860) → InferenceEngine(常驻 diar+wespeaker+vanilla Whisper)
                                         ↕ audio_utils(ffmpeg 转码 + add_noise/mix_overlap 混音)
                                   sessions/(enroll/test/mixed/target wav, 运行时产物)
```
`inference_engine.py` 复刻 `enroll_infer.py` vanilla 单条流程(零修改参赛链路, import 复用)。
