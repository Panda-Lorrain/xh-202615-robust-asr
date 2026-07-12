# 目标说话人 ASR Demo 网站 设计文档

- 日期：2026-07-12
- 关联：XH-202615 美的目标说话人 ASR 参赛方案
- 状态：设计已与用户对齐，待实现

## 1. 目标与范围

做一个**公网可访问**的演示网站，让访客在自己设备（手机/电脑）浏览器里录音，现场跑本机的目标说话人 ASR 引擎，直观看到「只转写目标说话人、拒识非目标、抗干扰」的效果。

**核心交互**：
1. 访客朗读 enrollment（声纹注册，~1.8s+）。
2. 访客朗读 test（测试指令）。
3. 可选：给 test 在线混入 babble 噪声或第二人声（演示抗干扰）。
4. 点「推理」→ 回显：转写文本 / 声纹匹配分(max_sim) / 拒识或接受徽章 / 各说话人 sim / 耗时 RTF / **切出的目标音回放**。

**在范围内**：
- FastAPI 后端 + 原生 HTML/JS 单页前端。
- 常驻推理引擎（复用 vanilla 路线：CER 0.664 主线）。
- 在线混音（`add_noise` / `mix_overlap`，复用 `simulate_pipeline`）。
- target timeline 切片回放（vanilla 的 `cut_target_timeline` 产物）。
- cloudflared quick tunnel 公网穿透。

**不在范围内**（YAGNI）：
- 多 GPU / 批量并发推理（单卡串行排队即可）。
- ASR 后端切换 UI（固定 vanilla；dicow/qwen/firered 不进 demo）。
- LLM 拒识 / content_gate / SE 前处理（demo 只展示核心 enroll→infer）。
- 用户账号 / 历史记录持久化（session 临时目录，可定期清理）。

## 2. 架构总览

```
访客浏览器(手机/电脑, MediaRecorder 录 webm/opus)
        │  HTTPS via cloudflared quick tunnel
        ▼
FastAPI  (本机 0.0.0.0:7860, 跑在 code/.venv)
  ├── /upload/enroll, /upload/test, /infer, /audio/{name}, /interferers, /
  ├── audio_utils.py   — ffmpeg webm→wav, add_noise/mix_overlap 包装
  └── InferenceEngine  — 启动时 load_models() 一次; 请求时 .infer() 5-15s
        │  import enroll_infer 的模块级工具函数(零修改 enroll_infer.py)
        │  DiariZen(diar+wespeaker) + Whisper-large-v3-turbo(vanilla)
        ▼
    单 GPU 串行(asyncio.Lock 排队)
```

**关键延迟预期**（给用户/评委预期）：服务启动加载模型 ~40s（一次性）；之后每条推理 5–15s（4060，batch=1）+ 公网上传音频延迟，端到端「点推理→出结果」约 10–20s。

## 3. 组件清单与职责

所有新增文件放 `code/demo_web/`。**不修改 `enroll_infer.py`**（参赛提交链路零风险）。

### 3.1 `inference_engine.py`（常驻推理引擎）
- 顶部**原样照搬** `enroll_infer.py` 的：
  - `inspect.getmodule` patch（speechbrain lazy 修复，2026-07-10 对抗审查固化点，不能丢）。
  - sys.path 注入（DiCoW-inference / DiariZen / pyannote-audio）。
- `import enroll_infer` 复用模块级工具函数：`get_diarization_mask`、`collect_clean_audio`。
- `from text_utils import cut_target_timeline, to_simplified, digit_postproc`。
- `from repro import resolve_model, set_global_seed, reset_peak_gpu, peak_gpu_mib`。
- 类 `InferenceEngine`：
  - `__init__(self, device="cuda:0", reject_threshold=0.5, vanilla_model=resolve_model("VANILLA"), diarization_model=resolve_model("DIAR"), seed=42)`：存配置，`self.lock = asyncio.Lock()`，不加载模型。
  - `load_models(self)`：加载 vanilla Whisper + tok + fe + DiariZen diar（照搬 enroll_infer main 的 137–159 行逻辑）。构造 `get_emb(wav_np)`（照搬 162–171，复用 `diar._embedding`）。打印加载耗时。
  - `infer(self, enroll_wav, rec_wav, save_target_dir) -> dict`：复刻 enroll_infer vanilla 单条流程（216–368 行）：
    1. enroll_emb（照搬 `get_enroll_emb`，无增强分支）。
    2. `librosa.load(rec)` → diar → speakers/per_spk → ifp/diar_mask → spk_embs → sims → target_idx/max_sim。
    3. `rejected = max_sim < reject_threshold`。
    4. 不拒：`cut_target_timeline` → 存 `save_target_dir/target.wav`（soundfile）→ fe → generate → batch_decode → `to_simplified` + `digit_postproc`。
    5. 返回 dict：`{transcript, max_sim, rejected, sims, target_idx, target_speaker, infer_sec, rtf, duration_s, target_audio_path}`。异常时返回 `{error, rejected:True, transcript:""}`（复刻 diar-fail 容错）。
- `asyncio.Lock` 由 server 层持有（引擎本身是同步阻塞推理，server 用 `run_in_executor` 包到线程池 + Lock 串行）。

### 3.2 `server.py`（FastAPI）
- 启动事件：`engine = InferenceEngine(); engine.load_models()`（挂全局，加载失败直接抛错退出，不挂空服务）。
- ffmpeg 启动检查：`shutil.which("ffmpeg")` 为空 → 报错并提示「装到 `E:\Tools\ffmpeg` 并入 PATH」（CLAUDE.md 规范）。
- 路由：
  - `GET /` → 返回 `static/index.html`。
  - `POST /upload/enroll`（multipart `file`）→ audio_utils 转码为 16k mono wav 存 session 目录 → `{enroll_id, duration_s, audio_url}`。
  - `POST /upload/test`（multipart `file` + form `mix_mode∈{none,babble,voice}`、`snr_db`、`interferer_id`、`overlap_ratio`）→ 转码 → 按 mix_mode 调 audio_utils.mix → 存 clean 与 mixed 两份 → `{test_id, clean_url, mixed_url, duration_s, mix_mode}`。
  - `GET /interferers` → 扫描预置第二人声素材目录（默认 `datasetA/neg/*.wav`）→ `[{id, name, duration_s}]`，供前端「第二人声」下拉。
  - `POST /infer`（json `{enroll_id, test_id}`）→ `async with engine.lock:` + `run_in_executor(engine.infer, ...)` → 返回 infer dict，并把 `target_audio_path` 转成可回放 `target_audio_url`。超时 120s → 504。
  - `GET /audio/{name}` → 返回 session 目录下 wav（enroll/test_clean/test_mixed/target）。
- session 管理：统一扁平 `sessions/` 目录，文件名约定：`{enroll_id}.wav`(enrollment)、`{test_id}.wav`(test 干净)、`{test_id}_mixed.wav`(test 混音；mix_mode=none 时 mixed_url=clean_url)、`{infer_id}_target.wav`(切出目标音，infer_id 每次唯一)。`GET /audio/{name}` 的 `name` 即上述文件名（含 `.wav`），按名直取。

### 3.3 `audio_utils.py`
- `to_wav_16k_mono(src_path, dst_path)`：调 `ffmpeg -i src -ar 16000 -ac 1 -sample_fmt s16 dst.wav`（subprocess）。webm/opus/wav/m4a 统一入口。
- `mix_babble(test_wav, snr_db, babble_pool) -> dst_wav`：`librosa.load` → 取 babble 片段（`_sample_babble` 风格，从 babble_pool 目录随机采，长度不足 tile）→ `add_noise(audio, noise, snr_db)`（from simulate_pipeline）→ soundfile 写出。
- `mix_voice(test_wav, interferer_wav, overlap_ratio) -> dst_wav`：`mix_overlap`（from simulate_pipeline）→ 写出。
- `duration_s(wav_path)`：librosa 取时长，用于前端展示 + <0.3s 拒收。
- babble_pool 目录可配（env `DEMO_BABBLE_DIR`），默认 `datasetA/neg`（用真实人声拼 babble，比白噪更贴题面）。

### 3.4 `static/index.html` + `app.js` + `style.css`
- 单页，三块布局：
  1. **enrollment 卡**：录音按钮（MediaRecorder）/ 停止 / 播放试听 / 上传状态。
  2. **test 卡**：录音 / 停止 / 播放 / **混音控件**（mix_mode 单选 none/babble/voice；babble 显示 SNR 滑块 −5~10dB；voice 显示第二人声下拉 + overlap 滑块 0~100%）/ 上传后展示「干净 test」与「混音 test」两个播放器对比。
  3. **结果区**：「开始推理」按钮 → loading（排队/推理中）→ 转写文本（大字）/ max_sim 仪表 / 拒识徽章（红）/ 接受徽章（绿）/ 各说话人 sim 列表 / infer_sec·RTF / **切出 target 播放器**。
- MediaRecorder 录音格式优先 `audio/webm`，fallback `audio/mp4`（Safari）。录完 Blob 直接 POST。
- 无外部 CDN（离线可用，公网演示稳）。样式手写 CSS，简洁现代即可。

### 3.5 `run_demo.sh` + `README.md`
- `run_demo.sh`：`cd 项目根` → `source code/setenv.sh` → `code/.venv/Scripts/python.exe -m uvicorn demo_web.server:app --host 0.0.0.0 --port 7860`（注：`demo_web` 需作为包可 import，用 `code/` 作 cwd 或加 sys.path）。
- `README.md`：本机启动步骤 + 装 fastapi/uvicorn/python-multipart 到 `code/.venv` 的命令 + cloudflared quick tunnel 命令 + 预期延迟 + 自检（跑 kws_0+cmd_0）。

## 4. API 契约

| 方法路径 | 请求 | 响应 |
|---|---|---|
| `POST /upload/enroll` | multipart `file`(webm/wav) | `{enroll_id, duration_s, audio_url}` |
| `POST /upload/test` | multipart `file` + form `mix_mode,snr_db,interferer_id,overlap_ratio` | `{test_id, clean_url, mixed_url, duration_s, mix_mode}` |
| `GET /interferers` | — | `[{id,name,duration_s}, ...]` |
| `POST /infer` | json `{enroll_id, test_id}` | `{transcript, max_sim, rejected, sims, target_idx, target_speaker, infer_sec, rtf, duration_s, target_audio_url}` 或 `{error}` |
| `GET /audio/{name}` | — | audio/wav |
| `GET /` | — | index.html |

`*_url` 形如 `/audio/enroll_{id}.wav`，由 GET /audio 回放。

## 5. 关键流程

### 5.1 启动
`run_demo.sh` → setenv（模型缓存指 E 盘 + 代理）→ uvicorn → startup 加载 vanilla Whisper + DiariZen（~40s，日志打印进度）→ 监听 7860。cloudflared quick tunnel 单独起（exposing-local-server skill），拿到 `*.trycloudflare.com` 公网链接。

### 5.2 录 enrollment / test
浏览器 MediaRecorder 录音 → Blob POST `/upload/{enroll|test}` → audio_utils `to_wav_16k_mono`（ffmpeg）→ 存 `sessions/{id}.wav` → 返回回放 url。duration<0.3s → 400「音频过短」。

### 5.3 混音（test 卡）
前端选 mix_mode 后随 test 一起上传（form 字段）。后端：
- `none`：mixed=clean。
- `babble`：`mix_babble(test, snr_db, babble_pool)`。
- `voice`：`mix_voice(test, interferer_wav, overlap_ratio)`。
存 `sessions/{test_id}_mixed.wav`，返回 clean_url + mixed_url。前端并排展示两个播放器——**亲耳听到加了多吵的噪声/重叠**，制造对比。

### 5.4 推理 + target 回放
POST `/infer` → 生成唯一 `infer_id` → engine.infer(enroll, mixed_test, save_target_dir=`sessions/`，target 文件名 `{infer_id}_target.wav`) → 返回结果 + target 切片 url（`/audio/{infer_id}_target.wav`）。前端展示：转写文本 / max_sim 仪表 / 徽章 / **target 播放器**（评委听到模型从重叠里抽出的纯目标语音）。

### 5.5 拒识演示
访客换一个人录 test（与 enrollment 不同人）→ diar 找不到匹配声纹 → max_sim 低 → `rejected=True` → 红色「拒识：目标不在场」徽章。

## 6. 错误处理与边界

- **模型加载失败**（OOM/缺权重）→ startup 抛错，进程退出，不挂空服务。
- **ffmpeg 缺失** → startup `shutil.which` 检查，提示装 `E:\Tools\ffmpeg`。
- **音频过短**（<0.3s）/ librosa 读失败 → 400 友好错误。
- **diar 失败**（pyannote 边界 bug）→ 复刻 enroll_infer 的 diar-fail 容错，返回 `{error, rejected:True}`。
- **推理排队超时**（>120s）→ 504，前端提示重试。
- **并发**：单 GPU 串行（`asyncio.Lock` + 线程池），第二个请求排队，前端显示「排队中」。
- **临时文件**：`sessions/` 按 id 命名，infer 后保留供回放；README 写清理命令（删 `sessions/*.wav`）。

## 7. 测试策略

### 7.1 单元（`audio_utils`）
- `to_wav_16k_mono`：用一段 webm（若无，用 wav 模拟）转出后 `wave` 读参数 = 16000/mono/int16。
- `mix_babble`：输出时长 == 输入时长；指定 snr 下实测 SNR 误差 <2dB。
- `mix_voice`：overlap_ratio=0 输出≈clean；=1.0 输出能量 > clean。

### 7.2 集成冒烟（自检脚本，demo 前跑）
服务启动后，`POST /infer` 用 `datasetA/pos/kws_0.wav`(enroll) + `datasetA/pos/cmd_0.wav`(test) 直接命中（绕过上传，构造 session 文件），断言：
- `transcript` 非空、`max_sim > 0`、`target_audio_url` 文件存在且时长>0。
- 这条同时验证 inference_engine 正确复刻了 enroll_infer vanilla 路径。

### 7.3 手动验收
浏览器实录两段（自己声音）+ 三种 mix_mode 各跑一次 + 一次换人拒识演示。

## 8. 文件清单（新增）

```
code/demo_web/
├── inference_engine.py    # InferenceEngine(load_models + infer), 复用 enroll_infer 工具
├── server.py              # FastAPI 路由 + 启动加载 + Lock 串行
├── audio_utils.py         # ffmpeg 转码 + add_noise/mix_overlap 包装
├── selfcheck.py           # 冒烟自检(kws_0+cmd_0)
├── run_demo.sh            # source setenv + uvicorn 启动
├── README.md              # 启动/公网/自检/依赖安装
├── __init__.py            # 让 demo_web 可作为包 import
└── static/
    ├── index.html
    ├── app.js
    └── style.css
```

依赖（装到 `code/.venv`，因为 inference_engine 必须在该 venv 复用 enroll_infer + torch/CUDA 上下文）：
`fastapi`、`uvicorn[standard]`、`python-multipart`。（均为无副作用轻依赖，不触发 speechbrain lazy 问题。）

## 9. 实现风险与备注

1. **不动 enroll_infer.py**：inference_engine 独立新建并 import 复用，参赛提交链路零风险。
2. **脆弱点保留**：`inspect.getmodule` patch + sys.path 注入必须原样照搬到 inference_engine.py 顶部（对抗审查固化点）。
3. **venv 选择**：服务必须在 `code/.venv` 跑（enroll_infer 依赖 torch/wespeaker/diar/DiCoW-inference sys.path + E 盘模型缓存）。FastAPI 装该 venv，不另开（另开无法共享 CUDA 上下文/import enroll_infer）。
4. **4060 vs L20**：demo 在本机 4060，推理比官方 L20 慢；预期 5–15s/条，demo 节奏按此设计（不承诺实时）。
5. **ffmpeg**：读 webm 必需，本机是否已装待实现时确认；缺失按 CLAUDE.md 装到 `E:\Tools\ffmpeg`。
6. **公网安全**：cloudflared quick tunnel 暴露后任何人可访问；demo 服务无鉴权（YAGNI），不跑高强度任务，session 不留敏感数据。README 提醒用完即关隧道。

## 10. 验收标准

- [x] `bash code/demo_web/run_demo.sh` 启动到 ready, 监听 7860（run_demo.sh 经 bash -n 语法检查 + 等价 uvicorn 命令实测启动到 ready; run_demo.sh 未单独实跑, 因 server 持续在跑供 demo）。
- [x] `selfcheck.py` 用 kws_0+cmd_0 返回非空 transcript（"空调开到自热调到二十五度风量调到百分之三十"21字）+ max_sim 0.254 + target.wav 存在。
- [x] 浏览器录音 enrollment + test 上传成功, 可回放（用户手机端到端测通）。
- [x] mix_mode = none/babble/voice 三种, 混音 test 可回放（curl 验证 mixed wav HTTP 200 + 用户测）。
- [x] `/infer` 返回转写 + 徽章 + target 切片可播放（curl 自匹配 max_sim 0.97 + target HTTP 200 + 用户测）。
- [x] cloudflared quick tunnel 公网链接在手机浏览器可录音 + 推理 + 出结果（用户测通; quick tunnel URL 每次重起变化, 见 README §4）。
- [x] 换人录 test 触发拒识徽章（默认 thr 0.27; 用户端到端测通）。
