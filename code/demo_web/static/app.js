// 录音 + 上传 + 混音 + 推理 + 回放
const $ = (id) => document.getElementById(id);

let enrBlob = null, testBlob = null;
let enrId = null, testId = null;
let mediaRec = null, chunks = [];

function pickMime() {
  const cands = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"];
  for (const c of cands) if (MediaRecorder.isTypeSupported(c)) return c;
  return "";
}

async function record(button, onDone) {
  if (button.dataset.rec === "1") {
    mediaRec && mediaRec.stop();
    return;
  }
  const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, sampleRate: 16000 } });
  chunks = [];
  mediaRec = new MediaRecorder(stream, pickMime() ? { mimeType: pickMime() } : undefined);
  mediaRec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  mediaRec.onstop = () => {
    stream.getTracks().forEach((t) => t.stop());
    button.dataset.rec = "0";
    button.textContent = "● 录音";
    button.classList.remove("recording");
    onDone(new Blob(chunks, { type: mediaRec.mimeType }));
  };
  mediaRec.start();
  button.dataset.rec = "1";
  button.textContent = "■ 停止";
  button.classList.add("recording");
}

function setBlob(blob, audioEl, playBtn, setter) {
  const url = URL.createObjectURL(blob);
  audioEl.src = url;
  playBtn.disabled = false;
  playBtn.onclick = () => audioEl.play();
  setter(blob);
}

$("enrRec").onclick = () =>
  record($("enrRec"), (b) => {
    setBlob(b, $("enrAudio"), $("enrPlay"), (x) => (enrBlob = x));
    $("enrStatus").textContent = `已录 ${(b.size / 1024).toFixed(0)} KB, 待上传`;
    maybeEnable();
  });

$("testRec").onclick = () =>
  record($("testRec"), (b) => {
    setBlob(b, $("testAudio"), $("testPlay"), (x) => (testBlob = x));
    $("testStatus").textContent = `已录 ${(b.size / 1024).toFixed(0)} KB, 待上传`;
    maybeEnable();
  });

// 混音控件联动
document.querySelectorAll("input[name=mix]").forEach((r) => r.addEventListener("change", renderMix));
function mixMode() {
  return document.querySelector("input[name=mix]:checked").value;
}
function renderMix() {
  const m = mixMode();
  $("babbleCtrl").classList.toggle("hidden", m !== "babble");
  $("voiceCtrl").classList.toggle("hidden", m !== "voice");
}
$("snr").oninput = (e) => ($("snrVal").textContent = e.target.value);
$("overlap").oninput = (e) => ($("overlapVal").textContent = e.target.value);

// 加载干扰人声列表
fetch("/interferers").then((r) => r.json()).then((list) => {
  const sel = $("interferer");
  sel.innerHTML = "";
  if (!list.length) { sel.innerHTML = "<option value=''>无可用</option>"; return; }
  list.forEach((it) => {
    const o = document.createElement("option");
    o.value = it.id; o.textContent = `${it.name} (${it.duration_s}s)`;
    sel.appendChild(o);
  });
}).catch(() => { $("interferer").innerHTML = "<option value=''>加载失败</option>"; });

function maybeEnable() {
  $("infer").disabled = !(enrBlob && testBlob);
}

async function uploadEnroll() {
  const fd = new FormData();
  fd.append("file", enrBlob, "enroll.webm");
  const r = await fetch("/upload/enroll", { method: "POST", body: fd });
  if (!r.ok) throw new Error("enroll 上传失败: " + (await r.text()));
  const j = await r.json();
  enrId = j.enroll_id;
  $("enrStatus").textContent = `已上传 ${j.duration_s}s`;
}

async function uploadTest() {
  const fd = new FormData();
  fd.append("file", testBlob, "test.webm");
  fd.append("mix_mode", mixMode());
  fd.append("snr_db", $("snr").value);
  fd.append("overlap_ratio", $("overlap").value / 100);
  fd.append("interferer_id", $("interferer").value);
  const r = await fetch("/upload/test", { method: "POST", body: fd });
  if (!r.ok) throw new Error("test 上传失败: " + (await r.text()));
  const j = await r.json();
  testId = j.test_id;
  $("testStatus").textContent = `已上传 ${j.duration_s}s [${j.mix_mode}]`;
  $("mixedAudio").src = j.mixed_url;
  return j;
}

$("infer").onclick = async () => {
  const btn = $("infer"); const st = $("inferStatus");
  btn.disabled = true; st.textContent = "上传 enrollment...";
  $("result").classList.add("hidden");
  try {
    if (!enrId) await uploadEnroll();
    st.textContent = "上传 test + 混音...";
    if (!testId) await uploadTest();
    st.textContent = "推理中(5~15s)...";
    const r = await fetch("/infer", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enroll_id: enrId, test_id: testId }),
    });
    if (!r.ok) throw new Error("推理失败: " + (await r.text()));
    const j = await r.json();
    renderResult(j);
    st.textContent = "完成";
  } catch (e) {
    st.textContent = "❌ " + e.message;
  } finally {
    btn.disabled = false;
  }
};

function renderResult(j) {
  $("result").classList.remove("hidden");
  if (j.error) {
    $("badge").className = "badge err"; $("badge").textContent = "出错: " + j.error;
    $("transcript").textContent = "-";
    return;
  }
  const rej = j.rejected;
  const badge = $("badge");
  badge.className = "badge " + (rej ? "rej" : "ok");
  badge.textContent = rej ? "🚫 拒识: 目标说话人不在场" : "✅ 接受: 已转写目标";
  $("simScore").textContent = j.max_sim.toFixed(3);
  $("transcript").textContent = rej ? "(拒识, 无转写)" : j.transcript;
  const sims = Object.entries(j.sims || {}).map(([k, v]) =>
    `<span class="sim-pill ${k === j.target_speaker ? "tgt" : ""}">${k}: ${v.toFixed(3)}${k === j.target_speaker ? " ★" : ""}</span>`).join("");
  $("simsList").innerHTML = sims;
  $("timing").textContent = `推理 ${j.infer_sec}s | 音频 ${j.duration_s}s | RTF ${j.rtf}`;
  if (j.target_audio_url) $("targetAudio").src = j.target_audio_url;
}
