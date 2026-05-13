const clips = [
  { name: "Lead Verse 01", time: "00:13.2", key: "F#m", confidence: "94%", mood: "close / smoky", length: "8 bars" },
  { name: "Harmony Stack A", time: "00:41.7", key: "A", confidence: "88%", mood: "wide / airy", length: "4 bars" },
  { name: "Bridge Adlib", time: "01:18.4", key: "F#m", confidence: "91%", mood: "urgent / bright", length: "6 bars" },
  { name: "Whisper Double", time: "01:50.9", key: "C#", confidence: "86%", mood: "intimate", length: "4 bars" }
];

const julesReplies = [
  "I hear a gentle 2.5 dB shelf around 12 kHz and a slower compressor release for this take.",
  "Zen readout is stable. Try muting the pad for two bars before the hook to make the lead vocal feel taller.",
  "The bridge adlib is sharp in a musical way. I would tune only the landing notes and keep the human drift.",
  "I can mock a three-part harmony map: low fifth, center octave, and breathy high response.",
  "For Pythonista later, this panel can stay as a WebView while native code feeds clips and meter data."
];

const eqBands = ["60", "125", "250", "500", "1k", "4k", "12k"];
const synthData = [
  { name: "Sub", blocks: [[5, 24], [36, 18], [68, 24]] },
  { name: "Pad", blocks: [[0, 42], [48, 44]] },
  { name: "Keys", blocks: [[12, 14], [31, 16], [57, 13], [76, 16]] },
  { name: "Arp", blocks: [[8, 9], [20, 9], [32, 9], [44, 9], [56, 9], [68, 9], [80, 9]] }
];

const chatLog = document.querySelector("#chatLog");
const chatForm = document.querySelector("#chatForm");
const chatInput = document.querySelector("#chatInput");
const clipList = document.querySelector("#clipList");
const clipInfo = document.querySelector("#clipInfo");
const eqContainer = document.querySelector("#eqBands");
const synthContainer = document.querySelector("#synthLanes");
const clockReadout = document.querySelector("#clockReadout");
const latencyValue = document.querySelector("#latencyValue");
const detectedNote = document.querySelector("#detectedNote");
const dspLoad = document.querySelector("#dspLoad");

let selectedClip = 0;
let replyCursor = 0;
let startTime = performance.now();
let transportMode = "play";

function appendMessage(author, text, type = "jules") {
  const message = document.createElement("article");
  message.className = `message ${type}`;
  message.innerHTML = `<strong>${author}</strong><span>${text}</span>`;
  chatLog.appendChild(message);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function renderClips() {
  clipList.innerHTML = "";
  clips.forEach((clip, index) => {
    const item = document.createElement("button");
    item.className = `clip-item ${index === selectedClip ? "active" : ""}`;
    item.type = "button";
    item.innerHTML = `
      <strong>${clip.name}</strong><em>${clip.time}</em>
      <small>${clip.mood}</small><small>${clip.confidence} pitch lock</small>
    `;
    item.addEventListener("click", () => {
      selectedClip = index;
      renderClips();
      renderClipInfo();
      appendMessage("Jules", `Loaded ${clip.name}. Pitch confidence is ${clip.confidence}; I am watching the ${clip.key} center.`, "jules");
    });
    clipList.appendChild(item);
  });
}

function renderClipInfo() {
  const clip = clips[selectedClip];
  clipInfo.innerHTML = `
    <dt>Name</dt><dd>${clip.name}</dd>
    <dt>Start</dt><dd>${clip.time}</dd>
    <dt>Key</dt><dd>${clip.key}</dd>
    <dt>Length</dt><dd>${clip.length}</dd>
    <dt>Confidence</dt><dd>${clip.confidence}</dd>
    <dt>Mood</dt><dd>${clip.mood}</dd>
  `;
}

function renderEq() {
  eqContainer.innerHTML = "";
  eqBands.forEach((band, index) => {
    const gain = 32 + Math.sin(index * 1.3) * 18 + index * 3;
    const node = document.createElement("div");
    node.className = "eq-band";
    node.innerHTML = `<div class="eq-slider" style="--gain: ${Math.max(14, gain)}%"></div><label>${band}</label>`;
    eqContainer.appendChild(node);
  });
}

function renderSynthLanes() {
  synthContainer.innerHTML = "";
  synthData.forEach((lane) => {
    const laneNode = document.createElement("div");
    laneNode.className = "lane";
    const blocks = lane.blocks.map(([left, width]) => `<span class="lane-block" style="left: ${left}%; width: ${width}%;"></span>`).join("");
    laneNode.innerHTML = `<span class="lane-name">${lane.name}</span><div class="lane-track">${blocks}</div>`;
    synthContainer.appendChild(laneNode);
  });
}

function updateMeters(t) {
  const values = {
    input: 46 + Math.sin(t / 380) * 24 + Math.sin(t / 97) * 7,
    output: 58 + Math.sin(t / 460 + 1.2) * 20,
    monitor: 42 + Math.sin(t / 520 + 2.1) * 16
  };

  Object.entries(values).forEach(([name, value]) => {
    const safeValue = Math.max(8, Math.min(96, value));
    document.querySelector(`[data-meter="${name}"]`).style.width = `${safeValue}%`;
    const db = Math.round(-34 + safeValue * 0.34);
    document.querySelector(`#${name}Db`).textContent = `${db} dB`;
  });

  dspLoad.style.width = `${48 + Math.sin(t / 820) * 14}%`;
  latencyValue.textContent = `${(7.2 + Math.sin(t / 1000) * 1.4).toFixed(1)} ms`;
}

function updateClock(now) {
  const elapsed = Math.floor((now - startTime) / 1000);
  const minutes = String(Math.floor(elapsed / 60)).padStart(2, "0");
  const seconds = String(elapsed % 60).padStart(2, "0");
  const frames = String(Math.floor((now / 33.3) % 30)).padStart(2, "0");
  clockReadout.textContent = `00:${minutes}:${seconds}.${frames}`;
}

function drawGrid(ctx, width, height) {
  ctx.strokeStyle = "rgba(57, 255, 136, 0.10)";
  ctx.lineWidth = 1;
  for (let x = 0; x < width; x += 50) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  for (let y = 0; y < height; y += 38) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
}

function drawWaveform(t) {
  const canvas = document.querySelector("#waveformCanvas");
  const ctx = canvas.getContext("2d");
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);
  drawGrid(ctx, width, height);

  const mid = height / 2;
  const gradient = ctx.createLinearGradient(0, 0, width, 0);
  gradient.addColorStop(0, "rgba(57,255,136,0.22)");
  gradient.addColorStop(0.5, "rgba(147,255,189,0.92)");
  gradient.addColorStop(1, "rgba(57,255,136,0.28)");

  ctx.strokeStyle = gradient;
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  for (let x = 0; x < width; x++) {
    const amp = 22 + Math.sin(x / 55 + t / 700) * 18 + Math.sin(x / 19) * 10;
    const y = mid + Math.sin(x / 9 + t / 220) * amp * Math.sin(x / 140 + 1.2);
    if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  const playhead = (t / 24) % width;
  ctx.strokeStyle = "rgba(255, 209, 102, 0.9)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(playhead, 12);
  ctx.lineTo(playhead, height - 12);
  ctx.stroke();
}

function drawPitchCurve(t) {
  const canvas = document.querySelector("#pitchCanvas");
  const ctx = canvas.getContext("2d");
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);
  drawGrid(ctx, width, height);

  ctx.strokeStyle = "rgba(57,255,136,0.95)";
  ctx.lineWidth = 3;
  ctx.beginPath();
  for (let x = 0; x < width; x++) {
    const step = Math.floor(x / 88) % 5;
    const y = height - 42 - step * 26 + Math.sin(x / 24 + t / 340) * 9;
    if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  ctx.fillStyle = "rgba(147,255,189,0.9)";
  for (let x = 28; x < width; x += 88) {
    const y = height - 42 - (Math.floor(x / 88) % 5) * 26 + Math.sin(x / 24 + t / 340) * 9;
    ctx.beginPath();
    ctx.arc(x, y, 4.5, 0, Math.PI * 2);
    ctx.fill();
  }
}

function animate(now) {
  updateMeters(now);
  updateClock(now);
  drawWaveform(now);
  drawPitchCurve(now);
  const notes = ["F#3", "A3", "C#4", "E4", "F#4"];
  detectedNote.textContent = notes[Math.floor(now / 900) % notes.length];
  requestAnimationFrame(animate);
}

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = chatInput.value.trim();
  if (!text) return;
  appendMessage("You", text, "user");
  chatInput.value = "";
  window.setTimeout(() => {
    appendMessage("Jules", julesReplies[replyCursor % julesReplies.length], "jules");
    replyCursor += 1;
  }, 420);
});

document.querySelectorAll(".transport-button").forEach((button) => {
  button.addEventListener("click", () => {
    transportMode = button.dataset.mode;
    document.querySelectorAll(".transport-button").forEach((item) => item.classList.toggle("active", item === button));
    appendMessage("Jules", `${transportMode.toUpperCase()} toggled. I am keeping the mock meters and pitch tracker alive.`, "jules");
  });
});

appendMessage("Jules", "Runtime awake. I am watching vocals, pitch confidence, synth lanes, and Zen state.", "jules");
appendMessage("Jules", "Reference mix target: dark neon intimacy, precise low end, and human vocal drift.", "jules");
renderClips();
renderClipInfo();
renderEq();
renderSynthLanes();
requestAnimationFrame(animate);
