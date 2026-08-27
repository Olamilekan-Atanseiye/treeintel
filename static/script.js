/* =====================================================
   PROCEDURAL ATMOSPHERE (fog + drifting particles)
   ===================================================== */
(function atmosphere() {
  const canvas = document.getElementById("atmosphere");
  const ctx = canvas.getContext("2d");
  let particles = [];
  let w, h;
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function resize() {
    w = canvas.width = window.innerWidth;
    h = canvas.height = window.innerHeight;
  }
  window.addEventListener("resize", resize);
  resize();

  function makeParticles(count) {
    particles = Array.from({ length: count }, () => ({
      x: Math.random() * w,
      y: Math.random() * h,
      r: Math.random() * 1.6 + 0.4,
      speedY: Math.random() * 0.18 + 0.04,
      drift: Math.random() * 0.3 - 0.15,
      alpha: Math.random() * 0.35 + 0.08,
    }));
  }
  makeParticles(reducedMotion ? 0 : Math.min(70, Math.floor((w * h) / 22000)));

  function tick() {
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "rgba(63, 224, 208, 1)";
    particles.forEach((p) => {
      ctx.globalAlpha = p.alpha;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fill();
      p.y -= p.speedY;
      p.x += p.drift;
      if (p.y < -10) { p.y = h + 10; p.x = Math.random() * w; }
    });
    ctx.globalAlpha = 1;
    if (!reducedMotion) requestAnimationFrame(tick);
  }
  if (!reducedMotion) requestAnimationFrame(tick);
})();

/* =====================================================
   NAVBAR
   ===================================================== */
const navbar = document.getElementById("navbar");
window.addEventListener("scroll", () => {
  navbar.classList.toggle("scrolled", window.scrollY > 40);
});

const navBurger = document.getElementById("navBurger");
const navMobile = document.getElementById("navMobile");
navBurger.addEventListener("click", () => navMobile.classList.toggle("open"));
navMobile.querySelectorAll("a").forEach((a) =>
  a.addEventListener("click", () => navMobile.classList.remove("open"))
);

/* =====================================================
   UPLOAD / SCAN / PREDICT
   ===================================================== */
const dropzone = document.getElementById("dropzone");
const dropzoneContent = document.getElementById("dropzoneContent");
const fileInput = document.getElementById("fileInput");
const previewImg = document.getElementById("previewImg");
const scanFrame = document.querySelector(".scan-frame");
const scanLine = document.getElementById("scanLine");
const identifyBtn = document.getElementById("identifyBtn");
const identifyBtnText = document.getElementById("identifyBtnText");
const errorMsg = document.getElementById("errorMsg");
const statusMsg = document.getElementById("statusMsg");

const pulseDot = document.getElementById("pulseDot");
const readoutStatus = document.getElementById("readoutStatus");
const readoutBody = document.getElementById("readoutBody");
const readoutLog = document.getElementById("readoutLog");

const dashboardEmpty = document.getElementById("dashboardEmpty");
const dashboardResult = document.getElementById("dashboardResult");
const speciesScientific = document.getElementById("speciesScientific");
const confidenceNumber = document.getElementById("confidenceNumber");
const barList = document.getElementById("barList");

const knowledgePulse = document.getElementById("knowledgePulse");
const knowledgeStatus = document.getElementById("knowledgeStatus");
const chatThread = document.getElementById("chatThread");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const chatSend = document.getElementById("chatSend");

let selectedFile = null;
let currentSpecies = null;

// Prevent the browser from opening a dropped image as a new page when the
// pointer lands just outside the drop target.
["dragover", "drop"].forEach((evt) =>
  window.addEventListener(evt, (e) => e.preventDefault())
);

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
});
fileInput.addEventListener("change", (e) => { if (e.target.files.length) handleFile(e.target.files[0]); });

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.add("dragover"); })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.remove("dragover"); })
);
dropzone.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) handleFile(f); });

function handleFile(file) {
  hideError();

  const allowed = ["image/png", "image/jpeg", "image/webp"];
  if (!allowed.includes(file.type)) {
    showError("Unsupported file type. Please upload a PNG, JPG, or WEBP image.");
    return;
  }
  if (file.size > 25 * 1024 * 1024) {
    showError("File is too large. Maximum size is 25 MB.");
    return;
  }

  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewImg.hidden = false;
    dropzoneContent.hidden = true;
    scanFrame.classList.add("active");
  };
  reader.readAsDataURL(file);

  identifyBtn.disabled = false;
  setReadoutIdle("Specimen loaded — ready to analyze");
}

identifyBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  hideError();
  identifyBtn.disabled = true;
  identifyBtnText.textContent = "Analyzing…";
  scanLine.classList.add("sweeping");
  setReadoutActive();

  const t0 = performance.now();
  const formData = new FormData();
  formData.append("image", selectedFile);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    const responseText = await response.text();
    let data;

    try {
      data = responseText ? JSON.parse(responseText) : {};
    } catch {
      throw new Error(
        `Server returned invalid JSON (HTTP ${response.status}). Response: ${
          responseText || "[empty response]"
        }`
      );
    }

    if (!response.ok) {
      throw new Error(
        data.error || `Prediction failed (HTTP ${response.status}).`
      );
    }

    const elapsedMs = Math.round(performance.now() - t0);
    await logReadout(data, elapsedMs);
    showResult(data);
  } catch (err) {
    showError(err.message || "Something went wrong while identifying the specimen.");
    setReadoutIdle("Analysis failed — see error above");
  } finally {
    identifyBtn.disabled = false;
    identifyBtnText.textContent = "Identify Specimen";
    scanLine.classList.remove("sweeping");
  }
});

function setReadoutIdle(msg) {
  pulseDot.classList.remove("active");
  readoutStatus.textContent = msg;
}

function setReadoutActive() {
  pulseDot.classList.add("active");
  readoutStatus.textContent = "Running inference…";
  readoutBody.innerHTML = "";
  readoutLog.innerHTML = "";
}

function appendLog(text, delay) {
  return new Promise((resolve) => {
    setTimeout(() => {
      const line = document.createElement("div");
      line.className = "log-line";
      line.innerHTML = `<span class="tag">›</span>${text}`;
      readoutLog.appendChild(line);
      resolve();
    }, delay);
  });
}

async function logReadout(data, elapsedMs) {
  await appendLog("Preprocessing image — resized to 224×224", 100);
  await appendLog("Running forward pass through classifier", 250);
  await appendLog(`Top match: <em>${formatSpeciesName(data.species)}</em> (${data.confidence}%)`, 300);
  await appendLog(`Inference completed in ${elapsedMs} ms`, 200);
  pulseDot.classList.remove("active");
  readoutStatus.textContent = "Analysis complete";
}

function showResult(data) {
  dashboardEmpty.hidden = true;
  dashboardResult.hidden = false;

  speciesScientific.textContent = formatSpeciesName(data.species);
  confidenceNumber.textContent = "0";
  animateNumber(confidenceNumber, data.confidence);

  currentSpecies = data.species;
  startKnowledgeChat(data.species);

  barList.innerHTML = "";
  data.top5.forEach((item) => {
    const li = document.createElement("li");
    li.className = "bar-item";
    li.innerHTML = `
      <div class="bar-top">
        <span class="bar-name">${formatSpeciesName(item.species)}</span>
        <span class="bar-value">${item.confidence}%</span>
      </div>
      <div class="bar-track"><div class="bar-fill"></div></div>
    `;
    barList.appendChild(li);
    requestAnimationFrame(() => {
      li.querySelector(".bar-fill").style.width = `${item.confidence}%`;
    });
  });

  dashboardResult.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function animateNumber(el, target) {
  const duration = 700;
  const start = performance.now();
  function step(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    el.textContent = (target * eased).toFixed(1);
    if (progress < 1) requestAnimationFrame(step);
    else el.textContent = target.toFixed(1);
  }
  requestAnimationFrame(step);
}

function formatSpeciesName(raw) {
  return raw.replace(/_/g, " ");
}

/* =====================================================
   AI KNOWLEDGE — chat-first flow
   ===================================================== */
async function startKnowledgeChat(species) {
  chatThread.innerHTML = "";
  chatInput.disabled = false;
  chatSend.disabled = false;
  chatInput.placeholder = `Ask about ${formatSpeciesName(species)}…`;

  knowledgePulse.classList.add("active");
  knowledgeStatus.textContent = `Preparing brief on ${formatSpeciesName(species)}…`;

  const pending = addChatBubble("assistant pending", "Preparing botanical brief…");

  try {
    const response = await fetch("/knowledge/overview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ species }),
});

const responseText = await response.text();

console.log("Knowledge HTTP status:", response.status);
console.log("Knowledge raw response:", responseText);

let data;

try {
    data = JSON.parse(responseText);
} catch (jsonError) {
    throw new Error(
        `Knowledge endpoint returned an invalid response (HTTP ${response.status}). ` +
        `Response: ${responseText || "[empty response]"}`
    );
}

if (!response.ok) {
    throw new Error(data.error || "Could not load knowledge brief.");
}
    if (!data.available) {
      pending.textContent = `I don't have documentation connected for ${formatSpeciesName(species)} yet (${data.reason}). You can still ask, but I may not be able to answer accurately.`;
      pending.className = "chat-bubble assistant";
    } else {
      pending.textContent = formatOverviewAsMessage(species, data.overview);
      pending.className = "chat-bubble assistant";
    }
  } catch (err) {
    pending.textContent = err.message || "Could not load the knowledge brief.";
    pending.className = "chat-bubble assistant error";
  } finally {
    knowledgePulse.classList.remove("active");
    knowledgeStatus.textContent = `Ready — ask about ${formatSpeciesName(species)}`;
  }
}

function formatOverviewAsMessage(species, overview) {
  return `Here's what I have on ${formatSpeciesName(species)}:

Description: ${overview.description}

Uses: ${overview.uses}

Ecology: ${overview.ecology}

Conservation status: ${overview.conservation}`;
}

/* =====================================================
   CHAT — ask questions about the identified species
   ===================================================== */
chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = chatInput.value.trim();
  if (!question || !currentSpecies) return;

  addChatBubble("user", question);
  chatInput.value = "";
  chatSend.disabled = true;

  const pending = addChatBubble("assistant pending", "Thinking…");

  try {
    const response = await fetch("/knowledge/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ species: currentSpecies, question }),
    });
    const responseText = await response.text();

console.log("Knowledge Ask HTTP status:", response.status);
console.log("Knowledge Ask raw response:", responseText);

let data;

try {
    data = JSON.parse(responseText);
} catch (jsonError) {
    throw new Error(
        `Knowledge server returned an invalid response (HTTP ${response.status}). ` +
        `Response: ${responseText || "[empty response]"}`
    );
}

if (!response.ok || data.error) {
    throw new Error(data.error || "Something went wrong.");
}
    pending.textContent = data.answer;
    pending.className = "chat-bubble assistant";
  } catch (err) {
    pending.textContent = err.message || "Something went wrong answering that.";
    pending.className = "chat-bubble assistant error";
  } finally {
    chatSend.disabled = false;
    chatThread.scrollTop = chatThread.scrollHeight;
  }
});

function addChatBubble(role, text) {
  const bubble = document.createElement("div");
  bubble.className = `chat-bubble ${role}`;
  bubble.textContent = text;
  chatThread.appendChild(bubble);
  chatThread.scrollTop = chatThread.scrollHeight;
  return bubble;
}

function showError(message) {
  errorMsg.textContent = message;
  errorMsg.hidden = false;
}
function hideError() {
  errorMsg.hidden = true;
  errorMsg.textContent = "";
}
