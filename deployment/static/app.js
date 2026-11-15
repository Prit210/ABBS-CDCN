// CDCN++ Face Anti-Spoofing Web Client (Clean & Modular)

let video = document.getElementById("webcam-video");
let overlayCanvas = document.getElementById("overlay-canvas");
let overlayCtx = overlayCanvas.getContext("2d");

let btnStartCam = document.getElementById("btn-start-cam");
let btnStopCam = document.getElementById("btn-stop-cam");
let btnCapture = document.getElementById("btn-capture");
let thresholdSlider = document.getElementById("threshold-slider");
let thresholdValueSpan = document.getElementById("threshold-val");

let resultsContainer = document.getElementById("results-container");
let emptyState = document.getElementById("empty-state");
let latencyBadge = document.getElementById("latency-badge");
let facesCountBadge = document.getElementById("faces-count-badge");
let videoPlaceholder = document.getElementById("video-placeholder");

// Settings Modal Elements
let btnSettings = document.getElementById("btn-settings");
let settingsModal = document.getElementById("settings-modal");
let btnCloseModal = document.getElementById("btn-close-modal");
let btnSaveSettings = document.getElementById("btn-save-settings");
let inputTheta = document.getElementById("setting-theta");

let mediaStream = null;
let isAnalyzing = false;
let currentResults = [];

// 1. Initialize UI & Settings
async function initApp() {
  thresholdSlider.addEventListener("input", onThresholdChange);
  btnStartCam.addEventListener("click", startWebcam);
  btnStopCam.addEventListener("click", stopWebcam);
  btnCapture.addEventListener("click", () => analyzeFrame());

  btnSettings.addEventListener("click", openSettingsModal);
  btnCloseModal.addEventListener("click", closeSettingsModal);
  btnSaveSettings.addEventListener("click", saveSettings);

  // Close modal when clicking outside
  settingsModal.addEventListener("click", (e) => {
    if (e.target === settingsModal) closeSettingsModal();
  });

  await fetchConfig();
}

async function fetchConfig() {
  try {
    const res = await fetch("/api/config");
    const data = await res.json();
    if (data.status === "success") {
      const cfg = data.config;
      inputTheta.value = cfg.theta !== undefined ? cfg.theta : 0.7;

      if (cfg.threshold !== undefined) {
        thresholdSlider.value = parseFloat(cfg.threshold).toFixed(2);
        thresholdValueSpan.textContent = parseFloat(cfg.threshold).toFixed(2);
      }
    }
  } catch (err) {
    console.warn("Could not load backend configuration:", err);
  }
}

function openSettingsModal() {
  fetchConfig();
  settingsModal.classList.add("open");
}

function closeSettingsModal() {
  settingsModal.classList.remove("open");
}

async function saveSettings() {
  const payload = {
    theta: parseFloat(inputTheta.value),
    threshold: parseFloat(thresholdSlider.value)
  };

  btnSaveSettings.disabled = true;
  btnSaveSettings.textContent = "Saving...";

  try {
    const res = await fetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (data.status === "success") {
      closeSettingsModal();
      await fetchConfig();
      if (currentResults.length > 0) {
        analyzeFrame();
      }
    } else {
      alert("Failed to save settings: " + (data.message || "Unknown error"));
    }
  } catch (err) {
    alert("Error updating settings: " + err.message);
  } finally {
    btnSaveSettings.disabled = false;
    btnSaveSettings.textContent = "Save Changes";
  }
}

function onThresholdChange() {
  const val = parseFloat(thresholdSlider.value);
  thresholdValueSpan.textContent = val.toFixed(2);
  if (currentResults.length > 0) {
    recalcLocalVerdicts(val);
  }
}

// 2. Webcam Operations
async function startWebcam() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: false
    });
    video.srcObject = mediaStream;
    await video.play();

    videoPlaceholder.style.display = "none";
    btnStartCam.disabled = true;
    btnStopCam.disabled = false;
    btnCapture.disabled = false;

    syncCanvasSize();
    video.addEventListener("loadedmetadata", syncCanvasSize);
  } catch (err) {
    alert("Error accessing webcam: " + err.message);
    console.error(err);
  }
}

function stopWebcam() {
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
  }
  video.srcObject = null;
  videoPlaceholder.style.display = "block";
  btnStartCam.disabled = false;
  btnStopCam.disabled = true;
  btnCapture.disabled = true;
  clearOverlay();
}

function syncCanvasSize() {
  if (video.videoWidth && video.videoHeight) {
    overlayCanvas.width = video.videoWidth;
    overlayCanvas.height = video.videoHeight;
  }
}

// 3. Frame Capture and API Request
async function analyzeFrame() {
  if (!mediaStream || isAnalyzing) return;
  if (!video.videoWidth || !video.videoHeight) return;

  isAnalyzing = true;
  syncCanvasSize();

  // Draw current video frame onto an off-screen canvas to get raw base64 JPEG
  const offscreen = document.createElement("canvas");
  offscreen.width = video.videoWidth;
  offscreen.height = video.videoHeight;
  const ctx = offscreen.getContext("2d");
  ctx.drawImage(video, 0, 0, offscreen.width, offscreen.height);

  const base64Data = offscreen.toDataURL("image/jpeg", 0.90);
  const currentThreshold = parseFloat(thresholdSlider.value);

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_base64: base64Data,
        threshold: currentThreshold
      })
    });

    const data = await res.json();
    if (data.status === "success") {
      currentResults = data.results || [];
      renderOverlay(currentResults);
      renderResultCards(currentResults);
      latencyBadge.textContent = `${data.latency_ms} ms`;
      facesCountBadge.textContent = `${data.faces_detected} face(s)`;
    } else {
      console.warn("Prediction returned error:", data);
    }
  } catch (err) {
    console.error("Inference request failed:", err);
  } finally {
    isAnalyzing = false;
  }
}

// 4. Canvas Bounding Box & Annotation Overlay
function clearOverlay() {
  overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
}

function renderOverlay(results) {
  clearOverlay();
  if (!results || results.length === 0) return;

  const currentThreshold = parseFloat(thresholdSlider.value);

  results.forEach((item) => {
    const [x1, y1, x2, y2] = item.box;
    const w = x2 - x1;
    const h = y2 - y1;

    const isReal = item.score >= currentThreshold;
    const color = isReal ? "#10b981" : "#ef4444";
    const label = `Face #${item.face_id}: ${isReal ? "REAL" : "SPOOF"} (${item.score.toFixed(3)})`;

    // Draw box
    overlayCtx.lineWidth = 3;
    overlayCtx.strokeStyle = color;
    overlayCtx.strokeRect(x1, y1, w, h);

    // Draw label background
    overlayCtx.font = "bold 14px sans-serif";
    const textWidth = overlayCtx.measureText(label).width;
    const textHeight = 22;
    const labelY = Math.max(y1 - textHeight, 0);

    overlayCtx.fillStyle = color;
    overlayCtx.fillRect(x1, labelY, textWidth + 12, textHeight);

    // Draw label text
    overlayCtx.fillStyle = "#ffffff";
    overlayCtx.fillText(label, x1 + 6, labelY + 16);
  });
}

// 5. Results Cards Rendering
function renderResultCards(results) {
  resultsContainer.innerHTML = "";

  if (!results || results.length === 0) {
    emptyState.style.display = "block";
    emptyState.textContent = "No faces detected in the captured frame.";
    resultsContainer.appendChild(emptyState);
    return;
  }

  emptyState.style.display = "none";
  const currentThreshold = parseFloat(thresholdSlider.value);

  results.forEach((item) => {
    const isReal = item.score >= currentThreshold;
    const card = document.createElement("div");
    card.className = `face-card ${isReal ? "real" : "spoof"}`;

    const scorePct = Math.min(Math.max(item.score * 100, 0), 100);

    card.innerHTML = `
      <div class="face-card-header">
        <span class="face-id">Face #${item.face_id}</span>
        <span class="verdict-tag ${isReal ? "real" : "spoof"}">
          ${isReal ? "✓ Real (Live)" : "✗ Spoof (Attack)"}
        </span>
      </div>

      <div class="face-card-body">
        <div class="img-panel">
          <span>Cropped Face</span>
          <img src="${item.face_crop_base64}" alt="Face #${item.face_id}" />
        </div>
        <div class="img-panel">
          <span>Estimated 3D Depth</span>
          <img src="${item.depth_map_base64}" alt="Depth Map #${item.face_id}" />
        </div>
      </div>

      <div class="score-metrics">
        <div class="metric-row">
          <span class="metric-label">Mean Depth Score:</span>
          <span class="metric-value" style="color: ${isReal ? '#10b981' : '#ef4444'}">
            ${item.score.toFixed(4)}
          </span>
        </div>
        <div class="metric-row">
          <span class="metric-label">Active Threshold:</span>
          <span class="metric-value">${currentThreshold.toFixed(2)}</span>
        </div>
        <div class="score-bar-bg" title="Score: ${item.score.toFixed(4)} (Threshold: ${currentThreshold.toFixed(2)})">
          <div class="score-bar-fill" style="width: ${scorePct}%; background-color: ${isReal ? '#10b981' : '#ef4444'};"></div>
        </div>
      </div>
    `;

    resultsContainer.appendChild(card);
  });
}

function recalcLocalVerdicts(newThreshold) {
  if (!currentResults || currentResults.length === 0) return;
  renderOverlay(currentResults);
  renderResultCards(currentResults);
}

// Start application
window.addEventListener("DOMContentLoaded", initApp);
