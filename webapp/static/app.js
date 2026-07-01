const tabs = document.querySelectorAll(".tab");
const panels = {
  camera: document.getElementById("camera-panel"),
  upload: document.getElementById("upload-panel"),
};

const video = document.getElementById("video");
const cameraPlaceholder = document.getElementById("camera-placeholder");
const cameraToggle = document.getElementById("camera-toggle");
const canvas = document.getElementById("canvas");

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const uploadPreview = document.getElementById("upload-preview");
const dropzoneText = document.getElementById("dropzone-text");

const resultEl = document.getElementById("result");
const resultBadge = document.getElementById("result-badge");
const resultLatency = document.getElementById("result-latency");
const meterFill = document.getElementById("meter-fill");
const meterScore = document.getElementById("meter-score");
const errorEl = document.getElementById("error");

let mediaStream = null;
let liveTimer = null;
let inFlight = false;

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    tabs.forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    Object.values(panels).forEach((p) => p.classList.remove("active"));
    panels[tab.dataset.tab].classList.add("active");
    if (tab.dataset.tab !== "camera") {
      stopCamera();
    }
    hideResult();
  });
});

function showError(message) {
  errorEl.textContent = message;
  errorEl.classList.remove("hidden");
}

function hideError() {
  errorEl.classList.add("hidden");
}

function hideResult() {
  resultEl.classList.add("hidden");
}

function renderResult(data) {
  hideError();
  resultEl.classList.remove("hidden");
  const percentScreen = Math.round(data.probability * 100);
  const isScreen = data.label === "screen";

  resultBadge.textContent = isScreen ? "SCREEN RECAPTURE" : "REAL PHOTO";
  resultBadge.className = "badge " + (isScreen ? "screen" : "real");
  resultLatency.textContent = `${data.latency_ms} ms`;
  meterFill.style.left = `${percentScreen}%`;
  meterScore.textContent = `${percentScreen}% screen`;
}

async function sendBlob(blob) {
  if (inFlight) return;
  inFlight = true;
  const formData = new FormData();
  formData.append("image", blob, "frame.jpg");
  try {
    const response = await fetch("/api/predict", { method: "POST", body: formData });
    const data = await response.json();
    if (!response.ok) {
      showError(data.error || "Prediction failed");
    } else {
      renderResult(data);
    }
  } catch (err) {
    showError("Could not reach the prediction server.");
  } finally {
    inFlight = false;
  }
}

// --- Camera mode ---

async function startCamera() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" },
      audio: false,
    });
  } catch (err) {
    showError("Camera access denied or unavailable: " + err.message);
    return;
  }
  video.srcObject = mediaStream;
  video.style.display = "block";
  cameraPlaceholder.style.display = "none";
  cameraToggle.textContent = "Stop Camera";
  cameraToggle.classList.add("stop");

  liveTimer = setInterval(captureFrame, 700);
}

function stopCamera() {
  if (liveTimer) {
    clearInterval(liveTimer);
    liveTimer = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
  video.style.display = "none";
  cameraPlaceholder.style.display = "flex";
  cameraToggle.textContent = "Start Camera";
  cameraToggle.classList.remove("stop");
}

function captureFrame() {
  if (!video.videoWidth) return;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  canvas.toBlob((blob) => {
    if (blob) sendBlob(blob);
  }, "image/jpeg", 0.85);
}

cameraToggle.addEventListener("click", () => {
  if (mediaStream) {
    stopCamera();
    hideResult();
  } else {
    startCamera();
  }
});

// --- Upload mode ---

function handleFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    showError("Please choose an image file.");
    return;
  }
  const url = URL.createObjectURL(file);
  uploadPreview.src = url;
  uploadPreview.classList.remove("hidden");
  dropzoneText.style.display = "none";
  sendBlob(file);
}

dropzone.addEventListener("click", (event) => {
  event.preventDefault();
  fileInput.click();
});

fileInput.addEventListener("change", () => {
  if (fileInput.files.length) handleFile(fileInput.files[0]);
});

["dragover", "dragenter"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
  });
});

dropzone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (file) handleFile(file);
});
