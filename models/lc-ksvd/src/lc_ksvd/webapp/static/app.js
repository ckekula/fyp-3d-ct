const STEP_LABELS = [
  "Load volume", "Resample", "Lung mask", "Extract patches",
  "Sparse code", "Classify", "Reconstruct map", "Visualise",
];

let currentJobId = null;
let pollTimer = null;

const el = (id) => document.getElementById(id);

const dropzone = el("dropzone");
const fileInput = el("file-input");
const dropzoneText = el("dropzone-text");
const uploadStatus = el("upload-status");
const runBtn = el("run-btn");
const runError = el("run-error");
const progressWrap = el("progress-wrap");
const progressSteps = el("progress-steps");
const logPanel = el("log-panel");
const step1Badge = el("step1-badge");
const step2 = el("step2");
const step2Badge = el("step2-badge");
const resultSummary = el("result-summary");
const slicerBtn = el("slicer-btn");
const slicerStatus = el("slicer-status");

function buildProgressSteps() {
  progressSteps.innerHTML = "";
  STEP_LABELS.forEach((label, i) => {
    const span = document.createElement("span");
    span.className = "progress-step";
    span.dataset.step = String(i + 1);
    span.textContent = `${i + 1}. ${label}`;
    progressSteps.appendChild(span);
  });
}

function updateProgressFromLogs(logs) {
  const text = logs.join("\n");
  const matches = [...text.matchAll(/\[(\d)\/8\]/g)];
  const lastStep = matches.length ? parseInt(matches[matches.length - 1][1], 10) : 0;
  document.querySelectorAll(".progress-step").forEach((node) => {
    const n = parseInt(node.dataset.step, 10);
    node.classList.toggle("done", n < lastStep);
    node.classList.toggle("active", n === lastStep);
  });
}

dropzone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];
  if (!file) return;

  dropzoneText.textContent = `Uploading ${file.name}...`;
  uploadStatus.textContent = "";
  uploadStatus.className = "inline-status busy";

  const form = new FormData();
  form.append("ct_file", file);

  try {
    const res = await fetch("/api/upload", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Upload failed.");

    currentJobId = data.job_id;
    dropzone.classList.add("has-file");
    dropzoneText.textContent = `Selected: ${data.filename}`;
    uploadStatus.textContent = "Ready to run.";
    uploadStatus.className = "inline-status ok";
    runBtn.disabled = false;
  } catch (err) {
    dropzoneText.textContent = "Click to choose a CT scan (.nii / .nii.gz)";
    uploadStatus.textContent = err.message;
    uploadStatus.className = "inline-status err";
  }
});

runBtn.addEventListener("click", async () => {
  if (!currentJobId) return;

  runBtn.disabled = true;
  runError.hidden = true;
  step1Badge.classList.add("active");
  progressWrap.hidden = false;
  buildProgressSteps();
  logPanel.textContent = "Starting inference...\n";

  try {
    const res = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_id: currentJobId }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Failed to start.");
    pollTimer = setInterval(pollStatus, 1500);
  } catch (err) {
    showRunError(err.message);
    runBtn.disabled = false;
  }
});

function showRunError(message) {
  runError.hidden = false;
  runError.textContent = message;
  step1Badge.classList.remove("active");
}

async function pollStatus() {
  if (!currentJobId) return;
  try {
    const res = await fetch(`/api/status/${currentJobId}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Status check failed.");

    logPanel.textContent = data.logs.length ? data.logs.join("\n") : "Working...";
    logPanel.scrollTop = logPanel.scrollHeight;
    updateProgressFromLogs(data.logs);

    if (data.status === "done") {
      clearInterval(pollTimer);
      step1Badge.classList.remove("active");
      step1Badge.classList.add("complete");
      onInferenceDone(data.result);
    } else if (data.status === "error") {
      clearInterval(pollTimer);
      runBtn.disabled = false;
      showRunError(data.error || "Inference failed.");
    }
  } catch (err) {
    clearInterval(pollTimer);
    runBtn.disabled = false;
    showRunError(err.message);
  }
}

function onInferenceDone(result) {
  step2.classList.remove("card-disabled");
  step2Badge.classList.add("active");

  resultSummary.hidden = false;
  el("result-scan-name").textContent = result.scan_name;
  el("result-ct-path").textContent = result.ct_resampled_nifti;
  el("result-labelmap-path").textContent = result.boundary_labelmap_nifti;

  slicerBtn.disabled = false;
}

slicerBtn.addEventListener("click", async () => {
  if (!currentJobId) return;
  slicerBtn.disabled = true;
  slicerStatus.textContent = "Launching 3D Slicer...";
  slicerStatus.className = "inline-status busy";

  try {
    const res = await fetch("/api/open-slicer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_id: currentJobId }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Failed to open 3D Slicer.");

    slicerStatus.textContent = "3D Slicer launched.";
    slicerStatus.className = "inline-status ok";
    step2Badge.classList.remove("active");
    step2Badge.classList.add("complete");
  } catch (err) {
    slicerStatus.textContent = err.message;
    slicerStatus.className = "inline-status err";
  } finally {
    slicerBtn.disabled = false;
  }
});
