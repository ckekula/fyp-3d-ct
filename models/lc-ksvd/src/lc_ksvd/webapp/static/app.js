const STEP_LABELS = [
  "Load volume", "Resample", "Lung mask", "Extract patches",
  "Sparse code", "Classify", "Reconstruct map", "Export outputs",
];

// Mirrors config.py's CLASS_ORDER / ABNORMALITY_CATEGORIES naming.
const CLASS_LABELS = { normal: "Normal", "2c": "GGO", "2d": "Lung Nodule" };

let selectedCtFile = null;
let selectedGtFile = null;
let currentJobId = null;
let pollTimer = null;
let currentFindings = [];

const el = (id) => document.getElementById(id);

const dropzone = el("dropzone");
const fileInput = el("file-input");
const dropzoneText = el("dropzone-text");
const uploadStatus = el("upload-status");
const gtBtn = el("gt-btn");
const gtFileInput = el("gt-file-input");
const gtStatus = el("gt-status");
const runBtn = el("run-btn");
const runError = el("run-error");
const findingsPanel = el("findings-panel");
const findingsList = el("findings-list");
const abnormalityPanel = el("abnormality-panel");
const abnormalityList = el("abnormality-list");
const progressWrap = el("progress-wrap");
const progressSteps = el("progress-steps");
const logPanel = el("log-panel");
const step1Badge = el("step1-badge");
const step2 = el("step2");
const step2Badge = el("step2-badge");
const resultSummary = el("result-summary");
const resultGtRow = el("result-gt-row");
const slicerBtn = el("slicer-btn");
const slicerStatus = el("slicer-status");

function renderFindings(findings) {
  findingsList.innerHTML = "";
  if (!findings || findings.length === 0) {
    findingsPanel.hidden = true;
    return;
  }
  findings.forEach((text) => {
    const li = document.createElement("li");
    li.textContent = text;
    findingsList.appendChild(li);
  });
  findingsPanel.hidden = false;
}

function renderAbnormalities(patchCounts, voxelCounts) {
  abnormalityList.innerHTML = "";
  if (!patchCounts) {
    abnormalityPanel.hidden = true;
    return;
  }
  const abnormalClasses = Object.keys(patchCounts).filter((c) => c !== "normal");
  const anyAbnormal = abnormalClasses.some((c) => patchCounts[c] > 0);

  if (!anyAbnormal) {
    const li = document.createElement("li");
    li.textContent = "No abnormal patches detected — scan classified as normal.";
    abnormalityList.appendChild(li);
  } else {
    abnormalClasses.forEach((cls) => {
      const nPatches = patchCounts[cls] || 0;
      if (nPatches === 0) return;
      const nVoxels = (voxelCounts && voxelCounts[cls]) || 0;
      const li = document.createElement("li");
      li.textContent = `${CLASS_LABELS[cls] || cls}: ${nPatches} patch(es), ${nVoxels.toLocaleString()} voxels`;
      abnormalityList.appendChild(li);
    });
  }
  abnormalityPanel.hidden = false;
}

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

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) return;

  selectedCtFile = file;
  dropzone.classList.add("has-file");
  dropzoneText.textContent = `Selected: ${file.name}`;
  uploadStatus.textContent = "Ready to run.";
  uploadStatus.className = "inline-status ok";
  runBtn.disabled = false;
  gtBtn.disabled = false;
  renderFindings(null);
});

gtBtn.addEventListener("click", () => gtFileInput.click());

gtFileInput.addEventListener("change", () => {
  const file = gtFileInput.files[0];
  if (!file) return;
  selectedGtFile = file;
  gtStatus.textContent = `Attached: ${file.name}`;
  gtStatus.className = "inline-status ok";
});

runBtn.addEventListener("click", async () => {
  if (!selectedCtFile) return;

  runBtn.disabled = true;
  gtBtn.disabled = true;
  runError.hidden = true;
  step1Badge.classList.add("active");
  progressWrap.hidden = false;
  buildProgressSteps();
  logPanel.textContent = "Uploading...\n";

  try {
    const form = new FormData();
    form.append("ct_file", selectedCtFile);
    if (selectedGtFile) form.append("gt_mask_file", selectedGtFile);

    const uploadRes = await fetch("/api/upload", { method: "POST", body: form });
    const uploadData = await uploadRes.json();
    if (!uploadRes.ok) throw new Error(uploadData.error || "Upload failed.");

    currentJobId = uploadData.job_id;
    currentFindings = uploadData.findings || [];

    logPanel.textContent = "Starting inference...\n";
    const runRes = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_id: currentJobId }),
    });
    const runData = await runRes.json();
    if (!runRes.ok) throw new Error(runData.error || "Failed to start.");

    pollTimer = setInterval(pollStatus, 1500);
  } catch (err) {
    showRunError(err.message);
    runBtn.disabled = false;
    gtBtn.disabled = false;
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
      gtBtn.disabled = false;
      showRunError(data.error || "Inference failed.");
    }
  } catch (err) {
    clearInterval(pollTimer);
    runBtn.disabled = false;
    gtBtn.disabled = false;
    showRunError(err.message);
  }
}

function onInferenceDone(result) {
  step2.classList.remove("card-disabled");
  step2Badge.classList.add("active");

  renderFindings(currentFindings);
  renderAbnormalities(result.patch_class_counts, result.abnormal_voxel_counts);

  resultSummary.hidden = false;
  el("result-scan-name").textContent = result.scan_name;
  el("result-ct-path").textContent = result.ct_resampled_nifti;
  el("result-labelmap-path").textContent = result.boundary_labelmap_nifti;

  if (result.ground_truth_mask_nifti) {
    const namedNote = result.ground_truth_class_order_aligned
      ? ""
      : " (generic segment names -- scan not found in dataset metadata, so category per region is unknown)";
    el("result-gt-path").textContent = result.ground_truth_mask_nifti + namedNote;
    resultGtRow.hidden = false;
  } else {
    resultGtRow.hidden = true;
  }

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
