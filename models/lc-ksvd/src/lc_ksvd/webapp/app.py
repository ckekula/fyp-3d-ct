"""
app.py
Flask web interface for the LC-KSVD chest CT abnormality detector.

Step 1: upload a CT volume (.nii/.nii.gz) and run the LC-KSVD inference
pipeline (lc_ksvd.inference.inference.run_inference) in the background,
watching live progress in the browser.
Step 2: once inference completes, open the two resulting NIfTI files
(resampled CT + boundary label map) directly in 3D Slicer.

Usage:
  python -m lc_ksvd.webapp.app
Then open http://127.0.0.1:5050 in a browser.
"""

# Force a non-interactive backend before anything imports pyplot -- inference
# runs on a background thread, and matplotlib's default GUI backend (e.g.
# MacOSX) is main-thread-only and will crash/hang when driven off it.
import matplotlib
matplotlib.use("Agg")

import logging
import os
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from lc_ksvd.config import INFERENCE_DIR
from lc_ksvd.inference.inference import run_inference

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

app = Flask(__name__)

UPLOAD_DIR = INFERENCE_DIR / "webapp_uploads"
RUNS_DIR = INFERENCE_DIR / "webapp_runs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# Path to the 3D Slicer executable -- the binary *inside* the app bundle, not
# the double-clickable Slicer.app itself. Override via the SLICER_EXECUTABLE
# environment variable if Slicer is installed somewhere else.
SLICER_EXECUTABLE = os.environ.get(
    "SLICER_EXECUTABLE", "/Applications/Slicer.app/Contents/MacOS/Slicer"
)

ALLOWED_SUFFIXES = (".nii", ".nii.gz")

# ─── In-memory job registry ────────────────────────────────────────────────
# Each uploaded scan gets a job_id with its own status/log/result. Only one
# inference job runs at a time -- matches the single-GPU/CPU reality of the
# underlying pipeline (lungmask + Batch-OMP are not meant to run concurrently
# against the same device).
JOBS: dict[str, dict] = {}
_current_job_id: Optional[str] = None
_current_job_lock = threading.Lock()


class _JobLogHandler(logging.Handler):
    """Appends formatted log records into a job's in-memory log list, so the
    browser can poll for the same progress lines that show up in the terminal."""

    def __init__(self, job: dict):
        super().__init__()
        self.job = job

    def emit(self, record: logging.LogRecord) -> None:
        self.job["logs"].append(self.format(record))


def _run_job(job_id: str, volume_path: Path) -> None:
    global _current_job_id
    job = JOBS[job_id]

    handler = _JobLogHandler(job)
    handler.setFormatter(logging.Formatter("%(message)s"))
    # lc_ksvd.inference.inference and lc_ksvd.inference.classify both log
    # under the "lc_ksvd" logger hierarchy -- attaching to the parent catches
    # every step/progress line from both.
    pipeline_logger = logging.getLogger("lc_ksvd")
    pipeline_logger.addHandler(handler)

    job["status"] = "running"
    try:
        output_dir = RUNS_DIR / job_id
        result = run_inference(volume_path=volume_path, output_dir=output_dir)
        job["result"] = {
            "scan_name": result["scan_name"],
            "overlay_png": str(result["overlay_png"]),
            "boundary_png": str(result["boundary_png"]),
            "ct_resampled_nifti": str(result["ct_resampled_nifti"]),
            "boundary_labelmap_nifti": str(result["boundary_labelmap_nifti"]),
        }
        job["status"] = "done"
    except Exception as exc:  # noqa: BLE001 -- surface any failure to the UI
        logger.exception("Inference job %s failed", job_id)
        job["error"] = str(exc)
        job["status"] = "error"
    finally:
        pipeline_logger.removeHandler(handler)
        with _current_job_lock:
            if _current_job_id == job_id:
                _current_job_id = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    file = request.files.get("ct_file")
    if file is None or file.filename == "":
        return jsonify({"error": "No file provided."}), 400

    filename = secure_filename(file.filename)
    if not filename.endswith(ALLOWED_SUFFIXES):
        return jsonify({"error": "Please upload a .nii or .nii.gz file."}), 400

    job_id = uuid.uuid4().hex
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    dest = job_dir / filename
    file.save(dest)

    JOBS[job_id] = {
        "status": "uploaded",
        "logs": [],
        "volume_path": str(dest),
        "result": None,
        "error": None,
    }
    return jsonify({"job_id": job_id, "filename": filename})


@app.route("/api/run", methods=["POST"])
def run():
    global _current_job_id
    body = request.get_json(silent=True) or {}
    job_id = body.get("job_id")
    job = JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown job_id."}), 404

    with _current_job_lock:
        if _current_job_id is not None:
            return jsonify({
                "error": "Another scan is currently running. Please wait for it to finish."
            }), 409
        _current_job_id = job_id

    thread = threading.Thread(
        target=_run_job, args=(job_id, Path(job["volume_path"])), daemon=True
    )
    thread.start()
    return jsonify({"status": "started"})


@app.route("/api/status/<job_id>")
def status(job_id):
    job = JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown job_id."}), 404
    return jsonify({
        "status": job["status"],
        "logs": job["logs"],
        "result": job["result"],
        "error": job["error"],
    })


@app.route("/api/open-slicer", methods=["POST"])
def open_slicer():
    body = request.get_json(silent=True) or {}
    job_id = body.get("job_id")
    job = JOBS.get(job_id)
    if job is None or job["result"] is None:
        return jsonify({"error": "No completed scan for this job_id."}), 404

    if not Path(SLICER_EXECUTABLE).exists():
        return jsonify({
            "error": f"3D Slicer executable not found at {SLICER_EXECUTABLE}. "
                     "Set the SLICER_EXECUTABLE environment variable to its path."
        }), 500

    ct_path = job["result"]["ct_resampled_nifti"]
    labelmap_path = job["result"]["boundary_labelmap_nifti"]
    # Loaded as: CT -> scalar volume, boundary label map -> segmentation
    # (Slicer turns each distinct label value in a label-map NIfTI into its
    # own segment automatically, same as Segmentations -> Import).
    code = (
        "import slicer;"
        f"slicer.util.loadVolume(r'{ct_path}');"
        f"slicer.util.loadSegmentation(r'{labelmap_path}')"
    )
    try:
        subprocess.Popen([SLICER_EXECUTABLE, "--python-code", code])
    except OSError as exc:
        return jsonify({"error": f"Failed to launch 3D Slicer: {exc}"}), 500

    return jsonify({"status": "launched"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
