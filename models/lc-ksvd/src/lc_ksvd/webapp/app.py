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

import json
import logging
import os
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Optional

import nibabel as nib
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from lc_ksvd.config import ABNORMALITY_CATEGORIES, CLASS_ORDER, INFERENCE_DIR, METADATA_JSON
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

# METADATA_JSON points at the training server's dataset copy, which isn't
# present in a local/dev checkout -- fall back to the copy checked into the
# repo itself (repo_root/data/rexgrounding-ct/dataset_2_ultimate.json).
_REPO_ROOT = Path(__file__).resolve().parents[5]
_FALLBACK_METADATA_JSON = _REPO_ROOT / "data" / "rexgrounding-ct" / "dataset_2_ultimate.json"


def _load_scan_metadata() -> tuple[dict[str, list[str]], dict[str, dict[int, str]]]:
    """
    Parse the RExGrounding-CT metadata JSON once into two per-scan-filename
    lookups:
      findings_by_scan    : scan filename -> list of reported finding strings
                             (display only, "Reported Findings" panel).
      finding_map_by_scan : scan filename -> {f_idx: category} (e.g. "2c"/"2d"
                             per f-slice index) -- mirrors
                             MetadataRegistry.get_finding_map(). Passed to
                             run_inference(gt_finding_map=...) so an uploaded
                             ground-truth mask for a *known* scan gets
                             labelled with real categories (the same
                             CLASS_ORDER convention the prediction uses)
                             instead of raw per-scan channel order -- see
                             inference.build_ground_truth_label_volume.
    """
    path = METADATA_JSON if METADATA_JSON.exists() else _FALLBACK_METADATA_JSON
    if not path.exists():
        logger.warning("Findings metadata JSON not found (looked at %s and %s); "
                        "uploaded scans will show no ground-truth findings.",
                        METADATA_JSON, _FALLBACK_METADATA_JSON)
        return {}, {}

    with open(path) as f:
        data = json.load(f)

    findings_by_scan: dict[str, list[str]] = {}
    finding_map_by_scan: dict[str, dict[int, str]] = {}
    for split in ("train", "test"):
        for entry in data.get(split, []):
            name = entry.get("name")
            if not name:
                continue

            findings = entry.get("findings")
            if findings:
                findings_by_scan[name] = list(findings.values())

            categories = entry.get("categories") or {}
            finding_map: dict[int, str] = {}
            for f_idx_str, category in categories.items():
                try:
                    finding_map[int(f_idx_str)] = str(category)
                except (TypeError, ValueError):
                    continue
            if finding_map:
                finding_map_by_scan[name] = finding_map

    return findings_by_scan, finding_map_by_scan


FINDINGS_BY_SCAN, FINDING_MAP_BY_SCAN = _load_scan_metadata()

# label value (== CLASS_ORDER index) -> display name, used to rename
# Slicer's auto-generated "Segment_<label>" entries after loading a label
# map into something meaningful. Prediction labels always follow this
# convention; ground-truth labels only do when the job result's
# "ground_truth_class_order_aligned" is True (see open_slicer() below).
SEGMENT_DISPLAY_NAMES = {
    i: ABNORMALITY_CATEGORIES[cls]
    for i, cls in enumerate(CLASS_ORDER)
    if cls in ABNORMALITY_CATEGORIES
}


def _ct_shape_error(path: Path) -> Optional[str]:
    """
    Fast header-only check (nib.load().shape doesn't touch voxel data) that
    catches an upload that can't possibly be a CT scan -- most commonly a
    ground-truth mask (nifti_io.load_mask's [F, H, W, D] finding-channel
    format) selected by mistake instead of the volume. Mirrors the
    squeezability rule nifti_io.ensure_3d_volume() applies for real, but
    checking it here means a bad upload is rejected in milliseconds instead
    of after a minute-plus of lung segmentation inside run_inference().
    """
    shape = nib.load(str(path)).shape
    if len(shape) <= 3:
        return None
    extra = shape[3:]
    if all(d == 1 for d in extra):
        return None
    return (
        f"This file has shape {shape}, which looks like a segmentation mask "
        "(nifti_io's [F, H, W, D] finding-mask format), not a CT scan -- a CT "
        "volume should be 3D (H, W, D). If you meant to attach a ground-truth "
        "mask instead, use the 'Attach ground-truth mask (optional)' button."
    )

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


def _run_job(
    job_id: str,
    volume_path: Path,
    gt_mask_path: Optional[Path],
    gt_finding_map: Optional[dict[int, str]],
) -> None:
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
        result = run_inference(
            volume_path=volume_path, output_dir=output_dir, gt_mask_path=gt_mask_path,
            gt_finding_map=gt_finding_map,
        )
        job["result"] = {
            "scan_name": result["scan_name"],
            "ct_resampled_nifti": str(result["ct_resampled_nifti"]),
            "boundary_labelmap_nifti": str(result["boundary_labelmap_nifti"]),
            "ground_truth_mask_nifti": (
                str(result["ground_truth_mask_nifti"])
                if result["ground_truth_mask_nifti"] else None
            ),
            "ground_truth_class_order_aligned": result["ground_truth_class_order_aligned"],
            "patch_class_counts": result["patch_class_counts"],
            "abnormal_voxel_counts": result["abnormal_voxel_counts"],
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

    try:
        shape_error = _ct_shape_error(dest)
    except Exception as exc:
        return jsonify({"error": f"Could not read NIfTI file: {exc}"}), 400
    if shape_error is not None:
        dest.unlink(missing_ok=True)
        return jsonify({"error": shape_error}), 400

    # Optional ground-truth mask: a raw [F,H,W,D] NIfTI on the *original*
    # (pre-resample) CT grid, same convention as nifti_io.load_mask -- if
    # provided, run_inference() resamples/crops it onto the prediction grid
    # and exports it alongside the predicted boundary map for comparison.
    gt_mask_path = None
    gt_file = request.files.get("gt_mask_file")
    if gt_file is not None and gt_file.filename != "":
        gt_filename = secure_filename(gt_file.filename)
        if not gt_filename.endswith(ALLOWED_SUFFIXES):
            return jsonify({"error": "Ground-truth mask must be a .nii or .nii.gz file."}), 400
        gt_dest = job_dir / gt_filename
        gt_file.save(gt_dest)
        gt_mask_path = str(gt_dest)

    findings = FINDINGS_BY_SCAN.get(filename, [])
    # Only meaningful when the uploaded scan matches a known dataset entry --
    # see _load_scan_metadata(). None otherwise, and run_inference() falls
    # back to raw channel-order ground-truth labelling in that case.
    gt_finding_map = FINDING_MAP_BY_SCAN.get(filename)

    JOBS[job_id] = {
        "status": "uploaded",
        "logs": [],
        "volume_path": str(dest),
        "gt_mask_path": gt_mask_path,
        "gt_finding_map": gt_finding_map,
        "result": None,
        "error": None,
        "findings": findings,
    }
    return jsonify({
        "job_id": job_id,
        "filename": filename,
        "findings": findings,
        "gt_mask_provided": gt_mask_path is not None,
    })


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

    gt_mask_path = job.get("gt_mask_path")
    thread = threading.Thread(
        target=_run_job,
        args=(
            job_id,
            Path(job["volume_path"]),
            Path(gt_mask_path) if gt_mask_path else None,
            job.get("gt_finding_map"),
        ),
        daemon=True,
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
    gt_path = job["result"].get("ground_truth_mask_nifti")
    gt_aligned = job["result"].get("ground_truth_class_order_aligned", False)

    def _rename_segment_lines(var_name: str) -> list[str]:
        """
        Statements that rename `var_name` (a loaded segmentation node)'s
        auto-generated "Segment_<label>" entries to their real abnormality
        name (SEGMENT_DISPLAY_NAMES, keyed by label value == CLASS_ORDER
        index) -- Slicer's labelmap importer names segments "Segment_1",
        "Segment_2", ... in ascending label-value order by default, which is
        what this assumes. Ternary form (not `if:`) because everything below
        is joined with ";" into one line -- see the note above `lines`.
        """
        out = [f"{var_name}Seg = {var_name}.GetSegmentation()"]
        for label_value, name in SEGMENT_DISPLAY_NAMES.items():
            seg_id_var = f"{var_name}SegId{label_value}"
            out.append(
                f"{seg_id_var} = {var_name}Seg.GetSegmentIdBySegmentName('Segment_{label_value}')"
            )
            out.append(
                f"{var_name}Seg.GetSegment({seg_id_var}).SetName({name!r}) if {seg_id_var} else None"
            )
        return out

    # Loaded as: CT -> scalar volume (+ GPU volume rendering turned on, so
    # the 3D pane isn't left blank -- Slicer's Volume Rendering module
    # requires an explicit CreateDefaultVolumeRenderingNodes()/
    # SetVisibility(True) call, loadVolume() alone never enables it),
    # boundary label map -> segmentation, renamed from "Segment_N" to the
    # real abnormality name (SEGMENT_DISPLAY_NAMES). Ground truth, when
    # present, loads as a second segmentation for a side-by-side comparison
    # -- renamed the same way only if gt_aligned (its label values follow
    # CLASS_ORDER; see build_ground_truth_label_volume), since otherwise its
    # labels are just raw per-scan channel order and "GGO"/"Lung Nodule"
    # would be a guess, not a fact.
    #
    # Slicer's own CLI argument parser (not Python's) chokes on a multi-line
    # --python-code string ("Problem parsing command line arguments") -- it
    # has to be one line, so statements are joined with ";" instead of "\n".
    # That makes an "if" statement dangerous mid-chain (`if x: a; b` binds
    # both a and b into the if-suite), so conditionals throughout are
    # written as ternary expression-statements instead of if/else blocks.
    lines = [
        "import slicer",
        f"volumeNode = slicer.util.loadVolume(r'{ct_path}')",
        "volRenLogic = slicer.modules.volumerendering.logic()",
        "displayNode = volRenLogic.CreateDefaultVolumeRenderingNodes(volumeNode)",
        "displayNode.SetVisibility(True)",
        "preset = volRenLogic.GetPresetByName('CT-Lung')",
        "displayNode.GetVolumePropertyNode().Copy(preset) if preset is not None else None",
        f"predSegNode = slicer.util.loadSegmentation(r'{labelmap_path}')",
    ]
    lines += _rename_segment_lines("predSegNode")
    if gt_path:
        lines.append(f"gtSegNode = slicer.util.loadSegmentation(r'{gt_path}')")
        if gt_aligned:
            lines += _rename_segment_lines("gtSegNode")
    lines += [
        "threeDView = slicer.app.layoutManager().threeDWidget(0).threeDView()",
        "threeDView.resetFocalPoint()",
    ]
    code = ";".join(lines)
    try:
        subprocess.Popen([SLICER_EXECUTABLE, "--python-code", code])
    except OSError as exc:
        return jsonify({"error": f"Failed to launch 3D Slicer: {exc}"}), 500

    return jsonify({"status": "launched"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
