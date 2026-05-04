"""
Status/diagnostics report for the work done so far (BioMed-Parse + CT-RATE pipeline).

What this script checks:
- Python + venv info
- numpy version (BioMed-Parse expects 1.26.4)
- whether OpenCV is installed (can conflict with numpy pin if it requires numpy>=2)
- BioMed-Parse repo layout (configs present, config name location)
- whether the required checkpoint path exists (and whether it's a broken symlink)
- whether CT-RATE train_labels.csv exists under /home/chest_ct/code/data (robust search)
- whether any NIfTI volumes exist under data/data_volumes/**/*.nii.gz

It prints "NEXT STEPS" tailored to your current state.
"""

from __future__ import annotations

import os
import sys
import platform
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List


REPO_ROOT = Path("/home/chest_ct/code").resolve()
MODELS_DIR = REPO_ROOT / "models" / "biomed-parse"
INFERENCE_PY = MODELS_DIR / "inference.py"
CONFIGS_DIR = MODELS_DIR / "configs"
EXPECTED_CKPT = MODELS_DIR / "model_weights" / "biomedparse_3D_AllData_MultiView_edge.ckpt"

DATA_DIR = REPO_ROOT / "data"
CT_RATE_DIR = DATA_DIR / "ct-rate"
CT_VOLUMES_ROOT = DATA_DIR / "data_volumes"


def run(cmd: List[str], cwd: Optional[Path] = None) -> Tuple[int, str]:
    try:
        p = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=False,
        )
        out = (p.stdout or "") + (p.stderr or "")
        return p.returncode, out.strip()
    except Exception as e:
        return 999, f"Failed to run {cmd}: {e}"


def find_one(pattern: str, root: Path) -> Optional[Path]:
    hits = sorted([p for p in root.rglob(pattern) if p.is_file()])
    if not hits:
        return None
    # Prefer data/ct-rate if present
    hits = sorted(hits, key=lambda p: (str(p).find("data/ct-rate") == -1, str(p)))
    return hits[0]


def is_broken_symlink(p: Path) -> bool:
    return p.is_symlink() and not p.exists()


def main() -> int:
    print("=== ENVIRONMENT ===")
    print("Platform:", platform.platform())
    print("Python:", sys.version.replace("\n", " "))
    print("Executable:", sys.executable)
    print("CWD:", os.getcwd())
    print("VIRTUAL_ENV:", os.environ.get("VIRTUAL_ENV"))

    # numpy
    numpy_ver = None
    try:
        import numpy as np  # noqa: F401

        numpy_ver = np.__version__
    except Exception as e:
        print("numpy: NOT INSTALLED:", e)

    if numpy_ver:
        print("numpy:", numpy_ver)

    # opencv
    opencv_installed = False
    try:
        import cv2  # noqa: F401

        opencv_installed = True
        print("opencv-python: INSTALLED (cv2 import OK)")
    except Exception:
        print("opencv-python: not installed (or not importable)")

    print("\n=== BIOMED-PARSE LAYOUT ===")
    print("MODELS_DIR:", MODELS_DIR)
    print("INFERENCE_PY exists:", INFERENCE_PY.exists())
    print("CONFIGS_DIR exists:", CONFIGS_DIR.exists())
    if CONFIGS_DIR.exists():
        # Key config file location we discovered
        config_candidate = CONFIGS_DIR / "model" / "biomedparse_3D.yaml"
        print("Has configs/model/biomedparse_3D.yaml:", config_candidate.exists())
        # show a few configs
        _, out = run(["bash", "-lc", "find configs -maxdepth 2 -type f | head -n 20"], cwd=MODELS_DIR)
        print("configs listing (head):")
        print(out if out else "(empty)")

    print("\n=== CHECKPOINT ===")
    print("Expected ckpt path:", EXPECTED_CKPT)
    if EXPECTED_CKPT.exists():
        print("Checkpoint status: OK (file exists)")
    elif EXPECTED_CKPT.is_symlink():
        print("Checkpoint status: SYMLINK")
        rc, target = run(["bash", "-lc", f"readlink -f {EXPECTED_CKPT} || true"])
        print("Symlink resolves to:", target if target else "(unresolved)")
        print("Broken symlink:", is_broken_symlink(EXPECTED_CKPT))
    else:
        print("Checkpoint status: MISSING")

    print("\n=== CT-RATE LABELS ===")
    expected_labels = CT_RATE_DIR / "train_labels.csv"
    if expected_labels.exists():
        print("train_labels.csv: OK at", expected_labels)
    else:
        alt = find_one("train_labels.csv", DATA_DIR) if DATA_DIR.exists() else None
        if alt:
            print("train_labels.csv: FOUND at", alt)
            print("Note: expected path is", expected_labels)
        else:
            print("train_labels.csv: NOT FOUND under", DATA_DIR)

    print("\n=== CT VOLUMES ===")
    if CT_VOLUMES_ROOT.exists():
        rc, out = run(["bash", "-lc", "find data/data_volumes -type f -name '*.nii.gz' | wc -l"], cwd=REPO_ROOT)
        print("NIfTI (*.nii.gz) count under data/data_volumes:", out)
    else:
        print("CT volumes root missing:", CT_VOLUMES_ROOT)

    print("\n=== INFERENCE CLI CHECK ===")
    if INFERENCE_PY.exists():
        rc, out = run([sys.executable, str(INFERENCE_PY), "-h"], cwd=MODELS_DIR)
        print(out)
    else:
        print("Cannot run inference.py - missing file.")

    print("\n=== NEXT STEPS (based on what we saw) ===")
    # 1) numpy pin
    if numpy_ver and numpy_ver != "1.26.4":
        print("- Pin numpy to 1.26.4 (BioMed-Parse repo pins this):")
        print("    python -m pip install --upgrade --force-reinstall numpy==1.26.4")
    elif numpy_ver == "1.26.4":
        print("- numpy pin looks OK (1.26.4).")

    # 2) opencv conflict warning
    if opencv_installed and numpy_ver == "1.26.4":
        print("- If opencv-python complains about numpy>=2, uninstall opencv-python to keep numpy==1.26.4.")

    # 3) checkpoint instructions
    if not EXPECTED_CKPT.exists():
        print("- Download the BioMed-Parse checkpoint and place it at:")
        print(f"    {EXPECTED_CKPT}")
        print("  Repo README suggests a HuggingFace checkpoint named biomedparse_v2.ckpt; you can:")
        print("    mkdir -p /home/chest_ct/code/models/biomed-parse/model_weights")
        print("    cd /home/chest_ct/code/models/biomed-parse/model_weights")
        print("    wget -O biomedparse_v2.ckpt https://huggingface.co/microsoft/BiomedParse/resolve/main/biomedparse_v2.ckpt")
        print("    ln -sf biomedparse_v2.ckpt biomedparse_3D_AllData_MultiView_edge.ckpt")

    # 4) config mismatch note
    if (CONFIGS_DIR / "model" / "biomedparse_3D.yaml").exists():
        print("- Your config is at configs/model/biomedparse_3D.yaml.")
        print("  If you see `MissingConfigException: Cannot find primary config 'biomedparse_3D'`,")
        print("  then inference.py likely needs compose(config_name='model/biomedparse_3D') or a top-level wrapper config.")

    # 5) labels
    if not expected_labels.exists() and not find_one("train_labels.csv", DATA_DIR):
        print("- CT-RATE labels missing. Place train_labels.csv under:")
        print(f"    {expected_labels.parent}/")
        print("  (or update your pipeline to point at the actual location).")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())