"""
merlin_ct_pipeline.py
Full pipeline: load CT volumes → Merlin embedding → classify findings → save results.
Classification uses keyword matching (no API key required).
"""

import os
import json
import time

import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.expanduser("~/Documents/GitHub/fyp-3d-ct")
DATA_ROOT    = os.path.join(PROJECT_ROOT, "data")
VOLUMES_DIR  = os.path.join(DATA_ROOT, "segmentations")
REXCT_DIR    = os.path.join(DATA_ROOT, "rexgrounding-ct")
DATASET_JSON = os.path.join(REXCT_DIR, "dataset.json")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "models/merlin/results_f")

VALID_CATEGORIES = ["Lung Nodule", "Lung opacity", "Consolidation", "Atelectasis"]


# ── CT loading ────────────────────────────────────────────────────────────────

def load_and_preprocess_ct(nii_path):
    """Load a NIfTI CT volume, clip HU values, and normalise to [0, 1]."""
    img    = nib.load(nii_path)
    volume = img.get_fdata()

    # Drop trailing singleton dimension if present (e.g. shape H,W,D,1)
    if volume.ndim == 4:
        volume = volume[..., 0]

    volume = np.clip(volume, -1000, 400)
    volume = (volume + 1000) / 1400.0
    volume = np.nan_to_num(volume, nan=0.0)

    # ── Downsample to reduce memory usage ────────────────────
    # Original is typically 512x512xD — resize to 128x128xD
    volume = volume[::4, ::4, ::2]   # ← ADD THIS LINE
    # ─────────────────────────────────────────────────────────

    # Shape → (1, 1, H, W, D)
    tensor = torch.tensor(volume, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor, img.affine, volume


# ── Finding classification (keyword matching) ─────────────────────────────────

def classify_finding(finding_text: str) -> str:
    """Classify a radiology finding string using keyword matching."""
    text = finding_text.lower()

    if "nodule" in text:
        return "Lung Nodule"
    elif "opacity" in text or "opacit" in text:
        return "Lung opacity"
    elif "consolidation" in text or "consolidat" in text:
        return "Consolidation"
    elif "atelectasis" in text or "collapse" in text or "atelectat" in text:
        return "Atelectasis"
    else:
        return "Others"


def classify_all_findings(findings: dict) -> dict:
    """Classify every finding in a sample and return an annotated dict."""
    classified = {}
    for idx, text in findings.items():
        category        = classify_finding(text)
        classified[idx] = {"text": text, "category": category}
    return classified


# ── Localisation mask ─────────────────────────────────────────────────────────

def build_localization_mask(volume_norm, emb_np):
    """Create a binary activation mask from the Merlin embedding (top-5% voxels)."""
    H, W, D    = volume_norm.shape
    emb_scalar = float(np.mean(np.abs(emb_np.flatten())))
    act        = np.zeros((H, W, D), dtype=np.float32)
    for d in range(D):
        act[:, :, d] = volume_norm[:, :, d] * emb_scalar
    threshold   = np.percentile(act, 95)
    binary_mask = (act >= threshold).astype(np.uint8)
    return binary_mask, threshold

def save_localization_png(volume_norm, binary_mask, out_dir, name):
    """Save middle axial, coronal, and sagittal slices with mask overlay as PNG."""
    H, W, D = volume_norm.shape
    mid_axial    = D // 2
    mid_coronal  = H // 2
    mid_sagittal = W // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Localization Mask — {name}", fontsize=10)

    # Axial slice (top-down view)
    axes[0].imshow(volume_norm[:, :, mid_axial],    cmap="gray", origin="lower")
    axes[0].imshow(binary_mask[:, :, mid_axial],    cmap="Reds", alpha=0.4, origin="lower")
    axes[0].set_title(f"Axial (slice {mid_axial})")
    axes[0].axis("off")

    # Coronal slice (front view)
    axes[1].imshow(volume_norm[mid_coronal, :, :].T, cmap="gray", origin="lower")
    axes[1].imshow(binary_mask[mid_coronal, :, :].T, cmap="Reds", alpha=0.4, origin="lower")
    axes[1].set_title(f"Coronal (slice {mid_coronal})")
    axes[1].axis("off")

    # Sagittal slice (side view)
    axes[2].imshow(volume_norm[:, mid_sagittal, :].T, cmap="gray", origin="lower")
    axes[2].imshow(binary_mask[:, mid_sagittal, :].T, cmap="Reds", alpha=0.4, origin="lower")
    axes[2].set_title(f"Sagittal (slice {mid_sagittal})")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "localization.png"), dpi=150, bbox_inches="tight")
    plt.close()

# ── Per-volume processing ─────────────────────────────────────────────────────

def process_one_volume(sample, model, ct_index, out_dir, device):
    """Run the full pipeline for a single CT volume and save all outputs."""
    name     = sample["name"]
    findings = sample["findings"]
    shape    = sample["shape"]
    cats     = sample.get("categories", {})
    pixels   = sample.get("pixels", {})
    protocol = sample.get("protocol", "unknown")

    stem = name.replace(".nii.gz", "")
    out  = os.path.join(out_dir, stem)

    # Skip if already processed
    if os.path.exists(os.path.join(out, "findings.json")):
        return "skipped"

    os.makedirs(out, exist_ok=True)

    try:
        # Load and move CT to device
        ct_tensor, affine, volume_norm = load_and_preprocess_ct(ct_index[name])
        ct_tensor = ct_tensor.to(device)

        # Merlin embedding
        with torch.no_grad():
            embedding = model(ct_tensor)
        emb_np = embedding.cpu().numpy()

        # Free GPU memory immediately
        del ct_tensor, embedding
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Classify findings via keyword matching
        classified = classify_all_findings(findings)

        # Build localisation mask
        binary_mask, threshold = build_localization_mask(volume_norm, emb_np)
        save_localization_png(volume_norm, binary_mask, out, name)

        # Build text report
        finding_lines = []
        for idx, info in classified.items():
            finding_lines.append(
                f"  Finding {int(idx)+1}\n"
                f"    Classification : {info['category']}\n"
                f"    Category code  : {cats.get(idx, 'N/A')}\n"
                f"    Pixel count    : {pixels.get(idx, 'N/A')}\n"
                f"    Text           : {info['text']}"
            )

        report = f"""
MERLIN CT ANALYSIS REPORT
{'='*55}
File       : {name}
Protocol   : {protocol}
Shape      : {shape[0]} x {shape[1]} x {shape[2]} voxels

EMBEDDING
  Shape    : {emb_np.shape}
  Norm     : {np.linalg.norm(emb_np):.4f}
  Mean     : {np.mean(emb_np):.4f}
  Std      : {np.std(emb_np):.4f}

FINDINGS ({len(findings)} total)
{chr(10).join(finding_lines)}

LOCALIZATION MASK
  Active voxels : {binary_mask.sum()}
  Threshold     : {threshold:.4f} (top 5% activation)
  Coverage      : {100*binary_mask.sum()/volume_norm.size:.2f}% of volume
{'='*55}
"""

        # Save outputs
        with open(os.path.join(out, "report.txt"), "w") as f:
            f.write(report)

        np.savez_compressed(
            os.path.join(out, "embedding.npz"),
            embedding=emb_np,
        )

        nib.save(
            nib.Nifti1Image(binary_mask, affine),
            os.path.join(out, "localization_mask.nii.gz"),
        )

        findings_out = {
            "name"               : name,
            "protocol"           : protocol,
            "shape"              : shape,
            "classified_findings": classified,
            "categories"         : cats,
            "pixels"             : pixels,
            "embedding_norm"     : float(np.linalg.norm(emb_np)),
            "mask_active_voxels" : int(binary_mask.sum()),
        }
        with open(os.path.join(out, "findings.json"), "w") as f:
            json.dump(findings_out, f, indent=2)

        return "done"

    except Exception as e:
        with open(os.path.join(out, "error.txt"), "w") as f:
            f.write(str(e))
        return f"error: {e}"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Index CT files on disk
    print("Indexing CT files...")
    ct_index = {}
    for root, _, files in os.walk(VOLUMES_DIR):
        for f in files:
            if f.endswith(".nii.gz"):
                ct_index[f] = os.path.join(root, f)
    print(f"CT files on disk : {len(ct_index)}")

    # Load metadata and match to disk files
    with open(DATASET_JSON) as f:
        metadata = json.load(f)

    samples = metadata if isinstance(metadata, list) else list(metadata.values())[0]
    matched = [s for s in samples if s["name"] in ct_index]

    print(f"Total in JSON    : {len(samples)}")
    print(f"Matched on disk  : {len(matched)}")
    print(f"Missing          : {len(samples) - len(matched)}")

    # ── TEST MODE: process only 1 volume ──────────────────────────────────────
    # Remove or comment out the next line to run the full dataset
    matched = matched[:1]
    # ─────────────────────────────────────────────────────────────────────────

    # Load Merlin model
    from merlin import Merlin  # noqa: PLC0415

    model = Merlin(ImageEmbedding=True)
    model.to(device)
    model.eval()
    print(f"Merlin loaded on {device}")

    # Run pipeline
    total      = len(matched)
    done       = 0
    skipped    = 0
    errors     = []
    log_path   = os.path.join(RESULTS_DIR, "run_log.txt")
    start_time = time.time()

    print(f"Starting full pipeline — {total} volumes")
    print(f"Results → {RESULTS_DIR}")
    print("=" * 55)

    with open(log_path, "a") as log:
        for i, sample in enumerate(matched):
            name   = sample["name"]
            status = process_one_volume(sample, model, ct_index, RESULTS_DIR, device)

            if status == "done":
                done += 1
            elif status == "skipped":
                skipped += 1
            else:
                errors.append({"name": name, "error": status})

            if (i + 1) % 10 == 0 or (i + 1) == total:
                elapsed   = time.time() - start_time
                per_vol   = elapsed / (i + 1)
                remaining = per_vol * (total - i - 1)
                print(
                    f"  [{i+1:4d}/{total}]  done={done}  "
                    f"skip={skipped}  err={len(errors)}  "
                    f"ETA={remaining/60:.1f}min"
                )
                log.write(f"[{i+1}/{total}] {name} → {status}\n")

    print("\n" + "=" * 55)
    print("FINISHED")
    print(f"  Processed  : {done}")
    print(f"  Skipped    : {skipped}  (already done)")
    print(f"  Errors     : {len(errors)}")
    print(f"  Total time : {(time.time()-start_time)/60:.1f} min")

    if errors:
        errors_path = os.path.join(RESULTS_DIR, "errors.json")
        with open(errors_path, "w") as f:
            json.dump(errors, f, indent=2)
        print(f"  Error log  : {errors_path}")

        # One automatic retry pass
        print(f"Retrying {len(errors)} failed volumes...")
        failed_names  = {e["name"] for e in errors}
        retry_samples = [s for s in matched if s["name"] in failed_names]
        retry_errors  = []

        for sample in retry_samples:
            stem     = sample["name"].replace(".nii.gz", "")
            err_file = os.path.join(RESULTS_DIR, stem, "error.txt")
            if os.path.exists(err_file):
                os.remove(err_file)

            status = process_one_volume(sample, model, ct_index, RESULTS_DIR, device)
            print(f"  {sample['name']} → {status}")
            if status.startswith("error"):
                retry_errors.append({"name": sample["name"], "error": status})

        print(f"Retry done. Still failing: {len(retry_errors)}")


if __name__ == "__main__":
    main()