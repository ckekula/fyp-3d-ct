import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import gc
import json
import time
import urllib.request
import traceback

import numpy as np
import nibabel as nib

import torch
import torch.nn.functional as F
torch.cuda.empty_cache()

PROJECT_ROOT = os.path.expanduser("/home/chest_ct/code")
DATA_ROOT    = os.path.join(PROJECT_ROOT, "data")
VOLUMES_DIR  = os.path.join(DATA_ROOT, "segmentations")
REXCT_DIR    = os.path.join(DATA_ROOT, "rexgrounding-ct")
DATASET_JSON = os.path.join(REXCT_DIR, "dataset.json")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "models/merlin/results_full")

os.makedirs(RESULTS_DIR, exist_ok=True)

print("PROJECT ROOT :", PROJECT_ROOT)
print("RESULTS DIR  :", RESULTS_DIR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device : {DEVICE}")

if DEVICE.type == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(
        f"VRAM   : "
        f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    )

print("Indexing CT files...")
ct_index = {}

for root, dirs, files in os.walk(VOLUMES_DIR):
    for f in files:
        if f.endswith(".nii.gz"):
            ct_index[f] = os.path.join(root, f)

print(f"CT files on disk : {len(ct_index)}")

with open(DATASET_JSON) as f:
    metadata = json.load(f)

samples = metadata["val"]

matched = [
    s for s in samples
    if s["name"] in ct_index
]

print(f"Total in JSON    : {len(samples)}")
print(f"Matched on disk  : {len(matched)}")
print(f"Missing          : {len(samples) - len(matched)}")

from merlin import Merlin

model = Merlin()
model.to(DEVICE)
model.eval()

# Register a forward hook to get dense features from layer4
activation = {}
def get_activation(name):
    def hook(mod, input, output):
        activation[name] = output.detach()
    return hook

model.model.encode_image.i3_resnet.layer4.register_forward_hook(get_activation('layer4'))

print(f"Merlin loaded on {DEVICE}")

def load_and_preprocess_ct(nii_path):
    img = nib.load(nii_path)
    volume = img.get_fdata(dtype=np.float32)
    np.nan_to_num(volume, nan=0.0, copy=False)

    if volume.ndim == 4:
        volume = np.mean(volume, axis=0, dtype=np.float32)

    volume = np.squeeze(volume)
    assert volume.ndim == 3, f"Expected 3D volume, got {volume.shape}"

    np.clip(volume, -1000, 400, out=volume)
    volume += 1000.0
    volume /= 1400.0

    tensor = torch.tensor(volume, dtype=torch.float32)
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    
    affine = img.affine.copy()
    del img

    return tensor, affine, volume

VALID_CATEGORIES = [
    "Lung Nodule",
    "Lung opacity",
    "Consolidation",
    "Atelectasis"
]

def classify_finding(finding_text: str) -> str:
    text_lower = finding_text.lower()
    if "nodule" in text_lower:
        return "Lung Nodule"
    elif "opacity" in text_lower or "opacities" in text_lower:
        return "Lung opacity"
    elif "consolidation" in text_lower:
        return "Consolidation"
    elif "atelectasis" in text_lower:
        return "Atelectasis"
    return "Others"

def classify_all_findings(findings: dict):
    classified = {}
    for idx, text in findings.items():
        category = classify_finding(text)
        classified[idx] = {
            "text": text,
            "category": category
        }
    return classified

def process_one_volume(sample, model, ct_index, out_dir):
    name     = sample["name"]
    findings = sample["findings"]
    shape    = sample["shape"]
    cats     = sample.get("categories", {})
    pixels   = sample.get("pixels", {})
    protocol = sample.get("protocol", "unknown")

    stem = name.replace(".nii.gz", "")
    out = os.path.join(out_dir, stem)

    if os.path.exists(os.path.join(out, "findings.json")):
        return "skipped"

    os.makedirs(out, exist_ok=True)

    try:
        ct_tensor, affine, volume_norm = load_and_preprocess_ct(ct_index[name])
        orig_shape = volume_norm.shape
        del volume_norm
        
        ct_tensor = ct_tensor.to(DEVICE)

        with torch.no_grad():
            # Run image encoder
            image_features, _ = model.model.encode_image(ct_tensor)
            emb_np = image_features.cpu().numpy()
            
            # Extract dense spatial features from layer4 via contrastive head
            layer4_out = activation['layer4']
            dense_contrastive = model.model.encode_image.i3_resnet.contrastive_head(layer4_out) # [1, 512, D', H', W']
            dense_contrastive = dense_contrastive / dense_contrastive.norm(dim=1, keepdim=True)
            
            classified = classify_all_findings(findings)
            
            # Evaluate masks for each finding
            localization_masks = {}
            for idx, finding_info in classified.items():
                text = finding_info["text"]
                class_name = finding_info["category"]
                
                # Get text embedding for this finding
                text_emb = model.model.encode_text([text]) # [1, 512]
                text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
                
                # Compute cosine similarity
                similarity = torch.einsum('b c d h w, b c -> b d h w', dense_contrastive, text_emb) # [1, D', H', W']
                similarity = similarity.unsqueeze(1) # [1, 1, D', H', W']
                
                # Upsample similarity to original CT shape
                similarity_up = F.interpolate(similarity, size=orig_shape, mode='trilinear', align_corners=False)
                similarity_up = similarity_up.squeeze().cpu().numpy()
                
                # Min-max normalization per class map
                sim_min, sim_max = similarity_up.min(), similarity_up.max()
                if sim_max > sim_min:
                    similarity_up = (similarity_up - sim_min) / (sim_max - sim_min)
                
                # Save mask based on normalized category name to match eval script
                cat_normalized = class_name.lower().replace(" ", "_")
                if cat_normalized == "others":
                    cat_normalized = text[:20].lower().replace(" ", "_") # Fallback
                localization_masks[cat_normalized + "_soft"] = similarity_up

        del ct_tensor
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        # Save class-specific localization masks as npz
        np.savez_compressed(
            os.path.join(out, "localization_masks.npz"),
            **localization_masks
        )

        np.savez_compressed(
            os.path.join(out, "embedding.npz"),
            embedding=emb_np
        )

        findings_out = {
            "name": name,
            "protocol": protocol,
            "shape": shape,
            "classified_findings": classified,
            "categories": cats,
            "pixels": pixels,
            "embedding_norm": float(np.linalg.norm(emb_np))
        }

        with open(os.path.join(out, "findings.json"), "w") as f:
            json.dump(findings_out, f, indent=2)

        return "done"

    except Exception as e:
        traceback.print_exc()
        with open(os.path.join(out, "error.txt"), "w") as f:
            f.write(str(e))
        return f"error: {e}"

total    = len(matched)
done     = 0
skipped  = 0
errors   = []

log_path = os.path.join(RESULTS_DIR, "run_log.txt")

print(f"Starting full pipeline — {total} volumes")

start_time = time.time()

with open(log_path, "a") as log:
    for i, sample in enumerate(matched):
        name = sample["name"]
        status = process_one_volume(sample, model, ct_index, RESULTS_DIR)

        if status == "done":
            done += 1
        elif status == "skipped":
            skipped += 1
        else:
            errors.append({"name": name, "error": status})

        if (i + 1) % 10 == 0 or (i + 1) == total:
            elapsed = time.time() - start_time
            per_vol = elapsed / (i + 1)
            remaining = per_vol * (total - i - 1)
            
            print(
                f"[{i+1}/{total}] "
                f"done={done} "
                f"skip={skipped} "
                f"err={len(errors)} "
                f"ETA={remaining/60:.1f}min"
            )

            log.write(
                f"[{i+1}/{total}] "
                f"{name} -> {status}\n"
            )
            log.flush()

print("\nFINISHED")
print(f"Processed : {done}")
print(f"Skipped   : {skipped}")
print(f"Errors    : {len(errors)}")

if errors:
    with open(os.path.join(RESULTS_DIR, "errors.json"), "w") as f:
        json.dump(errors, f, indent=2)
