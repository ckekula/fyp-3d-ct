import os
import sys
import json
import time
import torch
import numpy as np
import pandas as pd
import nibabel as nib

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rexground_pilot_pipeline import (
    load_ctclip_classifier, preprocess_ct, run_gradcam,
    DATASET_JSON, VOLUME_DIR, SEG_DIR, TRAIN_META, VALID_META,
    LUNG_NODULE_IDX, LUNG_OPACITY_IDX, DEVICE, ROOT,
    TARGET_SPACING, TARGET_SHAPE,
)
from data import resize_array

OUT_DIR = ROOT / "rexground_predictions/full"
os.makedirs(OUT_DIR, exist_ok=True)
PRED_CSV = OUT_DIR / "predictions_full.csv"
TOP20_CSV = OUT_DIR / "top20_for_gradcam.csv"
GRADCAM_SUMMARY_CSV = OUT_DIR / "gradcam_summary_top20.csv"

N_GRADCAM_PER_KIND = 10


def build_full_case_list():
    """Every dataset_2.json case tagged 2c and/or 2d, with a local volume+seg+metadata."""
    with open(DATASET_JSON) as f:
        data = json.load(f)
    available = set(os.listdir(VOLUME_DIR))
    seg_available = set(os.listdir(SEG_DIR))
    meta_set = set(pd.read_csv(TRAIN_META)["VolumeName"]) | set(pd.read_csv(VALID_META)["VolumeName"])

    cases = []
    for split_name, split_cases in data.items():
        for c in split_cases:
            name = c.get("name")
            if not name or name not in available or name not in seg_available or name not in meta_set:
                continue
            cats = set(c.get("categories", {}).values())
            if not cats or not cats.issubset({"2c", "2d"}):
                continue
            if cats == {"2d"}:
                kind = "nodule"
            elif cats == {"2c"}:
                kind = "ggo"
            else:
                kind = "mixed"
            cases.append({"split": split_name, "name": name, "kind": kind, "categories": sorted(cats), "case": c})
    return cases


def run_predictions(model, tokenizer, cases, train_meta, valid_meta):
    results = []
    t_start = time.time()
    for i, entry in enumerate(cases):
        name = entry["name"]
        try:
            meta_row = train_meta.loc[name] if name in train_meta.index else valid_meta.loc[name]
            image_tensor, _ = preprocess_ct(VOLUME_DIR / name, meta_row)
            image_tensor = image_tensor.to(DEVICE)

            text_tokens_empty = tokenizer(
                "", return_tensors="pt", padding="max_length", truncation=True, max_length=200
            ).to(DEVICE)
            with torch.no_grad():
                logits = model(text_tokens_empty, image_tensor, DEVICE)
                probs = torch.sigmoid(logits)[0].cpu().numpy()
                raw_logits = logits[0].cpu().numpy()

            findings_text = "; ".join(entry["case"].get("findings", {}).values())
            results.append({
                "name": name, "split": entry["split"], "kind": entry["kind"],
                "categories": ",".join(entry["categories"]),
                "pixels": sum(entry["case"].get("pixels", {}).values()),
                "findings": findings_text,
                "prob_lung_nodule": float(probs[LUNG_NODULE_IDX]),
                "prob_lung_opacity": float(probs[LUNG_OPACITY_IDX]),
                "logit_lung_nodule": float(raw_logits[LUNG_NODULE_IDX]),
                "logit_lung_opacity": float(raw_logits[LUNG_OPACITY_IDX]),
            })
        except Exception as e:
            results.append({"name": name, "split": entry["split"], "kind": entry["kind"], "error": str(e)})

        if (i + 1) % 25 == 0 or (i + 1) == len(cases):
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(cases) - (i + 1)) / rate if rate > 0 else 0
            print(f"[{i+1}/{len(cases)}] elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)
            pd.DataFrame(results).to_csv(PRED_CSV, index=False)

    return pd.DataFrame(results)


def select_top20(df):
    valid = df[df.get("error").isna()] if "error" in df.columns else df
    top = []
    nodule_pool = valid[valid["kind"].isin(["nodule", "mixed"])].sort_values("logit_lung_nodule", ascending=False)
    ggo_pool = valid[valid["kind"].isin(["ggo", "mixed"])].sort_values("logit_lung_opacity", ascending=False)
    for r in nodule_pool.head(N_GRADCAM_PER_KIND).to_dict("records"):
        r["gradcam_target"] = "nodule"
        top.append(r)
    for r in ggo_pool.head(N_GRADCAM_PER_KIND).to_dict("records"):
        r["gradcam_target"] = "ggo"
        top.append(r)
    return top


def gradcam_overlap(cam, seg_mask_orig, crop_info, meta_row):
    seg_zxy = np.transpose(seg_mask_orig.astype(np.float32), (2, 0, 1))
    seg_t = torch.tensor(seg_zxy).unsqueeze(0).unsqueeze(0)
    xy_spacing = float(meta_row["XYSpacing"][1:][:-2].split(",")[0])
    z_spacing = float(meta_row["ZSpacing"])
    seg_resized = resize_array(seg_t, (z_spacing, xy_spacing, xy_spacing), TARGET_SPACING)[0][0]
    seg_resized = np.transpose(seg_resized, (1, 2, 0))  # X,Y,Z

    dh, dw, dd = TARGET_SHAPE
    hs, ws, ds = crop_info["h_start"], crop_info["w_start"], crop_info["d_start"]
    seg_cropped = seg_resized[hs:hs + dh, ws:ws + dw, ds:ds + dd]
    seg_padded = np.pad(
        seg_cropped,
        ((crop_info["pad_h_before"], dh - seg_cropped.shape[0] - crop_info["pad_h_before"]),
         (crop_info["pad_w_before"], dw - seg_cropped.shape[1] - crop_info["pad_w_before"]),
         (crop_info["pad_d_before"], dd - seg_cropped.shape[2] - crop_info["pad_d_before"])),
        mode="constant", constant_values=0,
    )
    cam_xyz = np.transpose(cam, (1, 2, 0))
    gt_mask = seg_padded > 0.5

    best_dice, best_p = 0.0, 0
    for p in [70, 80, 90, 95]:
        thr = np.percentile(cam_xyz, p)
        cam_mask = cam_xyz >= thr
        inter = np.logical_and(cam_mask, gt_mask).sum()
        dice = 2 * inter / (cam_mask.sum() + gt_mask.sum() + 1e-8)
        if dice > best_dice:
            best_dice, best_p = dice, p

    max_point = np.unravel_index(np.argmax(cam_xyz), cam_xyz.shape)
    pointing_hit = bool(gt_mask[max_point]) if gt_mask.sum() > 0 else None
    return best_dice, best_p, pointing_hit


def main():
    cases = build_full_case_list()
    n_nodule = sum(1 for c in cases if c["kind"] == "nodule")
    n_ggo = sum(1 for c in cases if c["kind"] == "ggo")
    n_mixed = sum(1 for c in cases if c["kind"] == "mixed")
    print(f"Full case list: {len(cases)} total ({n_nodule} nodule-only, {n_ggo} ggo-only, {n_mixed} mixed)")

    model, tokenizer = load_ctclip_classifier()
    train_meta = pd.read_csv(TRAIN_META).set_index("VolumeName")
    valid_meta = pd.read_csv(VALID_META).set_index("VolumeName")

    print("Running predictions on full dataset...")
    df = run_predictions(model, tokenizer, cases, train_meta, valid_meta)
    df.to_csv(PRED_CSV, index=False)
    print(f"Saved full predictions to {PRED_CSV}")

    top20 = select_top20(df)
    top20_df = pd.DataFrame(top20)
    top20_df.to_csv(TOP20_CSV, index=False)
    print(f"\nTop {len(top20)} cases selected for Grad-CAM (ranked by target-pathology logit):")
    for r in top20:
        target = r["gradcam_target"]
        logit = r["logit_lung_nodule"] if target == "nodule" else r["logit_lung_opacity"]
        print(f"  [{target:6s}] {r['name']:30s} kind={r['kind']:6s} logit={logit:.3f}")

    print("\nRunning Grad-CAM on top 20 cases...")
    gradcam_rows = []
    for rec in top20:
        name = rec["name"]
        target = rec["gradcam_target"]
        target_idx = LUNG_NODULE_IDX if target == "nodule" else LUNG_OPACITY_IDX
        meta_row = train_meta.loc[name] if name in train_meta.index else valid_meta.loc[name]

        image_tensor, crop_info = preprocess_ct(VOLUME_DIR / name, meta_row)
        cam, probs = run_gradcam(model, tokenizer, image_tensor, target_idx)

        out_path = OUT_DIR / f"gradcam_{target}_{name.replace('.nii.gz', '')}.npy"
        np.save(out_path, cam.astype(np.float32))

        seg_nii = nib.load(str(SEG_DIR / name))
        seg = seg_nii.get_fdata()
        if seg.ndim == 4:
            seg = seg[0]
        seg_mask = seg > 0

        best_dice, best_p, pointing_hit = gradcam_overlap(cam, seg_mask, crop_info, meta_row)
        target_prob = probs[target_idx]
        print(f"  {target:6s} {name:30s} prob={target_prob:.3f} "
              f"best_dice@{best_p}%={best_dice:.4f} pointing_hit={pointing_hit}")

        gradcam_rows.append({
            **rec, "target_prob": float(target_prob), "gradcam_npy": str(out_path),
            "gradcam_best_dice": best_dice, "gradcam_best_pct": best_p,
            "gradcam_pointing_hit": pointing_hit,
        })

    gradcam_df = pd.DataFrame(gradcam_rows)
    gradcam_df.to_csv(GRADCAM_SUMMARY_CSV, index=False)
    print(f"\nSaved Grad-CAM summary to {GRADCAM_SUMMARY_CSV}")

    valid_gc = gradcam_df[gradcam_df["gradcam_pointing_hit"].notna()]
    print(f"\nAverage best Dice: {valid_gc['gradcam_best_dice'].mean():.4f}")
    print(f"Pointing accuracy: {valid_gc['gradcam_pointing_hit'].mean():.2%}")


if __name__ == "__main__":
    main()
