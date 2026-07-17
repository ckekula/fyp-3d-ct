"""
metrics.py -- evaluation suite for a Grad-CAM output.

    A) Faithfulness (does the CAM actually matter to the model?)
       - Average Drop % / Increase in Confidence   (Chattopadhay et al., 2018)
       - Deletion / Insertion AUC                    (Petsiuk et al., 2018 - RISE)

    B) Localization (does the CAM agree with a reference mask?)
       - Dice / IoU
       - Pointing Game hit rate

    C) Sanity
       - Coverage (fraction of voxels above threshold, should be << 1.0)
"""

import numpy as np
import torch


def average_drop_increase(predictor, image_tensor, cam, target, text=None):
    with torch.no_grad():
        orig_output = predictor.forward(image_tensor, text=text)
        orig_score = target(orig_output).item()

        cam_t = torch.tensor(cam, dtype=torch.float32, device=image_tensor.device)
        cam_t = cam_t.unsqueeze(0).unsqueeze(0)
        masked_input = image_tensor * cam_t

        masked_output = predictor.forward(masked_input, text=text)
        masked_score = target(masked_output).item()

    drop_percent = max(0.0, orig_score - masked_score) / (abs(orig_score) + 1e-8) * 100
    increase = 1.0 if masked_score > orig_score else 0.0

    return {
        "orig_score": orig_score,
        "masked_score": masked_score,
        "average_drop_percent": drop_percent,
        "increase_in_confidence": increase,
    }


def deletion_insertion_auc(predictor, image_tensor, cam, target, text=None, steps=10):
    flat_cam = cam.flatten()
    order = np.argsort(-flat_cam)
    n_voxels = flat_cam.size
    chunk = max(1, n_voxels // steps)

    orig_flat = image_tensor.squeeze().detach().cpu().numpy().flatten()

    def score_at(mask_flat):
        vol = (orig_flat * mask_flat).reshape(image_tensor.shape[2:])
        t = torch.tensor(vol, dtype=torch.float32, device=image_tensor.device)
        t = t.unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            out = predictor.forward(t, text=text)
        return target(out).item()

    del_mask = np.ones(n_voxels, dtype=np.float32)
    deletion_scores = [score_at(del_mask)]
    for i in range(steps):
        idx = order[i * chunk:(i + 1) * chunk]
        del_mask[idx] = 0.0
        deletion_scores.append(score_at(del_mask))

    ins_mask = np.zeros(n_voxels, dtype=np.float32)
    insertion_scores = [score_at(ins_mask)]
    for i in range(steps):
        idx = order[i * chunk:(i + 1) * chunk]
        ins_mask[idx] = 1.0
        insertion_scores.append(score_at(ins_mask))

    return {
        "deletion_auc": float(np.trapz(deletion_scores) / len(deletion_scores)),
        "insertion_auc": float(np.trapz(insertion_scores) / len(insertion_scores)),
    }


def dice_iou_with_mask(cam, gt_mask, threshold=0.5):
    cam_bin = (cam >= threshold).astype(np.uint8)
    gt_bin = (gt_mask >= threshold).astype(np.uint8)

    intersection = np.logical_and(cam_bin, gt_bin).sum()
    union = np.logical_or(cam_bin, gt_bin).sum()

    dice = (2 * intersection) / (cam_bin.sum() + gt_bin.sum() + 1e-8)
    iou = intersection / (union + 1e-8)

    return {"dice": float(dice), "iou": float(iou)}


def pointing_game(cam, gt_mask):
    idx = np.unravel_index(np.argmax(cam), cam.shape)
    hit = bool(gt_mask[idx] > 0)
    return {"pointing_game_hit": hit}


def cam_coverage(cam, threshold=0.5):
    return {"coverage_fraction": float((cam >= threshold).mean())}


def evaluate_gradcam(predictor, image_tensor, cam, target, text=None,
                      gt_mask=None, deletion_steps=10):
    metrics = {}
    metrics.update(average_drop_increase(predictor, image_tensor, cam, target, text=text))
    metrics.update(deletion_insertion_auc(predictor, image_tensor, cam, target,
                                           text=text, steps=deletion_steps))
    metrics.update(cam_coverage(cam))

    if gt_mask is not None:
        metrics.update(dice_iou_with_mask(cam, gt_mask))
        metrics.update(pointing_game(cam, gt_mask))

    return metrics
