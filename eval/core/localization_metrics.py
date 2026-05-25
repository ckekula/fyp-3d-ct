import numpy as np


def binarize(mask, threshold=0.5):
    return np.asarray(mask) >= threshold


def dice_score(pred, gt, eps=1e-8):
    pred = binarize(pred)
    gt = binarize(gt)

    intersection = np.logical_and(pred, gt).sum()
    denominator = pred.sum() + gt.sum()

    if denominator == 0:
        return np.nan

    return float((2.0 * intersection + eps) / (denominator + eps))


def iou_score(pred, gt, eps=1e-8):
    pred = binarize(pred)
    gt = binarize(gt)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return np.nan

    return float((intersection + eps) / (union + eps))


def hit_at_k(dice_values, k):
    dice_values = np.asarray(dice_values, dtype=float)
    dice_values = dice_values[~np.isnan(dice_values)]

    if len(dice_values) == 0:
        return np.nan

    return float(np.mean(dice_values > k))


def compute_localization_metrics(samples):
    per_case = []

    for sample in samples:
        dsc = dice_score(sample.pred_mask, sample.gt_mask)
        iou = iou_score(sample.pred_mask, sample.gt_mask)

        per_case.append({
            "case_id": sample.case_id,
            "class_name": sample.class_name,
            "model_name": sample.model_name,
            "dice": dsc,
            "iou": iou,
            "morphology": sample.morphology,
        })

    dice_values = [row["dice"] for row in per_case]

    summary = {
        "mean_dice": float(np.nanmean(dice_values)),
        "mean_iou": float(np.nanmean([row["iou"] for row in per_case])),
        "hit_at_5": hit_at_k(dice_values, 0.05),
        "hit_at_10": hit_at_k(dice_values, 0.10),
        "hit_at_25": hit_at_k(dice_values, 0.25),
        "num_cases": len(per_case),
    }

    return {
        "summary": summary,
        "per_case": per_case,
    }