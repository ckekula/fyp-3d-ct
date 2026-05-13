import numpy as np
from ..metrics.localization_metrics import iou_score, dice_score


def attribution_mask_iou(attribution_map, gt_mask, threshold=0.5):
    attr = normalize_map(attribution_map)
    attr_mask = attr >= threshold
    return iou_score(attr_mask, gt_mask)


def pointing_game(attribution_map, gt_mask):
    attr = np.asarray(attribution_map)
    gt = np.asarray(gt_mask).astype(bool)

    if gt.sum() == 0:
        return np.nan

    max_index = np.unravel_index(np.argmax(attr), attr.shape)
    return float(gt[max_index])


def energy_inside_mask(attribution_map, gt_mask, eps=1e-8):
    attr = np.asarray(attribution_map, dtype=float)
    attr = np.maximum(attr, 0.0)
    gt = np.asarray(gt_mask).astype(bool)

    total_energy = attr.sum()

    if total_energy <= eps:
        return np.nan

    return float(attr[gt].sum() / total_energy)


def grounded_accuracy(samples, dice_threshold=0.10):
    valid = []

    for sample in samples:
        correct = int(sample.y_true == sample.y_pred)
        dsc = dice_score(sample.attribution_map >= 0.5, sample.gt_mask)
        grounded = int(correct == 1 and dsc > dice_threshold)
        valid.append((correct, grounded))

    correct_count = sum(c for c, _ in valid)

    if correct_count == 0:
        return np.nan

    grounded_correct = sum(g for _, g in valid)
    return float(grounded_correct / correct_count)


def normalize_map(x, eps=1e-8):
    x = np.asarray(x, dtype=float)
    x = x - np.nanmin(x)
    max_value = np.nanmax(x)

    if max_value <= eps:
        return np.zeros_like(x)

    return x / max_value