import numpy as np
from scipy import ndimage


def binarize(mask, threshold=0.5):
    return np.asarray(mask) >= threshold


def dice_score(pred, gt):
    pred = binarize(pred)
    gt = binarize(gt)

    intersection = np.logical_and(pred, gt).sum()
    denominator = pred.sum() + gt.sum()

    # denominator == 0 already guards the only division-by-zero case, so no
    # epsilon smoothing is needed -- an eps in the numerator/denominator was
    # previously making true zero-overlap cases read as ~1e-8 instead of a
    # clean 0.0 in per-case tables/plots.
    if denominator == 0:
        return np.nan

    return float(2.0 * intersection / denominator)


def iou_score(pred, gt):
    pred = binarize(pred)
    gt = binarize(gt)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return np.nan

    return float(intersection / union)


def match_instances(pred_mask, gt_mask, dice_threshold=0.2):
    """
    Connected-component instance matching between a predicted and a
    ground-truth mask, per ReXGroundingCT's official protocol: a predicted
    instance is a true positive if its Dice against some GT instance is
    >= dice_threshold (default 0.2), one-to-one (greedy, highest-Dice-first).

    Each LocalizationSample's pred_mask/gt_mask is already a whole-case
    binary volume for one class (findings of the same class in one case are
    OR'd together upstream in the adapters), so instances here are connected
    components within that per-class mask -- multiple physically separate
    lesions of the same class in one case are correctly counted as separate
    instances via this decomposition, without requiring any adapter change.

    Returns tp/fp/fn counts (not precision/recall directly) so callers can
    aggregate them across cases before dividing -- ReXGroundingCT's Instance
    Precision/Recall are corpus-level (TP / (TP+FP) over the whole test set),
    not a per-case average of per-case precision/recall.
    """
    pred = binarize(pred_mask)
    gt = binarize(gt_mask)

    pred_labeled, n_pred = ndimage.label(pred)
    gt_labeled, n_gt = ndimage.label(gt)

    if n_pred == 0 and n_gt == 0:
        return {"tp": 0, "fp": 0, "fn": 0, "n_pred_instances": 0, "n_gt_instances": 0}

    candidates = []
    for p in range(1, n_pred + 1):
        pm = pred_labeled == p
        p_size = pm.sum()
        for g in range(1, n_gt + 1):
            gm = gt_labeled == g
            denom = p_size + gm.sum()
            if denom == 0:
                continue
            dsc = 2.0 * np.logical_and(pm, gm).sum() / denom
            if dsc >= dice_threshold:
                candidates.append((dsc, p, g))

    candidates.sort(key=lambda x: x[0], reverse=True)

    matched_pred, matched_gt = set(), set()
    for _, p, g in candidates:
        if p in matched_pred or g in matched_gt:
            continue
        matched_pred.add(p)
        matched_gt.add(g)

    tp = len(matched_pred)
    return {
        "tp": tp,
        "fp": n_pred - tp,
        "fn": n_gt - tp,
        "n_pred_instances": n_pred,
        "n_gt_instances": n_gt,
    }


def aggregate_instance_metrics(match_rows, tp_key="tp", fp_key="fp", fn_key="fn", prefix="instance"):
    """Corpus-level Precision/Recall/F1 (TP/(TP+FP) etc. over the whole set,
    not a per-case average) from a list of match_instances()-shaped dicts."""
    tp = sum(r[tp_key] for r in match_rows)
    fp = sum(r[fp_key] for r in match_rows)
    fn = sum(r[fn_key] for r in match_rows)

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
        f1 = np.nan
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_f1": f1,
        f"{prefix}_tp": tp,
        f"{prefix}_fp": fp,
        f"{prefix}_fn": fn,
    }


def _surface_mask(mask):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask
    eroded = ndimage.binary_erosion(mask, border_value=0)
    return mask & ~eroded


def assd(mask_a, mask_b, spacing):
    """Average Symmetric Surface Distance in mm (spacing-aware). NaN if
    either mask is empty (no surface to measure)."""
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)
    surf_a = _surface_mask(mask_a)
    surf_b = _surface_mask(mask_b)

    if not surf_a.any() or not surf_b.any():
        return np.nan

    dt_b = ndimage.distance_transform_edt(~mask_b, sampling=spacing)
    dt_a = ndimage.distance_transform_edt(~mask_a, sampling=spacing)
    d_a_to_b = dt_b[surf_a]
    d_b_to_a = dt_a[surf_b]

    return float((d_a_to_b.sum() + d_b_to_a.sum()) / (len(d_a_to_b) + len(d_b_to_a)))


def centroid_distance(mask_a, mask_b, spacing):
    """Euclidean distance in mm between two masks' centers of mass."""
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)

    if not mask_a.any() or not mask_b.any():
        return np.nan

    com_a = np.array(ndimage.center_of_mass(mask_a)) * np.array(spacing)
    com_b = np.array(ndimage.center_of_mass(mask_b)) * np.array(spacing)
    return float(np.linalg.norm(com_a - com_b))


def match_instances_by_distance(pred_mask, gt_mask, spacing, morphology="non_focal"):
    """
    Distance-based instance matching per ReXGroundingCT's official protocol:
    a predicted instance is a true positive if its ASSD (non-focal findings,
    e.g. consolidation) or centroid distance (focal findings, e.g. nodules)
    to some GT instance is <= 2x max voxel spacing, matched greedily
    closest-first, one-to-one -- mirrors match_instances()'s Dice-based
    matching but using physical distance instead of overlap.

    NOTE: spacing must be real physical mm spacing for this to be
    meaningful. Some adapters (e.g. Merlin) currently hardcode spacing to
    (1.0, 1.0, 1.0) because real spacing isn't tracked in their prediction
    output -- for those, this threshold is in index units, not mm, and the
    resulting distance metrics should not be compared across models until
    that's fixed at the adapter level.
    """
    pred = binarize(pred_mask)
    gt = binarize(gt_mask)
    pred_labeled, n_pred = ndimage.label(pred)
    gt_labeled, n_gt = ndimage.label(gt)

    if n_pred == 0 and n_gt == 0:
        return {"tp": 0, "fp": 0, "fn": 0, "n_pred_instances": 0, "n_gt_instances": 0}

    threshold = 2.0 * max(spacing)
    use_centroid = morphology == "focal"

    candidates = []
    for p in range(1, n_pred + 1):
        pm = pred_labeled == p
        for g in range(1, n_gt + 1):
            gm = gt_labeled == g
            d = centroid_distance(pm, gm, spacing) if use_centroid else assd(pm, gm, spacing)
            if not np.isnan(d) and d <= threshold:
                candidates.append((d, p, g))

    candidates.sort(key=lambda x: x[0])

    matched_pred, matched_gt = set(), set()
    for _, p, g in candidates:
        if p in matched_pred or g in matched_gt:
            continue
        matched_pred.add(p)
        matched_gt.add(g)

    tp = len(matched_pred)
    return {
        "tp": tp,
        "fp": n_pred - tp,
        "fn": n_gt - tp,
        "n_pred_instances": n_pred,
        "n_gt_instances": n_gt,
    }


def frac_dice_above(dice_values, threshold, inclusive=False):
    """Fraction of findings whose Dice exceeds `threshold`.

    ReXGroundingCT's official Global HIT Rate uses Dice >= 0.10 (inclusive).
    Use inclusive=True to reproduce that; other thresholds here are informal
    sensitivity checks, not the official metric.
    """
    dice_values = np.asarray(dice_values, dtype=float)
    dice_values = dice_values[~np.isnan(dice_values)]

    if len(dice_values) == 0:
        return np.nan

    if inclusive:
        return float(np.mean(dice_values >= threshold))
    return float(np.mean(dice_values > threshold))


def compute_localization_metrics(samples, instance_dice_threshold=0.2, compute_distance_metrics=True):
    per_case = []

    for sample in samples:
        dsc = dice_score(sample.pred_mask, sample.gt_mask)
        iou = iou_score(sample.pred_mask, sample.gt_mask)
        match = match_instances(sample.pred_mask, sample.gt_mask, dice_threshold=instance_dice_threshold)

        row = {
            "case_id": sample.case_id,
            "class_name": sample.class_name,
            "model_name": sample.model_name,
            "dice": dsc,
            "iou": iou,
            "morphology": sample.morphology,
            **match,
        }

        if compute_distance_metrics:
            dist_match = match_instances_by_distance(
                sample.pred_mask,
                sample.gt_mask,
                spacing=sample.spacing,
                morphology=sample.morphology or "non_focal",
            )
            row.update({f"dist_{k}": v for k, v in dist_match.items()})

        per_case.append(row)

    dice_values = [row["dice"] for row in per_case]

    summary = {
        "mean_dice": float(np.nanmean(dice_values)) if per_case else np.nan,
        "mean_iou": float(np.nanmean([row["iou"] for row in per_case])) if per_case else np.nan,
        "global_hit_rate": frac_dice_above(dice_values, 0.10, inclusive=True),
        "frac_dice_gt_0.05": frac_dice_above(dice_values, 0.05),
        "frac_dice_gt_0.25": frac_dice_above(dice_values, 0.25),
        "num_cases": len(per_case),
        **aggregate_instance_metrics(per_case),
        "instance_dice_threshold": instance_dice_threshold,
    }

    if compute_distance_metrics and per_case:
        dist_rows = [
            {"tp": row["dist_tp"], "fp": row["dist_fp"], "fn": row["dist_fn"]}
            for row in per_case
        ]
        summary.update(aggregate_instance_metrics(dist_rows, prefix="distance"))

    return {
        "summary": summary,
        "per_case": per_case,
    }