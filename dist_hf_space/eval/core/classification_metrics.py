import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    accuracy_score,
)


def safe_auroc(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if len(np.unique(y_true)) < 2:
        return np.nan

    return roc_auc_score(y_true, y_score)


def compute_ece(y_true, y_score, n_bins=10):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lower, upper = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_score >= lower) & (y_score <= upper)
        else:
            mask = (y_score >= lower) & (y_score < upper)

        if mask.sum() == 0:
            continue

        confidence = y_score[mask].mean()
        accuracy = y_true[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(confidence - accuracy)

    return float(ece)


def compute_classification_metrics(samples, class_names, threshold=0.5):
    results = {}

    # Rank-based metrics (AUROC, AP, ECE) require a real confidence
    # distribution. If any sample's score is a hard 0/1 decision rather than
    # a probability, those metrics are not meaningful and must be reported
    # as N/A instead of computed on a degenerate score.
    score_types = {getattr(s, "score_type", "probabilistic") for s in samples}
    is_probabilistic = score_types == {"probabilistic"}

    for class_name in class_names:
        y_true = [s.y_true[class_name] for s in samples]
        y_score = [s.y_score[class_name] for s in samples]
        y_pred = [1 if score >= threshold else 0 for score in y_score]
        support = int(sum(y_true))

        if support == 0:
            # No positive ground truth for this class in the evaluated set
            # (e.g. a model/test split with no examples of this abnormality).
            # precision/recall/f1 would silently be zero_division defaults
            # here, which reads as "the model failed this class" rather than
            # "this class was never evaluated" -- report N/A instead so it's
            # excluded from the macro average rather than dragging it down.
            results[class_name] = {
                "auroc": np.nan,
                "average_precision": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "f1": np.nan,
                "accuracy": np.nan,
                "ece": np.nan,
                "support": 0,
                "not_evaluated": True,
            }
            continue

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0,
        )

        if is_probabilistic:
            ap = average_precision_score(y_true, y_score) if len(set(y_true)) > 1 else np.nan
            auroc = safe_auroc(y_true, y_score)
            ece = compute_ece(y_true, y_score)
        else:
            ap = np.nan
            auroc = np.nan
            ece = np.nan

        results[class_name] = {
            "auroc": auroc,
            "average_precision": float(ap) if ap is not None else np.nan,
            # "support" here is the positive-class count (sum(y_true)), not
            # the sklearn-report convention of total sample count -- kept as
            # "support" for output-file compatibility, but don't read it as
            # len(y_true).
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "ece": ece,
            "support": support,
        }

    if is_probabilistic:
        macro_keys = ["auroc", "average_precision", "precision", "recall", "f1", "ece"]
        results["macro"] = {
            key: float(np.nanmean([results[c][key] for c in class_names]))
            for key in macro_keys
        }
        results["macro"]["score_type"] = "probabilistic"
    else:
        macro_keys = ["precision", "recall", "f1"]
        results["macro"] = {
            key: float(np.nanmean([results[c][key] for c in class_names]))
            for key in macro_keys
        }
        results["macro"]["auroc"] = None
        results["macro"]["average_precision"] = None
        results["macro"]["ece"] = None
        results["macro"]["score_type"] = "hard_label"
        results["macro"]["note"] = (
            "auroc/average_precision/ece are N/A: model produces hard 0/1 "
            "decisions, not calibrated probabilities, so rank-based metrics "
            "are not computable and are excluded from the macro average."
        )

    return results