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
        mask = (y_score >= lower) & (y_score < upper)

        if mask.sum() == 0:
            continue

        confidence = y_score[mask].mean()
        accuracy = y_true[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(confidence - accuracy)

    return float(ece)


def compute_classification_metrics(samples, class_names, threshold=0.5):
    results = {}

    for class_name in class_names:
        y_true = [s.y_true[class_name] for s in samples]
        y_score = [s.y_score[class_name] for s in samples]
        y_pred = [1 if score >= threshold else 0 for score in y_score]

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0,
        )

        ap = average_precision_score(y_true, y_score) if len(set(y_true)) > 1 else np.nan

        results[class_name] = {
            "auroc": safe_auroc(y_true, y_score),
            "average_precision": float(ap),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "ece": compute_ece(y_true, y_score),
            "support": int(sum(y_true)),
        }

    macro_keys = ["auroc", "average_precision", "precision", "recall", "f1", "ece"]
    results["macro"] = {
        key: float(np.nanmean([results[c][key] for c in class_names]))
        for key in macro_keys
    }

    return results