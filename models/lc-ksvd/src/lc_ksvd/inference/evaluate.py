import logging
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


def evaluate(predictions, labels):
    acc = accuracy_score(labels, predictions)

    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(labels, predictions))

    print("\nConfusion Matrix:")
    print(confusion_matrix(labels, predictions))