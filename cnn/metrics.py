import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)

def classification_metrics(y_true, y_pred, num_classes=10):
    labels = range(num_classes)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0,
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0,
    )

    return cm, {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision, "recall": recall, "f1": f1,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
    }


def roc_one_vs_rest(y_true, probs, num_classes=10):
    curves = {}
    for c in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true == c, probs[:, c], drop_intermediate=False)
        curves[c] = (fpr, tpr, auc(fpr, tpr))
    return curves
