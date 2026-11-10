import numpy as np

def confusion_matrix(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def classification_metrics(y_true, y_pred, num_classes=10):
    cm = confusion_matrix(y_true, y_pred, num_classes)

    tp = np.diag(cm).astype(float)      # верно найденные объекты класса c
    fp = cm.sum(axis=0) - tp            # чужие объекты, отнесённые к классу c (столбец без диагонали)
    fn = cm.sum(axis=1) - tp            # объекты класса c, отнесённые к другим (строка без диагонали)

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(precision + recall > 0,
                      2 * precision * recall / (precision + recall), 0.0)

    return cm, {
        "accuracy": tp.sum() / cm.sum(),
        "precision": precision, "recall": recall, "f1": f1,
        "precision_macro": precision.mean(),
        "recall_macro": recall.mean(),
        "f1_macro": f1.mean(),
    }

def roc_curve_binary(is_positive, scores):
    is_positive = np.asarray(is_positive, dtype=bool)
    order = np.argsort(-scores, kind="mergesort")
    s, pos = scores[order], is_positive[order]

    tp_cum = np.cumsum(pos)
    fp_cum = np.cumsum(~pos)
    # берём только последние позиции перед сменой значения оценки
    last = np.r_[np.where(np.diff(s) != 0)[0], len(s) - 1]

    tpr = np.r_[0, tp_cum[last] / pos.sum()]
    fpr = np.r_[0, fp_cum[last] / (~pos).sum()]
    return fpr, tpr

def auc_trapezoid(fpr, tpr):
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))

def roc_one_vs_rest(y_true, probs, num_classes=10):
    curves = {}
    for c in range(num_classes):
        fpr, tpr = roc_curve_binary(y_true == c, probs[:, c])
        curves[c] = (fpr, tpr, auc_trapezoid(fpr, tpr))
    return curves