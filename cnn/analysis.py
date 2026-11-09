import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

plotsd_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plots")
os.makedirs(plotsd_dir, exist_ok=True)

def _save(name):
    path = os.path.join(plotsd_dir, name)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()

def tsne_2d(features, seed=42):
    return TSNE(n_components=2, init="pca", perplexity=30, random_state=seed).fit_transform(features)


def plot_tsne(points, labels, title, filename, errors=None):
    plt.figure(figsize=(8, 7))
    for c in range(10):
        m = labels == c
        plt.scatter(points[m, 0], points[m, 1], s=4, alpha=0.6, label=str(c), color=plt.cm.tab10(c))
        # подпись класса в центре его облака
        cx, cy = np.median(points[m], axis=0)
        plt.text(cx, cy, str(c), fontsize=14, weight="bold")
    if errors is not None and errors.any():
        plt.scatter(points[errors, 0], points[errors, 1], s=30, marker="x", color="black",
                    label="ошибка классификации")
    plt.title(title)
    plt.legend(markerscale=3, fontsize=8, loc="best")
    _save(filename)


def plot_loss(history, target):
    plt.figure(figsize=(7, 4))
    plt.semilogy(range(1, len(history) + 1), history, marker="o", label="средняя ошибка на обучении")
    plt.axhline(target, color="red", linestyle="--", label=f"цель = {target}")
    plt.xlabel("эпоха")
    plt.ylabel("категориальная кросс-энтропия (лог. шкала)")
    plt.title("Средняя ошибка на обучающей выборке по эпохам")
    plt.legend()
    _save("loss.png")


def plot_confusion(cm):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar(label="число объектов")
    for i in range(10):
        for j in range(10):
            plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.xlabel("предсказанный класс")
    plt.ylabel("истинный класс")
    plt.title("Матрица неточностей (тестовая выборка)")
    _save("confusion_matrix.png")


def plot_roc(curves):
    plt.figure(figsize=(7, 6))
    for c, (fpr, tpr, a) in curves.items():
        plt.plot(fpr, tpr, label=f"цифра {c}: AUC = {a:.5f}", color=plt.cm.tab10(c))
    plt.plot([0, 1], [0, 1], "k--", label="случайный классификатор")
    plt.xlabel("FPR (доля ложноположительных)")
    plt.ylabel("TPR (полнота)")
    plt.title("ROC-кривые, один-против-всех (тест)")
    plt.legend(fontsize=8, loc="lower right")
    _save("roc.png")

    plt.figure(figsize=(7, 6))
    for c, (fpr, tpr, a) in curves.items():
        plt.plot(fpr, tpr, label=f"цифра {c}", color=plt.cm.tab10(c))
    plt.xlim(0, 0.05)
    plt.ylim(0.9, 1.0)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC-кривые, увеличенный левый верхний угол")
    plt.legend(fontsize=8, loc="lower right")
    _save("roc_zoom.png")


def plot_errors(images, y_true, y_pred, probs, n=24):
    wrong = np.where(y_true != y_pred)[0]
    # самые уверенные ошибки: сеть уверена, но неправа
    wrong = wrong[np.argsort(-probs[wrong, y_pred[wrong]])][:n]
    cols = 8
    rows = int(np.ceil(len(wrong) / cols))
    plt.figure(figsize=(cols * 1.6, rows * 2.4 + 0.6))
    for k, idx in enumerate(wrong):
        plt.subplot(rows, cols, k + 1)
        plt.imshow(images[idx], cmap="gray")
        plt.title(f"ист {y_true[idx]} → {y_pred[idx]}\np={probs[idx, y_pred[idx]]:.2f}", fontsize=8)
        plt.axis("off")
    plt.suptitle("Ошибочно классифицированные примеры (самые уверенные ошибки)")
    _save("errors.png")


def top_confusions(cm, k=10):
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = np.dstack(np.unravel_index(np.argsort(-off, axis=None), off.shape))[0][:k]
    return [(int(i), int(j), int(off[i, j])) for i, j in pairs if off[i, j] > 0]
