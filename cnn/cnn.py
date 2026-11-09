import mnist
import numpy as np
from conv import convolutional_layer
from maxpool import MaxPoll
from softmax import SoftMax
import mnist
import matplotlib.pyplot as plt
from dense import Dense
import analysis
import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

np.random.seed(42)

mnist.datasets_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"

images = np.concatenate((mnist.train_images(), mnist.test_images()), axis = 0)
labels = np.concatenate((mnist.train_labels(), mnist.test_labels()), axis = 0)

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.3, train_size=0.7, random_state=42, stratify=labels)

print("Обучающая выборка:", train_images.shape)
print("Тестовая выборка:", test_images.shape)

conv = convolutional_layer(num_filters=6, kernel_size=5)
pool = MaxPoll()
dense = Dense(14 * 14 * 6, 128)
softmax = SoftMax(128, 10)

# до обучения
tsne_n = 3000
analysis.plot_tsne(analysis.tsne_2d(test_images[:tsne_n].reshape(tsne_n, -1) / 255), test_labels[:tsne_n], "t-SNE исходных изображений (пиксели), до обучения","tsne_raw.png",)


# feed forward
def forward(image, label):
  image = np.pad(image, ((2, 2), (2, 2)), mode="constant")

  out = conv.forward((image / 255) - 0.5)
  out = pool.forward(out)
  out = dense.forward(out)
  out = softmax.forward(out)

  # Calculate cross-entropy loss and accuracy. np.log() is the natural log.
#   loss = -np.log(out[label])
  loss = -softmax.last_log_probs[label]

  acc = 1 if np.argmax(out) == label else 0

  return out, loss, acc


def train(im, label, lr=0.005):
    out, loss, acc = forward(im, label)

    gradient = softmax.backprob(label, lr)
    gradient = dense.backprob(gradient, lr)
    gradient = pool.backprob(gradient)
    conv.backprob(gradient, lr)

    return loss, acc

def evaluate(images, labels):
  total_loss = 0
  num_correct = 0

  for image, label in zip(images, labels):
    _, loss, acc = forward(image, label)
    total_loss += loss # прибавляет ошибку к картинке
    num_correct += acc # увеливает число верных ответов, если цифра угадана

  return total_loss / len(images), num_correct / len(images)

print('MNIST CNN initialized!')

target_loss = 0.0001
train_loss_history = []
max_epochs = 60
lr_start = 0.01

for epoch in range(max_epochs):
  lr = lr_start / (1 + 0.1 * epoch)
  print('Epoch %d (lr = %.5f)' % (epoch + 1, lr))
  permutation = np.random.permutation(len(train_images))
  train_images = train_images[permutation]
  train_labels = train_labels[permutation]

  # Train!
  loss = 0
  num_correct = 0

  for i, (im, label) in enumerate(zip(train_images, train_labels)):
    l, acc = train(im, label, lr)
    loss += l
    num_correct += acc

    if (i + 1) % 10000 == 0:
      print(
          f"[Step {i + 1}] Past 10000 steps: "
          f"Average Loss {loss / 10000:.5f} | "
          f"Accuracy: {num_correct / 100:.2f}%"
      )
      loss = 0
      num_correct = 0

  mean_loss, mean_accuracy = evaluate(train_images, train_labels)
  train_loss_history.append(mean_loss)

  print(
      f"Epoch {epoch + 1}: "
      f"Train Loss = {mean_loss:.6f} | "
      f"Train Accuracy = {mean_accuracy:.2%}"
  )

  if mean_loss <= target_loss:
    print("Целевая средняя обучающая ошибка достигнута.")
    break

# Test the CNN
print('\n Testing the CNN')

loss = 0
test_probs = np.zeros((len(test_images), 10))   # вероятности по классам (нужны для ROC)
test_features = np.zeros((tsne_n, 128))          # выход скрытого слоя (для t-SNE)

for k, (im, label) in enumerate(zip(test_images, test_labels)):
  out, l, _ = forward(im, label)
  test_probs[k] = out
  if k < tsne_n:
    test_features[k] = dense.last_output
  loss += l

num_tests = len(test_images)
test_pred = np.argmax(test_probs, axis=1)
cm, m = metrics.classification_metrics(test_labels, test_pred)

print('Test Loss:', loss / num_tests)
print('Test Accuracy:', m['accuracy'])

analysis.plot_loss(train_loss_history, target_loss)

# # матрица неточностей
# print('\nМатрица неточностей (строки — истинный класс, столбцы — предсказанный):')
# print(cm)
# print(f"\nAccuracy        = {m['accuracy']:.4f}")
# print(f"Precision macro = {m['precision_macro']:.4f}")
# print(f"Recall macro    = {m['recall_macro']:.4f}")
# print(f"F1 macro        = {m['f1_macro']:.4f}")
# print('\nкласс  precision  recall  F1')
# for c in range(10):
#   print(f"  {c}     {m['precision'][c]:.4f}    {m['recall'][c]:.4f}  {m['f1'][c]:.4f}")
# curves = metrics.roc_one_vs_rest(test_labels, test_probs)
# aucs = np.array([curves[c][2] for c in range(10)])
# print('\nAUC по классам:', np.round(aucs, 5))
# auc_macro = roc_auc_score(test_labels, test_probs, multi_class='ovr', average='macro')
# print(f'AUC macro = {auc_macro:.5f}')

# analysis.plot_confusion(cm)
# analysis.plot_roc(curves)

# # анализ ошибок
# print('\nЧаще всего путаются (истинный -> предсказанный: сколько раз):')
# for i, j, n in analysis.top_confusions(cm):
#   print(f'  {i} -> {j}: {n}')
# analysis.plot_tsne(
#     analysis.tsne_2d(test_features),
#     test_labels[:tsne_n],
#     "t-SNE признаков скрытого слоя (после обучения), × — ошибки",
#     "tsne_features.png",
#     errors=(test_pred[:tsne_n] != test_labels[:tsne_n]),
# )
# analysis.plot_errors(test_images, test_labels, test_pred, test_probs)
