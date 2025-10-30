from google.colab import drive
drive.mount('/content/drive')

# 作業ディレクトリを指定
work_dir = 'drive/MyDrive/Colab Notebooks/DLBasics2025_colab'

import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import inspect


#学習データ
x_train = np.load(work_dir + '/Lecture03/data/x_train.npy')
t_train = np.load(work_dir + '/Lecture03/data/y_train.npy')

#テストデータ
x_test = np.load(work_dir + '/Lecture03/data/x_test.npy')

# データの前処理（正規化， one-hot encoding)
x_train, x_test = x_train / 255., x_test / 255.
x_train, x_test = x_train.reshape(x_train.shape[0], -1), x_test.reshape(x_test.shape[0], -1)
t_train = np.eye(N=10)[t_train.astype("int32").flatten()]

# データの分割
x_train, x_val, t_train, t_val =\
    train_test_split(x_train, t_train, test_size=10000)

def np_log(x):
    return np.log(np.clip(x, 1e-10, 1e+10))


def create_batch(data, batch_size):
    """
    :param data: np.ndarray，入力データ
    :param batch_size: int，バッチサイズ
    """
    num_batches, mod = divmod(data.shape[0], batch_size)
    batched_data = np.split(data[: batch_size * num_batches], num_batches)
    if mod:
        batched_data.append(data[batch_size * num_batches:])

    return batched_data

# シード値を変えることで何が起きるかも確かめてみてください．
rng = np.random.RandomState(1234)
random_state = 42


# 発展: 今回の講義で扱っていない活性化関数について調べ，実装してみましょう
def relu(x):
    return np.maximum(0.0, x)


def deriv_relu(x):
    grad = np.zeros_like(x)
    grad[x > 0] = 1.0
    return grad


def softmax(x):
    # Stable softmax for batches (rows are samples)
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def deriv_softmax(x):
    # Returns the Jacobian matrices for each sample in the batch.
    # Shape: (batch, C, C)
    s = softmax(x)
    batch, C = s.shape
    J = np.zeros((batch, C, C), dtype=s.dtype)
    for i in range(batch):
        si = s[i]
        J[i] = np.diag(si) - np.outer(si, si)
    return J


def crossentropy_loss(t, y):
    # t: one-hot targets (batch, C); y: probabilities (batch, C)
    return -np.mean(np.sum(t * np_log(y), axis=1))


class Dense:
    def __init__(self, in_dim, out_dim, activation=None, rng=None):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation  # 'relu' or None
        self.rng = rng if rng is not None else np.random.RandomState()

        # He init for ReLU, Xavier otherwise
        if self.activation == 'relu':
            scale = np.sqrt(2.0 / in_dim)
        else:
            scale = np.sqrt(1.0 / in_dim)
        self.W = self.rng.randn(in_dim, out_dim) * scale
        self.b = np.zeros(out_dim, dtype=np.float64)

        # caches and grads
        self.x = None
        self.z = None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        self.z = x @ self.W + self.b
        if self.activation == 'relu':
            return relu(self.z)
        else:
            return self.z

    def backward(self, grad_out):
        # grad_out is dL/dA where A is layer output; convert to dL/dZ
        if self.activation == 'relu':
            grad_z = grad_out * deriv_relu(self.z)
        else:
            grad_z = grad_out

        # gradients w.r.t parameters
        self.dW = self.x.T @ grad_z
        self.db = np.sum(grad_z, axis=0)

        # gradient w.r.t input to this layer
        grad_x = grad_z @ self.W.T
        return grad_x

    def update(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db


class Model:
    def __init__(self, input_dim, hidden_dim, output_dim, rng=None):
        self.rng = rng if rng is not None else np.random.RandomState()
        self.l1 = Dense(input_dim, hidden_dim, activation='relu', rng=self.rng)
        self.l2 = Dense(hidden_dim, output_dim, activation=None, rng=self.rng)
        # caches
        self.y = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        h = self.l1.forward(x)
        scores = self.l2.forward(h)
        self.y = softmax(scores)
        return self.y

    def backward(self, t):
        # assumes forward called before; t is one-hot
        batch_size = t.shape[0]
        # dL/dscores for softmax + cross-entropy
        dscores = (self.y - t) / batch_size
        dh = self.l2.backward(dscores)
        _ = self.l1.backward(dh)

    def update(self, lr):
        self.l1.update(lr)
        self.l2.update(lr)

lr = 0.1
n_epochs = 10
batch_size = 128

mlp = Model(input_dim=x_train.shape[1], hidden_dim=256, output_dim=10, rng=rng)

def train_model(mlp, x_train, t_train, x_val, t_val, n_epochs=10):
    for epoch in range(n_epochs):
        losses_train = []
        losses_valid = []
        train_num = 0
        train_true_num = 0
        valid_num = 0
        valid_true_num = 0

        x_train, t_train = shuffle(x_train, t_train)
        x_train_batches, t_train_batches = create_batch(x_train, batch_size), create_batch(t_train, batch_size)

        x_val, t_val = shuffle(x_val, t_val)
        x_val_batches, t_val_batches = create_batch(x_val, batch_size), create_batch(t_val, batch_size)

        # モデルの訓練
        for x, t in zip(x_train_batches, t_train_batches):
            # 順伝播
            y = mlp(x)

            # 損失の計算
            loss = crossentropy_loss(t, y)
            losses_train.append(loss.tolist())

            # パラメータの更新
            mlp.backward(t)
            mlp.update(lr)

            # 精度を計算
            acc = accuracy_score(t.argmax(axis=1), y.argmax(axis=1), normalize=False)
            train_num += x.shape[0]
            train_true_num += acc

        # モデルの評価
        for x, t in zip(x_val_batches, t_val_batches):
            # 順伝播
            y = mlp(x)

            # 損失の計算
            loss = crossentropy_loss(t, y)
            losses_valid.append(loss.tolist())

            acc = accuracy_score(t.argmax(axis=1), y.argmax(axis=1), normalize=False)
            valid_num += x.shape[0]
            valid_true_num += acc

        print('EPOCH: {}, Train [Loss: {:.3f}, Accuracy: {:.3f}], Valid [Loss: {:.3f}, Accuracy: {:.3f}]'.format(
            epoch,
            np.mean(losses_train),
            train_true_num/train_num,
            np.mean(losses_valid),
            valid_true_num/valid_num
        ))


train_model(mlp, x_train, t_train, x_val, t_val, n_epochs)

t_pred = []
for x in x_test:
    # 順伝播
    x = x[np.newaxis, :]
    y = mlp(x)

    # モデルの出力を予測値のスカラーに変換
    pred = y.argmax(1).tolist()

    t_pred.extend(pred)

submission = pd.Series(t_pred, name='label')
submission.to_csv(work_dir + '/Lecture03/submission_pred_03.csv', header=True, index_label='id')