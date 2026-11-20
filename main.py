# %%
import gzip
import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

DATA_DIR = Path(r"D:\2ndSem2025\GitRepo\CNN-RCNN\data\gzip")

def load_images(path):
    with gzip.open(path, "rb") as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    images = data.reshape(num, rows, cols)
    return np.transpose(images, (0, 2, 1))

def load_labels(path):
    with gzip.open(path, "rb") as f:
        _, num = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

X_train = load_images(DATA_DIR / "emnist-digits-train-images-idx3-ubyte.gz")
y_train = load_labels(DATA_DIR / "emnist-digits-train-labels-idx1-ubyte.gz")

print("Images:", X_train.shape)
print("Labels:", y_train.shape)

fig, axes = plt.subplots(2, 5, figsize=(8, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i], cmap="gray")
    ax.set_title(f"Label: {y_train[i]}")
    ax.axis("off")

plt.tight_layout()
plt.savefig("digits_preview.png")
plt.show()
# %%

from torch.utils.data import DataLoader, TensorDataset

X = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1) / 255
y = torch.tensor(y_train, dtype=torch.float32)

train_dataset = TensorDataset(X, y)
dataLoader = DataLoader(train_dataset, batch_size=64, shuffle=True)

