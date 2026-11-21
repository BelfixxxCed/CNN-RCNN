
# %%
import torchvision
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from torchvision import transforms
# %%
data = torchvision.datasets.EMNIST(root="data", split="balanced", download=True, transform=transforms.ToTensor())
# %%

fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(8, 8))

for i, ax in enumerate(axes.flat):
    img, label = data[i]
    img_upright = img.squeeze().T
    char = data.classes[label]
    ax.imshow(img_upright, cmap="gray")
    ax.set_title(f"{label}: {char}")
    ax.axis("off")
plt.show()

# %%

transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.transpose(1, 2)),
    transforms.RandomRotation(degrees=5),
    transforms.RandomAffine(translate=(0.1, 0.1), degrees=0),
])

data.transform = transformer

# %%
import torch
from torch.utils.data import random_split
import torch.nn as nn
# %%

generator = torch.Generator().manual_seed(64)

data_train, data_test = random_split(data, [0.8, 0.2], generator=generator)

train_loader = DataLoader(data_train, batch_size=200, shuffle=True)
test_loader = DataLoader(data_test, batch_size=200, shuffle=True)


# %%

class CNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1) 
        self.relu1 = nn.ReLU()
        self.AvgPool = nn.AvgPool2d(kernel_size=2, stride=2) # (in_channels, 14, 14)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=126, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2) # (in_channels, 7, 7)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=126*7*7, out_features=47)

    def forward(self, in_channels):
        x = self.AvgPool(self.relu1(self.conv1(in_channels)))
        x = self.MaxPool(self.relu2(self.conv2(x)))
        x = self.linear(self.flatten(x))
        return x
CNN_Model = CNN(1)
# %%
muon_params = [p for p in CNN_Model.parameters() if p.ndim == 2]
other_params = [p for p in CNN_Model.parameters() if p.ndim != 2]

adam_optimizer = torch.optim.Adam(params=other_params, lr=1e-3)
muon_optimizer = torch.optim.Muon(params=muon_params, lr=0.02, momentum=0.95)
from tqdm.auto import tqdm

criterion = nn.CrossEntropyLoss()

# %%
# Training
EPOCH = 100
patience = 10
device = torch.device('cpu')
torch.set_num_threads(4)

CNN_Model.to(device)
CNN_Model.train()
losses_history = []
for epoch in range(EPOCH):

    epoch_loss = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        adam_optimizer.zero_grad()
        muon_optimizer.zero_grad()

        pred = CNN_Model(images)
        loss = criterion(pred, labels)

        loss.backward()
        adam_optimizer.step()
        muon_optimizer.step()

        epoch_loss += loss.item()

        # Indented 8 spaces so it runs inside the batch loop
        if batch_idx == 0 or (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
            print(f"Epoch {epoch + 1} | Batch {batch_idx + 1}/{len(train_loader)} | Loss: {loss.item():.4f}", flush=True)

    losses_history.append(epoch_loss)
        
    print(f"Loss for {epoch}: {epoch_loss}", flush=True)

    if len(losses_history) >= 2 and losses_history[-1] > losses_history[-2]:
        patience -= 1
    if patience <= 0:
        break

# %%
torch.save(CNN_Model.state_dict(), "emnist_cnn.pth")
