import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------- Params ----------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

CLASSES = [
    "cat","dog","airplane","car","tree",
    "apple","banana","bird","basketball","book",
    "butterfly","chair","cloud","cow","flower",
    "hand","horse","umbrella","star","sun"
]

IMG_SIZE = 28
BATCH_SIZE = 256
EPOCHS = 45                 # ↓ Reduced max epochs (early stopping still applies)
LR = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLES_PER_CLASS = 20000
PATIENCE = 5                # ↓ Slightly tighter early stopping

# ---------------------- Dataset ----------------------
class QuickDrawDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

def load_quickdraw_data(data_dir, classes, samples_per_class):
    images, labels = [], []
    for label, cls in enumerate(classes):
        data = np.load(os.path.join(data_dir, f"{cls}.npy"))[:samples_per_class]
        data = data.astype(np.float32) / 255.0
        data = data.reshape(-1, 1, IMG_SIZE, IMG_SIZE)
        images.append(data)
        labels.append(np.full(len(data), label))
    return np.concatenate(images), np.concatenate(labels)

images, labels = load_quickdraw_data(DATA_DIR, CLASSES, SAMPLES_PER_CLASS)

# Stratified 70 / 15 / 15 split
X_temp, X_test, y_temp, y_test = train_test_split(
    images, labels, test_size=0.15, stratify=labels, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=42
)

# ---------------------- Data Augmentation ----------------------
# ↓ Weaker augmentation: keeps robustness but improves accuracy + speed
train_transform = T.Compose([
    T.RandomAffine(
        degrees=5,                  # ↓ was 10
        translate=(0.05, 0.05),      # ↓ was 0.1
        scale=(0.95, 1.05)           # ↓ narrower range
    )
])

val_transform = None

# ↓ DataLoader speedup (very important)
train_loader = DataLoader(
    QuickDrawDataset(X_train, y_train, train_transform),
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=4,
    persistent_workers=True
)

val_loader = DataLoader(
    QuickDrawDataset(X_val, y_val, val_transform),
    batch_size=BATCH_SIZE,
    pin_memory=True,
    num_workers=4,
    persistent_workers=True
)

test_loader = DataLoader(
    QuickDrawDataset(X_test, y_test, val_transform),
    batch_size=BATCH_SIZE,
    pin_memory=True,
    num_workers=4,
    persistent_workers=True
)

# ---------------------- Model ----------------------
class QuickDrawCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).squeeze(-1).squeeze(-1)
        return self.classifier(x)

model = QuickDrawCNN(len(CLASSES)).to(DEVICE)

# ---------------------- Loss & Optimization ----------------------
criterion = nn.CrossEntropyLoss(
    label_smoothing=0.05     # ↓ Reduced from 0.1 to recover accuracy
)

optimizer = optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)

# ↓ Scheduler that reacts to validation stagnation (earlier convergence)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
)

# ---------------------- Training ----------------------
best_val_acc = 0.0
early_stop_counter = 0

train_losses, val_accs = [], []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    val_acc = correct / total
    val_accs.append(val_acc)

    scheduler.step(val_acc)

    print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(MODELS_DIR, "best_model_20cls.pth"))
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= PATIENCE:
            print("Early stopping triggered")
            break

# ---------------------- Test Evaluation ----------------------
model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "best_model_20cls.pth")))
model.eval()

correct, total = 0, 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

test_acc = correct / total
print(f"Test Accuracy: {test_acc*100:.2f}%")

# ---------------------- Plot Curves ----------------------
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, marker='o')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(val_accs, marker='o', color='green')
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, "loss_acc_curves_optCNN_20cls_2.png"))
plt.close()