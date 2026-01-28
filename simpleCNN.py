import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# params
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
CLASSES = ["cat", "dog", "airplane", "car", "tree"]
IMG_SIZE = 28
BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLES_PER_CLASS = 20000

# Dataset class
class QuickDrawDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# Load .npy data
def load_quickdraw_data(data_dir, classes, samples_per_class=20000):
    images, labels = [], []
    for label, cls in enumerate(classes):
        data = np.load(os.path.join(data_dir, f"{cls}.npy"))[:samples_per_class]
        data = data.astype(np.float32)/255.0
        data = data.reshape(-1,1,IMG_SIZE,IMG_SIZE)
        images.append(data)
        labels.append(np.full(len(data), label))
    return np.concatenate(images), np.concatenate(labels)

images, labels = load_quickdraw_data(DATA_DIR, CLASSES, SAMPLES_PER_CLASS)

# Split data 70/15/15
X_temp, X_test, y_temp, y_test = train_test_split(images, labels, test_size=0.15, stratify=labels, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=42)

# DataLoaders
train_loader = DataLoader(QuickDrawDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(QuickDrawDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=BATCH_SIZE)
test_loader = DataLoader(QuickDrawDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=BATCH_SIZE)

# CNN model
class QuickDrawCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128*3*3,256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256,num_classes)
        )
    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

model = QuickDrawCNN(len(CLASSES)).to(DEVICE)

# Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(dim=1)
            correct += (preds==y).sum().item()
            total += y.size(0)
    val_acc = correct/total
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

# Test evaluation
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        preds = model(x).argmax(dim=1)
        correct += (preds==y).sum().item()
        total += y.size(0)
print(f"Test Accuracy: {correct/total*100:.2f}%")

# Save model
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
model_path = os.path.join(models_dir, "simplecnn.pth")
torch.save(model.state_dict(), model_path)
print("Model saved to", model_path)

all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        preds = model(x).argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)

# Plot
plt.figure(figsize=(8,8))
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix")
plt.show()
plot_path = os.path.join(models_dir, "confusion_matrix.png")
plt.savefig(plot_path)
print("Confusion matrix saved to", plot_path)
plt.close()