import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ---------------------- Params ----------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "..", "data")
MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

CLASSES = [
    "cat", "dog", "airplane", "car", "tree",
    "apple", "banana", "bird", "basketball", "book",
    "butterfly", "chair", "cloud", "cow", "flower",
    "hand", "horse", "umbrella", "star", "sun"
]

IMG_SIZE = 28
BATCH_SIZE = 128
EPOCHS = 50
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLES_PER_CLASS = 20000
PATIENCE = 5  # early stopping patience

# Dataset
class QuickDrawDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def load_quickdraw_data(data_dir, classes, samples_per_class=SAMPLES_PER_CLASS):
    images, labels = [], []
    for label, cls in enumerate(classes):
        data = np.load(os.path.join(data_dir, f"{cls}.npy"))[:samples_per_class]
        data = data.astype(np.float32)/255.0
        data = data.reshape(-1,1,IMG_SIZE,IMG_SIZE)
        images.append(data)
        labels.append(np.full(len(data), label))
    return np.concatenate(images), np.concatenate(labels)

images, labels = load_quickdraw_data(DATA_DIR, CLASSES, SAMPLES_PER_CLASS)

# Split 70/15/15
X_temp, X_test, y_temp, y_test = train_test_split(images, labels, test_size=0.15, stratify=labels, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=42)

# DataLoaders
train_loader = DataLoader(QuickDrawDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(QuickDrawDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=BATCH_SIZE)
test_loader = DataLoader(QuickDrawDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=BATCH_SIZE)

# Model
class QuickDrawCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.2),

            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.3),

            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.4)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128*3*3,256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256,num_classes)
        )

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

model = QuickDrawCNN(len(CLASSES)).to(DEVICE)

# Loss, Optimizer, Scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# Training
best_val_loss = np.inf
early_stop_counter = 0

train_losses, val_accs = [], []

for epoch in range(EPOCHS):
    # Training
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
    train_losses.append(train_loss)

    # Validation
    model.eval()
    val_loss = 0
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            val_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds==y).sum().item()
            total += y.size(0)
    val_loss /= len(val_loader)
    val_acc = correct/total
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

    # Scheduler step
    scheduler.step(val_loss)

    # Early stopping
    # if val_loss < best_val_loss:
    #     best_val_loss = val_loss
    #     torch.save(model.state_dict(), os.path.join(MODELS_DIR, "best_model.pth"))
    #     early_stop_counter = 0
    # else:
    #     early_stop_counter += 1
    #     if early_stop_counter >= PATIENCE:
    #         print("Early stopping triggered")
    #         break

#Test Evaluation
model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "best_model_20cls.pth")))
model.eval()
correct, total = 0, 0
all_preds, all_labels = [], []
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds==y).sum().item()
        total += y.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
test_acc = correct/total
print(f"Test Accuracy: {test_acc*100:.2f}%")

# Plot Loss & Accuracy
plt.figure(figsize=(10,5))

# Plot training loss
plt.subplot(1,2,1)
plt.plot(range(1,len(train_losses)+1), train_losses, marker='o', label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.grid(True)

# Plot validation accuracy
plt.subplot(1,2,2)
plt.plot(range(1,len(val_accs)+1), val_accs, marker='o', color='green', label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title(f"Validation Accuracy over Epochs\nTest Accuracy: {test_acc*100:.2f}%")
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, "loss_acc_curves_20classes.png"))
plt.close()
print("Loss and accuracy curves saved")



# Save off training (data if needed)
# import pandas as pd

# data = {
#    "Train_Loss": train_losses,
#    "Val_Accuracy": val_accs,
#    "Test_Accuracy": [test_acc]*len(train_losses)  # repeated for convenience
# }
#
# df = pd.DataFrame(data)

# Save CSV
# csv_file = os.path.join(MODELS_DIR, "simpleCNN_20classes_training_data.csv")
# df.to_csv(csv_file, index=False)
# print(f"Training data saved to {csv_file}")
