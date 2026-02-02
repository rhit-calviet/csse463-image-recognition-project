import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'  # GPU 4

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

torch.cuda.empty_cache()

DATA_DIR = "../data/"
SAMPLES_PER_CLASS = 5000   
BATCH_SIZE = 16  
EPOCHS = 15
LR = 0.001

# 5 classes
# classes = [
#    "airplane","apple","banana","basketball","bird"
# ]

#all 20 classes
classes = [
    "airplane","apple","banana","basketball","bird","book","butterfly",
    "car","cat","chair","cloud","cow","dog","flower","hand",
    "horse","star","sun","tree","umbrella"
]
# DATASET 
class QuickDrawDataset(Dataset):
    def __init__(self, data_dir, classes, samples_per_class, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform

        print("Loading QuickDraw data...")
        for label, cname in enumerate(tqdm(classes, desc="Loading classes")):
            path = os.path.join(data_dir, cname + ".npy")
            data = np.load(path)
            data = data[:samples_per_class]

            for img in data:
                img = img.reshape(28, 28).astype(np.uint8)
                self.images.append(img)
                self.labels.append(label)

        print(f"Loaded {len(self.images)} images.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# TRANSFORMS 
transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = QuickDrawDataset(DATA_DIR, classes, SAMPLES_PER_CLASS, transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)

# MODEL 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)

for param in model.features.parameters():
    param.requires_grad = False

model.classifier[6] = nn.Linear(4096, len(classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=LR)

# TRAINING 
train_losses = []
val_losses = []
best_val_acc = 0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0

    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")

    for images, labels in train_pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    #  VALIDATION 
    model.eval()
    val_running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            val_running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_running_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    val_accuracy = 100 * correct / total

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"Val Acc: {val_accuracy:.2f}%")

    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        torch.save(model.state_dict(), "best_model.pth")

#  LOSS CURVE PLOT 
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig("alexnet_loss_curve_weston.png", dpi=300, bbox_inches='tight')

#  CONFUSION MATRIX 
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8,6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix")
plt.savefig("alexnet_confusion_matrix_weston.png", dpi=300, bbox_inches='tight')

print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")

