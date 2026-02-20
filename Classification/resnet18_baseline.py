# ResNet18 Baseline for QuickDraw dataset classification. Code was created with the help of GPT 5.2.

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3' #GPU 3

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from tqdm import tqdm

torch.cuda.empty_cache()  # Clear GPU memory

#------------- SAME AS ALEXNET DATASET CONSTRUCTION -------------------#

DATA_DIR = "/work/csse463/202620/04/data/"
MODELS_DIR = "/work/csse463/202620/04/models"
SAMPLES_PER_CLASS = 5000   
BATCH_SIZE = 16  
EPOCHS = 15
LR = 0.001

# 5 classes
classes = [
    "airplane","apple","banana","basketball","bird"
]

#all 20 classes
# classes = [
#    "airplane","apple","banana","basketball","bird","book","butterfly",
#    "car","cat","chair","cloud","cow","dog","flower","hand",
#    "horse","star","sun","tree","umbrella"
#]

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
    
# PREPROCESSING (ResNet)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],                 # ImageNet normalization - since transfer learning
        std=[0.229, 0.224, 0.225]
    )
])

dataset = QuickDrawDataset(DATA_DIR, classes, SAMPLES_PER_CLASS, transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    num_workers=2,
    pin_memory=True
)

# --- ResNet Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Freeze pretrained layers
for param in model.parameters():
    param.requires_grad = False

# Replace final fully-connected layer and set optimizer based on LR
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(classes))

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)

train_losses = []
val_accs = []
best_val_acc = 0.0

# Training loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
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
    
    # Average training loss for this epoch
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/{EPOCHS}] Average Training Loss: {avg_loss:.4f}")
    
    # Validation
    model.eval()
    correct = 0
    total = 0
    
    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
    
    with torch.no_grad():
        for images, labels in val_pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            current_acc = 100 * correct / total
            val_pbar.set_postfix({'accuracy': f'{current_acc:.2f}%'})
    
    val_accuracy = 100 * correct / total
    val_accs.append(val_accuracy)
    print(f"Epoch [{epoch+1}/{EPOCHS}] Validation Accuracy: {val_accuracy:.2f}%")
    
    # Save off best model
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_accuracy': val_accuracy,
            'classes': classes
        }, os.path.join(MODELS_DIR, 'resnet18_5classes.pth'))
        print(f"Saved best ResNet18 model with accuracy: {val_accuracy:.2f}%")

# Saving training data (if needed)
# import pandas as pd

# df = pd.DataFrame({
#    'epoch': list(range(1, EPOCHS+1)),
#    'train_loss': train_losses,
#    'val_accuracy': val_accs
#})
#df.to_csv(os.path.join(MODELS_DIR, 'resnet18_20classes_training_data.csv'), index=False)
#print("ResNet18 training data saved to CSV.")
