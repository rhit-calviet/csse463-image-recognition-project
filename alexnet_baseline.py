import os, subprocess, copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import ParameterGrid
import pandas as pd

# GPU AUTO SELECT (Idea taken from Chat-GPT as a way to find unused GPU)
def get_free_gpu():
    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
        ).decode("utf-8")
        free_mem = [int(x) for x in smi.strip().split("\n")]
        return str(free_mem.index(max(free_mem)))
    except:
        return "0"

os.environ["CUDA_VISIBLE_DEVICES"] = get_free_gpu()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.backends.cudnn.benchmark = True
scaler = torch.amp.GradScaler("cuda")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparamaters
DATA_DIR = "../data/"
SAMPLES_PER_CLASS = 5000
EPOCHS = 20
PATIENCE = 5  # Early stopping

param_grid = {
    'batch_size': [128, 256],
    'lr': [5e-4, 1e-4],
    'weight_decay': [1e-4],
    'dropout_rate': [0.4, 0.5]
}

classes = ["airplane","apple","banana","basketball","bird","book","butterfly",
           "car","cat","chair","cloud","cow","dog","flower","hand",
           "horse","star","sun","tree","umbrella"]

# Dataset
class QuickDrawDataset(Dataset):
    def __init__(self, data_dir, classes, samples_per_class):
        self.images, self.labels = [], []
        print("Loading data into memory...")
        for label, cname in enumerate(tqdm(classes)):
            data = np.load(os.path.join(data_dir, cname + ".npy"))[:samples_per_class]
            for img in data:
                # Resize to AlexNet standard input size
                img = Image.fromarray(img.reshape(28,28).astype(np.uint8)).resize((227,227))
                self.images.append(np.array(img))
                self.labels.append(label)

    def __len__(self): return len(self.images)

full_dataset = QuickDrawDataset(DATA_DIR, classes, SAMPLES_PER_CLASS)

train_size = int(0.7 * len(full_dataset))
val_size   = int(0.15 * len(full_dataset))
test_size  = len(full_dataset) - train_size - val_size

train_idx, val_idx, test_idx = torch.utils.data.random_split(
    range(len(full_dataset)),
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

# Translations
train_tf = transforms.Compose([
    transforms.RandomAffine(15, translate=(0.1, 0.1)),
    transforms.RandomHorizontalFlip(),
    transforms.Grayscale(3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

val_tf = transforms.Compose([
    transforms.Grayscale(3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

class Subset(Dataset):
    def __init__(self, dataset, indices, tf):
        self.dataset, self.indices, self.tf = dataset, indices.indices, tf
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        img = Image.fromarray(self.dataset.images[self.indices[i]])
        label = self.dataset.labels[self.indices[i]]
        return self.tf(img), label

train_set = Subset(full_dataset, train_idx, train_tf)
val_set   = Subset(full_dataset, val_idx, val_tf)
test_set  = Subset(full_dataset, test_idx, val_tf)

# model
class ModifiedAlexNet(nn.Module):
    def __init__(self, num_classes, dropout):
        super().__init__()
        self.base = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        self.base.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256*6*6, 4096), nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096), nn.ReLU(True),
            nn.Linear(4096, num_classes),
        )
    def forward(self, x): return self.base(x)

    def freeze_features(self):
        for p in self.base.features.parameters(): p.requires_grad = False

    def unfreeze_last(self):
        # Fine-tuning: enable gradients for the last few layers of the backbone
        for p in list(self.base.features.parameters())[-4:]: 
            p.requires_grad = True

# Training
def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler):
    best_acc = 0
    best_wts = None
    patience_ctr = 0
    history = {"train": [], "val": []}

    

    for epoch in range(EPOCHS):
        # Unfreeze backbone at epoch 4 for fine-tuning
        if epoch == 4:
            print("\n[INFO] Epoch 4: Unfreezing backbone for fine-tuning.")
            if isinstance(model, nn.DataParallel):
                model.module.unfreeze_last()
            else:
                model.unfreeze_last()

        model.train()
        correct = total = 0
        t_loader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for imgs, labels in t_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast("cuda"):
                out = model(imgs)
                loss = criterion(out, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            pred = out.argmax(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            t_loader.set_postfix({"acc": f"{100*correct/total:.2f}%"})

        train_acc = 100 * correct / total
        history["train"].append(train_acc)

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                pred = out.argmax(1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        
        val_acc = 100 * correct / total
        history["val"].append(val_acc)
        scheduler.step(1 - (val_acc/100))

        print(f"Epoch {epoch+1} Results - Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            best_wts = copy.deepcopy(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
        
        if patience_ctr >= PATIENCE:
            print("Early stopping triggered.")
            break

    model.load_state_dict(best_wts)
    return model, best_acc, history

# GRID SEARCH 
results = []
best_overall_acc = 0
best_overall_params = None
best_overall_state = None

for params in ParameterGrid(param_grid):
    print(f"\n{'#'*40}\nTesting Params: {params}\n{'#'*40}")
    
    t_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True, num_workers=8, pin_memory=True)
    v_loader = DataLoader(val_set, batch_size=params['batch_size'], num_workers=8, pin_memory=True)
    
    model = ModifiedAlexNet(len(classes), params['dropout_rate']).to(device)
    model.freeze_features()
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    model, val_acc, history = train_model(model, t_loader, v_loader, optimizer, criterion, scheduler)
    
    results.append({"params": params, "val_acc": val_acc, "history": history})

    if val_acc > best_overall_acc:
        best_overall_acc = val_acc
        best_overall_params = params
        best_overall_state = copy.deepcopy(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict())

# TEST
torch.save(best_overall_state, "best_gridsearch_model.pth")
print(f"\nBest Params Found: {best_overall_params} with {best_overall_acc:.2f}% Val Acc")

final_model = ModifiedAlexNet(len(classes), best_overall_params['dropout_rate']).to(device)
final_model.load_state_dict(torch.load("best_gridsearch_model.pth"))
final_model.eval()

test_loader = DataLoader(test_set, batch_size=best_overall_params['batch_size'], num_workers=4)
all_preds, all_labels = [], []

with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc="Final Testing"):
        imgs = imgs.to(device)
        pred = final_model(imgs).argmax(1).cpu().numpy()
        all_preds.extend(pred)
        all_labels.extend(labels.numpy())

test_acc = 100 * (np.array(all_preds) == np.array(all_labels)).mean()
print(f"\nFINAL TEST ACCURACY: {test_acc:.2f}%")

# plots
plt.figure(figsize=(10,6))
best_run = max(results, key=lambda x: x["val_acc"])
plt.plot(best_run["history"]["train"], label="Train Acc")
plt.plot(best_run["history"]["val"], label="Val Acc")
plt.title(f"Best Model Curves (LR: {best_overall_params['lr']})")
plt.legend()
plt.savefig("gs_performance_curves.png")

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12,10))
ConfusionMatrixDisplay(cm, display_labels=classes).plot(xticks_rotation=45, cmap="Blues")
plt.savefig("gs_confusion_matrix.png")

print("Grid search and evaluation complete.")
