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

#  GPU AUTO SELECT (Idea taken from Chat-GPT as a way to find unused GPU)
def get_free_gpu():
    smi = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
    ).decode("utf-8")
    free_mem = [int(x) for x in smi.strip().split("\n")]
    return str(free_mem.index(max(free_mem)))

os.environ["CUDA_VISIBLE_DEVICES"] = get_free_gpu()
torch.backends.cudnn.benchmark = True
scaler = torch.amp.GradScaler("cuda")

# CONFIG 
DATA_DIR = "../../data/"
SAMPLES_PER_CLASS = 5000
BATCH_SIZE = 256  
LR = 5e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.4
EPOCHS = 20
EARLYSTOP = 4

classes = ["airplane","apple","banana","basketball","bird","book","butterfly",
           "car","cat","chair","cloud","cow","dog","flower","hand",
           "horse","star","sun","tree","umbrella"]

# DATASET 
class QuickDrawDataset(Dataset):
    def __init__(self, data_dir, classes, samples_per_class):
        self.images, self.labels = [], []
        for label, cname in enumerate(tqdm(classes)):
            data = np.load(os.path.join(data_dir, cname + ".npy"))[:samples_per_class]
            for img in data:
                img = Image.fromarray(img.reshape(28,28).astype(np.uint8)).resize((227,227))
                self.images.append(np.array(img))
                self.labels.append(label)

    def __len__(self): return len(self.images)

full_dataset = QuickDrawDataset(DATA_DIR, classes, SAMPLES_PER_CLASS)

# 70 / 15 / 15 split
train_size = int(0.7 * len(full_dataset))
val_size   = int(0.15 * len(full_dataset))
test_size  = len(full_dataset) - train_size - val_size

train_idx, val_idx, test_idx = torch.utils.data.random_split(
    range(len(full_dataset)),
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

train_transform = transforms.Compose([
    transforms.RandomAffine(15, translate=(0.1,0.1)),
    transforms.RandomHorizontalFlip(),
    transforms.Grayscale(3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])

val_transform = transforms.Compose([
    transforms.Grayscale(3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])

class Subset(Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset, self.indices, self.transform = dataset, indices.indices, transform
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        img = Image.fromarray(self.dataset.images[self.indices[idx]])
        label = self.dataset.labels[self.indices[idx]]
        return self.transform(img), label

train_set = Subset(full_dataset, train_idx, train_transform)
val_set   = Subset(full_dataset, val_idx, val_transform)
test_set  = Subset(full_dataset, test_idx, val_transform)

train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True, num_workers=12, pin_memory=True, persistent_workers=True)
val_loader   = DataLoader(val_set,   BATCH_SIZE, num_workers=12, pin_memory=True)
test_loader  = DataLoader(test_set,  BATCH_SIZE, num_workers=12, pin_memory=True)

#  MODEL 
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
    def forward(self,x): return self.base(x)
    def freeze_features(self):
        for p in self.base.features.parameters(): p.requires_grad=False
    def unfreeze_last(self):
        for p in list(self.base.features.parameters())[-4:]: p.requires_grad=True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ModifiedAlexNet(len(classes), DROPOUT)
model.freeze_features()
model = model.to(device)

if torch.cuda.device_count()>1:
    model = nn.DataParallel(model)

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

#  TRAIN 
history = {"train_acc":[], "val_acc":[]}
best_acc = 0
patience_ctr = 0

for epoch in range(EPOCHS):
    model.train()
    correct=total=0

    for imgs,labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        imgs,labels=imgs.to(device),labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            out=model(imgs)
            loss=criterion(out,labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        pred=out.argmax(1)
        correct+=(pred==labels).sum().item()
        total+=labels.size(0)

    train_acc=100*correct/total
    history["train_acc"].append(train_acc)

    model.eval()
    correct=total=0
    with torch.no_grad():
        for imgs,labels in val_loader:
            imgs,labels=imgs.to(device),labels.to(device)
            out=model(imgs)
            pred=out.argmax(1)
            correct+=(pred==labels).sum().item()
            total+=labels.size(0)

    val_acc=100*correct/total
    history["val_acc"].append(val_acc)
    scheduler.step(1-val_acc)

    print(f"Train {train_acc:.2f} | Val {val_acc:.2f}")

    if val_acc>best_acc:
        best_acc=val_acc
        torch.save(model.module.state_dict() if isinstance(model,nn.DataParallel) else model.state_dict(),"best_model.pth")
        patience_ctr=0
    else:
        patience_ctr+=1
    if patience_ctr>=EARLYSTOP: break

    if epoch==4:
        model.module.unfreeze_last() if isinstance(model,nn.DataParallel) else model.unfreeze_last()

# TEST ACC 
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
correct=total=0
with torch.no_grad():
    for imgs,labels in test_loader:
        imgs,labels=imgs.to(device),labels.to(device)
        pred=model(imgs).argmax(1)
        correct+=(pred==labels).sum().item()
        total+=labels.size(0)

test_acc=100*correct/total
print(f"\nTEST ACCURACY: {test_acc:.2f}%")

# PLOTS 
plt.figure(figsize=(8,5))
plt.plot(history["train_acc"], label="Train")
plt.plot(history["val_acc"], label="Val")
plt.legend()
plt.title("Training Curves")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("training_curves.png", dpi=300)

# Confusion matrix (test set)
all_preds,all_labels=[],[]
with torch.no_grad():
    for imgs,labels in test_loader:
        imgs=imgs.to(device)
        pred=model(imgs).argmax(1).cpu().numpy()
        all_preds.extend(pred)
        all_labels.extend(labels.numpy())

cm=confusion_matrix(all_labels,all_preds)
disp=ConfusionMatrixDisplay(cm,display_labels=classes)
disp.plot(xticks_rotation=45,cmap="Blues")
plt.savefig("confusion_matrix.png",dpi=300)