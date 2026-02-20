import os
import glob
import numpy as np
from PIL import Image
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

from custom_encoder_cnn import SketchEncoder

# DATASET: Handles 12k sketches vs 4k npy files
class SketchPointDataset(Dataset):
    def __init__(self, root, cat_id="02691156"):
        self.sketch_paths = sorted(glob.glob(os.path.join(root, "2d", cat_id, "*.png")))
        npy_list = glob.glob(os.path.join(root, "npy", cat_id, "*.npy"))
        
        self.npy_map = {os.path.splitext(os.path.basename(p))[0]: p for p in npy_list}
        
        if len(self.sketch_paths) == 0:
            raise FileNotFoundError(f"No files found for category {cat_id} in {root}")
            
        self.tf = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.sketch_paths)

    def __getitem__(self, idx):
        img_path = self.sketch_paths[idx]
        file_id = os.path.basename(img_path).split('_')[0]

        img = Image.open(img_path).convert("RGB")
        img = self.tf(img)

        pts = np.load(self.npy_map[file_id]).astype(np.float32)
        pts = torch.from_numpy(pts)

        return img, pts


# MODELS: Encoder-Decoder 

class PointDecoder(nn.Module):
    def __init__(self, latent=512, num_points=4096):
        super().__init__()
        self.num_points = num_points
        self.fc = nn.Sequential(
            nn.Linear(latent, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_points * 3)
        )

    def forward(self, z):
        x = self.fc(z)
        return x.view(-1, self.num_points, 3)

class Sketch2Point(nn.Module):
    def __init__(self):
        super().__init__()
        # Using the custom CNN 
        self.encoder = SketchEncoder(in_channels=3, latent_dim=512)
        self.decoder = PointDecoder(latent=512)

    def forward(self, x):
        z = self.encoder(x)
        pts = self.decoder(z)
        return pts


# LOSS & UTILS: Chamfer Distance and Visualization
def chamfer_distance(pred, gt):
    if pred.dim() == 2: pred = pred.unsqueeze(0)
    if gt.dim() == 2: gt = gt.unsqueeze(0)

    diff = pred.unsqueeze(2) - gt.unsqueeze(1)
    dist = torch.sum(diff ** 2, dim=-1)

    min_pred_to_gt, _ = dist.min(dim=2)
    min_gt_to_pred, _ = dist.min(dim=1)

    return min_pred_to_gt.mean() + min_gt_to_pred.mean()

def save_sketch_comparison(model, dataset, device, epoch, folder="plots"):
    """Generates the side-by-side 2D sketch vs 3D point cloud output."""
    os.makedirs(folder, exist_ok=True)
    model.eval()
    
    idx = random.randint(0, len(dataset) - 1)
    img_tensor, _ = dataset[idx]
    
    with torch.no_grad():
        pred = model(img_tensor.unsqueeze(0).to(device))
    
    input_img = img_tensor.permute(1, 2, 0).cpu().numpy()
    pred_pts = pred[0].cpu().numpy()
    
    fig = plt.figure(figsize=(10, 5))
    
    ax1 = fig.add_subplot(121)
    ax1.imshow(input_img)
    ax1.set_title("Original Sketch")
    ax1.axis('off')

    ax2 = fig.add_subplot(122)
    ax2.scatter(pred_pts[:, 0], pred_pts[:, 1], s=1, c='red', alpha=0.6)
    ax2.set_title(f"Predicted 3D Points (Epoch {epoch})")
    ax2.set_aspect('equal')
    ax2.axis('off')
    
    plt.savefig(f"{folder}/epoch_{epoch}.png")
    plt.close()

# 4. TRAINING LOOP
def main():
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_root = "../../data/ShapeNet" 
    airplane_id = "02691156"

    full_ds = SketchPointDataset(data_root, airplane_id)
    
    train_size = int(0.9 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=16)

    model = Sketch2Point().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_loss = 1e9
    for epoch in range(101):
        model.train()
        train_loss = 0
        for imgs, gt in train_loader:
            imgs, gt = imgs.to(device), gt.to(device)
            optimizer.zero_grad()
            pred = model(imgs)
            loss = chamfer_distance(pred, gt)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, gt in val_loader:
                imgs, gt = imgs.to(device), gt.to(device)
                pred = model(imgs)
                val_loss += chamfer_distance(pred, gt).item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch}: Train {avg_train:.5f} | Val {avg_val:.5f}")

        if epoch % 10 == 0:
            save_sketch_comparison(model, val_ds, device, epoch)

        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), "reconstruction_model.pt")
            print("--- Checkpoint Saved ---")

if __name__ == "__main__":
    main()