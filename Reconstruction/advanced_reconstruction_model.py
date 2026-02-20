import os
import glob
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from custom_encoder_cnn import SketchEncoder

# dataset
class SketchPointDataset(Dataset):
    def __init__(self, root, cat_id="02691156"):

        self.sketch_paths = sorted(glob.glob(os.path.join(root, "2d", cat_id, "*.png")))
        npy_list = glob.glob(os.path.join(root, "npy", cat_id, "*.npy"))

        self.npy_map = {os.path.splitext(os.path.basename(p))[0]: p for p in npy_list}

        if len(self.sketch_paths) == 0:
            raise RuntimeError("No sketches found")

        self.tf = transforms.Compose([
            transforms.Resize((28,28)),
            transforms.RandomRotation(10),
            transforms.RandomAffine(10, translate=(0.05,0.05)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.sketch_paths)

    def normalize_pc(self, pts):
        pts -= pts.mean(0)
        scale = torch.norm(pts, dim=1).max()
        pts /= scale
        return pts

    def __getitem__(self, idx):

        img_path = self.sketch_paths[idx]
        file_id = os.path.basename(img_path).split("_")[0]

        img = Image.open(img_path).convert("RGB")
        img = self.tf(img)

        pts = torch.from_numpy(np.load(self.npy_map[file_id])).float()
        pts = self.normalize_pc(pts)

        return img, pts


# folding decoder
class FoldingDecoder(nn.Module):
    def __init__(self, latent=512, num_points=2048):
        super().__init__()

        self.num_points = num_points

        self.mlp = nn.Sequential(
            nn.Linear(latent + 2, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,3)
        )

        grid = torch.rand(num_points,2)*2-1
        self.register_buffer("grid", grid)

    def forward(self, z):

        B = z.shape[0]

        grid = self.grid.unsqueeze(0).repeat(B,1,1)
        z = z.unsqueeze(1).repeat(1,self.num_points,1)

        x = torch.cat([grid,z], dim=-1)

        return self.mlp(x)


# model
class Sketch2Point(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = SketchEncoder(in_channels=3, latent_dim=512)
        self.decoder = FoldingDecoder()

    def forward(self,x):
        z = self.encoder(x)
        return self.decoder(z)


# loss
def chamfer(pred, gt):

    B, N, _ = pred.shape
    _, M, _ = gt.shape

    loss = 0

    for b in range(B):
        d = torch.cdist(pred[b], gt[b])
        loss += d.min(1)[0].mean() + d.min(0)[0].mean()

    return loss / B



def repulsion_loss(x, k=20):

    dist = torch.cdist(x,x)
    _, idx = dist.topk(k+1, largest=False)

    knn = torch.gather(dist,2,idx[:,:,1:])
    return torch.mean(torch.exp(-knn))


# train
def train_epoch(model, loader, opt, device):

    model.train()
    total = 0

    for imgs,gt in loader:

        imgs,gt = imgs.to(device), gt.to(device)

        pred = model(imgs)

        cd = chamfer(pred,gt)
        rep = repulsion_loss(pred)

        loss = cd + 0.01*rep

        opt.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)

        opt.step()

        total += loss.item()

    return total/len(loader)


@torch.no_grad()
def eval_epoch(model, loader, device):

    model.eval()
    total = 0

    for imgs,gt in loader:

        imgs,gt = imgs.to(device), gt.to(device)
        pred = model(imgs)

        total += chamfer(pred,gt).item()

    return total/len(loader)

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

# main
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = "../../data/ShapeNet"
    cat = "02691156"

    full = SketchPointDataset(root,cat)

    train_ds, val_ds = random_split(full,[int(0.9*len(full)), len(full)-int(0.9*len(full))])

    train_loader = DataLoader(train_ds,batch_size=16,shuffle=True,num_workers=4)
    val_loader = DataLoader(val_ds,batch_size=16)

    model = Sketch2Point().to(device)

    opt = torch.optim.Adam(model.parameters(),1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(opt,30,0.5)

    best = 1e9

    for epoch in range(200):

        tr = train_epoch(model,train_loader,opt,device)
        vl = eval_epoch(model,val_loader,device)

        scheduler.step()

        print(f"Epoch {epoch}: train {tr:.5f} val {vl:.5f}")

        if epoch % 10 == 0:
            save_sketch_comparison(model, val_ds, device, epoch)

        if vl < best:
            best = vl
            torch.save(model.state_dict(),"advanced_encoder_decoder_model.pt")
            print("Saved checkpoint")


if __name__=="__main__":
    main()
