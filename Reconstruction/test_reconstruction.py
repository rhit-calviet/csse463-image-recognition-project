import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import plotly.graph_objects as go

from reconstruction_model import Sketch2Point

# --- Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
MODEL_FOLDER = os.path.join(SCRIPT_DIR, "..", "models")

# --- Device ---
device = torch.device("cpu")  # safe for any machine

# --- Transform ---
tf = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.ToTensor(),
])

# --- Load model ---
model = Sketch2Point().to(device)
ckpt = torch.load(os.path.join(MODEL_FOLDER, "best_model.pt"), map_location=device)
model.load_state_dict(ckpt, strict=True)
model.eval()


# --- Functions ---
def extract(class_name="airplane", index=0):
    arr = np.load(os.path.join(DATA_DIR, f"{class_name}.npy"))
    img = arr[index].reshape(28,28).astype(np.uint8)
    return Image.fromarray(img, "L").convert("RGB")


def predict_points(img):
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        points = model(x)[0].cpu().numpy()  # Nx3 points
    return points


def show_pointcloud_plotly(points, title="Predicted 3D points"):
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:,0],
        y=points[:,1],
        z=points[:,2],
        mode='markers',
        marker=dict(size=2, color=points[:,2], colorscale='Viridis')
    )])
    fig.update_layout(scene=dict(aspectmode='data'), title=title)
    fig.show()  # opens browser window for interaction


# --- Main ---
if __name__ == "__main__":
    class_name = "airplane"
    index = 0  # choose which sample to test

    # Show reference sketch
    img = extract(class_name, index)
    img.show(title=f"{class_name} #{index}")

    # Predict 3D points
    points = predict_points(img)

    # Show interactive 3D point cloud
    show_pointcloud_plotly(points, title=f"{class_name} #{index} Prediction")
