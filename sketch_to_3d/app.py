import os
import base64
import torch
import numpy as np
import open3d as o3d
from flask import Flask, render_template, request, jsonify
from io import BytesIO
from PIL import Image, ImageOps
from torchvision import transforms
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import plotly.graph_objects as go
import plotly.io as pio

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

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool, self.max_pool = nn.AdaptiveAvgPool2d(1), nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False), nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        return x * self.sigmoid(avg_out + max_out).view(b, c, 1, 1)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.attention = ChannelAttention(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride, bias=False), nn.BatchNorm2d(out_channels))
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.attention(out)
        return F.relu(out + self.shortcut(x))

class Config: pass # Dummy for pickle loading

class AdvancedSketchEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=512):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = nn.Sequential(ResidualBlock(64, 128, 2), ResidualBlock(128, 128))
        self.layer2 = nn.Sequential(ResidualBlock(128, 256, 2), ResidualBlock(256, 256))
        self.layer3 = nn.Sequential(ResidualBlock(256, 512, 2), ResidualBlock(512, 512))
        self.gap, self.gmp = nn.AdaptiveAvgPool2d(1), nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, latent_dim))
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer3(self.layer2(self.layer1(x)))
        x = torch.cat([self.gap(x).view(x.size(0), -1), self.gmp(x).view(x.size(0), -1)], dim=1)
        return self.fc(x)

class FoldingDecoder(nn.Module):
    def __init__(self, latent_dim=512, num_points=2048):
        super().__init__()
        self.num_points = num_points
        self.fold1 = nn.Sequential(nn.Linear(latent_dim + 2, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 3))
        self.fold2 = nn.Sequential(nn.Linear(latent_dim + 3, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 3))
        sqrt_n = int(np.ceil(np.sqrt(num_points)))
        grid = np.stack(np.meshgrid(np.linspace(-1, 1, sqrt_n), np.linspace(-1, 1, sqrt_n)), axis=-1).reshape(-1, 2)[:num_points]
        self.register_buffer("grid", torch.from_numpy(grid).float())
    def forward(self, z):
        grid = self.grid.unsqueeze(0).repeat(z.size(0), 1, 1)
        z_ext = z.unsqueeze(1).repeat(1, self.num_points, 1)
        x1 = self.fold1(torch.cat([grid, z_ext], dim=-1))
        return self.fold2(torch.cat([x1, z_ext], dim=-1))

class Sketch2Point(nn.Module):
    def __init__(self, latent_dim=512, num_points=2048):
        super().__init__()
        self.encoder = AdvancedSketchEncoder(in_channels=3, latent_dim=latent_dim)
        self.decoder = FoldingDecoder(latent_dim=latent_dim, num_points=num_points)
    def forward(self, x): return self.decoder(self.encoder(x))

app = Flask(__name__)

# Constants
CLASSES = ["cat", "dog", "airplane", "car", "tree", "apple", "banana", "bird", "basketball", "book",
           "butterfly", "chair", "cloud", "cow", "flower", "hand", "horse", "umbrella", "star", "sun",
           "bicycle", "cake", "fish", "guitar", "house", "moon", "mug", "octopus", "pencil", "fork"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Models
clf_model = QuickDrawCNN(len(CLASSES)).to(DEVICE)
clf_model.load_state_dict(torch.load("../../models/simpleCNN_30classes.pth", map_location=DEVICE))
clf_model.eval()

recon_model = Sketch2Point().to(DEVICE)
checkpoint = torch.load("../../models/complete_model.pt", map_location=DEVICE, weights_only=False)
recon_model.load_state_dict(checkpoint["model_state_dict"])
recon_model.eval()

# Store raw points between requests so /generate_mesh can reuse them
_cached_points = {}

def local_surface_upsample(pcd, k=6):
    points = np.asarray(pcd.points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    new_points = []
    for i, p in enumerate(points):
        _, idx, _ = kdtree.search_knn_vector_3d(p, k)
        for j in idx[1:]:
            new_points.append((p + points[j]) / 2.0)
    all_pts = np.vstack([points, np.array(new_points)])
    res_pcd = o3d.geometry.PointCloud()
    res_pcd.points = o3d.utility.Vector3dVector(all_pts)
    return res_pcd

def build_advanced_mesh(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2*avg_dist, max_nn=60))
    pcd.orient_normals_consistent_tangent_plane(50)

    dense_pcd = local_surface_upsample(pcd, k=6)
    dense_pcd = dense_pcd.voxel_down_sample(voxel_size=avg_dist * 0.5)
    cl, ind = dense_pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
    dense_pcd = dense_pcd.select_by_index(ind)

    combined_pcd = pcd + dense_pcd
    distances2 = combined_pcd.compute_nearest_neighbor_distance()
    avg_dist2 = np.mean(distances2)
    combined_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=4*avg_dist2, max_nn=100))
    combined_pcd.orient_normals_consistent_tangent_plane(90)

    # BPA sweep
    radii = o3d.utility.DoubleVector([avg_dist2 * f for f in [1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4, 2.8, 3, 3.5, 4, 4.5]])
    mesh_bpa = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(combined_pcd, radii)

    # Boundary detection for hole filling
    triangles = np.asarray(mesh_bpa.triangles)
    edge_count = defaultdict(int)
    for tri in triangles:
        for e in [tuple(sorted([tri[0], tri[1]])), tuple(sorted([tri[1], tri[2]])), tuple(sorted([tri[2], tri[0]]))]:
            edge_count[e] += 1
    boundary_edges = [e for e, c in edge_count.items() if c == 1]
    adj = defaultdict(list)
    for v1, v2 in boundary_edges:
        adj[v1].append(v2); adj[v2].append(v1)
    visited, loops = set(), []
    for start in adj:
        if start in visited: continue
        loop, current, prev = [], start, None
        while True:
            loop.append(current); visited.add(current)
            next_vertex = next((n for n in adj[current] if n != prev), None)
            if next_vertex is None or next_vertex in visited: break
            prev, current = current, next_vertex
        if len(loop) > 2: loops.append(loop)

    # Poisson smoothing pass
    pcd_from_mesh = mesh_bpa.sample_points_poisson_disk(number_of_points=100000)
    pcd_from_mesh.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=avg_dist2*4, max_nn=100))
    pcd_from_mesh.orient_normals_consistent_tangent_plane(100)
    mesh_p, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_from_mesh, depth=9, scale=1.05, linear_fit=True)
    mesh_p.remove_vertices_by_mask(np.asarray(densities) < np.quantile(np.asarray(densities), 0.02))

    # Hole filling
    final_tris = list(np.asarray(mesh_p.triangles))
    for loop in loops:
        v0 = loop[0]
        for i in range(1, len(loop) - 1):
            final_tris.append([v0, loop[i], loop[i+1]])
    mesh_p.triangles = o3d.utility.Vector3iVector(np.array(final_tris))

    mesh_p = mesh_p.filter_smooth_laplacian(number_of_iterations=5)
    mesh_p.compute_vertex_normals()
    return mesh_p, combined_pcd

@app.route('/static/<path:filename>')
def static_files(filename):
    return app.send_static_file(filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['image']
    image_data = base64.b64decode(data.split(',')[1])
    raw_img = Image.open(BytesIO(image_data)).convert("L").resize((28, 28))

    img_clf = ImageOps.invert(raw_img)
    tensor_clf = transforms.ToTensor()(img_clf).unsqueeze(0).to(DEVICE)

    print("Saving classifier input image to debug/last_classifier_input.png")
    img_clf.save("debug/last_classifier_input.png")
    print("Tensor stats:", tensor_clf.min().item(), tensor_clf.max().item(), tensor_clf.mean().item())

    with torch.no_grad():
        logits = clf_model(tensor_clf)
        probs = torch.softmax(logits, dim=1).squeeze()
        top_probs, top_indices = torch.topk(probs, 5)

    predictions = [{"label": CLASSES[idx], "confidence": float(p)}
                   for p, idx in zip(top_probs, top_indices)]

    return jsonify({"predictions": predictions})


@app.route('/generate_pointcloud', methods=['POST'])
def generate_pointcloud():
    """Step 1: Run model inference and return the dense point cloud immediately."""
    data = request.json['image']
    image_data = base64.b64decode(data.split(',')[1])
    img_rgb = Image.open(BytesIO(image_data)).convert("RGB").resize((28, 28))
    tensor_recon = transforms.ToTensor()(img_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        points = recon_model(tensor_recon).squeeze(0).cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2*avg_dist, max_nn=60))
    pcd.orient_normals_consistent_tangent_plane(50)
    dense_pcd = local_surface_upsample(pcd, k=6)
    dense_pcd = dense_pcd.voxel_down_sample(voxel_size=avg_dist * 0.5)
    cl, ind = dense_pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
    dense_pcd = dense_pcd.select_by_index(ind)
    combined_pcd = pcd + dense_pcd
    dense_points = np.asarray(combined_pcd.points)

    _cached_points['latest'] = points

    # Save point cloud viz HTML
    os.makedirs("static", exist_ok=True)
    fig_points = go.Figure(data=[go.Scatter3d(
        x=dense_points[:,0], y=dense_points[:,1], z=dense_points[:,2],
        mode='markers',
        marker=dict(size=2.0, color=dense_points[:,2], colorscale='Plasma', opacity=0.9)
    )])
    fig_points.update_layout(
        paper_bgcolor='#0f172a',
        scene=dict(aspectmode='data', bgcolor='#0f172a',
                   xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
        margin=dict(l=0, r=0, b=0, t=0)
    )
    fig_points.write_html("static/points_viz.html")
    print(f"Point cloud ready: {len(dense_points)} points")

    return jsonify({"status": "success", "points_count": len(dense_points)})


@app.route('/generate_mesh', methods=['POST'])
def generate_mesh():
    """Step 2: Build the mesh from the cached points (slow part)."""
    points = _cached_points.get('latest')
    if points is None:
        return jsonify({"status": "error", "message": "No point cloud cached. Call /generate_pointcloud first."})

    print("Building advanced mesh...")
    mesh, _ = build_advanced_mesh(points)

    verts = np.asarray(mesh.vertices)
    tris  = np.asarray(mesh.triangles)
    print(f"Mesh: {len(verts)} vertices, {len(tris)} triangles")

    fig_mesh = go.Figure(data=[go.Mesh3d(
        x=verts[:,0], y=verts[:,1], z=verts[:,2],
        i=tris[:,0], j=tris[:,1], k=tris[:,2],
        color='orange', opacity=1.0
    )])
    fig_mesh.update_layout(
        paper_bgcolor='#0f172a',
        scene=dict(aspectmode='data', bgcolor='#0f172a',
                   xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    os.makedirs("static", exist_ok=True)
    o3d.io.write_triangle_mesh("static/airplane_mesh.obj", mesh)
    fig_mesh.write_html("static/mesh_viz.html")

    return jsonify({"status": "success", "vertices": len(verts), "triangles": len(tris)})


if __name__ == '__main__':
    app.run(port=5000, debug=True)