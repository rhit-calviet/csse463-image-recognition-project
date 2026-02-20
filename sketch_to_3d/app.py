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

# tommaso and weston code for models
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

class SketchEncoder(nn.Module):

    def __init__(self, in_channels=3, latent_dim=512):
        super().__init__()
        
        # Input: (batch, in_channels, 28, 28)
        self.features = nn.Sequential(
            # Block 1: 28x28 -> 14x14
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            # Block 2: 14x14 -> 7x7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),

            # Block 3: 7x7 -> 3x3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4)
        )
        
        # After features, the shape is (batch, 128, 3, 3)
        # 128 * 3 * 3 = 1152
        
        self.latent_layer = nn.Sequential(
            nn.Linear(128 * 3 * 3, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, latent_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        z = self.latent_layer(x)
        return z

# model
class Sketch2Point(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = SketchEncoder(in_channels=3, latent_dim=512)
        self.decoder = FoldingDecoder()

    def forward(self,x):
        z = self.encoder(x)
        return self.decoder(z)

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
recon_model.load_state_dict(torch.load("../../models/advanced_encoder_decoder_model.pt", map_location=DEVICE))
recon_model.eval()

# Store raw points between requests so /generate_mesh can reuse them
_cached_points = {}

# connors code
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

    radii = o3d.utility.DoubleVector([avg_dist2 * f for f in [1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4, 2.8, 3, 3.5, 4, 4.5]])
    mesh_bpa = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(combined_pcd, radii)

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

# putting everything together
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