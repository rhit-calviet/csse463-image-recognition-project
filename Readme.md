# Sketch Classification & 3D Reconstruction
Matteo Calviello, Tommaso Calviello, Conner O'Connell, Weston Seybold

This repository contains all of the experimental code for our CSSE 463 final project. It spans a pipeline thhat includes:

- Sketch classification models for 20–30 QuickDraw categories.
- Sketch → 3D point-cloud reconstruction of ShapeNet airplanes.

Everything here is documented so you can recreate our results, run our trained models, or extend the experiments.

---

## Repository Layout

- `Classification/` – PyTorch scripts for the AlexNet, ResNet18, and custom CNN experiments.
- `Reconstruction/` – End-to-end training code plus data-prep notebooks for the sketch-to-point-cloud model.
- `Meshing/` – Standalone scripts for testing different meshing strategies on reconstructed point clouds.
- `sketch_to_3d/` – Flask + Plotly web demo.
- `test_results/` – Example `.ply` meshes that we generated while validating the pipeline.

There is also a `models` and `data` folder that is not in the repository. Save the models trained in the `models` folder and download the data in the `data` folder. How and where to download the data is explained below.

## Environment Setup

1. Install Python 3.10+ and create a virtual environment:
2. Install PyTorch
3. Install the remaining Python packages:
   ```bash
   pip install numpy pandas scikit-learn matplotlib pillow tqdm flask plotly open3d
   ```

## Data Preparation

### 1. QuickDraw Sketch Classification

1. Visit the [QuickDraw Cloud bucket](https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap) and download the `.npy` files for the 20 classes used in the optimized CNNs:
   ```
   cat, dog, airplane, car, tree,
   apple, banana, bird, basketball, book,
   butterfly, chair, cloud, cow, flower,
   hand, horse, umbrella, star, sun
   ```
   There is also the improved version to expand to a 30-class list (see `sketch_to_3d/app.py`).
2. Place the `.npy` files in `data/` (e.g., `data/cat.npy`, `data/dog.npy`, ...).

### 2. ShapeNet Sketch → Point Cloud Reconstruction

1. Download the ShapeNet data from [HuggingFace](https://huggingface.co/datasets/ShapeNet/ShapeNetCore). It is a gated dataset so you will need to register in order to get access.
2. Unpack the archive so the raw `.obj` / `.off` files are in `data/ShapeNet/3d`.
3. Run `Reconstruction/2d-3dprocessing.ipynb`. The notebook:
   - Converts ShapeNet meshes to dense `.npy` point clouds (`data/ShapeNet/npy/02691156/*.npy`).
   - Generates paired sketch renderings (`data/ShapeNet/2d/02691156/*_sketch.png`).

### 3. Trained Weights

- The classification scripts save checkpoints to `models/` (e.g., `optimized_cnn_epochXX.pt`).
- `Reconstruction/reconstruction_model.py` writes `models/best_model.pt` and intermediate plots to `Reconstruction/plots/`.
- The Flask demo expects:
  - `models/simpleCNN_30classes.pth` (sketch classifier).
  - `models/complete_model.pt` (advanced reconstruction model checkpoint).
Feel free to rename or replace these lines with whatever model you end up running.

## Interactive Demo (`sketch_to_3d/`)

1. Ensure the checkpoints listed in [Trained Weights](#3-trained-weights) are present.
2. Launch the Flask app:
   ```bash
   python sketch_to_3d/app.py
   ```
3. Navigate to `http://127.0.0.1:5000/`. Draw a sketch, run classification, generate the point cloud, and finally build the mesh.  
