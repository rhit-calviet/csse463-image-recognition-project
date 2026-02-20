# 3d_meshing_examples.py
#
# This code is intended to provide examples of the different 3d meshing outputs generated during testing. 
# It saves off images of each generated mesh, to show the strengths, weaknesses, and differences between the
# different meshing algorithms. Generated with help from GPT 5.2.

import numpy as np
import plotly.graph_objects as go
import open3d as o3d

# Load points
points = np.load("/work/csse463/202620/04/data/ShapeNet/npy/02691156/8bb827904cd9acd36c1cd53dbc9f7b8e.npy")

#----------------------------------------------------------------------#
#------------------Original Meshing without Upsampling-----------------#


# Convert to Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# ---Poisson Surface Reconstruction--- #
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1,
        max_nn=30
    )
)
pcd.orient_normals_consistent_tangent_plane(50)
# Poisson reconstruction
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd,
    depth=8
)
# Remove low-density triangles (clean floating junk)
densities = np.asarray(densities)
density_threshold = np.quantile(densities, 0.02)
vertices_to_remove = densities < density_threshold
mesh.remove_vertices_by_mask(vertices_to_remove)

mesh = mesh.filter_smooth_laplacian(number_of_iterations=5)
mesh.compute_vertex_normals()

# Save mesh
o3d.io.write_triangle_mesh("new_reconstructed_poisson_1.ply", mesh)

# ---Ball Pivoting Algorithm Reconstruction--- #
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.05,
        max_nn=30
    )
)
pcd.orient_normals_consistent_tangent_plane(50)
# Compute average neighbor distance
distances = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
# Adaptive normal estimation
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=2 * avg_dist,
        max_nn=60
    )
)
pcd.orient_normals_consistent_tangent_plane(50)
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd, radii=o3d.utility.DoubleVector([
    avg_dist,
    avg_dist * 1.2,
    avg_dist * 2,
    avg_dist * 3
    ])
)

#Save mesh
o3d.io.write_triangle_mesh("new_reconstructed_bpa_2.ply", mesh)

# ---Ball Pivoting Algorithm Reconstruction (Fine tuning radii)--- #
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.05,
        max_nn=30
    )
)
pcd.orient_normals_consistent_tangent_plane(50)
# Compute average neighbor distance
distances = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
# Adaptive normal estimation
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=2 * avg_dist,
        max_nn=60
    )
)
pcd.orient_normals_consistent_tangent_plane(50)
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd, radii=o3d.utility.DoubleVector([
    avg_dist,
    avg_dist * 1.2,
    avg_dist * 1.4,
    avg_dist * 1.6,
    avg_dist * 1.8,
    avg_dist * 2.0,
    avg_dist * 2.4,
    avg_dist * 2.8,
    avg_dist * 3,
    avg_dist * 3.5,
    avg_dist * 4,
    avg_dist * 4.5
    ])
)

#Save mesh
o3d.io.write_triangle_mesh("new_reconstructed_bpa_fine_tuned_radii_3.ply", mesh)

#----------------------------------------------------------------------#
#------------------Meshing without Upsampling (Dense Point Cloud)-----------------#

def local_surface_upsample(pcd, k=6):
    points = np.asarray(pcd.points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    new_points = []

    for i, p in enumerate(points):
        _, idx, _ = kdtree.search_knn_vector_3d(p, k)

        for j in idx[1:]:
            midpoint = (p + points[j]) / 2.0
            new_points.append(midpoint)

    all_points = np.vstack([points, np.array(new_points)])

    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(all_points)

    return new_pcd

def remove_sparse_edges(pcd, nb_neighbors=10, std_ratio=2.0):
    """
    Remove sparse or isolated points (typically at edges or outliers).
    
    Args:
        pcd: Open3D point cloud
        nb_neighbors: number of neighbors to consider
        std_ratio: threshold in standard deviations above mean distance
    
    Returns:
        filtered_pcd: cleaned Open3D point cloud
    """
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                             std_ratio=std_ratio)
    filtered_pcd = pcd.select_by_index(ind)
    return filtered_pcd

distances = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)

# Estimate normals
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=2 * avg_dist,
        max_nn=60
    )
)
pcd.orient_normals_consistent_tangent_plane(50)
dense_pcd = local_surface_upsample(pcd, k=6) # upsample point cloud using midpoints
# Downsample to reduce noise
dense_pcd = dense_pcd.voxel_down_sample(
    voxel_size=avg_dist * 0.5
)
# Remove sparse edges to clean noise again
dense_pcd = remove_sparse_edges(dense_pcd, nb_neighbors=10, std_ratio=2.0)
pcd = pcd + dense_pcd
# Recompute distances and normals for dense cloud
distances = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)

pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=4 * avg_dist,
        max_nn=100
    )
)
pcd.orient_normals_consistent_tangent_plane(90)

# Fine tuning with multiple radii
radii = o3d.utility.DoubleVector([
    avg_dist,
    avg_dist * 1.2,
    avg_dist * 1.4,
    avg_dist * 1.6,
    avg_dist * 1.8,
    avg_dist * 2.0,
    avg_dist * 2.4,
    avg_dist * 2.8,
    avg_dist * 3,
    avg_dist * 3.5,
    avg_dist * 4,
    avg_dist * 4.5
    ])

mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd, radii
)
# Added cleanup
mesh = mesh.remove_degenerate_triangles()
mesh = mesh.remove_duplicated_triangles()
mesh = mesh.remove_duplicated_vertices()
mesh = mesh.remove_non_manifold_edges()
mesh.remove_unreferenced_vertices()

o3d.io.write_triangle_mesh("new_reconstructed_bpa_upsampled_4.ply", mesh)


#----------------------------------------------------------------------#
#---Meshing without Upsampling (Dense Point Cloud) and Poisson Smoothing---#
distances = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)

# Adaptive normal estimation
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=2 * avg_dist,
        max_nn=60
    )
)
pcd.orient_normals_consistent_tangent_plane(50)

dense_pcd = local_surface_upsample(pcd, k=6)

dense_pcd = dense_pcd.voxel_down_sample(
    voxel_size=avg_dist * 0.5
)
dense_pcd = remove_sparse_edges(dense_pcd, nb_neighbors=10, std_ratio=2.0)
pcd = pcd + dense_pcd
distances = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
# New normal estimation on dense cloud
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=4 * avg_dist,
        max_nn=100
    )
)
pcd.orient_normals_consistent_tangent_plane(90)

radii = o3d.utility.DoubleVector([
    avg_dist,
    avg_dist * 1.2,
    avg_dist * 1.4,
    avg_dist * 1.6,
    avg_dist * 1.8,
    avg_dist * 2.0,
    avg_dist * 2.4,
    avg_dist * 2.8,
    avg_dist * 3,
    avg_dist * 3.5,
    avg_dist * 4,
    avg_dist * 4.5
    ])
# BPA meshing
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd, radii
)
# Cleanup
mesh = mesh.remove_degenerate_triangles()
mesh = mesh.remove_duplicated_triangles()
mesh = mesh.remove_duplicated_vertices()
mesh = mesh.remove_non_manifold_edges()
mesh.remove_unreferenced_vertices()
mesh = mesh.filter_smooth_laplacian(20)

# Poisson reconstruction based on BPA output resampling
mesh_bpa = mesh
pcd_from_mesh = mesh_bpa.sample_points_poisson_disk(
    number_of_points=200000
)

pcd_from_mesh.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=avg_dist * 4,
        max_nn=100
    )
)

pcd_from_mesh.orient_normals_consistent_tangent_plane(100)

mesh_watertight, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd_from_mesh,
    depth=9,
    scale=1.05,
    linear_fit=True
)

densities = np.asarray(densities)
density_threshold = np.quantile(densities, 0.02)

vertices_to_remove = densities < density_threshold
mesh_watertight.remove_vertices_by_mask(vertices_to_remove)

mesh_watertight = mesh_watertight.filter_smooth_laplacian(1)

mesh = mesh_watertight
o3d.io.write_triangle_mesh("new_reconstructed_final_5.ply", mesh)

