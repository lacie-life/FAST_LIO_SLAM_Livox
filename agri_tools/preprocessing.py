import open3d as o3d
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import math
import random

# ---------- CONFIG ----------
PCD_PATH = "test.pcd"
VOXEL_SIZE = 0.01
ROTATION_ANGLE_DEG = 45
ROI_BOUNDS = {
    "x": (-4, 2),
    "y": (-3.5, 3),
    "z": (0.0, 2.5)
}
ROW_Y_GAP = 0.8
PLANT_MIN_HEIGHT = 0.3
GROUND_Z = 0.0
GROUND_TOLERANCE = 0.1

# Clustering params
PLANT_CLUSTER_EPS = 0.1
PLANT_CLUSTER_MIN_POINTS = 5

# ---------- STEP 1: Load & De-rotate ----------
print("\U0001F504 Loading point cloud...")
pcd = o3d.io.read_point_cloud(PCD_PATH)

theta = math.radians(ROTATION_ANGLE_DEG)
Rx = np.array([
    [1, 0, 0],
    [0, math.cos(theta), -math.sin(theta)],
    [0, math.sin(theta), math.cos(theta)]
])
pcd.rotate(Rx, center=(0, 0, 0))

# ---------- STEP 2: Crop ROI ----------
points = np.asarray(pcd.points)
x_min, x_max = ROI_BOUNDS["x"]
y_min, y_max = ROI_BOUNDS["y"]
z_min, z_max = ROI_BOUNDS["z"]

roi_mask = (
    (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
    (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
    (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
)
roi_points = points[roi_mask]
roi_pcd = o3d.geometry.PointCloud()
roi_pcd.points = o3d.utility.Vector3dVector(roi_points)

# ---------- STEP 3: Denoising ----------
roi_pcd, _ = roi_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
roi_pcd = roi_pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)

# ---------- STEP 4: Row detection ----------
clean_points = np.asarray(roi_pcd.points)
row_geometries = []
row_bboxes = []
row_centers = []

bins = {}
for pt in clean_points:
    bin_key = round(pt[1] / ROW_Y_GAP)
    bins.setdefault(bin_key, []).append(pt)

for bin_id, bin_pts in bins.items():
    bin_pts = np.array(bin_pts)
    if len(bin_pts) < 100:
        continue

    min_z = np.min(bin_pts[:, 2])
    if abs(min_z - GROUND_Z) > GROUND_TOLERANCE:
        continue

    row_pcd = o3d.geometry.PointCloud()
    row_pcd.points = o3d.utility.Vector3dVector(bin_pts)
    row_pcd.paint_uniform_color([random.random(), random.random(), random.random()])
    row_geometries.append(row_pcd)
    row_centers.append(np.mean(bin_pts, axis=0))

    row_bbox = row_pcd.get_axis_aligned_bounding_box()
    row_bbox.color = [0.2, 0.5, 1.0]  # blue for rows
    row_bboxes.append(row_bbox)

# ---------- STEP 5: Plant detection (improved with convex hull filtering) ----------
plant_records = []
plant_geometries = []

for row_pcd in row_geometries:
    row_pts = np.asarray(row_pcd.points)
    clustering = DBSCAN(eps=PLANT_CLUSTER_EPS, min_samples=PLANT_CLUSTER_MIN_POINTS).fit(row_pts)
    labels = clustering.labels_
    for label in set(labels):
        if label == -1:
            continue
        cluster_pts = row_pts[labels == label]
        cluster_pcd = o3d.geometry.PointCloud()
        cluster_pcd.points = o3d.utility.Vector3dVector(cluster_pts)

        # Lọc nhiễu trong cụm
        cluster_pcd, _ = cluster_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
        cluster_pcd = cluster_pcd.voxel_down_sample(voxel_size=0.01)

        if len(cluster_pcd.points) < 10:
            continue

        z_vals = np.asarray(cluster_pcd.points)[:, 2]
        plant_height = np.percentile(z_vals, 95) - np.percentile(z_vals, 5)
        if plant_height < PLANT_MIN_HEIGHT:
            continue

        z_min_cluster = np.min(z_vals)
        if z_min_cluster > 0.2:
            continue

        # Convex hull filtering (to get tighter shape and remove flying points)
        try:
            hull, _ = cluster_pcd.compute_convex_hull()
            hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
            hull_ls.paint_uniform_color([1.0, 0, 0])
            bbox = cluster_pcd.get_axis_aligned_bounding_box()
            bbox.color = [1.0, 0, 0]
            plant_geometries.extend([cluster_pcd, bbox, hull_ls])
        except:
            continue

        color = [random.random(), random.random(), random.random()]
        cluster_pcd.paint_uniform_color(color)

        plant_records.append({
            "x": bbox.get_center()[0],
            "y": bbox.get_center()[1],
            "z": bbox.get_center()[2],
            "height_m": round(plant_height, 3),
            "bbox_min": bbox.get_min_bound().tolist(),
            "bbox_max": bbox.get_max_bound().tolist()
        })

# ---------- Export ----------
df = pd.DataFrame(plant_records)
df.to_csv("plants_detected.csv", index=False)
print("✅ Xuất kết quả: plants_detected.csv")
print(df.head())

# ---------- Visualization ----------
axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
roi_pcd.paint_uniform_color([0.7, 0.7, 0.7])

o3d.visualization.draw_geometries([roi_pcd, axes] + row_bboxes + plant_geometries)