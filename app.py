import cv2
import numpy as np
from pipeline import *
from bundle_adjustment import *

# List of image paths (replace these with your own file paths)
image_paths = [
    "images/00.jpg", "images/01.jpg", "images/02.jpg", "images/03.jpg", "images/04.jpg",
    "images/05.jpg", "images/06.jpg", "images/07.jpg", "images/08.jpg", "images/09.jpg",
    "images/10.jpg"
]

# Load images
images = [cv2.imread(p) for p in image_paths]

# Correspondences
# correspondence_graph = {}
# print("Detecting and matching SIFT features between consecutive images...")
# for i in range(1, len(images)):
#     ptsA, ptsB, matches, kp1, kp2 = detect_and_match_sift(images[i - 1], images[i])
#     if i - 1 not in correspondence_graph:
#         correspondence_graph[i - 1] = {}
#     if i not in correspondence_graph:
#         correspondence_graph[i] = {}
#     correspondence_graph[i - 1][i] = (ptsA, ptsB)
#     correspondence_graph[i][i - 1] = (ptsB, ptsA)
#     print(f"Image {i-1} to Image {i}: {len(matches)} matches found.")

#     # Save these points
#     np.savez(f"correspondences/{i-1}_{i}.npz", ptsA=ptsA, ptsB=ptsB)

# Load points
correspondences_graph = {}
for i in range(1, len(images)):
    data = np.load(f"correspondences/{i-1}_{i}.npz")
    ptsA = data['ptsA']
    ptsB = data['ptsB']
    if i - 1 not in correspondences_graph:
        correspondences_graph[i - 1] = {}
    if i not in correspondences_graph:
        correspondences_graph[i] = {}
    correspondences_graph[i - 1][i] = (ptsA, ptsB)
    correspondences_graph[i][i - 1] = (ptsB, ptsA)

print("Loaded correspondences from files.")

# Construct 3d points
K = compute_intrinsics_from_exif(image_paths[0])
Rs, ts, all_pts_3d, colors, points_mapping = structure_from_motion(images, correspondences_graph, K)

# Save results
points_3d = np.array(all_pts_3d)
Rs = np.array(Rs)
ts = np.array(ts)
colors = np.array(colors)
points_mapping = np.array(points_mapping)

np.savez("structure_from_motion.npz",
         Rs=Rs,
         ts=ts,
         points_3d=points_3d,
         colors=colors,
         points_mapping=points_mapping)

# Load results
data = np.load("structure_from_motion.npz")
Rs = data['Rs']
ts = data['ts']
points_3d = data['points_3d']
colors = data['colors']
points_mapping = data['points_mapping']

# # Plot point cloud
# # Combine all points into a single point cloud
# all_pts_3d = np.array(points_3d)
# colors = np.array(colors)
# # Plot the final 3D point cloud
# plot_point_cloud(points_3d, colors / 255, title="Final 3D Point Cloud Reconstruction")

# Run bundle adjustment
points_2d = points_mapping[:, 3:5]
camera_indices = points_mapping[:, 0].astype(int)
point_indices = points_mapping[:, 2].astype(int)
Rs, ts, points_3d = run_bundle_adjustment(
    Rs, ts, points_3d,
    K,
    camera_indices,
    point_indices,
    points_2d,
)

# Save results after BA
np.savez("structure_from_motion_ba.npz",
         Rs=Rs,
         ts=ts,
         points_3d=points_3d,
         colors=colors,
         points_mapping=points_mapping)

# Plot point cloud after BA
# Combine all points into a single point cloud
all_pts_3d = np.array(points_3d)
colors = np.array(colors)
# Plot the final 3D point cloud
plot_point_cloud(points_3d, colors / 255, title="3D Point Cloud After Bundle Adjustment")
