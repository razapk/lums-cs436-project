import cv2
import numpy as np
from pipeline import *
from bundle_adjustment import *
import json

# List of image paths (replace these with your own file paths)
with open('config.json', 'r') as f:
    config = json.load(f)

image_paths = config['images']
panoramas = config['panoramas']
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
correspondence_graph = {}
for i in range(1, len(images)):
    data = np.load(f"correspondences/{i-1}_{i}.npz")
    ptsA = data['ptsA']
    ptsB = data['ptsB']
    if i - 1 not in correspondence_graph:
        correspondence_graph[i - 1] = {}
    if i not in correspondence_graph:
        correspondence_graph[i] = {}
    correspondence_graph[i - 1][i] = (ptsA, ptsB)
    correspondence_graph[i][i - 1] = (ptsB, ptsA)

print("Loaded correspondences from files.")

# Construct 3d points
K = compute_intrinsics_from_exif(image_paths[0])
print("Computing structure from motion...")
Rs, ts, points_3d, colors, points_mapping = structure_from_motion(images, correspondence_graph, K)

# Save results
points_3d = np.array(points_3d)
Rs = np.array(Rs)
ts = np.array(ts)
colors = np.array(colors)
points_mapping = np.array(points_mapping)

np.savez("results/structure_from_motion.npz",
         Rs=Rs,
         ts=ts,
         points_3d=points_3d,
         colors=colors,
         points_mapping=points_mapping)

# Load results
data = np.load("results/structure_from_motion.npz")
Rs = data['Rs']
ts = data['ts']
points_3d = data['points_3d']
colors = data['colors']
points_mapping = data['points_mapping']

# Plot point cloud
# Combine all points into a single point cloud
points_3d = np.array(points_3d)
colors = np.array(colors)
# Plot the final 3D point cloud
# plot_point_cloud(points_3d, colors / 255, title="Final 3D Point Cloud Reconstruction")

# Run bundle adjustment
print("Running bundle adjustment...")
points_2d = points_mapping[:, 3:5]
camera_indices = points_mapping[:, 0].astype(int)
point_indices = points_mapping[:, 2].astype(int)
Rs, ts, points_3d = run_optimized_ba(
    Rs, ts, points_3d,
    K,
    camera_indices,
    point_indices,
    points_2d,
    max_nfev=1000
)

# Save results after BA
np.savez("results/structure_from_motion_ba.npz",
         Rs=Rs,
         ts=ts,
         points_3d=points_3d,
         colors=colors,
         points_mapping=points_mapping)

# Load results after BA
data = np.load("results/structure_from_motion_ba.npz")
Rs = data['Rs']
ts = data['ts']
points_3d = data['points_3d']
colors = data['colors']
points_mapping = data['points_mapping']

# Estimate camera locations
camera_positions = []
for i in range(len(Rs)):
    R = cv2.Rodrigues(Rs[i])[0]
    t = ts[i].reshape(3, 1)
    cam_location = -R.T @ t
    camera_positions.append(cam_location.flatten())
camera_positions = np.array(camera_positions)

# Plot the final 3D point cloud
plot_point_cloud(camera_positions, points_3d, colors, title="3D Point Cloud After Bundle Adjustment")
save_point_cloud_to_file(camera_positions, points_3d, colors, filename="results/point_cloud_after_ba.ply")
print("Saved point cloud after bundle adjustment to results/point_cloud_after_ba.ply")

# Plot point and camera locations in each image
print("Plotting trajectory on images...")
camera_locations_results = []
for i, img in enumerate(images):
    copy = img.copy()
    R = cv2.Rodrigues(Rs[i])[0]
    t = ts[i].reshape(3, 1)
    camera_object = {
        'id': f'camera_{i}',
        'image': f'./{image_paths[i]}',
        'location': camera_positions[i].tolist(),
        'R': R.tolist(),
        't': t.tolist(),
        'K': K.tolist(),
        'width': img.shape[1],
        'height': img.shape[0],
        'other_cameras': []
    }

    for j in range(len(Rs)):
        # Project camera center
        cam2d = K @ (R @ camera_positions[j].reshape(3, 1) + t)
        cam2d = cam2d.flatten()
        if cam2d[2] <= 1e-6:
            camera_object['other_cameras'].append({
                'id': f'camera_{j}',
                'visible': False
            })
            continue
        cam2d /= cam2d[2]
        if np.isnan(cam2d).any() or np.isinf(cam2d).any():
            print(f"Warning: Invalid projection for camera {j} in image {i}. Skipping.")                
            continue
        if cam2d[0] < 0 or cam2d[0] >= img.shape[1] or cam2d[1] < 0 or cam2d[1] >= img.shape[0]:
            camera_object['other_cameras'].append({
                'id': f'camera_{j}',
                'visible': False
            })
            continue
        camera_object['other_cameras'].append({
            'id': f'camera_{j}',
            'visible': True,
            'pixel_location': [float(cam2d[0]), float(cam2d[1])]
        })
        cv2.circle(copy, (int(cam2d[0]), int(cam2d[1])), 15, (0, 0, 255), -1)
        
    camera_locations_results.append(camera_object)
    cv2.imwrite(f'results/image_{i}_with_points_and_cameras.png', copy)

# Panorama stitching
print("Starting panorama stitching...")
panorama_results = []
for i in range(len(panoramas)):
    points = []
    labels = []
    ids = []
    for j in range(len(panoramas)):
        if i == j:
            continue
        first_image_i = image_paths.index(panoramas[i][0])
        first_image_j = image_paths.index(panoramas[j][0])
        camera_positions_j = camera_positions[first_image_j]
        R = cv2.Rodrigues(Rs[first_image_i])[0]
        t = ts[first_image_i].reshape(3, 1)
        location_pixels = K @ (R @ camera_positions_j.reshape(3, 1) + t)
        location_pixels /= location_pixels[2]
        points.append([location_pixels[0][0], location_pixels[1][0]])
        labels.append(f"View {j + 1}")
        ids.append(f'view_{j}')

    panorama_images = [cv2.imread(p) for p in panoramas[i]]
    stitched, modified_points = create_panorama(
        panorama_images,
        points
    )

    cv2.imwrite(f'results/panorama_{i}_stitched.jpg', stitched)
    panorama_results.append({
        'id': f'view_{i}',
        'stitched_image': f'./results/panorama_{i}_stitched.jpg',
        'image_width': stitched.shape[1],
        'image_height': stitched.shape[0],
        'links': [{'point': pt.tolist(), 'label': lbl, 'id': id_} for pt, lbl, id_ in zip(modified_points, labels, ids)]
    })

# Save panorama results to a JS file
with open('results/panorama_results.js', 'w') as f:
    f.write("const panoramaResults = ")
    f.write(json.dumps(panorama_results))
    f.write(";\n")
    f.write("const cameraLocations = ")
    f.write(json.dumps(camera_locations_results))
    f.write(";\n")
