import cv2
import numpy as np
from matplotlib import pyplot as plt
from exif import Image
from scipy.spatial import cKDTree
import open3d as o3d

def detect_and_match_sift(img1, img2):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find the keypoints and descriptors with SIFT
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Extract location of good matches
    points1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    return points1, points2, good_matches, kp1, kp2

def show_correspondences(img1, img2, pts1, pts2, title="Feature Correspondences"):
    # Ensure inputs are numpy arrays
    pts1, pts2 = np.array(pts1), np.array(pts2)

    # Combine images horizontally
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape
    h = max(h1, h2)
    combined = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    combined[:h1, :w1] = img1
    combined[:h2, w1:w1 + w2] = img2

    # Plot combined image
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(title)

    # Draw correspondences
    for (x1, y1), (x2, y2) in zip(pts1, pts2):
        plt.plot([x1, x2 + w1], [y1, y2], 'r-', linewidth=1)
        plt.plot(x1, y1, 'go', markersize=5)
        plt.plot(x2 + w1, y2, 'bo', markersize=5)

    plt.tight_layout()
    plt.show()

def compute_intrinsics_from_exif(image_path):
    # --- Load EXIF ---
    with open(image_path, "rb") as f:
        img = Image(f)

    if not img.has_exif:
        raise ValueError("Image has no EXIF data")

    # --- Extract required fields ---
    if not hasattr(img, "focal_length"):
        raise ValueError("No focal_length in EXIF")

    if not hasattr(img, "focal_length_in_35mm_film"):
        raise ValueError("No focal_length_in_35mm_film in EXIF")

    f_mm = float(img.focal_length)                      # e.g. 5.54 mm
    f35 = float(img.focal_length_in_35mm_film)          # e.g. 23 mm
    W = img.pixel_x_dimension                           # e.g. 4080 px
    H = img.pixel_y_dimension                           # e.g. 2296 px

    # --- Compute crop factor ---
    crop = f35 / f_mm                                   # ≈ 4.15

    # --- Compute sensor physical size ---
    SENSOR_WIDTH_35MM = 36.0
    SENSOR_HEIGHT_35MM = 24.0

    sensor_width_mm = SENSOR_WIDTH_35MM / crop
    sensor_height_mm = SENSOR_HEIGHT_35MM / crop

    # --- Compute pixel size ---
    pixel_size_x = sensor_width_mm / W
    pixel_size_y = sensor_height_mm / H

    # --- Compute focal length in pixels ---
    fx = f_mm / pixel_size_x
    fy = f_mm / pixel_size_y

    # --- Principal point (image center) ---
    cx = W / 2
    cy = H / 2

    # --- Build intrinsic matrix ---
    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ])

    return K

def triangulate_dlt(P1, P2, pt1, pt2):
    A = np.zeros((4, 4), dtype=np.float64)
    u1, v1 = pt1
    u2, v2 = pt2

    # Camera 1 constraints (u1, v1)
    A[0, :] = u1 * P1[2, :] - P1[0, :]
    A[1, :] = v1 * P1[2, :] - P1[1, :]
    
    # Camera 2 constraints (u2, v2)
    A[2, :] = u2 * P2[2, :] - P2[0, :]
    A[3, :] = v2 * P2[2, :] - P2[1, :]

    # Solve A * X = 0 using SVD (Singular Value Decomposition)
    _, _, V = np.linalg.svd(A)
    
    # The solution is the last column of V (or last row of V in numpy's output V matrix)
    X_homogeneous = V[-1]
    
    # Convert from homogeneous (X, Y, Z, W) to Cartesian (X, Y, Z)
    if np.isclose(X_homogeneous[3], 0):
        # Handle case where point is at infinity
        print("Point at infinity encountered during triangulation.")
        return None

    X_cartesian = X_homogeneous[:3] / X_homogeneous[3]

    # Calculate reconstruction error
    pt1_reproj = P1 @ X_homogeneous
    pt1_reproj /= pt1_reproj[2]
    error1 = np.linalg.norm(pt1_reproj[:2] - pt1)
    pt2_reproj = P2 @ X_homogeneous
    pt2_reproj /= pt2_reproj[2]
    error2 = np.linalg.norm(pt2_reproj[:2] - pt2)

    if error1 > 15 or error2 > 15:
        return None

    return X_cartesian


def plot_point_cloud(pts_3D, colors=None, title="3D Point Cloud Reconstruction"):
    """
    Visualizes a 3D point cloud using Open3D.
    
    Args:
        pts_3D: Numpy array of shape (N, 3)
        colors: Numpy array of shape (N, 3) (RGB). Optional.
        title: Window title string.
    """
    pts_3D = np.asarray(pts_3D)
    
    if pts_3D.size == 0:
        print("No points to plot.")
        return

    # 1. Initialize Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_3D)

    # 2. Handle Colors
    if colors is not None and len(colors) == len(pts_3D):
        colors = np.asarray(colors)
        
        # Open3D expects colors in range [0, 1] (Float)
        # Common bug: Passing [0, 255] (Int) results in all points being black/white.
        if colors.max() > 1.1: 
            colors = colors.astype(np.float64) / 255.0
            
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        # Paint uniform blue if no colors provided
        pcd.paint_uniform_color([0.1, 0.1, 0.7])

    # 3. Create Camera/Origin Marker
    # Instead of a single point, we draw a Coordinate Frame (Red=X, Green=Y, Blue=Z)
    # The 'size' parameter adjusts how big the axes look relative to your cloud
    avg_span = np.abs(pts_3D.max() - pts_3D.min()) * 0.1
    camera_marker = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=avg_span, origin=[0, 0, 0]
    )

    # 4. Visualization Settings
    # We create a visualization window with the point cloud and the origin
    print(f"[{title}] Opening visualizer...")
    print("Controls:\n  [Left Click + Drag]: Rotate\n  [Scroll Wheel]: Zoom\n  [Right Click + Drag]: Pan\n  [+/-]: Increase/Decrease point size")
    
    o3d.visualization.draw_geometries(
        [pcd, camera_marker],
        window_name=title,
        width=1280,
        height=720,
        left=50,
        top=50,
        point_show_normal=False
    )


def find_rotation_translation(K, ptsA, ptsB):
    F, inlier_mask = cv2.findFundamentalMat(
        ptsA, ptsB, 
        method=cv2.FM_RANSAC, 
        ransacReprojThreshold=3, # Reprojection error threshold in pixels
        confidence=0.99
    )
    
    if F is None:
        print("Error: Could not estimate Fundamental Matrix (F).")
        return None

    # Filter points to keep only inliers
    ptsA_inliers = ptsA[inlier_mask.ravel() == 1]
    ptsB_inliers = ptsB[inlier_mask.ravel() == 1]
    
    if ptsA_inliers.shape[0] < 8: # Minimum 8 points for essential matrix estimation
        print("Error: Too few inlier points remaining for reliable pose recovery.")
        return None

    # E is calculated from F and K (E = K^T * F * K).
    E, mask_e = cv2.findEssentialMat(
        ptsA_inliers, ptsB_inliers, 
        cameraMatrix=K, 
        method=cv2.RANSAC, 
        prob=0.99, 
        threshold=3
    )
    
    if E is None:
        print("Error: Could not estimate Essential Matrix (E).")
        return None

    # Filter points again based on E mask
    ptsA_final = ptsA_inliers[mask_e.ravel() == 1]
    ptsB_final = ptsB_inliers[mask_e.ravel() == 1]
    inlier_idx = np.where(inlier_mask.ravel() == 1)[0][mask_e.ravel() == 1]
    
    # Recovers the 4 possible (R, T) pairs and selects the one
    # that places points in front of both cameras (cheirality check).
    _, R, T, _ = cv2.recoverPose(E, ptsA_final, ptsB_final, K)

    return inlier_idx, R, T

def find_3D_points(K, R, T, ptsA, ptsB):
    # Projection matrix for camera 1 (assumed at origin)
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))

    # Projection matrix for camera 2
    P2 = K @ np.hstack((R, T))

    pts_3D = []
    for ptA, ptB in zip(ptsA, ptsB):
        X = triangulate_dlt(P1, P2, ptA, ptB)
        pts_3D.append(X)

    return pts_3D

def find_matching_features(ptsA, ptsB, threshold=2.0):
    """
    Finds pairs of points (i, j) where distance(ptsA[i], ptsB[j]) < threshold.
    Optimized using cKDTree.
    """
    # Ensure inputs are numpy arrays
    ptsA = np.atleast_2d(ptsA)
    ptsB = np.atleast_2d(ptsB)

    # 1. Build the K-D Tree on the second set of points (ptsB)
    # Construction complexity: O(M log M)
    tree = cKDTree(ptsB)

    # 2. Query the tree for points in ptsB within 'threshold' of points in ptsA
    indices_list = tree.query_ball_point(ptsA, r=threshold)

    # 3. Flatten the results into the expected format [(i, j), ...]
    matches = []
    for i, neighbors_in_B in enumerate(indices_list):
        for j in neighbors_in_B:
            matches.append((i, j))

    return matches

def structure_from_motion(images, correspondences_graph, K):
    Rs = [np.zeros((3))]
    ts = [np.zeros(3)]
    all_pts_3d = []
    colors = []
    points_mapping = []

    for i in range(1, len(images)):
        ptsA, ptsB = correspondences_graph[i - 1][i]
        inlier_idx, R, t = find_rotation_translation(K, ptsA, ptsB)

        if i == 1:
            ptsA_inliers = ptsA[inlier_idx]
            ptsB_inliers = ptsB[inlier_idx]
            pts_3D = find_3D_points(K, R, t, ptsA_inliers, ptsB_inliers)
            inlier_idx = [idx for j, idx in enumerate(inlier_idx) if pts_3D[j] is not None]
            pts_3D = [pt for pt in pts_3D if pt is not None]
            # Convert R to rodrigues from matrix form
            Rs.append(cv2.Rodrigues(R)[0].reshape(3))
            ts.append(t.reshape(3))

            for j, idx in enumerate(inlier_idx):
                all_pts_3d.append(pts_3D[j])
                # Frame Idx, PointIdx, 3d Point Idx, 2d Point x , 2d Point y, 3d Point X, Y, Z
                points_mapping.append((i - 1, idx, len(all_pts_3d) - 1, ptsA_inliers[j][0], ptsA_inliers[j][1], pts_3D[j][0], pts_3D[j][1], pts_3D[j][2]))
                points_mapping.append((i, idx, len(all_pts_3d) - 1, ptsB_inliers[j][0], ptsB_inliers[j][1], pts_3D[j][0], pts_3D[j][1], pts_3D[j][2]))
                # Get color
                x, y = map(int, ptsB_inliers[j])
                color = images[i][y, x]
                colors.append(color[::-1])

            print(f"Image {i}: {len(pts_3D)} 3D points added from first image pair.")

        else:
            ptsA_inliers = ptsA[inlier_idx]
            ptsB_inliers = ptsB[inlier_idx]
            ptsA_records = [record for record in points_mapping if record[0] == i - 1]
            ptsA_2d_points = [record[3:5] for record in ptsA_records]
            ptsA_3d_points = [record[5:8] for record in ptsA_records]
            ptsA_3d_idx = [record[2] for record in ptsA_records]
            matching_idx = find_matching_features(ptsA_2d_points, ptsA_inliers)
            print(f"Image {i}: {len(matching_idx)} matching features found.")

            ptsA_3d_points_filtered = [ptsA_3d_points[idxA] for idxA, idxB in matching_idx]
            ptsB_2d_points_filtered = [ptsB_inliers[idxB] for idxA, idxB in matching_idx]
            _, rvec, t, inliers = cv2.solvePnPRansac(
                np.array(ptsA_3d_points_filtered),
                np.array(ptsB_2d_points_filtered),
                K,
                None,
                reprojectionError=8.0,
                confidence=0.99,
                iterationsCount=100
            )
            t = t.flatten()
            Rs.append(rvec.reshape(3))
            ts.append(t.reshape(3))
            R = cv2.Rodrigues(rvec)[0]

            # Add correspondences for existing points
            existing_idx = set([idxB for idxA, idxB in matching_idx])
            # for idxA, idxB in matching_idx:
            #     pt_3D = ptsA_3d_points[idxA]
            #     pt_3D_idx = ptsA_3d_idx[idxA]
            #     pt_2D = ptsB_inliers[idxB]
            #     pt_2D_idx = inlier_idx[idxB]
            #     points_mapping.append((i, pt_2D_idx, pt_3D_idx, pt_2D[0], pt_2D[1], pt_3D[0], pt_3D[1], pt_3D[2]))

            # Remove points with existing 3d coordinates
            inlier_idx = [idx for j, idx in enumerate(inlier_idx) if j not in existing_idx]
            ptsA_inliers = [ptsA_inliers[j] for j in range(len(ptsA_inliers)) if j not in existing_idx]
            ptsB_inliers = [ptsB_inliers[j] for j in range(len(ptsB_inliers)) if j not in existing_idx]

            # Find 3d points
            bad_points = 0
            for j in range(len(ptsA_inliers)):
                P1 = K @ np.hstack((cv2.Rodrigues(Rs[i - 1])[0], ts[i - 1].reshape(3, 1)))
                P2 = K @ np.hstack((cv2.Rodrigues(Rs[i])[0], ts[i].reshape(3, 1)))
                pt_3D = triangulate_dlt(P1, P2, ptsA_inliers[j], ptsB_inliers[j])
                if pt_3D is None:
                    bad_points += 1
                    continue

                all_pts_3d.append(pt_3D)
                points_mapping.append((i - 1, inlier_idx[j], len(all_pts_3d) - 1, ptsA_inliers[j][0], ptsA_inliers[j][1], pt_3D[0], pt_3D[1], pt_3D[2]))
                points_mapping.append((i, inlier_idx[j], len(all_pts_3d) - 1, ptsB_inliers[j][0], ptsB_inliers[j][1], pt_3D[0], pt_3D[1], pt_3D[2]))
                # Get color
                x, y = map(int, ptsB_inliers[j])
                color = images[i][y, x]
                colors.append(color[::-1])

            print(f"Image {i}: {len(ptsA_inliers) - bad_points} new 3D points added. Bad points: {bad_points}")

    return Rs, ts, all_pts_3d, colors, points_mapping