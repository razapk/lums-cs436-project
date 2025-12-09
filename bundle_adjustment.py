import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.sparse import coo_matrix
import time

def project_vectorized(points_3d, camera_params, K):
    """
    Vectorized projection that handles full K matrix (including skew).
    points_3d: (N, 3) 
    camera_params: (N, 6) [r_vec, t]
    K: (3, 3) Intrinsic matrix
    """
    # 1. Rotate (World -> Camera)
    # Note: Rotation.from_rotvec is the bottleneck here, but necessary for correctness
    r_vecs = camera_params[:, :3]
    t_vecs = camera_params[:, 3:]
    
    rot = Rotation.from_rotvec(r_vecs)
    points_cam = rot.apply(points_3d)
    
    # 2. Translate
    points_cam += t_vecs
    
    # 3. Normalize (X/Z, Y/Z)
    # Avoid division by zero by adding a tiny epsilon if needed, 
    # though valid points should have Z > 0
    z = points_cam[:, 2, np.newaxis]
    points_normalized = points_cam[:, :2] / z
    
    # 4. Apply Intrinsics (K @ [x, y, 1])
    # Equiv: u = fx*x + s*y + cx, v = fy*y + cy
    # We use matrix multiplication for full correctness: 
    # [u, v] = [x, y] @ K_2x2^T + [cx, cy]
    
    k_2x2 = K[:2, :2] # [[fx, s], [0, fy]]
    k_trans = K[:2, 2] # [cx, cy]
    
    projected = points_normalized @ k_2x2.T + k_trans
    
    return projected

def vectorized_reprojection_error(params, num_cameras, num_points, camera_indices, point_indices, points_2d, K):
    # Unpack
    camera_params = params[:num_cameras * 6].reshape((num_cameras, 6))
    points_3d = params[num_cameras * 6:].reshape((num_points, 3))
    
    # Select params per observation
    cameras_for_obs = camera_params[camera_indices]
    points_for_obs = points_3d[point_indices]
    
    # Project
    projected_points = project_vectorized(points_for_obs, cameras_for_obs, K)
    
    # Residuals
    return (projected_points - points_2d).ravel()

def build_bundle_adjustment_sparsity(num_cameras, num_points, camera_indices, point_indices):
    """
    Constructs the Jacobian sparsity matrix using COO format for speed.
    """
    n_obs = len(camera_indices)
    
    # We have 2 residuals (u, v) per observation
    m = n_obs * 2
    n = num_cameras * 6 + num_points * 3
    
    # --- 1. Camera Block Indices ---
    # Each observation i affects residuals 2*i and 2*i+1
    # It depends on camera parameters 6*c to 6*c+5
    
    obs_idxs = np.arange(n_obs)
    
    # Repeat observation indices for u and v rows
    row_u = obs_idxs * 2
    row_v = obs_idxs * 2 + 1
    
    # We need to link these rows to 6 camera columns
    # We repeat the rows 6 times (for the 6 params)
    rows_cam = np.repeat(np.stack([row_u, row_v], axis=1), 6, axis=None) # Shape (12 * n_obs,)
    
    # Columns: For each obs, the cam index is c. The cols are 6*c ... 6*c+5
    cam_base_cols = camera_indices * 6
    # Create the offsets [0, 1, 2, 3, 4, 5]
    col_offsets = np.tile(np.arange(6), 2 * n_obs) 
    # Repeat the base cols for u and v, then repeat for the offsets logic... 
    # Actually, simpler to use broadcasting:
    
    # Let's do it explicitly to avoid broadcasting confusion:
    rows_list = []
    cols_list = []
    
    # Camera Params (6 per obs, applied to both u and v residuals)
    for p in range(6):
        # Param p affects u-residual
        rows_list.append(row_u)
        cols_list.append(camera_indices * 6 + p)
        # Param p affects v-residual
        rows_list.append(row_v)
        cols_list.append(camera_indices * 6 + p)

    # Point Params (3 per obs, applied to both u and v residuals)
    for p in range(3):
        # Param p affects u-residual
        rows_list.append(row_u)
        cols_list.append(num_cameras * 6 + point_indices * 3 + p)
        # Param p affects v-residual
        rows_list.append(row_v)
        cols_list.append(num_cameras * 6 + point_indices * 3 + p)
        
    rows = np.hstack(rows_list)
    cols = np.hstack(cols_list)
    data = np.ones(rows.size, dtype=int)
    
    # Create COO matrix
    A = coo_matrix((data, (rows, cols)), shape=(m, n))
    return A

def run_optimized_ba(initial_Rs_vec, initial_ts, initial_points_3d, K, 
                     camera_indices, point_indices, points_2d, max_nfev=100):
    
    num_cameras = len(initial_Rs_vec)
    num_points = len(initial_points_3d)
    
    camera_params = np.hstack((initial_Rs_vec, initial_ts)).flatten()
    initial_params = np.hstack((camera_params, initial_points_3d.flatten()))
    
    print("Computing sparsity pattern...")
    A = build_bundle_adjustment_sparsity(num_cameras, num_points, camera_indices, point_indices)

    print(f"Starting Vectorized BA ({num_cameras} cams, {num_points} points)...")
    t0 = time.time()
    
    result = least_squares(
        vectorized_reprojection_error,
        initial_params,
        jac_sparsity=A, 
        verbose=2,
        x_scale='jac',
        ftol=1e-4,
        max_nfev=max_nfev,
        method='trf',
        args=(num_cameras, num_points, camera_indices, point_indices, points_2d, K)
    )
    
    t1 = time.time()
    print(f"Optimization took {t1 - t0:.2f} seconds.")
    
    # Unpack results
    optimized_params = result.x
    camera_params_end = num_cameras * 6
    optimized_camera_params = optimized_params[:camera_params_end].reshape((num_cameras, 6))
    optimized_Rs_vec = optimized_camera_params[:, :3]
    optimized_ts = optimized_camera_params[:, 3:]
    optimized_points_3d = optimized_params[camera_params_end:].reshape((num_points, 3))

    return optimized_Rs_vec, optimized_ts, optimized_points_3d
