import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

# --- Helper Functions ---

def _to_projection_matrix(K, R_vec, t):
    """
    Constructs the 3x4 Projection Matrix (P = K[R|t]) from intrinsics and extrinsic parameters.
    R_vec is a 3-vector (Rodrigues) which is converted to a 3x3 rotation matrix R.
    """
    # Convert Rodrigues vector (R_vec) to 3x3 rotation matrix R
    R = Rotation.from_rotvec(R_vec).as_matrix()

    # Form the 3x4 Extrinsic matrix [R|t]
    Rt = np.hstack((R, t.reshape(3, 1)))

    # Form the 3x4 Projection matrix P = K @ [R|t]
    P = K @ Rt
    return P

def _project_points(P, points_3d):
    """
    Projects 3D points (homogeneous coordinates) onto the image plane using P.
    Returns 2D inhomogeneous coordinates (u, v).
    """
    # Convert 3D points to homogeneous coordinates (4xN)
    points_4d = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))]).T

    # Project points: (3x4) @ (4xN) = (3xN)
    projected_homogeneous = P @ points_4d

    # Convert from homogeneous to inhomogeneous (2D) coordinates
    # Divide by the last row (Z component in camera coordinates)
    u = projected_homogeneous[0] / projected_homogeneous[2]
    v = projected_homogeneous[1] / projected_homogeneous[2]

    # Stack u and v back into Nx2 array
    return np.vstack([u, v]).T

# --- Core Bundle Adjustment Function ---

def _reprojection_error(params, num_cameras, num_points, camera_indices, point_indices, points_2d, K):
    """
    Computes the vector of reprojection errors (residuals).

    NOTE: The number of cameras (num_cameras) and 3D points (num_points) must be
    passed correctly to unpack the 'params' vector.

    Args:
        params (np.array): 1D array of all optimized parameters (R_vecs, Ts, 3D Points).
        num_cameras (int): The total count of camera poses being optimized.
        num_points (int): The total count of 3D points being optimized.
        camera_indices (np.array): Index of the camera for each observation.
        point_indices (np.array): Index of the 3D point for each observation.
        points_2d (np.array): Lx2 array of observed 2D pixel coordinates.
        K (np.array): 3x3 Intrinsic camera matrix (fixed).
    """
    
    # 1. Unpack the optimization parameters (FIXED LOGIC)
    
    # Camera parameters: 6 parameters (3 R_vec, 3 t) per camera
    camera_params_end = num_cameras * 6
    
    # Use the correct num_cameras for reshaping
    camera_params = params[:camera_params_end].reshape((num_cameras, 6))
    
    # 3D points: 3 coordinates (X, Y, Z) per point
    # Use the correct num_points for reshaping
    points_3d = params[camera_params_end:].reshape((num_points, 3))

    # 2. Compute residuals
    
    residuals = []
    
    # Iterate over all observed 2D points (the correspondences)
    for i in range(points_2d.shape[0]):
        # Get the index of the camera (frame) and the 3D point (structure)
        cam_idx = camera_indices[i]
        pt_idx = point_indices[i]
        
        # Get the parameters for the relevant camera and point
        cam_params = camera_params[cam_idx]
        pt_3d = points_3d[pt_idx]
        
        # 3D points used for projection must be passed as (1, 3) array
        pt_3d_arr = pt_3d.reshape(1, 3) 
        
        # Construct the Projection Matrix P
        P = _to_projection_matrix(K, cam_params[:3], cam_params[3:])
        
        # Project the single 3D point to 2D
        projected_2d = _project_points(P, pt_3d_arr).flatten()
        
        # Calculate the reprojection error (observed - projected)
        observed_2d = points_2d[i]
        error = observed_2d - projected_2d
        
        # Append the error (u_error, v_error)
        residuals.append(error)

    # Return the flattened residual vector for least_squares
    return np.array(residuals).flatten()

def run_bundle_adjustment(initial_Rs_vec, initial_ts, initial_points_3d, K, 
                          camera_indices, point_indices, points_2d):
    """
    Runs Bundle Adjustment optimization to refine camera poses and 3D structure.

    Args:
        initial_Rs_vec (np.array): Nx3 array of initial rotation vectors (Rodrigues).
        initial_ts (np.array): Nx3 array of initial translation vectors.
        initial_points_3d (np.array): Mx3 array of initial 3D point coordinates.
        K (np.array): 3x3 Intrinsic camera matrix (fixed).
        camera_indices (np.array): Array (length L) of camera indices (0 to N-1).
        point_indices (np.array): Array (length L) of 3D point indices (0 to M-1).
        points_2d (np.array): Lx2 array of observed 2D pixel coordinates.

    Returns:
        tuple: (optimized_Rs_vec, optimized_ts, optimized_points_3d)
    """
    num_cameras = len(initial_Rs_vec)
    num_points = len(initial_points_3d)
    
    print(f"Starting BA with {num_cameras} cameras and {num_points} points...")

    # 1. Prepare initial parameters vector
    # Stack R_vec and t for each camera: [R1, t1, R2, t2, ...]
    camera_params = np.hstack((initial_Rs_vec, initial_ts))
    
    # Concatenate camera parameters and 3D points into a single 1D vector
    initial_params = np.hstack((camera_params.flatten(), initial_points_3d.flatten()))
    
    # 2. Run the least squares optimization
    # Pass the correct counts (num_cameras, num_points) as fixed arguments
    result = least_squares(
        _reprojection_error,
        initial_params,
        args=(num_cameras, num_points, camera_indices, point_indices, points_2d, K),
        jac='3-point', # Use finite difference Jacobian estimation
        verbose=1,
        x_scale='jac', # Scale variables based on the Jacobian norm
        ftol=1e-8,
        xtol=1e-8,
        method='trf' # Trust Region Reflective is often effective for BA
    )

    # 3. Unpack optimized parameters
    optimized_params = result.x
    
    # Unpack optimized camera parameters
    camera_params_end = num_cameras * 6
    optimized_camera_params = optimized_params[:camera_params_end].reshape((num_cameras, 6))
    
    optimized_Rs_vec = optimized_camera_params[:, :3]
    optimized_ts = optimized_camera_params[:, 3:]
    
    # Unpack optimized 3D points
    optimized_points_3d = optimized_params[camera_params_end:].reshape((-1, 3))
    
    # Compute final mean reprojection error
    final_residuals = result.fun
    final_mean_error = np.sqrt(np.mean(final_residuals**2))
    
    print("\n--- Bundle Adjustment Results ---")
    print(f"Optimization Status: {result.message}")
    print(f"Final Mean Reprojection Error (RMSE): {final_mean_error:.4f} (Pixels)")
    
    return optimized_Rs_vec, optimized_ts, optimized_points_3d


# --- Example Usage ---

if __name__ == '__main__':
    # --- 1. Synthetic Data Setup ---
    np.random.seed(42)

    # Intrinsics (Fixed)
    K = np.array([
        [1000, 0, 500],
        [0, 1000, 500],
        [0, 0, 1]
    ], dtype=float)

    # Use smaller numbers for a quick test
    N_CAMERAS = 5
    N_POINTS = 50
    
    # Generate true 3D points (M x 3)
    true_points_3d = np.random.rand(N_POINTS, 3) * 10 

    # Generate true camera poses (N x 6: R_vec, t)
    true_Rs_vec = np.random.rand(N_CAMERAS, 3) * 0.1 
    true_ts = np.array([[0, 0, 0], [1, 0.1, 0.1], [0.5, 0.5, 0.2], [-0.5, 0.1, 0.3], [1.5, 0.2, 0.1]]) 
    true_ts[:, 2] += 10 

    # Generate initial 2D observations (correspondences)
    points_2d = []
    camera_indices = []
    point_indices = []
    
    for cam_idx in range(N_CAMERAS):
        R_vec = true_Rs_vec[cam_idx]
        t = true_ts[cam_idx]
        P = _to_projection_matrix(K, R_vec, t)
        
        projected_2d = _project_points(P, true_points_3d)
        
        noise = np.random.randn(N_POINTS, 2) * 0.5 
        noisy_2d = projected_2d + noise
        
        # Only keep points that are roughly in the image plane (0 to 1000)
        valid_mask = (noisy_2d[:, 0] > 0) & (noisy_2d[:, 0] < 1000) & \
                     (noisy_2d[:, 1] > 0) & (noisy_2d[:, 1] < 1000)
        
        points_2d.append(noisy_2d[valid_mask])
        camera_indices.append(np.full(np.sum(valid_mask), cam_idx))
        point_indices.append(np.where(valid_mask)[0])

    # Final arrays for BA input
    points_2d = np.vstack(points_2d)
    camera_indices = np.concatenate(camera_indices).astype(int)
    point_indices = np.concatenate(point_indices).astype(int)

    print(f"Total {len(points_2d)} 2D observations generated.")

    # Introduce intentional error in initial guess for BA to correct
    initial_Rs_vec = true_Rs_vec + np.random.randn(N_CAMERAS, 3) * 0.05
    initial_ts = true_ts + np.random.randn(N_CAMERAS, 3) * 0.1
    initial_points_3d = true_points_3d + np.random.randn(N_POINTS, 3) * 0.5

    # Fix the first camera pose (Identity rotation, zero translation)
    initial_Rs_vec[0] = [0, 0, 0]
    initial_ts[0] = [0, 0, 0]
    
    # --- 2. Run Bundle Adjustment ---
    optimized_Rs_vec, optimized_ts, optimized_points_3d = run_bundle_adjustment(
        initial_Rs_vec,
        initial_ts,
        initial_points_3d,
        K,
        camera_indices,
        point_indices,
        points_2d
    )
    
    # --- 3. Evaluate results ---
    
    # Calculate the average L2 distance error for 3D points
    point_error = np.mean(np.linalg.norm(optimized_points_3d - true_points_3d, axis=1))
    print(f"\nAverage 3D Point Refinement Error (L2): {point_error:.4f} meters")
