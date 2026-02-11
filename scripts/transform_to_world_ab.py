# scripts/transform_to_world_ab.py
"""
Transform joint data from camera frame to world frame using the simplified A-B method.

This mimics the old reference implementation which:
1. Uses camera position as A
2. Uses a look-at point as B  
3. Builds a rotation matrix assuming the camera is level (no roll)

This is NOT a true world coordinate transformation - it's an approximation that
assumes the camera has no roll/tilt. Use transform_to_world.py for accurate
world coordinates from full calibration.

Information lost compared to full calibration:
- Camera roll (rotation around the look axis)
- Camera pitch accuracy (vertical tilt is approximated)
- Any non-level mounting of the camera
"""
import json
import hydra
import numpy as np
from pathlib import Path
from omegaconf import DictConfig


def load_extrinsics(filepath):
    """Load camera extrinsics from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    R_world_to_cam = np.array(data["camera_rotation_world_to_cam"])
    camera_position_world = np.array(data["camera_position_world"])  # in meters
    
    return R_world_to_cam, camera_position_world


def get_ab_from_calibration(R_world_to_cam, camera_position_world, distance=1.0):
    """
    Extract A (position) and B (look-at point) from calibration.
    
    Args:
        R_world_to_cam: 3x3 rotation matrix from calibration
        camera_position_world: camera position in world frame
        distance: distance from A to B (default 1.0m)
    
    Returns:
        A: camera position
        B: point the camera is looking at
    """
    A = camera_position_world
    
    # Camera +Z in world is the third column of R_world_to_cam
    look_dir = R_world_to_cam[:, 2]
    
    B = A + distance * look_dir
    
    return A, B


def get_transform_matrix_ab(A, B):
    """
    Build rotation matrix from A-B representation.
    This is the same method as the old reference implementation.
    
    Assumes:
    - Camera +Z points from A toward B
    - Camera is level (no roll) - uses world up to derive other axes
    
    Returns:
        R: 3x3 rotation matrix (camera to world)
        T: translation (camera position = A)
    """
    T = np.array(A)
    
    # Z axis = look direction (A toward B)
    z_axis = np.array(B) - T
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    # X axis = world_up × z_axis (perpendicular to both)
    world_up = np.array([0, 0, 1])
    x_axis = np.cross(world_up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # Y axis = z_axis × x_axis (completes right-handed system)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    # R transforms camera coords to world coords: world = R @ cam + T
    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    
    return R, T


def transform_camera_to_world_ab(points_cam, R_cam_to_world, camera_position):
    """
    Transform points from camera frame to world frame using A-B method.
    
    Args:
        points_cam: Nx3 array or single [x,y,z] in camera frame (mm)
        R_cam_to_world: 3x3 rotation matrix (camera to world)
        camera_position: [x,y,z] camera position in world frame (meters)
    
    Returns:
        points_world: Nx3 array or single [x,y,z] in world frame (meters)
    """
    points_cam = np.atleast_2d(points_cam)
    
    # Convert camera frame points from mm to meters
    points_cam_m = points_cam / 1000.0
    
    # Transform: world_point = R @ cam_point + T
    points_world_m = (R_cam_to_world @ points_cam_m.T).T + camera_position
    
    return points_world_m.squeeze()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print(f"--- Transform to World Coordinates (A-B Method) ---")
    print(f"Data session: {cfg.session_id}")
    print(f"Camera: {cfg.camera.id}")
    print(f"\n⚠️  Note: This uses simplified A-B transformation (assumes level camera)")
    print(f"    For accurate world coords, use transform_to_world.py instead.\n")
    
    root_dir = Path(hydra.utils.get_original_cwd())
    
    # 1. Load extrinsics
    extrinsics_path = root_dir / cfg.extrinsics_file
    if not extrinsics_path.exists():
        print(f"Error: Extrinsics file not found at {extrinsics_path}")
        print(f"Please run calibration first: python scripts/calibrate.py camera={cfg.camera.id}")
        return
    
    print(f"Loading extrinsics from: {extrinsics_path}")
    R_world_to_cam, camera_position_world = load_extrinsics(extrinsics_path)
    
    # 2. Extract A-B from calibration
    A, B = get_ab_from_calibration(R_world_to_cam, camera_position_world, distance=1.0)
    
    print(f"\nA-B representation (from calibration):")
    print(f"  A (camera position): [{A[0]:.3f}, {A[1]:.3f}, {A[2]:.3f}]")
    print(f"  B (look-at point):   [{B[0]:.3f}, {B[1]:.3f}, {B[2]:.3f}]")
    
    # 3. Build rotation matrix using A-B method
    R_cam_to_world, T = get_transform_matrix_ab(A, B)
    
    print(f"\nR_cam_to_world (from A-B construction):")
    print(f"  X-axis in world: [{R_cam_to_world[0,0]:.3f}, {R_cam_to_world[1,0]:.3f}, {R_cam_to_world[2,0]:.3f}]")
    print(f"  Y-axis in world: [{R_cam_to_world[0,1]:.3f}, {R_cam_to_world[1,1]:.3f}, {R_cam_to_world[2,1]:.3f}]")
    print(f"  Z-axis in world: [{R_cam_to_world[0,2]:.3f}, {R_cam_to_world[1,2]:.3f}, {R_cam_to_world[2,2]:.3f}]")
    
    # Compare to calibrated rotation
    R_cam_to_world_calib = R_world_to_cam.T
    print(f"\nR_cam_to_world (from full calibration):")
    print(f"  X-axis in world: [{R_cam_to_world_calib[0,0]:.3f}, {R_cam_to_world_calib[1,0]:.3f}, {R_cam_to_world_calib[2,0]:.3f}]")
    print(f"  Y-axis in world: [{R_cam_to_world_calib[0,1]:.3f}, {R_cam_to_world_calib[1,1]:.3f}, {R_cam_to_world_calib[2,1]:.3f}]")
    print(f"  Z-axis in world: [{R_cam_to_world_calib[0,2]:.3f}, {R_cam_to_world_calib[1,2]:.3f}, {R_cam_to_world_calib[2,2]:.3f}]")
    
    # 4. Load joint data (camera frame)
    joint_data_path = root_dir / cfg.joint_data_file
    if not joint_data_path.exists():
        print(f"\nError: Joint data file not found at {joint_data_path}")
        return
    
    print(f"\nLoading joint data from: {joint_data_path}")
    with open(joint_data_path, 'r') as f:
        joint_data_cam = json.load(f)
    
    print(f"Found {len(joint_data_cam)} frames with joint data\n")
    
    # 5. Transform each frame
    joint_data_world = {}
    
    for frame_name, joint_pos_cam in joint_data_cam.items():
        joint_pos_cam = np.array(joint_pos_cam)  # in mm
      
        joint_pos_world = transform_camera_to_world_ab(
            joint_pos_cam, 
            R_cam_to_world, 
            T
        )  # returns in meters

        joint_data_world[frame_name] = joint_pos_world.tolist()
    
    # 6. Save transformed data
    output_path = root_dir / cfg.joint_data_world_file
    # Modify output filename to indicate A-B method
    output_path = output_path.parent / output_path.name.replace('.json', '_ab.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(joint_data_world, f, indent=2)
    
    print(f"Transformed joint data saved to: {output_path}")
    print(f"Output units: meters\n")
    
    # 7. Print sample for verification
    sample_frame = list(joint_data_world.keys())[0]
    sample_cam = np.array(joint_data_cam[sample_frame])
    print(f"Sample transformation ({sample_frame}):")
    print(f"  Camera frame (mm): {joint_data_cam[sample_frame]}")
    print(f"  Camera frame (m):  [{sample_cam[0]/1000:.3f}, {sample_cam[1]/1000:.3f}, {sample_cam[2]/1000:.3f}]")
    print(f"  World frame (m):   {joint_data_world[sample_frame]}")
    
    # Sanity check
    world_pos = np.array(joint_data_world[sample_frame])
    if np.any(world_pos < -0.1) or np.any(world_pos > 15):
        print(f"\n⚠️  WARNING: World coordinates seem unusual. Check coordinate system definitions.")
    else:
        print(f"\n✓ World coordinates look reasonable for a typical room.")


if __name__ == "__main__":
    main()