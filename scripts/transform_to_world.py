# scripts/transform_to_world.py
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


def transform_camera_to_world(points_cam, R_world_to_cam, camera_position_world):
    """
    Transform points from camera frame to world frame.
    
    Args:
        points_cam: Nx3 array or single [x,y,z] in camera frame (mm)
        R_world_to_cam: 3x3 rotation matrix (world to camera)
        camera_position_world: [x,y,z] camera position in world frame (meters)
    
    Returns:
        points_world: Nx3 array or single [x,y,z] in world frame (meters)
    """
    points_cam = np.atleast_2d(points_cam)
    
    # Convert camera frame points from mm to meters
    points_cam_m = points_cam / 1000.0
    
    # Transform: world_point = R_world_to_cam.T @ cam_point + camera_position
    points_world_m = (R_world_to_cam.T @ points_cam_m.T).T + camera_position_world
    
    return points_world_m.squeeze()

def compute_look_at_point(R_world_to_cam, camera_position_world, distance=1.0):
    """
    Compute a point B that the camera is looking at.
    
    Args:
        R_world_to_cam: 3x3 rotation matrix (world to camera)
        camera_position_world: [x,y,z] camera position in world (meters)
        distance: How far in front of camera to place point B (meters)
    
    Returns:
        B: [x,y,z] point in world that camera is looking at
    """
    # Camera's +Z axis in world frame (where camera is looking)
    cam_z_in_world = R_world_to_cam.T @ np.array([0, 0, 1])
    
    # Point B is camera position + distance along look direction
    B = camera_position_world + distance * cam_z_in_world
    
    return B

def apply_coordinate_correction(positions):
    """
    Apply coordinate system transformation to match old reference frame.
    Transformation: (Y, X, Z) with signs (-, +, -)
    
    Args:
        positions: Nx3 array in new coordinate system
    Returns:
        Nx3 array in old coordinate system
    """
    corrected = np.zeros_like(positions)
    corrected[:, 0] = -positions[:, 1]  # new_X = -old_Y
    corrected[:, 1] = +positions[:, 0]  # new_Y = +old_X
    corrected[:, 2] = -positions[:, 2]  # new_Z = -old_Z
    return corrected

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print(f"--- Transform to World Coordinates ---")
    print(f"Data session: {cfg.session_id}")
    print(f"Camera: {cfg.camera.id}\n")
    
    root_dir = Path(hydra.utils.get_original_cwd())
    
    # 1. Load extrinsics
    extrinsics_path = root_dir / cfg.extrinsics_file
    if not extrinsics_path.exists():
        print(f"Error: Extrinsics file not found at {extrinsics_path}")
        print(f"Please run calibration first: python scripts/calibrate.py camera={cfg.camera.id}")
        return
    
    print(f"Loading extrinsics from: {extrinsics_path}")
    R_world_to_cam, camera_position_world = load_extrinsics(extrinsics_path)
    
    print(f"Camera position (world, meters): {camera_position_world}")
    print(f"Camera orientation:\n{R_world_to_cam}\n")

    # Compute look-at point B
    A = camera_position_world
    B = compute_look_at_point(R_world_to_cam, camera_position_world, distance=1.0)
    print(f"In A-B representation:")
    print(f"  A (camera position): [{A[0]:.2f}, {A[1]:.2f}, {A[2]:.2f}]")
    print(f"  B (look-at point):   [{B[0]:.2f}, {B[1]:.2f}, {B[2]:.2f}]")
    print(f"  Camera looking from A toward B\n")
    
    # 2. Load joint data (camera frame)
    joint_data_path = root_dir / cfg.joint_data_file
    if not joint_data_path.exists():
        print(f"Error: Joint data file not found at {joint_data_path}")
        return
    
    print(f"Loading joint data from: {joint_data_path}")
    with open(joint_data_path, 'r') as f:
        joint_data_cam = json.load(f)
    
    print(f"Found {len(joint_data_cam)} frames with joint data\n")
    
    # 3. Transform each frame
    joint_data_world = {}
    
    for frame_name, joint_pos_cam in joint_data_cam.items():
        joint_pos_cam = np.array(joint_pos_cam)  # in mm

        # CORRECTION: Un-flip X coordinate since joint data was computed on flipped images
        # but calibration was done on un-flipped images
        # joint_pos_cam[2] = -joint_pos_cam[2]
        # joint_pos_cam[1] = -joint_pos_cam[1]

        # Apply camera frame transformation: (Y, Z, X) with signs (+, -, -)
        # This reorders and flips axes before world transformation
        # transformed_cam = np.array([
        #     +joint_pos_cam[1],  # new X = +old Y
        #     +joint_pos_cam[0],  # new Y = -old Z
        #     +joint_pos_cam[2]   # new Z = -old X
        # ])                                                                                        
      
        joint_pos_world = transform_camera_to_world(
            joint_pos_cam, 
            R_world_to_cam, 
            camera_position_world
        )  # returns in meters

        joint_data_world[frame_name] = joint_pos_world.tolist()
    
    # 4. Save transformed data
    output_path = root_dir / cfg.joint_data_world_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(joint_data_world, f, indent=2)
    
    print(f"Transformed joint data saved to: {output_path}")
    print(f"Output units: meters\n")
    
    # 5. Print sample for verification
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