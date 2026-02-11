# scripts/test_camera_coordinate_combinations.py
import json
import hydra
import numpy as np
from pathlib import Path
from omegaconf import DictConfig
from itertools import permutations, product


def load_extrinsics(filepath):
    """Load camera extrinsics from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    R_world_to_cam = np.array(data["camera_rotation_world_to_cam"])
    camera_position_world = np.array(data["camera_position_world"])
    
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


def apply_transform(positions, perm, signs):
    """
    Apply axis permutation and sign flips.
    
    Args:
        positions: Nx3 array
        perm: tuple of (0,1,2) in some order
        signs: tuple of 3 signs
    """
    transformed = positions[:, perm] * np.array(signs)
    return transformed


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print(f"=== Testing Camera Coordinate Combinations ===")
    print(f"Data session: {cfg.session_id}")
    print(f"Camera: {cfg.camera.id}\n")
    
    root_dir = Path(hydra.utils.get_original_cwd())
    
    # 1. Load extrinsics
    extrinsics_path = root_dir / cfg.extrinsics_file
    if not extrinsics_path.exists():
        print(f"Error: Extrinsics file not found at {extrinsics_path}")
        return
    
    R_world_to_cam, camera_position_world = load_extrinsics(extrinsics_path)
    
    # 2. Load joint data (camera frame)
    joint_data_path = root_dir / cfg.joint_data_file
    if not joint_data_path.exists():
        print(f"Error: Joint data file not found at {joint_data_path}")
        return
    
    with open(joint_data_path, 'r') as f:
        joint_data_cam = json.load(f)
    
    # Convert to array
    frames = sorted(joint_data_cam.keys())
    joint_positions_cam = np.array([joint_data_cam[frame] for frame in frames])
    
    print(f"Loaded {len(frames)} frames\n")
    print(f"Original camera frame data range (mm):")
    print(f"  X: [{joint_positions_cam[:, 0].min():.2f}, {joint_positions_cam[:, 0].max():.2f}]")
    print(f"  Y: [{joint_positions_cam[:, 1].min():.2f}, {joint_positions_cam[:, 1].max():.2f}]")
    print(f"  Z: [{joint_positions_cam[:, 2].min():.2f}, {joint_positions_cam[:, 2].max():.2f}]")
    print()
    
    # Try all combinations
    axis_perms = list(permutations([0, 1, 2]))
    sign_combos = list(product([1, -1], repeat=3))
    
    print("=" * 100)
    print("Testing all 48 coordinate transformations on camera frame data BEFORE world transform:")
    print("=" * 100)
    
    results = []
    
    for perm in axis_perms:
        for signs in sign_combos:
            # Apply transformation to camera frame data
            transformed_cam = apply_transform(joint_positions_cam, perm, signs)
            
            # Now transform to world
            world_positions = []
            for i, frame in enumerate(frames):
                joint_cam_transformed = transformed_cam[i]
                joint_world = transform_camera_to_world(
                    joint_cam_transformed,
                    R_world_to_cam,
                    camera_position_world
                )
                world_positions.append(joint_world)
            
            world_positions = np.array(world_positions)
            
            # Compute ranges
            x_range = (world_positions[:, 0].min(), world_positions[:, 0].max())
            y_range = (world_positions[:, 1].min(), world_positions[:, 1].max())
            z_range = (world_positions[:, 2].min(), world_positions[:, 2].max())
            
            z_positive = (world_positions[:, 2] > 0).all()
            
            results.append({
                'perm': perm,
                'signs': signs,
                'x_range': x_range,
                'y_range': y_range,
                'z_range': z_range,
                'z_positive': z_positive
            })
    
    # Sort by Z being positive, then by Z range being reasonable (0.5-1.5m)
    def score(r):
        z_min, z_max = r['z_range']
        if not r['z_positive']:
            return 1000  # Bad
        # Prefer Z around 0.9m (human waist/hand height)
        z_mid = (z_min + z_max) / 2
        return abs(z_mid - 0.9)
    
    results.sort(key=score)
    
    # Print results
    print("\nTop 20 transformations (sorted by reasonable Z height):")
    print("=" * 100)
    
    for i, r in enumerate(results):
        perm_str = f"({'XYZ'[r['perm'][0]]}, {'XYZ'[r['perm'][1]]}, {'XYZ'[r['perm'][2]]})"
        sign_str = f"({'+' if r['signs'][0] > 0 else '-'}, {'+' if r['signs'][1] > 0 else '-'}, {'+' if r['signs'][2] > 0 else '-'})"
        z_flag = "✓" if r['z_positive'] else "✗"
        
        print(f"\n{i+1}. Camera frame transform: {perm_str}, Signs: {sign_str}")
        print(f"   World X range: [{r['x_range'][0]:6.2f}, {r['x_range'][1]:6.2f}] m")
        print(f"   World Y range: [{r['y_range'][0]:6.2f}, {r['y_range'][1]:6.2f}] m")
        print(f"   World Z range: [{r['z_range'][0]:6.2f}, {r['z_range'][1]:6.2f}] m  {z_flag}")


if __name__ == "__main__":
    main()