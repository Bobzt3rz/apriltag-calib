# scripts/calibrate_single_tag.py
"""
Calibrate using a single tag only (default: tag 0).
Usage: python scripts/calibrate_single_tag.py tag_id=0
       python scripts/calibrate_single_tag.py tag_id=1
"""
import cv2
import json
import hydra
import numpy as np
from pathlib import Path
from omegaconf import DictConfig
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation as R


def get_tag_pose_in_world(tag_cfg):
    """
    Converts config (measured corner + wall facing) -> tag center + rotation matrix.
    """
    width_m = tag_cfg.width_m
    height_m = tag_cfg.height_m

    half_w = width_m / 2.0
    half_h = height_m / 2.0
    corner_pos_world = np.array(tag_cfg.position_xyz)

    if tag_cfg.measured_corner == "top_left":
        local_offset = np.array([half_w, half_h, 0])
    elif tag_cfg.measured_corner == "top_right":
        local_offset = np.array([-half_w, half_h, 0])
    elif tag_cfg.measured_corner == "bottom_right":
        local_offset = np.array([-half_w, -half_h, 0])
    elif tag_cfg.measured_corner == "bottom_left":
        local_offset = np.array([half_w, -half_h, 0])
    else:
        raise ValueError(f"Invalid corner: {tag_cfg.measured_corner}")

    facing = tag_cfg.wall_facing

    if facing == "neg_x":
        r_matrix = np.array([
            [ 0,  0,  1],
            [-1,  0,  0],
            [ 0, -1,  0]
        ])
    elif facing == "pos_x":
        r_matrix = np.array([
            [ 0,  0, -1],
            [ 1,  0,  0],
            [ 0, -1,  0]
        ])
    elif facing == "neg_y":
        r_matrix = np.array([
            [ 1,  0,  0],
            [ 0,  0,  1],
            [ 0, -1,  0]
        ])
    elif facing == "pos_y":
        r_matrix = np.array([
            [-1,  0,  0],
            [ 0,  0, -1],
            [ 0, -1,  0]
        ])
    else:
        raise ValueError(f"Unknown facing: {facing}")

    center_pos_world = corner_pos_world + (r_matrix @ local_offset)
    return center_pos_world, r_matrix, width_m, height_m


def load_intrinsics(filepath):
    """Loads a 3x3 camera matrix from CSV. Assumes zero distortion."""
    try:
        K = np.loadtxt(filepath, delimiter=',')
    except Exception as e:
        raise ValueError(f"Failed to load intrinsics at {filepath}: {e}")

    if K.shape != (3, 3):
        raise ValueError(f"Expected 3x3 matrix, got shape {K.shape}")

    D = np.zeros(5)
    return K, D


def get_world_corners(center, rotation, width_m, height_m):
    """
    Returns 4 world-frame corners in pupil_apriltags detection order.
    """
    half_w = width_m / 2.0
    half_h = height_m / 2.0
    
    local_corners = np.array([
        [-half_w,  half_h, 0],  # bottom-left
        [ half_w,  half_h, 0],  # bottom-right
        [ half_w, -half_h, 0],  # top-right
        [-half_w, -half_h, 0]   # top-left
    ])
    return center + (rotation @ local_corners.T).T


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Which single tag to use
    target_tag_id = cfg.get("tag_id", 0)
    
    print(f"--- Single-Tag Calibration ---")
    print(f"Session: {cfg.session_id}")
    print(f"Using ONLY Tag {target_tag_id}\n")

    # 1. Load World Map (Scene) - only the target tag
    world_tags = {}
    for tag in cfg.scene.tags:
        if tag.id == target_tag_id:
            center, rotation, width_m, height_m = get_tag_pose_in_world(tag)
            world_tags[tag.id] = {
                "center_3d": center,
                "rotation": rotation,
                "width_m": width_m,
                "height_m": height_m
            }
            print(f"Tag {tag.id} ({tag.wall_facing}): center={center}, normal={-rotation[:, 2]}")

    if not world_tags:
        print(f"Error: Tag {target_tag_id} not found in scene config")
        return

    # 2. Load Camera Intrinsics
    root_dir = Path(hydra.utils.get_original_cwd())
    intrinsics_path = root_dir / cfg.camera.intrinsics_file
    K, D = load_intrinsics(intrinsics_path)

    # 3. Setup Detector
    at_detector = Detector(
        families=cfg.scene.tag_family,
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
    )

    # 4. Find processed frames
    frames_dir = root_dir / "data" / "processed" / cfg.session_id / "frames"
    if not frames_dir.exists():
        print(f"Error: No frames found at {frames_dir}")
        return

    print(f"Scanning frames in {frames_dir}...\n")

    # 5. Process Frames - collect results from multiple frames
    all_results = []

    for img_path in sorted(frames_dir.glob("*.png")):
        img = cv2.imread(str(img_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        detections = at_detector.detect(gray)
        detected_ids = [d.tag_id for d in detections]

        if target_tag_id not in detected_ids:
            continue

        # Get the target detection
        target_det = [d for d in detections if d.tag_id == target_tag_id][0]

        # Build 2D-3D correspondences (single tag = 4 corners)
        world_corners = get_world_corners(
            world_tags[target_tag_id]["center_3d"],
            world_tags[target_tag_id]["rotation"],
            world_tags[target_tag_id]["width_m"],
            world_tags[target_tag_id]["height_m"]
        )

        obj_points = np.array(world_corners, dtype=np.float64)
        img_points = np.array(target_det.corners, dtype=np.float64)

        # Solve PnP
        success, rvec, tvec = cv2.solvePnP(
            obj_points, img_points, K, D, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            continue

        R_cam, _ = cv2.Rodrigues(rvec)
        camera_position_world = (-R_cam.T @ tvec).flatten()
        R_world_to_cam = R_cam  # solvePnP returns world-to-camera

        # Reprojection error
        reproj, _ = cv2.projectPoints(obj_points, rvec, tvec, K, D)
        reproj = reproj.reshape(-1, 2)
        reproj_err = np.linalg.norm(img_points - reproj, axis=1)

        # Camera look direction in world
        cam_look_dir = R_world_to_cam[2, :]  # third row

        all_results.append({
            'frame': img_path.name,
            'camera_position_world': camera_position_world,
            'R_world_to_cam': R_world_to_cam,
            'cam_look_dir': cam_look_dir,
            'rvec': rvec,
            'tvec': tvec,
            'reproj_err_mean': reproj_err.mean(),
            'reproj_err_max': reproj_err.max(),
        })

    if not all_results:
        print("No frames with target tag detected!")
        return

    print(f"Found {len(all_results)} frames with Tag {target_tag_id}\n")

    # 6. Print all results for comparison
    print("=" * 90)
    print(f"{'Frame':<45} {'Position (x,y,z)':<35} {'Reproj (mean)':<15}")
    print("=" * 90)

    for r in all_results[:20]:  # show first 20
        pos = r['camera_position_world']
        print(f"{r['frame']:<45} [{pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}]   {r['reproj_err_mean']:.2f}px")

    # 7. Pick the best result (lowest reprojection error)
    best = min(all_results, key=lambda x: x['reproj_err_mean'])

    print(f"\n{'=' * 60}")
    print(f"BEST RESULT (lowest reprojection error):")
    print(f"{'=' * 60}")
    print(f"  Frame: {best['frame']}")
    print(f"  Position: [{best['camera_position_world'][0]:.4f}, "
          f"{best['camera_position_world'][1]:.4f}, {best['camera_position_world'][2]:.4f}]")
    print(f"  Look direction: [{best['cam_look_dir'][0]:.3f}, "
          f"{best['cam_look_dir'][1]:.3f}, {best['cam_look_dir'][2]:.3f}]")
    print(f"  Reprojection error: mean={best['reproj_err_mean']:.2f}px, max={best['reproj_err_max']:.2f}px")

    # 8. Save best result
    rot = R.from_matrix(best['R_world_to_cam'])
    euler_angles = rot.as_euler('xyz', degrees=True)

    extrinsics_output = {
        "camera_id": cfg.camera.id,
        "scene_name": cfg.scene.name,
        "calibration_session_id": cfg.session_id,
        "calibration_tag_id": int(target_tag_id),
        "frame": best['frame'],
        "camera_position_world": best['camera_position_world'].tolist(),
        "camera_rotation_world_to_cam": best['R_world_to_cam'].tolist(),
        "euler_angles_xyz_deg": euler_angles.tolist(),
        "rvec": best['rvec'].flatten().tolist(),
        "tvec": best['tvec'].flatten().tolist(),
        "reprojection_error_mean_px": float(best['reproj_err_mean']),
        "reprojection_error_max_px": float(best['reproj_err_max']),
        "num_frames_detected": len(all_results),
    }

    output_path = root_dir / "outputs" / f"extrinsics_{cfg.camera.id}_tag{target_tag_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(extrinsics_output, f, indent=2)

    print(f"\n  Saved to: {output_path}")

    # 9. Compare with reference
    REFERENCE_CAMERAS = {
        "har_01": {"A": [5.03, 8.45, 0.90], "B": [5.73, 8.18, 0.90]},
    }
    ref = REFERENCE_CAMERAS.get(cfg.camera.id)
    if ref:
        ref_A = np.array(ref["A"])
        ref_dir = np.array(ref["B"]) - ref_A
        ref_dir = ref_dir / np.linalg.norm(ref_dir)

        pos_diff = np.linalg.norm(best['camera_position_world'] - ref_A)
        dot = np.clip(np.dot(best['cam_look_dir'], ref_dir), -1, 1)
        angle_diff = np.degrees(np.arccos(dot))

        print(f"\n  vs Reference: pos_diff={pos_diff:.3f}m, angle_diff={angle_diff:.1f}°")


if __name__ == "__main__":
    main()