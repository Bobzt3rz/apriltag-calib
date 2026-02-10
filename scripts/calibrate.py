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

    AprilTag local frame (defined from the viewer looking AT the tag):
        +X = viewer's right
        +Y = down
        +Z = X × Y = into the tag / into the wall (away from viewer)

    The outward normal (toward the camera) is therefore -Z in tag frame.

    "wall_facing" in config = direction of the outward normal = tag's -Z in world.
    So tag +Z in world = opposite of wall_facing.

    All tags are upright on walls, so tag +Y = world -Z (down) always.

    To determine tag +X: stand in front of the tag, facing it. Your right hand
    direction in world coordinates is tag +X. When facing direction D:
        facing +X => right is -Y
        facing -X => right is +Y
        facing +Y => right is +X
        facing -Y => right is -X

    World frame: +X = right, +Y = forward (into room from door), +Z = up.
    """
    size_m = tag_cfg.size_mm / 1000.0
    half_s = size_m / 2.0
    corner_pos_world = np.array(tag_cfg.position_xyz)

    # Offset from measured corner to tag center in tag-local frame
    # Tag frame: +X = right, +Y = down, origin at center
    if tag_cfg.measured_corner == "top_left":
        local_offset = np.array([half_s, half_s, 0])
    elif tag_cfg.measured_corner == "top_right":
        local_offset = np.array([-half_s, half_s, 0])
    elif tag_cfg.measured_corner == "bottom_right":
        local_offset = np.array([-half_s, -half_s, 0])
    elif tag_cfg.measured_corner == "bottom_left":
        local_offset = np.array([half_s, -half_s, 0])
    else:
        raise ValueError(f"Invalid corner: {tag_cfg.measured_corner}")

    facing = tag_cfg.wall_facing

    # Rotation matrix R: world_vec = R @ tag_vec
    # Columns are [tag_+X_in_world, tag_+Y_in_world, tag_+Z_in_world]

    if facing == "neg_x":
        # Outward normal = -X. Viewer faces +X. Right = -Y.
        # tag +X = -Y,  tag +Y = -Z,  tag +Z = (-Y)×(-Z) = +X
        r_matrix = np.array([
            [ 0,  0,  1],
            [-1,  0,  0],
            [ 0, -1,  0]
        ])
    elif facing == "pos_x":
        # Outward normal = +X. Viewer faces -X. Right = +Y.
        # tag +X = +Y,  tag +Y = -Z,  tag +Z = (+Y)×(-Z) = -X
        r_matrix = np.array([
            [ 0,  0, -1],
            [ 1,  0,  0],
            [ 0, -1,  0]
        ])
    elif facing == "neg_y":
        # Outward normal = -Y. Viewer faces +Y. Right = +X.
        # tag +X = +X,  tag +Y = -Z,  tag +Z = (+X)×(-Z) = +Y
        r_matrix = np.array([
            [ 1,  0,  0],
            [ 0,  0,  1],
            [ 0, -1,  0]
        ])
    elif facing == "pos_y":
        # Outward normal = +Y. Viewer faces -Y. Right = -X.
        # tag +X = -X,  tag +Y = -Z,  tag +Z = (-X)×(-Z) = -Y
        r_matrix = np.array([
            [-1,  0,  0],
            [ 0,  0, -1],
            [ 0, -1,  0]
        ])
    else:
        raise ValueError(f"Unknown facing: {facing}")

    center_pos_world = corner_pos_world + (r_matrix @ local_offset)
    return center_pos_world, r_matrix


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


def get_world_corners(center, rotation, size_m):
    """
    Returns 4 world-frame corners in pupil_apriltags detection order.

    The library's homography maps from ideal tag corners at:
        (-1,+1), (+1,+1), (+1,-1), (-1,-1)
    In tag-local coords (X=right, Y=down):
        corner 0: (-half, +half, 0) = bottom-left
        corner 1: (+half, +half, 0) = bottom-right
        corner 2: (+half, -half, 0) = top-right
        corner 3: (-half, -half, 0) = top-left
    """
    s = size_m / 2.0
    local_corners = np.array([
        [-s,  s, 0],
        [ s,  s, 0],
        [ s, -s, 0],
        [-s, -s, 0]
    ])
    return center + (rotation @ local_corners.T).T


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print(f"--- Calibration Session: {cfg.session_id} ---\n")

    # 1. Load World Map (Scene)
    world_tags = {}
    for tag in cfg.scene.tags:
        center, rotation = get_tag_pose_in_world(tag)
        world_tags[tag.id] = {
            "center_3d": center,
            "rotation": rotation,
            "size_m": tag.size_mm / 1000.0
        }
        print(f"Tag {tag.id} ({tag.wall_facing}): center={center}, normal={-rotation[:, 2]}")

    # 2. Load Camera Intrinsics
    root_dir = Path(hydra.utils.get_original_cwd())
    intrinsics_path = root_dir / cfg.camera.intrinsics_file
    K, D = load_intrinsics(intrinsics_path)

    # 3. Setup Detector
    at_detector = Detector(
        families='tag36h11',
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

    # 5. Process Frames
    for img_path in sorted(frames_dir.glob("*.png")):
        img = cv2.imread(str(img_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # The stored frames are horizontally mirrored upstream.
        # Flip to recover the true physical view before detection.
        img_physical = cv2.flip(gray, 1)

        detections = at_detector.detect(img_physical)
        detected_ids = [d.tag_id for d in detections]

        # Check if frame contains all required tags
        required = set(cfg.camera.required_tags)
        if not required.issubset(set(detected_ids)):
            continue

        print(f"Frame {img_path.name}: Found tags {detected_ids} -> SOLVING PnP")

        # 6. Build 2D-3D Correspondences
        obj_points = []
        img_points = []

        for d in detections:
            if d.tag_id not in world_tags:
                continue

            world_corners = get_world_corners(
                world_tags[d.tag_id]["center_3d"],
                world_tags[d.tag_id]["rotation"],
                world_tags[d.tag_id]["size_m"]
            )

            obj_points.extend(world_corners)
            img_points.extend(d.corners)

        if not obj_points:
            continue

        obj_points = np.array(obj_points, dtype=np.float64)
        img_points = np.array(img_points, dtype=np.float64)

        # 7. Solve PnP
        success, rvec, tvec = cv2.solvePnP(
            obj_points, img_points, K, D, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            print(f"  solvePnP failed, skipping frame.")
            continue

        R_cam, _ = cv2.Rodrigues(rvec)
        camera_position_world = (-R_cam.T @ tvec).flatten()
        R_world_to_cam = R_cam.T

        # Reprojection error
        reproj, _ = cv2.projectPoints(obj_points, rvec, tvec, K, D)
        reproj = reproj.reshape(-1, 2)
        reproj_err = np.linalg.norm(img_points - reproj, axis=1)

        print(f"  Camera Position (x,y,z): [{camera_position_world[0]:.4f}, "
              f"{camera_position_world[1]:.4f}, {camera_position_world[2]:.4f}]")
        print(f"  Camera look direction (cam -Z in world): {-R_world_to_cam[2, :]}")
        print(f"  Reprojection error: mean={reproj_err.mean():.2f}px, max={reproj_err.max():.2f}px")

        # 8. Save result
        rot = R.from_matrix(R_world_to_cam)
        euler_angles = rot.as_euler('xyz', degrees=True)

        calib_output = {
            "session_id": cfg.session_id,
            "frame": img_path.name,
            "camera_position_world": camera_position_world.tolist(),
            "camera_rotation_world_to_cam": R_world_to_cam.tolist(),
            "euler_angles_xyz_deg": euler_angles.tolist(),
            "rvec": rvec.flatten().tolist(),
            "tvec": tvec.flatten().tolist(),
            "reprojection_error_mean_px": float(reproj_err.mean()),
            "reprojection_error_max_px": float(reproj_err.max()),
        }

        output_path = root_dir / "outputs" / f"calibration_{cfg.session_id}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(calib_output, f, indent=2)

        print(f"  Saved to: {output_path}")
        break


if __name__ == "__main__":
    main()