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
    Converts config (Corner + Facing) -> Center + Rotation
    """
    # 1. Size Check (assuming mm -> meters)
    size_m = tag_cfg.size_mm / 1000.0
    half_s = size_m / 2.0
    corner_pos_world = np.array(tag_cfg.position_xyz)
    
    # 2. Local Offset (Vector from Corner -> Center)
    # Tag Frame: X=Right, Y=Down
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

    # 3. World Rotation (Tag Frame -> World Frame)
    # Tag Z is Normal. Tag Up is World Z.
    facing = tag_cfg.wall_facing
    
    if facing == "neg_y":   # Looking North (-Y)
        r_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, -1, 0]])
    elif facing == "pos_y": # Looking South (+Y)
        r_matrix = np.array([[-1, 0, 0], [0, 0, -1], [0, 1, 0]])
    elif facing == "neg_x": # Looking West (-X)
        r_matrix = np.array([[0, -1, 0], [0, 0, -1], [-1, 0, 0]])
    elif facing == "pos_x": # Looking East (+X)
        r_matrix = np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]])
    else:
        raise ValueError(f"Unknown facing: {facing}")

    # 4. Apply: Center = Corner + (R * Local_Offset)
    center_pos_world = corner_pos_world + (r_matrix @ local_offset)
    
    return center_pos_world, r_matrix

def load_intrinsics(filepath):
    """
    Loads a 3x3 Camera Matrix from a CSV file.
    Assumes zero distortion since not provided in the CSV.
    """
    # 1. Load the 3x3 Matrix (K)
    # delimiter=',' handles the commas
    # numpy automatically handles scientific notation (1.11e+03)
    try:
        K = np.loadtxt(filepath, delimiter=',')
    except Exception as e:
        raise ValueError(f"Failed to load CSV at {filepath}. Error: {e}")

    if K.shape != (3, 3):
        raise ValueError(f"Expected 3x3 matrix, got shape {K.shape}")

    # 2. Handle Distortion (D)
    # Since your file doesn't have them, we initialize a zero vector.
    # Standard OpenCV model uses 5 coefficients: (k1, k2, p1, p2, k3)
    D = np.zeros(5) 
    
    return K, D

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print(f"--- Calibration Session: {cfg.session_id} ---")
    
    # 1. Load World Map (Scene)
    world_tags = {}
    for tag in cfg.scene.tags:
        center, rotation = get_tag_pose_in_world(tag)
        world_tags[tag.id] = {
            "center_3d": center,
            "rotation": rotation,
            "size_m": tag.size_mm / 1000.0
        }
        print(f"Loaded Tag {tag.id}: World Pos {center}")

    # 2. Load Camera Intrinsics
    # Resolving path relative to project root
    root_dir = Path(hydra.utils.get_original_cwd())
    intrinsics_path = root_dir / cfg.camera.intrinsics_file
    K, D = load_intrinsics(intrinsics_path)

    # 3. Setup Detector
    # Note: family must match config (tag36h11)
    at_detector = Detector(families='tag36h11', 
                           nthreads=1, 
                           quad_decimate=1.0,
                           quad_sigma=0.0,
                           refine_edges=1,
                           decode_sharpening=0.25,
                           debug=0)

    # 4. Find the processed frames
    frames_dir = root_dir / "data" / "processed" / cfg.session_id / "frames"
    if not frames_dir.exists():
        print(f"Error: No frames found at {frames_dir}")
        return

    # 5. Process Frames
    print(f"Scanning frames in {frames_dir}...")

    
    for img_path in sorted(frames_dir.glob("*.png")):
        img = cv2.imread(str(img_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # CRITICAL: Un-flip if your raw data was mirrored!
        # Assuming you want to solve for the TRUE physical camera:
        img_clean = cv2.flip(gray, 1) 
        
        detections = at_detector.detect(img_clean)
        detected_ids = [d.tag_id for d in detections]

        # Check Requirements
        required = set(cfg.required_tags)
        if not required.issubset(set(detected_ids)):
            continue # Skip frames that don't see all tags

        print(f"Frame {img_path.name}: Found tags {detected_ids} -> SOLVING PnP")

        # 6. Build 2D-3D Correspondences
        # We use the 4 corners of the tag for maximum accuracy (PnP uses points)
        obj_points = [] # 3D World
        img_points = [] # 2D Image

        for d in detections:
            if d.tag_id not in world_tags:
                continue
            
            # A. Get World Corners for this tag
            # We reconstruct them from the World Center + World Rotation we calculated
            t_world_center = world_tags[d.tag_id]["center_3d"]
            R_world = world_tags[d.tag_id]["rotation"]
            s = world_tags[d.tag_id]["size_m"]
            
            # Local corners in Tag Frame (Counter-Clockwise from Bottom-Left)
            # 0: BL (-x, -y) ?? Wait, AprilTag standard is:
            # Corner 0: Bottom-Left (-x, +y)? No, let's stick to the library output order.
            # pupil_apriltags output.corners are: LB, RB, RT, LT (Counter-Clockwise)
            # Tag Frame: X=Right, Y=Down. 
            # LB = (-s/2, s/2), RB = (s/2, s/2), RT = (s/2, -s/2), LT = (-s/2, -s/2)
            
            local_corners = np.array([
                [-s/2,  s/2, 0], # LB
                [ s/2,  s/2, 0], # RB
                [ s/2, -s/2, 0], # RT
                [-s/2, -s/2, 0]  # LT
            ])
            
            # Transform local corners to World
            world_corners = t_world_center + (R_world @ local_corners.T).T
            
            obj_points.extend(world_corners)
            img_points.extend(d.corners)

        # 7. Solve PnP
        if not obj_points:
            continue

        obj_points = np.array(obj_points, dtype=np.float32)
        img_points = np.array(img_points, dtype=np.float32)

        success, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, D, flags=cv2.SOLVEPNP_ITERATIVE)

        if success:
            print(f"  -> Camera Position (x,y,z): {tvec.flatten()}")
            
            # Optional: Save this result?
            # For now, just print the first valid one and break if you only need one
            break 

if __name__ == "__main__":
    main()