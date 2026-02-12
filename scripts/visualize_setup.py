# scripts/visualize_setup.py
"""
Visualize the room setup: AprilTag positions, camera location, and look direction.
Provides a sanity check that calibration matches the physical layout.
"""
import json
import hydra
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
from pathlib import Path
from omegaconf import DictConfig


# Hardcoded reference camera positions (for comparison)
REFERENCE_CAMERAS = {
    "har_01": {"A": [5.03, 8.45, 0.90], "B": [5.73, 8.18, 0.90]},
}


def get_tag_pose_in_world(tag_cfg):
    """
    Converts config (measured corner + wall facing) -> tag center + rotation matrix.
    Duplicated from calibrate.py for standalone use.
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
    
    # Outward normal is -Z in tag frame, so in world it's -R[:, 2]
    outward_normal = -r_matrix[:, 2]
    
    return center_pos_world, r_matrix, outward_normal, width_m, height_m


def load_extrinsics(filepath):
    """Load camera extrinsics from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    R_world_to_cam = np.array(data["camera_rotation_world_to_cam"])
    camera_position_world = np.array(data["camera_position_world"])
    
    return R_world_to_cam, camera_position_world


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print(f"--- Visualizing Room Setup ---")
    print(f"Scene: {cfg.scene.name}")
    print(f"Camera: {cfg.camera.id}\n")
    
    root_dir = Path(hydra.utils.get_original_cwd())
    
    # 1. Load tag information from scene config
    tags_info = []
    for tag in cfg.scene.tags:
        center, rotation, normal, width, height = get_tag_pose_in_world(tag)
        tags_info.append({
            'id': tag.id,
            'center': center,
            'normal': normal,
            'width': width,
            'height': height,
            'facing': tag.wall_facing,
            'measured_corner': tag.measured_corner,
            'corner_pos': np.array(tag.position_xyz)
        })
        print(f"Tag {tag.id}:")
        print(f"  Measured corner ({tag.measured_corner}): {tag.position_xyz}")
        print(f"  Center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
        print(f"  Outward normal: [{normal[0]:.2f}, {normal[1]:.2f}, {normal[2]:.2f}] ({tag.wall_facing})")
        print()
    
    # 2. Load camera extrinsics (calibrated)
    extrinsics_path = root_dir / cfg.extrinsics_file
    camera_loaded = False
    
    if extrinsics_path.exists():
        R_world_to_cam, camera_pos = load_extrinsics(extrinsics_path)
        camera_loaded = True
        
        cam_look_dir = R_world_to_cam[2, :]
        
        print(f"Calibrated Camera (red):")
        print(f"  Position: [{camera_pos[0]:.3f}, {camera_pos[1]:.3f}, {camera_pos[2]:.3f}]")
        print(f"  Look direction: [{cam_look_dir[0]:.2f}, {cam_look_dir[1]:.2f}, {cam_look_dir[2]:.2f}]")
    else:
        print(f"Warning: Extrinsics file not found at {extrinsics_path}")
        print(f"Only showing tag positions.\n")
    
    # 3. Load reference camera (hardcoded)
    ref_camera = REFERENCE_CAMERAS.get(cfg.camera.id)
    if ref_camera:
        ref_A = np.array(ref_camera["A"])
        ref_B = np.array(ref_camera["B"])
        ref_look_dir = ref_B - ref_A
        ref_look_dir = ref_look_dir / np.linalg.norm(ref_look_dir)  # normalize
        
        print(f"\nReference Camera (yellow):")
        print(f"  Position A: [{ref_A[0]:.3f}, {ref_A[1]:.3f}, {ref_A[2]:.3f}]")
        print(f"  Look-at B:  [{ref_B[0]:.3f}, {ref_B[1]:.3f}, {ref_B[2]:.3f}]")
        print(f"  Look direction: [{ref_look_dir[0]:.2f}, {ref_look_dir[1]:.2f}, {ref_look_dir[2]:.2f}]")
    
    # 4. Create figure with two views
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # ===== TOP-DOWN VIEW (XY plane) =====
    ax1 = axes[0]
    ax1.set_title("Top-Down View (XY Plane)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("X (meters) - Right →", fontsize=12)
    ax1.set_ylabel("Y (meters) - Forward into room →", fontsize=12)
    
    # Draw room outline (approximate)
    ax1.axhline(y=0, color='brown', linestyle='--', alpha=0.5, label='Door wall (Y=0)')
    ax1.axvline(x=0, color='gray', linestyle=':', alpha=0.3)
    
    # Draw origin marker (door)
    ax1.plot(0, 0, 'ks', markersize=12, label='Origin (door)')
    ax1.annotate('DOOR', (0, -0.3), ha='center', fontsize=10, fontweight='bold')
    
    # Draw tags
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    for i, tag in enumerate(tags_info):
        color = colors[i % len(colors)]
        cx, cy, cz = tag['center']
        nx, ny, nz = tag['normal']
        
        # Plot tag center
        ax1.plot(cx, cy, 'o', color=color, markersize=12, 
                label=f"Tag {tag['id']} ({tag['facing']})")
        
        # Plot measured corner
        corner = tag['corner_pos']
        ax1.plot(corner[0], corner[1], 's', color=color, markersize=8, alpha=0.5)
        
        # Draw outward normal as arrow (in XY plane)
        arrow_len = 0.8
        ax1.annotate('', xy=(cx + nx*arrow_len, cy + ny*arrow_len), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))
        
        # Label
        ax1.annotate(f'Tag {tag["id"]}', (cx + 0.2, cy + 0.2), fontsize=10, color=color)
    
    # Draw calibrated camera (red)
    if camera_loaded:
        ax1.plot(camera_pos[0], camera_pos[1], 'r^', markersize=15, 
                label='Calibrated Camera', zorder=10)
        
        # Draw look direction
        look_len = 1.5
        ax1.annotate('', xy=(camera_pos[0] + cam_look_dir[0]*look_len, 
                            camera_pos[1] + cam_look_dir[1]*look_len),
                    xytext=(camera_pos[0], camera_pos[1]),
                    arrowprops=dict(arrowstyle='->', color='red', lw=3))
        
        ax1.annotate('CAL', (camera_pos[0] + 0.2, camera_pos[1] - 0.3), 
                    fontsize=10, fontweight='bold', color='red')
    
    # Draw reference camera (yellow)
    if ref_camera:
        ax1.plot(ref_A[0], ref_A[1], '^', color='gold', markersize=15, 
                markeredgecolor='black', markeredgewidth=1,
                label='Reference Camera', zorder=10)
        
        # Draw look direction (A to B, extended)
        look_len = 1.5
        ax1.annotate('', xy=(ref_A[0] + ref_look_dir[0]*look_len, 
                            ref_A[1] + ref_look_dir[1]*look_len),
                    xytext=(ref_A[0], ref_A[1]),
                    arrowprops=dict(arrowstyle='->', color='gold', lw=3))
        
        # Also plot point B
        ax1.plot(ref_B[0], ref_B[1], 'o', color='gold', markersize=8, 
                markeredgecolor='black', markeredgewidth=1, alpha=0.7)
        
        ax1.annotate('REF', (ref_A[0] - 0.5, ref_A[1] - 0.3), 
                    fontsize=10, fontweight='bold', color='goldenrod')
    
    ax1.set_xlim(-1, 10)
    ax1.set_ylim(-1, 10)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=9)
    
    # ===== SIDE VIEW (YZ plane) =====
    ax2 = axes[1]
    ax2.set_title("Side View (YZ Plane) - Looking from +X", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Y (meters) - Forward into room →", fontsize=12)
    ax2.set_ylabel("Z (meters) - Up ↑", fontsize=12)
    
    # Draw floor
    ax2.axhline(y=0, color='brown', linestyle='-', lw=2, label='Floor (Z=0)')
    
    # Draw tags
    for i, tag in enumerate(tags_info):
        color = colors[i % len(colors)]
        cx, cy, cz = tag['center']
        
        # Plot tag center
        ax2.plot(cy, cz, 'o', color=color, markersize=12, 
                label=f"Tag {tag['id']} (Z={cz:.2f}m)")
        
        # Draw tag height extent (approximate)
        half_h = tag['height'] / 2
        ax2.plot([cy, cy], [cz - half_h, cz + half_h], '-', color=color, lw=3, alpha=0.5)
    
    # Draw calibrated camera (red)
    if camera_loaded:
        ax2.plot(camera_pos[1], camera_pos[2], 'r^', markersize=15, 
                label=f'Calibrated (Z={camera_pos[2]:.2f}m)', zorder=10)
        
        # Draw look direction in YZ plane
        look_len = 1.0
        ax2.annotate('', xy=(camera_pos[1] + cam_look_dir[1]*look_len, 
                            camera_pos[2] + cam_look_dir[2]*look_len),
                    xytext=(camera_pos[1], camera_pos[2]),
                    arrowprops=dict(arrowstyle='->', color='red', lw=3))
    
    # Draw reference camera (yellow)
    if ref_camera:
        ax2.plot(ref_A[1], ref_A[2], '^', color='gold', markersize=15,
                markeredgecolor='black', markeredgewidth=1,
                label=f'Reference (Z={ref_A[2]:.2f}m)', zorder=10)
        
        # Draw look direction in YZ plane
        look_len = 1.0
        ax2.annotate('', xy=(ref_A[1] + ref_look_dir[1]*look_len, 
                            ref_A[2] + ref_look_dir[2]*look_len),
                    xytext=(ref_A[1], ref_A[2]),
                    arrowprops=dict(arrowstyle='->', color='gold', lw=3))
    
    ax2.set_xlim(-1, 10)
    ax2.set_ylim(-0.5, 3)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    # 5. Save figure
    output_path = root_dir / "outputs" / f"setup_visualization_{cfg.camera.id}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    # 6. Show plot
    if cfg.get('show_plot', True):
        plt.show()
    
    # 7. Print comparison summary
    if camera_loaded and ref_camera:
        print("\n" + "=" * 60)
        print("CALIBRATION vs REFERENCE COMPARISON")
        print("=" * 60)
        
        pos_diff = np.linalg.norm(camera_pos - ref_A)
        print(f"\nPosition difference: {pos_diff:.3f} m")
        print(f"  Calibrated: [{camera_pos[0]:.3f}, {camera_pos[1]:.3f}, {camera_pos[2]:.3f}]")
        print(f"  Reference:  [{ref_A[0]:.3f}, {ref_A[1]:.3f}, {ref_A[2]:.3f}]")
        print(f"  Delta:      [{camera_pos[0]-ref_A[0]:.3f}, {camera_pos[1]-ref_A[1]:.3f}, {camera_pos[2]-ref_A[2]:.3f}]")
        
        # Angle between look directions
        dot = np.clip(np.dot(cam_look_dir, ref_look_dir), -1, 1)
        angle_diff = np.degrees(np.arccos(dot))
        print(f"\nLook direction difference: {angle_diff:.1f}°")
        print(f"  Calibrated: [{cam_look_dir[0]:.3f}, {cam_look_dir[1]:.3f}, {cam_look_dir[2]:.3f}]")
        print(f"  Reference:  [{ref_look_dir[0]:.3f}, {ref_look_dir[1]:.3f}, {ref_look_dir[2]:.3f}]")


if __name__ == "__main__":
    main()