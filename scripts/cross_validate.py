# scripts/cross_validate.py
"""
Cross-validation of single-tag and joint calibrations.
Uses extrinsics solved from Tag 0, Tag 1, and joint (both tags) to project
each tag's corners. Cross-projection reprojection error reveals world model consistency.

Usage: python scripts/cross_validate.py --config-name cross-validation
"""
import cv2
import json
import hydra
import numpy as np
from pathlib import Path
from omegaconf import DictConfig
from pupil_apriltags import Detector


def get_tag_pose_in_world(tag_cfg):
    """Converts config (measured corner + wall facing) -> tag center + rotation matrix."""
    width_m = tag_cfg.width_m
    height_m = tag_cfg.height_m
    half_w = width_m / 2.0
    half_h = height_m / 2.0
    corner_pos_world = np.array(tag_cfg.position_xyz)

    offsets = {
        "top_left":     np.array([ half_w,  half_h, 0]),
        "top_right":    np.array([-half_w,  half_h, 0]),
        "bottom_right": np.array([-half_w, -half_h, 0]),
        "bottom_left":  np.array([ half_w, -half_h, 0]),
    }
    local_offset = offsets[tag_cfg.measured_corner]

    rotations = {
        "neg_x": np.array([[ 0,  0,  1], [-1,  0,  0], [ 0, -1,  0]]),
        "pos_x": np.array([[ 0,  0, -1], [ 1,  0,  0], [ 0, -1,  0]]),
        "neg_y": np.array([[ 1,  0,  0], [ 0,  0,  1], [ 0, -1,  0]]),
        "pos_y": np.array([[-1,  0,  0], [ 0,  0, -1], [ 0, -1,  0]]),
    }
    r_matrix = rotations[tag_cfg.wall_facing]

    center_pos_world = corner_pos_world + (r_matrix @ local_offset)
    return center_pos_world, r_matrix, width_m, height_m


def get_world_corners(center, rotation, width_m, height_m):
    """Returns 4 world-frame corners in pupil_apriltags detection order."""
    half_w = width_m / 2.0
    half_h = height_m / 2.0
    local_corners = np.array([
        [-half_w,  half_h, 0],
        [ half_w,  half_h, 0],
        [ half_w, -half_h, 0],
        [-half_w, -half_h, 0],
    ])
    return center + (rotation @ local_corners.T).T


def load_extrinsics(filepath):
    """Load camera extrinsics from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    rvec = np.array(data["rvec"]).reshape(3, 1)
    tvec = np.array(data["tvec"]).reshape(3, 1)
    R_world_to_cam = np.array(data["camera_rotation_world_to_cam"])
    camera_position_world = np.array(data["camera_position_world"])
    frame = data.get("frame", "unknown")
    return rvec, tvec, R_world_to_cam, camera_position_world, frame


def load_intrinsics(filepath):
    K = np.loadtxt(filepath, delimiter=',')
    D = np.zeros(5)
    return K, D


CORNER_LABELS = ["BL", "BR", "TR", "TL"]


def draw_corners(img, corners, color, label_prefix, thickness=2, radius=8):
    """Draw detected or projected corners with labels."""
    for i, (x, y) in enumerate(corners):
        pt = (int(round(x)), int(round(y)))
        cv2.circle(img, pt, radius, color, thickness)
        label = f"{label_prefix} {CORNER_LABELS[i]}"
        cv2.putText(img, label, (pt[0] + 10, pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_quad(img, corners, color, thickness=2):
    """Draw lines connecting 4 corners into a quad."""
    pts = np.array(corners, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)


def draw_error_lines(img, detected, projected, color, thickness=1):
    """Draw lines from detected to projected corners showing the error."""
    for (dx, dy), (px, py) in zip(detected, projected):
        pt_d = (int(round(dx)), int(round(dy)))
        pt_p = (int(round(px)), int(round(py)))
        cv2.line(img, pt_d, pt_p, color, thickness, cv2.LINE_AA)


def draw_legend(img, x, y, entries):
    """Draw a color legend on the image."""
    cv2.rectangle(img, (x - 5, y - 5), (x + 300, y + len(entries) * 25 + 5),
                  (0, 0, 0), -1)
    cv2.rectangle(img, (x - 5, y - 5), (x + 300, y + len(entries) * 25 + 5),
                  (255, 255, 255), 1)
    for i, (color, text) in enumerate(entries):
        cy = y + i * 25 + 15
        cv2.circle(img, (x + 10, cy - 4), 6, color, -1)
        cv2.putText(img, text, (x + 25, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def visualize_cross(img, det_by_id, world_tags, rvec_src, tvec_src,
                    src_label, src_tag_id, tgt_tag_id, K, D, output_path):
    """
    Visualize cross-projection: src extrinsics projecting both tags.
    src_tag_id: tag whose extrinsics we're using (or None for joint)
    tgt_tag_id: the "other" tag for cross-projection
    """
    vis = img.copy()
    h, w = vis.shape[:2]

    GREEN = (0, 255, 0)
    ORANGE = (0, 165, 255)
    RED = (0, 0, 255)
    YELLOW = (0, 255, 255)
    WHITE = (255, 255, 255)

    info_lines = []

    # --- Self-projection on source tag (if single-tag) ---
    if src_tag_id is not None:
        det_src = np.array(det_by_id[src_tag_id].corners)
        reproj_src, _ = cv2.projectPoints(
            world_tags[src_tag_id]["world_corners"], rvec_src, tvec_src, K, D
        )
        reproj_src = reproj_src.reshape(-1, 2)
        err_self = np.linalg.norm(det_src - reproj_src, axis=1)

        draw_quad(vis, det_src, GREEN, 2)
        draw_corners(vis, det_src, GREEN, f"T{src_tag_id} det", radius=6)
        draw_quad(vis, reproj_src, ORANGE, 2)
        draw_corners(vis, reproj_src, ORANGE, f"T{src_tag_id} self", radius=4, thickness=1)
        info_lines.append(f"Self-reproj Tag {src_tag_id}: mean={err_self.mean():.1f}px max={err_self.max():.1f}px")

        # Cross-project target tag
        det_tgt = np.array(det_by_id[tgt_tag_id].corners)
        reproj_tgt, _ = cv2.projectPoints(
            world_tags[tgt_tag_id]["world_corners"], rvec_src, tvec_src, K, D
        )
        reproj_tgt = reproj_tgt.reshape(-1, 2)
        err_cross = np.linalg.norm(det_tgt - reproj_tgt, axis=1)

        draw_quad(vis, det_tgt, GREEN, 2)
        draw_corners(vis, det_tgt, GREEN, f"T{tgt_tag_id} det", radius=6)
        draw_quad(vis, reproj_tgt, RED, 2)
        draw_corners(vis, reproj_tgt, RED, f"T{tgt_tag_id} cross", radius=6)
        draw_error_lines(vis, det_tgt, reproj_tgt, YELLOW, 2)

        for i, (dx, dy) in enumerate(det_tgt):
            mid_x = int((dx + reproj_tgt[i][0]) / 2)
            mid_y = int((dy + reproj_tgt[i][1]) / 2)
            cv2.putText(vis, f"{err_cross[i]:.1f}px", (mid_x + 5, mid_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW, 1, cv2.LINE_AA)

        info_lines.append(f"Cross-proj  Tag {tgt_tag_id}: mean={err_cross.mean():.1f}px max={err_cross.max():.1f}px")

        legend_entries = [
            (GREEN,  "Detected corners"),
            (ORANGE, "Self-projected (sanity)"),
            (RED,    "Cross-projected (test)"),
            (YELLOW, "Error line"),
        ]

    else:
        # Joint: project BOTH tags, both are "cross-projections" in a sense
        BLUE = (255, 100, 0)
        MAGENTA = (255, 0, 255)

        err_all = {}
        for tid, color, proj_label in [(0, BLUE, "T0 joint"), (1, MAGENTA, "T1 joint")]:
            det_corners = np.array(det_by_id[tid].corners)
            reproj, _ = cv2.projectPoints(
                world_tags[tid]["world_corners"], rvec_src, tvec_src, K, D
            )
            reproj = reproj.reshape(-1, 2)
            err = np.linalg.norm(det_corners - reproj, axis=1)
            err_all[tid] = err

            draw_quad(vis, det_corners, GREEN, 2)
            draw_corners(vis, det_corners, GREEN, f"T{tid} det", radius=6)
            draw_quad(vis, reproj, color, 2)
            draw_corners(vis, reproj, color, f"{proj_label}", radius=5, thickness=1)
            draw_error_lines(vis, det_corners, reproj, YELLOW, 2)

            for i, (dx, dy) in enumerate(det_corners):
                mid_x = int((dx + reproj[i][0]) / 2)
                mid_y = int((dy + reproj[i][1]) / 2)
                cv2.putText(vis, f"{err[i]:.1f}px", (mid_x + 5, mid_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW, 1, cv2.LINE_AA)

            info_lines.append(f"Joint → Tag {tid}: mean={err.mean():.1f}px max={err.max():.1f}px")

        legend_entries = [
            (GREEN,   "Detected corners"),
            (BLUE,    "Joint → Tag 0"),
            (MAGENTA, "Joint → Tag 1"),
            (YELLOW,  "Error line"),
        ]

    # --- Title and info text ---
    cv2.putText(vis, src_label, (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2, cv2.LINE_AA)
    for i, line in enumerate(info_lines):
        cv2.putText(vis, line, (20, 70 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)

    draw_legend(vis, 20, h - 20 - len(legend_entries) * 25, legend_entries)

    cv2.imwrite(str(output_path), vis)


def compute_reproj_errors(det_by_id, world_tags, rvec, tvec, K, D):
    """Compute reprojection errors for both tags given one set of extrinsics."""
    errors = {}
    for tid in [0, 1]:
        det_corners = np.array(det_by_id[tid].corners)
        reproj, _ = cv2.projectPoints(
            world_tags[tid]["world_corners"], rvec, tvec, K, D
        )
        reproj = reproj.reshape(-1, 2)
        err = np.linalg.norm(det_corners - reproj, axis=1)
        errors[tid] = {"mean": err.mean(), "max": err.max()}
    return errors


@hydra.main(version_base=None, config_path="../conf", config_name="cross-validation")
def main(cfg: DictConfig):
    print("=" * 70)
    print("  CROSS-VALIDATION: Single-Tag & Joint Calibration Quality")
    print("=" * 70)

    root_dir = Path(hydra.utils.get_original_cwd())

    # 1. Load intrinsics
    intrinsics_path = root_dir / cfg.camera.intrinsics_file
    K, D = load_intrinsics(intrinsics_path)

    # 2. Load world tag info
    world_tags = {}
    for tag in cfg.scene.tags:
        center, rotation, width_m, height_m = get_tag_pose_in_world(tag)
        world_tags[tag.id] = {
            "center": center,
            "rotation": rotation,
            "width_m": width_m,
            "height_m": height_m,
            "world_corners": get_world_corners(center, rotation, width_m, height_m),
        }
        print(f"\nTag {tag.id} ({tag.wall_facing}):")
        print(f"  Center: {center}")
        print(f"  Outward normal: {-rotation[:, 2]}")

    # 3. Load all extrinsics
    tag0_path = root_dir / cfg.tag0_extrinsics_file
    tag1_path = root_dir / cfg.tag1_extrinsics_file
    joint_path = root_dir / cfg.joint_extrinsics_file

    missing = []
    if not tag0_path.exists(): missing.append(f"Tag 0: {tag0_path}")
    if not tag1_path.exists(): missing.append(f"Tag 1: {tag1_path}")
    if not joint_path.exists(): missing.append(f"Joint: {joint_path}")
    if missing:
        print(f"\nError: Missing extrinsics files:")
        for m in missing:
            print(f"  ✗ {m}")
        return

    rvec0, tvec0, R0, pos0, frame0 = load_extrinsics(tag0_path)
    rvec1, tvec1, R1, pos1, frame1 = load_extrinsics(tag1_path)
    rvec_j, tvec_j, R_j, pos_j, frame_j = load_extrinsics(joint_path)

    print(f"\n{'=' * 70}")
    print(f"  Loaded Extrinsics")
    print(f"{'=' * 70}")
    sources = [
        ("Tag 0 only", pos0, frame0),
        ("Tag 1 only", pos1, frame1),
        ("Joint",      pos_j, frame_j),
    ]
    for label, pos, frame in sources:
        print(f"  {label:<12} (from {frame}):")
        print(f"    Camera pos: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

    print(f"\n  Position differences:")
    print(f"    Tag0 vs Tag1:  {np.linalg.norm(pos0 - pos1):.4f} m")
    print(f"    Tag0 vs Joint: {np.linalg.norm(pos0 - pos_j):.4f} m")
    print(f"    Tag1 vs Joint: {np.linalg.norm(pos1 - pos_j):.4f} m")

    # 4. Setup detector
    at_detector = Detector(
        families=cfg.scene.tag_family,
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )

    frames_dir = root_dir / "data" / "processed" / cfg.session_id / "frames"
    if not frames_dir.exists():
        print(f"\nError: Frames not found at {frames_dir}")
        return

    # 5. Process all frames with both tags visible
    print(f"\n{'=' * 70}")
    print(f"  Cross-Projection Results")
    print(f"{'=' * 70}")

    results = []
    vis_frame = None

    for img_path in sorted(frames_dir.glob("*.png")):
        img = cv2.imread(str(img_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detections = at_detector.detect(gray)

        det_by_id = {d.tag_id: d for d in detections}
        if 0 not in det_by_id or 1 not in det_by_id:
            continue

        if vis_frame is None:
            vis_frame = (img.copy(), {k: v for k, v in det_by_id.items()}, img_path.name)

        frame_result = {"frame": img_path.name}

        # Errors from each extrinsics source
        for label, rvec, tvec, key_prefix in [
            ("tag0", rvec0, tvec0, "ext0"),
            ("tag1", rvec1, tvec1, "ext1"),
            ("joint", rvec_j, tvec_j, "joint"),
        ]:
            errs = compute_reproj_errors(det_by_id, world_tags, rvec, tvec, K, D)
            frame_result[f"{key_prefix}_to_tag0_mean"] = errs[0]["mean"]
            frame_result[f"{key_prefix}_to_tag0_max"] = errs[0]["max"]
            frame_result[f"{key_prefix}_to_tag1_mean"] = errs[1]["mean"]
            frame_result[f"{key_prefix}_to_tag1_max"] = errs[1]["max"]

        results.append(frame_result)

    if not results:
        print("No frames found with both tags visible!")
        return

    # 6. Print per-frame table
    print(f"\n{'Frame':<40} {'E0→T0':>7} {'E0→T1':>7} {'E1→T0':>7} {'E1→T1':>7} {'J→T0':>7} {'J→T1':>7}")
    print("-" * 100)
    for r in results[:15]:
        print(f"{r['frame']:<40} "
              f"{r['ext0_to_tag0_mean']:6.2f}px {r['ext0_to_tag1_mean']:6.2f}px "
              f"{r['ext1_to_tag0_mean']:6.2f}px {r['ext1_to_tag1_mean']:6.2f}px "
              f"{r['joint_to_tag0_mean']:6.2f}px {r['joint_to_tag1_mean']:6.2f}px")
    if len(results) > 15:
        print(f"  ... ({len(results) - 15} more frames)")

    # 7. Summary statistics
    def avg(key): return np.mean([r[key] for r in results])
    def mx(key):  return np.max([r[key] for r in results])

    print(f"\n{'=' * 70}")
    print(f"  SUMMARY ({len(results)} frames with both tags)")
    print(f"{'=' * 70}")

    print(f"\n  {'Source':<20} {'→ Tag 0 (mean)':>15} {'→ Tag 0 (max)':>15} {'→ Tag 1 (mean)':>15} {'→ Tag 1 (max)':>15}")
    print(f"  {'-'*80}")

    rows = [
        ("Tag 0 ext (self→0)", "ext0_to_tag0", "ext0_to_tag1"),
        ("Tag 1 ext (self→1)", "ext1_to_tag0", "ext1_to_tag1"),
        ("Joint ext",          "joint_to_tag0", "joint_to_tag1"),
    ]
    for label, k0, k1 in rows:
        print(f"  {label:<20} {avg(k0+'_mean'):14.2f}px {mx(k0+'_max'):14.2f}px "
              f"{avg(k1+'_mean'):14.2f}px {mx(k1+'_max'):14.2f}px")

    # Interpretation
    joint_avg = (avg("joint_to_tag0_mean") + avg("joint_to_tag1_mean")) / 2
    cross_0to1 = avg("ext0_to_tag1_mean")
    cross_1to0 = avg("ext1_to_tag0_mean")
    single_avg = (cross_0to1 + cross_1to0) / 2

    print(f"\n  Single-tag cross-projection average: {single_avg:.1f} px")
    print(f"  Joint reprojection average:          {joint_avg:.1f} px")

    if joint_avg < 3.0:
        print(f"  ✓ Joint calibration is excellent ({joint_avg:.1f} px)")
    elif joint_avg < 8.0:
        print(f"  ~ Joint calibration is good ({joint_avg:.1f} px)")
    else:
        print(f"  ⚠ Joint calibration has notable error ({joint_avg:.1f} px)")

    improvement = single_avg / joint_avg if joint_avg > 0 else float('inf')
    print(f"  Joint is {improvement:.1f}x better than single-tag cross-projection")

    # 8. Visualizations
    if vis_frame is not None:
        vis_img, vis_dets, vis_name = vis_frame
        vis_dir = root_dir / "outputs" / "cross_validation_vis"
        vis_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  Generating visualizations from frame: {vis_name}")

        # Image 1: Tag 0 ext → project both
        out1 = vis_dir / f"cross_val_ext0_proj1_{cfg.camera.id}.png"
        visualize_cross(vis_img, vis_dets, world_tags,
                        rvec0, tvec0, "Extrinsics from Tag 0 -> Project Tag 1",
                        src_tag_id=0, tgt_tag_id=1, K=K, D=D, output_path=out1)
        print(f"    Saved: {out1}")

        # Image 2: Tag 1 ext → project both
        out2 = vis_dir / f"cross_val_ext1_proj0_{cfg.camera.id}.png"
        visualize_cross(vis_img, vis_dets, world_tags,
                        rvec1, tvec1, "Extrinsics from Tag 1 -> Project Tag 0",
                        src_tag_id=1, tgt_tag_id=0, K=K, D=D, output_path=out2)
        print(f"    Saved: {out2}")

        # Image 3: Joint ext → project both tags
        out3 = vis_dir / f"cross_val_joint_{cfg.camera.id}.png"
        visualize_cross(vis_img, vis_dets, world_tags,
                        rvec_j, tvec_j, "Joint Extrinsics -> Project Both Tags",
                        src_tag_id=None, tgt_tag_id=None, K=K, D=D, output_path=out3)
        print(f"    Saved: {out3}")

    # 9. Save results
    output = {
        "session_id": cfg.session_id,
        "camera_id": cfg.camera.id,
        "num_frames": len(results),
        "extrinsics_sources": {
            "tag0": {"frame": frame0, "position": pos0.tolist()},
            "tag1": {"frame": frame1, "position": pos1.tolist()},
            "joint": {"frame": frame_j, "position": pos_j.tolist()},
        },
        "position_differences_m": {
            "tag0_vs_tag1": float(np.linalg.norm(pos0 - pos1)),
            "tag0_vs_joint": float(np.linalg.norm(pos0 - pos_j)),
            "tag1_vs_joint": float(np.linalg.norm(pos1 - pos_j)),
        },
        "summary": {
            "ext0_to_tag0_mean_px": float(avg("ext0_to_tag0_mean")),
            "ext0_to_tag1_mean_px": float(avg("ext0_to_tag1_mean")),
            "ext1_to_tag0_mean_px": float(avg("ext1_to_tag0_mean")),
            "ext1_to_tag1_mean_px": float(avg("ext1_to_tag1_mean")),
            "joint_to_tag0_mean_px": float(avg("joint_to_tag0_mean")),
            "joint_to_tag1_mean_px": float(avg("joint_to_tag1_mean")),
        },
        "per_frame": results,
    }

    output_path = root_dir / "outputs" / f"cross_validation_{cfg.camera.id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()