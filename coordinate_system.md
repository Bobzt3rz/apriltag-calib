# Coordinate System Reference

## World Frame

- **Origin**: At the door, directly to the left of a person entering the room
- **+X**: Right (when entering the room)
- **+Y**: Forward (deeper into the room from the door)
- **+Z**: Up (toward the ceiling)

## AprilTag Local Frame

Defined from the perspective of a viewer **looking at the front face of the tag**:

- **+X**: Viewer's right
- **+Y**: Down (gravity)
- **+Z**: Into the tag surface (into the wall, away from viewer)

This is **right-handed**: X × Y = Z (right × down = into wall). The outward normal (toward the camera/viewer) is **-Z**.

### Corner ordering (from `pupil_apriltags` detector)

The library's homography maps from ideal corners `(-1,+1), (+1,+1), (+1,-1), (-1,-1)`. In tag-local coordinates:

| Corner index | Ideal coords | Tag-local position | Label |
|---|---|---|---|
| 0 | (-1, +1) | (-half, +half, 0) | Bottom-left |
| 1 | (+1, +1) | (+half, +half, 0) | Bottom-right |
| 2 | (+1, -1) | (+half, -half, 0) | Top-right |
| 3 | (-1, -1) | (-half, -half, 0) | Top-left |

Corners wrap **counter-clockwise** around the tag in the image.

## Camera Frame (OpenCV convention)

- **+X**: Right in the image
- **+Y**: Down in the image
- **+Z**: Forward (out of the camera lens, into the scene)

## Viewer's Right Direction

When standing in front of a tag and facing it, the viewer's right hand direction in world coordinates depends on which direction the viewer is facing:

| Viewer faces | Viewer's right (= tag +X in world) |
|---|---|
| +X | -Y |
| -X | +Y |
| +Y | +X |
| -Y | -X |

**How to remember**: If you enter the room facing +Y and turn right, you face +X and your right is now -Y. Each 90° right turn cycles through: facing +Y→right is +X, facing +X→right is -Y, facing -Y→right is -X, facing -X→right is +Y.

## Tag-to-World Rotation Matrices

The rotation matrix `R` transforms tag-local coordinates to world coordinates: `world_vec = R @ tag_vec`. Columns of `R` are where each tag axis points in world.

`wall_facing` = direction of the tag's **outward normal** = where tag **-Z** points in world. So tag +Z points **opposite** to `wall_facing` (into the wall).

### `neg_x` (outward normal = -X, tag on right wall)

- Viewer faces +X. Right = -Y.
- tag +X = -Y, tag +Y = -Z, tag +Z = +X

```python
R = [[ 0,  0,  1],
     [-1,  0,  0],
     [ 0, -1,  0]]
```

### `pos_x` (outward normal = +X, tag on left wall)

- Viewer faces -X. Right = +Y.
- tag +X = +Y, tag +Y = -Z, tag +Z = -X

```python
R = [[ 0,  0, -1],
     [ 1,  0,  0],
     [ 0, -1,  0]]
```

### `neg_y` (outward normal = -Y, tag faces toward door)

- Viewer faces +Y. Right = +X.
- tag +X = +X, tag +Y = -Z, tag +Z = +Y

```python
R = [[ 1,  0,  0],
     [ 0,  0,  1],
     [ 0, -1,  0]]
```

### `pos_y` (outward normal = +Y, tag faces away from door)

- Viewer faces -Y. Right = -X.
- tag +X = -X, tag +Y = -Z, tag +Z = -Y

```python
R = [[-1,  0,  0],
     [ 0,  0, -1],
     [ 0, -1,  0]]
```

### Verification

All matrices satisfy:
- `det(R) = +1` (proper rotation, no reflection)
- `R @ R.T = I` (orthogonal)
- Column 2 = Column 0 × Column 1 (right-hand rule)
- Column 1 = `[0, 0, -1]` for all cases (tag +Y = world down)
- `-R[:, 2]` matches the `wall_facing` direction

## Image Flip

The raw stored frames are **horizontally mirrored** by an upstream process. The flipped image (`cv2.flip(gray, 1)`) recovers the true physical view. Detection is performed on the flipped image.

The original camera intrinsics matrix `K` is used directly with the flipped image — no `cx` adjustment is needed — because the intrinsics describe the physical optical path, which matches the flipped (corrected) image.

## Key Lessons

1. **Viewer's right is not intuitive.** Facing +X does NOT mean right is +Y. Use the table above.
2. **Tag +Z goes into the wall**, not out. The outward normal is -Z. Getting this wrong produces mirrored geometry that solvePnP can still fit individually but fails catastrophically on cross-projection between tags.
3. **Single-tag PnP has planar ambiguity.** Always use multiple tags at different orientations for reliable pose estimation. Two coplanar points alone admit two valid solutions with nearly identical reprojection error.
4. **Cross-projection is the best diagnostic.** Solve PnP from one tag, project the other tag's corners. If the error is large, the world model is wrong. Self-reprojection can be misleadingly low due to degenerate solutions.