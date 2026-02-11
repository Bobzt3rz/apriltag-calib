# scripts/find_coordinate_transform.py
import json
import numpy as np
from pathlib import Path
from itertools import permutations, product


def load_json(filepath):
    """Load joint data from JSON."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    frames = sorted(data.keys())
    positions = np.array([data[frame] for frame in frames])
    return frames, positions


def apply_transform(positions, perm, signs):
    """
    Apply axis permutation and sign flips.
    
    Args:
        positions: Nx3 array
        perm: tuple of (0,1,2) in some order, e.g., (1,0,2) means (Y,X,Z)
        signs: tuple of 3 signs, e.g., (1,-1,1) means (+X,-Y,+Z)
    """
    transformed = positions[:, perm] * np.array(signs)
    return transformed


def compute_error(new_positions, old_positions):
    """
    Compute RMS error between two trajectories.
    Direct absolute position comparison.
    """
    diff = new_positions - old_positions
    rms = np.sqrt((diff ** 2).mean())
    return rms


def main():
    root_dir = Path(__file__).resolve().parent.parent
    
    # Load your new transformed data (without X flip for now)
    new_file = root_dir / "outputs" / "3d_joint_data_world_2026_02_09_17_03_39_SB-40808C-2.json"
    new_frames, new_positions = load_json(new_file)
    
    # Load old reference data
    old_file = root_dir / "outputs" / "har1_traj_old.json"
    old_frames, old_positions = load_json(old_file)
    
    # Ensure same frames
    assert len(new_frames) == len(old_frames), "Frame count mismatch"
    
    print(f"Testing {len(new_frames)} frames\n")
    print(f"Old data range:")
    print(f"  X: [{old_positions[:, 0].min():.2f}, {old_positions[:, 0].max():.2f}]")
    print(f"  Y: [{old_positions[:, 1].min():.2f}, {old_positions[:, 1].max():.2f}]")
    print(f"  Z: [{old_positions[:, 2].min():.2f}, {old_positions[:, 2].max():.2f}]\n")
    
    print(f"New data range (before transform):")
    print(f"  X: [{new_positions[:, 0].min():.2f}, {new_positions[:, 0].max():.2f}]")
    print(f"  Y: [{new_positions[:, 1].min():.2f}, {new_positions[:, 1].max():.2f}]")
    print(f"  Z: [{new_positions[:, 2].min():.2f}, {new_positions[:, 2].max():.2f}]\n")
    
    # Try all permutations and sign combinations
    best_error = float('inf')
    best_transform = None
    results = []
    
    # Axis permutations
    axis_perms = list(permutations([0, 1, 2]))
    
    # Sign combinations
    sign_combos = list(product([1, -1], repeat=3))
    
    total_tests = len(axis_perms) * len(sign_combos)
    print(f"Testing {total_tests} transformations...\n")
    
    for perm in axis_perms:
        for signs in sign_combos:
            # Apply transformation
            transformed = apply_transform(new_positions, perm, signs)
            
            # Compute error
            error = compute_error(transformed, old_positions)
            
            # Check if all Z values are positive (physical constraint)
            z_positive = (transformed[:, 2] > 0).all()
            
            # Check if ranges are reasonable (within typical room dimensions)
            x_range = (transformed[:, 0].min(), transformed[:, 0].max())
            y_range = (transformed[:, 1].min(), transformed[:, 1].max())
            z_range = (transformed[:, 2].min(), transformed[:, 2].max())
            
            results.append({
                'perm': perm,
                'signs': signs,
                'error': error,
                'z_positive': z_positive,
                'x_range': x_range,
                'y_range': y_range,
                'z_range': z_range
            })
            
            if error < best_error and z_positive:
                best_error = error
                best_transform = (perm, signs)
    
    # Sort by error
    results.sort(key=lambda x: x['error'])
    
    # Print top 10 results
    print("=" * 80)
    print("TOP 10 TRANSFORMATIONS (by absolute RMS error):")
    print("=" * 80)
    
    for i, r in enumerate(results[:10]):
        perm_str = f"({'XYZ'[r['perm'][0]]}, {'XYZ'[r['perm'][1]]}, {'XYZ'[r['perm'][2]]})"
        sign_str = f"({'+' if r['signs'][0] > 0 else '-'}, {'+' if r['signs'][1] > 0 else '-'}, {'+' if r['signs'][2] > 0 else '-'})"
        z_flag = "✓" if r['z_positive'] else "✗"
        
        print(f"\n{i+1}. Permutation: {perm_str}, Signs: {sign_str}")
        print(f"   RMS Error: {r['error']:.4f} m")
        print(f"   X range: [{r['x_range'][0]:.2f}, {r['x_range'][1]:.2f}] m")
        print(f"   Y range: [{r['y_range'][0]:.2f}, {r['y_range'][1]:.2f}] m")
        print(f"   Z range: [{r['z_range'][0]:.2f}, {r['z_range'][1]:.2f}] m  {z_flag}")
    
    # Show best valid transformation
    if best_transform:
        perm, signs = best_transform
        print(f"\n" + "=" * 80)
        print(f"BEST VALID TRANSFORMATION (Z > 0, lowest absolute error):")
        print(f"=" * 80)
        perm_str = f"({'XYZ'[perm[0]]}, {'XYZ'[perm[1]]}, {'XYZ'[perm[2]]})"
        sign_str = f"({'+' if signs[0] > 0 else '-'}, {'+' if signs[1] > 0 else '-'}, {'+' if signs[2] > 0 else '-'})"
        print(f"Permutation: {perm_str}")
        print(f"Signs: {sign_str}")
        print(f"RMS Error: {best_error:.4f} m")
        
        # Apply best transformation and save
        best_transformed = apply_transform(new_positions, perm, signs)
        
        print(f"\nTransformed data range:")
        print(f"  X: [{best_transformed[:, 0].min():.2f}, {best_transformed[:, 0].max():.2f}] m (target: [{old_positions[:, 0].min():.2f}, {old_positions[:, 0].max():.2f}])")
        print(f"  Y: [{best_transformed[:, 1].min():.2f}, {best_transformed[:, 1].max():.2f}] m (target: [{old_positions[:, 1].min():.2f}, {old_positions[:, 1].max():.2f}])")
        print(f"  Z: [{best_transformed[:, 2].min():.2f}, {best_transformed[:, 2].max():.2f}] m (target: [{old_positions[:, 2].min():.2f}, {old_positions[:, 2].max():.2f}])")
        
        output_dict = {frame: best_transformed[i].tolist() 
                      for i, frame in enumerate(new_frames)}
        
        output_file = root_dir / "outputs" / "3d_joint_data_world_CORRECTED.json"
        with open(output_file, 'w') as f:
            json.dump(output_dict, f, indent=2)
        
        print(f"\nSaved corrected data to: {output_file}")


if __name__ == "__main__":
    main()