# scripts/visualize_trajectory.py
import json
import hydra
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import DictConfig


def load_world_joint_data(filepath):
    """Load transformed joint data in world coordinates."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Extract frame names and positions
    frames = sorted(data.keys())
    positions = np.array([data[frame] for frame in frames])
    
    return frames, positions


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print(f"--- Visualizing Joint Trajectory ---")
    print(f"Data session: {cfg.session_id}\n")
    
    root_dir = Path(hydra.utils.get_original_cwd())
    
    # 1. Load world coordinate joint data
    joint_data_path = root_dir / cfg.joint_data_world_file
    if not joint_data_path.exists():
        print(f"Error: World joint data not found at {joint_data_path}")
        print(f"Please run transformation first: python scripts/transform_to_world.py session_id={cfg.session_id}")
        return
    
    print(f"Loading joint data from: {joint_data_path}")
    frames, positions = load_world_joint_data(joint_data_path)
    
    print(f"Loaded {len(frames)} frames")
    print(f"Position range:")
    print(f"  X: [{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}] m")
    print(f"  Y: [{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}] m")
    print(f"  Z: [{positions[:, 2].min():.2f}, {positions[:, 2].max():.2f}] m\n")
    
    # 2. Create simple 2D plot
    plt.figure(figsize=(10, 10))
    
    # Plot trajectory line
    plt.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.3, linewidth=1)
    
    # Plot points color-coded by time
    scatter = plt.scatter(positions[:, 0], positions[:, 1], 
                         c=range(len(positions)), 
                         cmap='viridis', 
                         s=30, 
                         alpha=0.8)
    
    # Mark start and end
    plt.plot(positions[0, 0], positions[0, 1], 'go', markersize=15, label='Start', zorder=5)
    plt.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=15, label='End', zorder=5)
    
    plt.xlabel('X (meters)', fontsize=14)
    plt.ylabel('Y (meters)', fontsize=14)
    plt.title('Joint Trajectory (Top-Down View)', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Set fixed bounds
    plt.xlim(6, 7)
    plt.ylim(6, 7)
    plt.gca().set_aspect('equal')
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Frame Index', fontsize=12)
    
    plt.tight_layout()
    
    # 3. Save figure
    output_path = root_dir / "outputs" / f"trajectory_{cfg.session_id}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    # 4. Show plot
    if cfg.get('show_plot', True):
        plt.show()


if __name__ == "__main__":
    main()