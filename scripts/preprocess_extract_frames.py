import cv2
import sys
from pathlib import Path
import shutil

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

def process_session(session_path):
    """
    Extracts frames for a single session folder.
    """
    session_id = session_path.name
    
    # 1. Check for required files
    video_path = session_path / "rgb.avi"
    ts_path = session_path / "rgb_ts.txt"
    
    if not video_path.exists() or not ts_path.exists():
        print(f"[SKIP] {session_id}: Missing rgb.avi or rgb_ts.txt")
        return

    # 2. Prepare output directory
    output_dir = PROCESSED_DIR / session_id / "frames"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Load Timestamps
    with open(ts_path, 'r') as f:
        timestamps = [line.strip() for line in f.readlines()]

    # 4. Open Video
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if len(timestamps) != total_frames:
        print(f"[WARN] {session_id}: Video has {total_frames} frames but {len(timestamps)} timestamps. Alignment may drift.")

    print(f"[START] Processing {session_id} ({total_frames} frames)...")
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        
        # Stop if video ends or we run out of timestamps
        if not ret or frame_idx >= len(timestamps):
            break
            
        timestamp = timestamps[frame_idx]
        
        # Flip horizontally to recover true physical view
        # (raw video is mirrored by upstream process)
        frame_flipped = cv2.flip(frame, 1)
        
        # Save frame: frame_0001_163223.png
        filename = f"frame_{frame_idx:05d}_{timestamp}.png"
        cv2.imwrite(str(output_dir / filename), frame_flipped)
        
        frame_idx += 1

    cap.release()
    print(f"[DONE] {session_id}: Extracted {frame_idx} frames (flipped to physical view).")

def main():
    # Ensure raw directory exists
    if not RAW_DIR.exists():
        print(f"Error: Raw data directory not found at {RAW_DIR}")
        return

    # Iterate over all items in data/raw
    sessions = [p for p in RAW_DIR.iterdir() if p.is_dir()]
    
    if not sessions:
        print("No session folders found in data/raw/")
        return

    print(f"Found {len(sessions)} sessions to process.")
    
    for session_path in sessions:
        process_session(session_path)

if __name__ == "__main__":
    main()