import cv2
import os
import csv
import numpy as np
from src.detection import detect_players
from src.tracker import PlayerTracker

# Generate a list of distinct colors for visualization
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(100, 3), dtype="uint8")

def main():
    # --- Configuration ---
    VIDEO_PATH = os.path.join('data', 'videos', '15sec_input_720p.mp4')
    MODEL_PATH = os.path.join('models', 'best.pt')
    OUTPUT_VIDEO_PATH = os.path.join('outputs', 'annotated_videos', 'annotated_output.mp4')
    OUTPUT_CSV_PATH = os.path.join('outputs', 'logs', 'tracking_log.csv')
    
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(OUTPUT_VIDEO_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)

    # --- Video I/O ---
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video {VIDEO_PATH}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

    # --- Initialize Tracker and CSV Logger ---
    tracker = PlayerTracker(iou_threshold=0.3, max_misses=15)
    csv_file = open(OUTPUT_CSV_PATH, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['frame_number', 'player_id', 'x1', 'y1', 'x2', 'y2'])

    # --- Main Processing Loop ---
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Detection
        detections = detect_players(frame, MODEL_PATH, conf_threshold=0.4)

        # 2. Tracking
        tracked_objects = tracker.update(detections, frame)

        # 3. Visualization and Logging
        for obj in tracked_objects:
            player_id = obj['id']
            bbox = obj['bbox']
            x1, y1, x2, y2 = map(int, bbox)

            # Get a unique color for the player ID
            color = [int(c) for c in COLORS[player_id % len(COLORS)]]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            # Prepare label text
            label = f"Player {player_id}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            # Draw label background and text
            cv2.rectangle(frame, (x1, y1 - h - 15), (x1 + w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Log data to CSV
            csv_writer.writerow([frame_num, player_id, x1, y1, x2, y2])

        # Write the frame to the output video
        out.write(frame)
        
        # Display the frame (optional)
        cv2.imshow('Player Re-Identification', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        frame_num += 1
        print(f"Processed frame {frame_num}")

    # --- Cleanup ---
    cap.release()
    out.release()
    csv_file.close()
    cv2.destroyAllWindows()
    print(f"Processing complete.")
    print(f"Annotated video saved to: {OUTPUT_VIDEO_PATH}")
    print(f"Tracking log saved to: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()