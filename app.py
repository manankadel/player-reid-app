import os
import cv2
import numpy as np
from flask import Flask, request, render_template, send_from_directory, url_for
from werkzeug.utils import secure_filename
from src.detection import detect_players
from src.tracker import PlayerTracker

# --- App Configuration ---
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static/outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Helper Function ---
def process_video(input_path, output_filename):
    """The core video processing logic from your main.py, adapted for Flask."""
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    
    cap = cv2.VideoCapture(input_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    MODEL_PATH = os.path.join('models', 'best.pt')
    tracker = PlayerTracker(iou_threshold=0.3, max_misses=20, reid_feature_threshold=0.75)
    
    COLORS = np.random.randint(0, 255, size=(100, 3), dtype="uint8")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        detections = detect_players(frame, MODEL_PATH, conf_threshold=0.4)
        tracked_objects = tracker.update(detections, frame)

        for obj in tracked_objects:
            player_id = obj['id']
            bbox = obj['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            color = [int(c) for c in COLORS[player_id % len(COLORS)]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            label = f"Player {player_id}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - h - 15), (x1 + w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)

    cap.release()
    out.release()
    return f"outputs/{output_filename}"

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'video' not in request.files:
            return "No video file part", 400
        file = request.files['video']
        if file.filename == '':
            return "No selected file", 400
        if file:
            filename = secure_filename(file.filename)
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(input_path)
            
            # Process the video and get the output path
            output_file_path = process_video(input_path, f"processed_{filename}")
            
            # Clean up the uploaded file
            os.remove(input_path)
            
            return render_template('result.html', video_file=output_file_path)

    return render_template('index.html')

if __name__ == '__main__':
    # For local testing
    app.run(debug=True)