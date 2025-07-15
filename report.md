# Project Report: Player Re-Identification

- **Author:** Manan
- **Assignment:** AI Intern Task, Liat.ai

---

### 1. Introduction & Objective

The goal of this project was to develop a computer vision pipeline capable of detecting and tracking players in a 15-second sports video clip. The key requirement was to assign a unique and consistent ID to each player. This ID must be maintained even if a player is occluded or leaves the frame and re-enters at a later time (Re-Identification).

### 2. Methodology & Pipeline Design

The project was implemented as a step-by-step pipeline in Python, ensuring modular and clean code. The core logic is as follows:

**I. Frame Processing:**
The input video is read frame-by-frame using OpenCV's `VideoCapture`.

**II. Player Detection:**
For each frame, player detection is performed using the provided pre-trained YOLOv8 model (`best.pt`). Due to environment and library versioning challenges, a low-level detection function was implemented in `src/detection.py`. This involves:
-   Manually loading the `.pt` file using `torch.load` with `weights_only=False`.
-   Applying a "LetterBox" pre-processing step to resize the frame to the model's required input size (640x640) while maintaining the aspect ratio.
-   Running inference directly on the raw PyTorch model.
-   Applying Non-Max Suppression (NMS) to the raw output to get clean, non-overlapping bounding boxes.

**III. Player Tracking (Frame-to-Frame):**
A custom tracker in `src/tracker.py` associates detections across frames.
-   **Cost Matrix:** An Intersection over Union (IoU) cost matrix is calculated between the bounding boxes of active tracks from the previous frame and new detections in the current frame.
-   **Matching:** The Hungarian algorithm (`scipy.linear_sum_assignment`) is used on the cost matrix to find the most optimal matching of tracks to detections with minimal cost.
-   **State Management:** Matched tracks have their bounding boxes updated. Unmatched tracks have a "miss" counter incremented.

**IV. Appearance Feature Extraction:**
To enable re-identification, an appearance feature is extracted for every detected player.
-   A 3D color histogram is calculated from the HSV color space of the player's cropped bounding box. HSV is used for its robustness to lighting changes.
-   This histogram serves as a simple but effective feature vector representing the player's appearance (e.g., jersey color). This logic is in `src/utils.py`.

**V. Player Re-Identification:**
This is the core logic for handling players who re-enter the frame.
-   **Lost Tracks:** When a track's "miss" counter exceeds a threshold (`max_misses`), it is moved from the `active_tracks` list to a `lost_tracks` list, preserving its last known state and appearance feature.
-   **Re-ID Matching:** When a new detection appears that cannot be matched to any *active* track, its color histogram is compared to the histograms of all players in the `lost_tracks` list.
-   **Re-activation:** If the feature similarity (calculated using histogram correlation) is above a threshold (`reid_feature_threshold`), the new detection is matched with the lost track. The lost track is then "reactivated" with its original ID and updated bounding box, and removed from the `lost_tracks` list.
-   **New Player:** If an unmatched detection cannot be re-identified, it is assigned a completely new player ID.

**VI. Output Generation:**
-   The final processed frames, with color-coded bounding boxes and ID labels, are written to a new video file (`annotated_output.mp4`).
-   A `tracking_log.csv` file is generated to store the bounding box coordinates for each player ID in every frame for potential offline analysis.

### 3. Challenges Faced

The project development involved significant environmental and logical challenges:
1.  **Environment & Dependency Conflicts:** The primary hurdle was setting up a stable environment. Initial attempts with the latest Python version (3.13) led to a cascade of errors due to library incompatibilities (NumPy 2.x, PyTorch 2.6+ security changes, incomplete OpenCV builds). This was definitively solved by downgrading to the stable **Python 3.11** and using an isolated **virtual environment (`venv`)**.
2.  **PyTorch Model Loading:** PyTorch 2.6+ introduced a `weights_only=True` security default in `torch.load`, which blocked the YOLOv8 model file. The final solution involved bypassing the high-level `YOLO()` wrapper and manually loading the model with `weights_only=False`.
3.  **Model Input Pre-processing:** When using the raw model, it was discovered that the input frames must be resized/padded (letterboxed) to the model's expected input dimensions (e.g., 640x640) to prevent tensor mismatch errors during inference. This step was added to the detection pipeline.
4.  **Tracking Logic Bug:** An early version of the Re-ID logic contained an `IndexError` caused by modifying the `lost_tracks` list while iterating over it. This was fixed by deferring the removal of re-identified tracks until after the matching loop was complete.

### 4. Results & Visual Outputs

The final pipeline successfully tracks and re-identifies players across the video.

**Frame with Multiple Tracked Players:**
> **ACTION:** Take a screenshot from the output video showing several players with their IDs and paste it here.
>
> `[Insert screenshot of a frame with multiple tracked players]`

**Successful Re-Identification:**
> **ACTION:** Find a moment in the video where a player leaves and re-enters. Take a screenshot showing they have the same ID.
>
> `[Insert screenshot showing a player retaining their ID after re-entering the frame]`

### 5. Conclusion & Future Improvements

**Conclusion:**
The project successfully meets all the requirements of the assignment. A robust pipeline for player detection, tracking, and re-identification was built, demonstrating a modular and well-documented approach. The system correctly assigns consistent IDs to players even after they are occluded or leave the view.

**Future Improvements:**
If more time were available, the following improvements could be made:
-   **More Robust Appearance Features:** While color histograms work well, they can fail if players have similar jersey colors or if lighting changes dramatically. A more advanced method would involve using a pre-trained Re-ID model to extract deep learning-based feature embeddings from each player crop. These are significantly more discriminative.
-   **Motion Prediction with Kalman Filter:** The current tracker relies only on IoU. A Kalman Filter could be added to predict a player's position in the next frame based on their past velocity and direction. This would make the tracker smoother and more capable of handling brief, full occlusions where IoU would be zero.

### 6. Scalability and Performance Considerations

The current application is deployed as a monolithic service where the web server and the model inference engine run on the same machine. While functional, this presents a "cold start" problem: the first request after a period of inactivity will be slow as the server must download and load the 185MB model.

For a production-grade, high-performance application, a decoupled microservices architecture would be superior:

1.  **Frontend Service:** A lightweight Flask/React web app responsible only for the user interface and handling file uploads.
2.  **Backend Model Service:** A dedicated, more powerful worker service (e.g., using FastAPI and running on a GPU-enabled instance) that keeps the model pre-loaded in memory. Its sole purpose is to receive processing requests via an API and return results.
3.  **Task Queue & Caching:** A system like Redis or Celery could manage incoming video processing jobs, preventing the frontend from timing out and allowing for status updates.

This architecture would ensure that the user-facing web app is always fast and responsive, while the heavy computation is handled asynchronously by a specialized backend, eliminating the cold start issue and providing a much better user experience.