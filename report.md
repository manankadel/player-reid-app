
# Project Report: Player Re-Identification

- **Author:** Manan
- **Assignment:** AI Intern Task, Liat.ai
- **GitHub Repository:** https://github.com/manankadel/player-reid-app
- **Live Demo:** https://player-reid-app.onrender.com

---

### 1. My Approach and Methodology

My goal for this assignment was to build a modular, robust, and end-to-end computer vision pipeline that is both functional and deployable. I structured the project into discrete components, each with a clear responsibility, simulating a real-world software engineering workflow.

**A. Core Pipeline (`src/`):**

1.  **Detection (`detection.py`):** I utilized the provided YOLOv8 model. A key part of my methodology was to create a robust loading mechanism. The final implementation includes a "LetterBox" pre-processing step to resize frames to the model's required 640x640 input, preventing inference errors.

2.  **Tracking (`tracker.py`):** For frame-to-frame tracking of visible players, I built a custom tracker based on Intersection over Union (IoU) and the Hungarian algorithm for optimal assignment. This provides efficient and smooth tracking when players are in view.

3.  **Re-Identification (`tracker.py` & `utils.py`):** To handle cases where players leave and re-enter the frame, I implemented a re-identification system.
    - **Appearance Features:** I used HSV color histograms as an "appearance fingerprint" for each player, chosen for their resilience to lighting changes.
    - **Memory Bank:** The tracker maintains a memory of "lost" players. When a new player appears, their features are compared against this memory. If a strong match is found (via histogram correlation), the original ID is reassigned, ensuring ID persistence.

**B. Application Layer (`app.py`):**

To make the solution interactive and demonstrate full-stack capability, I wrapped the core pipeline in a **Flask web application**. This provides a user-friendly interface for video uploads and displays the processed results directly in the browser.

---

### 2. Techniques I Tried and Their Outcomes

-   **Model Hosting Strategy:** My initial plan was to host the large (185MB) model file using GitHub Releases.
    - **Outcome:** This approach failed. The download links provided by GitHub for large files were not direct raw links but rather Git-LFS pointer files. This corrupted the downloaded file, making it unreadable by `torch.load()`.
    - **Final Technique:** I pivoted to **Hugging Face Hub**, which is industry-standard for hosting ML models. This provided a reliable, direct download link and resolved the issue completely.

-   **Deployment Platform:** I considered both Streamlit Community Cloud and Render.
    - **Outcome:** I chose **Render** because its support for custom `Procfile` configurations provided the necessary control to solve deployment-specific issues, such as worker timeouts.

---

### 3. Challenges Encountered

This project was an excellent exercise in real-world problem-solving, with challenges extending beyond the core AI logic.

1.  **Environment Instability:** My biggest initial hurdle was environment setup. Using the latest Python version (3.13) led to a cascade of dependency conflicts. I solved this by adopting a professional standard: locking the project to a stable, long-term support version (**Python 3.11**) and using an isolated **virtual environment (`venv`)**.

2.  **Large File and Deployment Logistics:** The 185MB model file was too large for a standard Git workflow. This led me to implement the professional pattern of decoupling the model from the codebase, hosting it externally, and building a dynamic download mechanism into the application logic.

3.  **Production Server Timeouts:** The initial deployment on Render failed due to a `WORKER TIMEOUT` error. The combined time to download and load the large model on a free-tier server exceeded the default 30-second limit. I debugged this by analyzing the server logs and resolved it by specifying a longer timeout (`--timeout 300`) in the `Procfile`, making the application resilient to "cold starts".

---

### 4. Future Improvements

Given more time, I would focus on enhancing its accuracy and performance to make it a production-grade system.

-   **Improve Re-ID Accuracy:** Color histograms are effective but can be confused by players with similar jerseys. The next step would be to replace this with a **deep learning-based Re-ID model** (e.g., a pre-trained Siamese network) to extract more discriminative feature embeddings from player images.

-   **Improve Tracking Robustness:** I would augment the IoU tracker with a **Kalman Filter**. This would add motion prediction to the tracker, allowing it to predict a player's position based on their velocity and better handle brief, full occlusions where IoU would be zero.
```