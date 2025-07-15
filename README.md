# Player Re-Identification in Sports Footage

## Project Overview
This project implements a real-time computer vision pipeline to detect, track, and re-identify players in a sports video. The primary objective is to assign a consistent ID to each player, even if they leave the frame and re-enter later. This solution was developed as part of an AI Intern assignment for Liat.ai.

The pipeline leverages a YOLOv8 model for player detection and a custom tracker that uses a combination of IoU (Intersection over Union) for frame-to-frame tracking and color histogram features for re-identification.

## Folder Structure
project2/
│
├── data/
│ └── videos/
│ └── 15sec_input_720p.mp4
│
├── models/
│ └── best.pt
│
├── outputs/ (Generated automatically)
│ ├── annotated_videos/
│ │ └── annotated_output.mp4
│ └── logs/
│ └── tracking_log.csv
│
├── src/
│ ├── init.py
│ ├── detection.py (Player detection logic)
│ ├── tracker.py (Tracking and Re-ID logic)
│ └── utils.py (Feature extraction utilities)
│
├── final_submission/ (Folder for final deliverables)
│
├── main.py (Main script to run the pipeline)
├── requirements.txt (Project dependencies)
├── README.md (This file)
└── report.md (Detailed project report)


## Setup Instructions

This project was built and tested using **Python 3.11**.

1.  **Clone the repository and navigate to the project directory:**
    ```bash
    cd path/to/your/project2
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Use the py launcher to specify Python 3.11
    py -3.11 -m venv venv

    # Activate the environment
    .\venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

With the virtual environment activated, run the main script from the project root:
```bash
python main.py

The script will process the video located at data/videos/15sec_input_720p.mp4.

Outputs
Annotated Video: A video with bounding boxes and player IDs will be saved to outputs/annotated_videos/annotated_output.mp4.
Tracking Log: A CSV file containing frame-by-frame tracking data will be saved to outputs/logs/tracking_log.csv.
