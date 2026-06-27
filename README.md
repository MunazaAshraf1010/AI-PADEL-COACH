# AI padel coach

This project analyzes padel game videos to extract player and ball tracking data. It processes an input video, applies various computer vision models to track players, the ball, and court keypoints, and then outputs the collected analytics into a CSV file.

## Core Functionality

-   **Player Tracking**: Identifies and tracks players on the court.
-   **Ball Tracking**: Detects and follows the ball's movement.
-   **Court Keypoint Detection**: Uses pre-defined court keypoints for spatial reference.
-   **Player Keypoints Tracking**: Estimates player poses.
-   **Data Export**: Saves the analyzed tracking data (positions, velocities, etc.) into a CSV file for further analysis.

## Setup

1.  **Clone Repository**:

    ```bash
    git install lfs
    git clone https://huggingface.co/OmarhAhmed/ai-padel-coach
    cd ai-padel-coach
    ```

2.  **Create Virtual Environment** (Recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare Court Keypoints**:
    -   This project requires a JSON file containing exactly 12 pre-defined court keypoints. These keypoints are crucial for establishing a reference for the court's geometry.
    -   Create a JSON file (e.g., `court_keypoints.json`) with the coordinates of these 12 points. Example format:
        ```json
        [
          [x1, y1], [x2, y2], [x3, y3], [x4, y4],
          [x5, y5], [x6, y6], [x7, y7], [x8, y8],
          [x9, y9], [x10, y10], [x11, y11], [x12, y12]
        ]
        ```
    -   Update the `FIXED_COURT_KEYPOINTS_LOAD_PATH` in `config.py` to point to this JSON file.

## Configuration

Before running the project, review and update the `config.py` file. Key settings include:

-   `input_video`: Path to the padel game video you want to analyze.
-   `output_video`: Path where the output CSV data file will be saved.
-   `keypoint_tracking_model`: Path to your JSON file containing the 12 court keypoints.
-   `maximum_frame_count`: Set to `None` to process the entire video, or an integer to limit the number of frames.
-   `save_data`: Set to `True` to enable saving the output CSV file.

## Running the Analysis

Execute the main script from the project's root directory:

```bash
python main.py
```
