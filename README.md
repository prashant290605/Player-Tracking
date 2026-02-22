# Player Tracking using YOLOv8 + Deep SORT

Consistent multi-object tracking of football players across video frames, combining **YOLOv8** for detection and **Deep SORT** for temporal identity association.

---

## Motivation

Multi-object tracking is a fundamental problem in computer vision, particularly in sports analytics where consistent player identity across frames is critical for downstream tasks such as movement analysis, tactical reconstruction, and event detection.

This project implements a lightweight tracking pipeline that:

- Uses **YOLOv8** for robust, frame-level player detection
- Uses **Deep SORT** for temporally consistent ID assignment via Kalman filtering and appearance-based re-identification
- Maintains stable player IDs across frames, even under partial occlusion and motion blur

---

## Pipeline Overview

```
Input Video
    │
    ▼
Frame Extraction
    │
    ▼
Player Detection (YOLOv8)
    │
    ▼
Feature Embedding Extraction (Deep SORT appearance model)
    │
    ▼
Kalman Filter — State Prediction & Update
    │
    ▼
Hungarian Algorithm — Detection-to-Track Assignment
    │
    ▼
Annotated Video Rendering
```

Each detection is associated with an existing track (or initialized as a new one) using the Hungarian algorithm on a combined motion + appearance cost matrix. Tracks below a confidence threshold are marked tentative until confirmed over successive frames.

---

## Project Structure

```
your_project/
├── main.py                   # Main tracking script
├── 15sec_input_720p.mp4      # Input video (15 seconds, 720p)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── output/                   # Auto-created; stores output video and logs
    ├── output_tracking.mp4
    └── tracking_log.csv      # (optional, see below)
```

> `best.pt` (YOLOv8 weights) is downloaded automatically on first run via `gdown`.

---

## Setup and Usage

**1. Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the tracker**
```bash
python main.py
```

This will load the input video, run per-frame detection and tracking, and write the annotated output to `output/`.

---

## Output Files

### `output/output_tracking.mp4`

Annotated video with bounding boxes and player ID labels overlaid on each frame.

Bounding box colors indicate track confidence:
- **Green** — confirmed track (stable, high-quality ID)
- **Yellow** — tentative track (recently initialized, awaiting confirmation)

### `output/tracking_log.csv` *(optional)*

Per-frame log of all tracked detections:

| frame_id | player_id | x1 | y1 | x2 | y2 | quality_score |
|----------|-----------|----|----|----|----|---------------|
| 0 | 1 | ... | ... | ... | ... | 0.84 |
| 1 | 2 | ... | ... | ... | ... | 0.78 |

---

## Requirements

```
torch
torchvision
torchaudio
ultralytics==8.0.197
opencv-python==4.8.1.78
pandas==2.2.2
numpy==1.26.4
deep_sort_realtime
gdown
```

> PyTorch is left unpinned to avoid environment conflicts across CUDA versions. Install a CUDA-specific build manually if needed: see [pytorch.org/get-started](https://pytorch.org/get-started/locally/).

---

## Limitations

- No team classification or jersey color segmentation
- No jersey number recognition
- No cross-camera re-identification
- Evaluated on a single 15-second clip; performance on longer sequences is untested
- Deep SORT ID switches may increase under severe occlusion or fast camera motion

---

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Deep SORT Realtime](https://github.com/levan92/deep_sort_realtime)
