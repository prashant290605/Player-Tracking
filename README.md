
# Player Tracking using YOLOv8 and DeepSORT

This project performs player re-identification and tracking in a football video using a trained YOLOv8 model and DeepSORT. The output includes an annotated video with consistent player IDs and a CSV file logging player positions and quality metrics across frames.

---

## Folder Setup

Ensure your project folder has the following structure **before running the code**:

```
your_project/
├── main.py                        # Main tracking script
├── best.pt                        # Your trained YOLOv8 model
├── 15sec_input_720p.mp4           # The input video (15 seconds, 720p)
├── requirements.txt               # Python dependencies to run the project
├── README.md                      # Setup and usage instructions
└── output/                        # Auto-created folder for outputs (video + CSV)


```

You can create a folder, move everything into it, and then proceed with setup.

---

## Setup Instructions

### 1. Install Python (>= 3.8)

Ensure you have Python 3.8 or newer:
```bash
python --version
```

### 2. Create and Activate Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate    # On Windows
# OR
source venv/bin/activate # On Mac/Linux
```

### 3. Install All Dependencies

Install dependencies manually or using the included `requirements.txt` file.

```bash
pip install -r requirements.txt
```


##  How to Run

Once the dependencies are installed:

```bash
python main.py
```

This will:

- Run detection & tracking
- Save the output video with annotations
- Save tracking logs in CSV format

---

## Output Files

All outputs are saved in the `output/` folder:

### 1. `output_tracking.mp4`

- The input video with bounding boxes and consistent player IDs

### 2. `tracking_log.csv`

Detailed per-frame tracking log:

| frame_id | player_id | x1  | y1  | x2  | y2  | quality_score |
|----------|-----------|-----|-----|-----|-----|----------------|
| 0        | 1         | ... | ... | ... | ... | 0.84           |
| 0        | 2         | ... | ... | ... | ... | 0.79           |
| 1        | 1         | ... | ... | ... | ... | 0.85           |

- Only **confirmed players** are logged
- `player_id` is stable across frames
- `quality_score` is a metric indicating tracking confidence and stability

---

## Requirements (Full List)

These are all the dependencies your environment must have:

```
ultralytics==8.0.197
deep_sort_realtime==1.3.1
opencv-python
torch==2.5.1
torchvision
torchaudio
pandas
numpy

```


## Notes

- If DeepSORT or YOLO fails to detect a player momentarily, ID stability is maintained through motion & size heuristics.
- Bounding boxes with green colors indicate **high-quality stable IDs**.
- Yellow boxes with `P?` are tentative, potentially lower confidence.

---
