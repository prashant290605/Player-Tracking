# Player Tracking using YOLOv8 + Deep SORT

Track football players consistently using a 15-second match video. This project combines **YOLOv8** for detection and **Deep SORT** for tracking players across frames, automatically assigning and maintaining player IDs.

---

## Folder Setup

Ensure your project folder has the following structure **before running the code**:

```
your_project/
├── main.py                        # Main tracking script
├── 15sec_input_720p.mp4           # Input video (15 seconds, 720p)
├── requirements.txt               # Python dependencies (used by the bat file)
├── README.md                      # Setup and usage instructions
├── output/                        # Auto-created folder for outputs (video + CSV)
└── (best.pt will be auto-downloaded) 
```


## How to Run

Create an environment(optional but reccomended):
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows


Run this in terminal:

```bash
pip install -r requirements.txt # install dependencies
python main.py
```

This will:

1. Load the input video
2. Detect players using YOLOv8
3. Track them across frames using Deep SORT 
4. Save output video with bounding boxes + tracking IDs in output/

---

## Output Files

All outputs are saved inside the `output/` folder:

### `output_tracking.mp4`

- Input video with bounding boxes + player ID labels  
- Bounding box color:
  - Green = stable high-quality ID
  - Yellow = tentative player (P?)

### `tracking_log.csv` (Optional future version)

Logs each detected/tracked player per frame (if implemented):

| frame_id | player_id | x1  | y1  | x2  | y2  | quality_score |
|----------|-----------|-----|-----|-----|-----|----------------|
| 0        | 1         | ... | ... | ... | ... | 0.84           |
| 1        | 2         | ... | ... | ... | ... | 0.78           |

---

## Requirements

```
torch==2.5.1
torchvision==0.20.1
torchaudio==2.5.1
ultralytics==8.0.197
opencv-python==4.8.1.78
pandas==2.2.2
numpy==1.26.4
deep_sort_realtime
gdown
```

---

5. **Run the Tracker**  
   Executes `main.py`:
   - Loads video
   - Loads YOLOv8 model
   - Runs detection on each frame
   - Uses Deep SORT to assign and maintain IDs
   - Writes the result as a new video with tracking overlays

---
