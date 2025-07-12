# Player Tracking using YOLOv8 + Deep SORT

Track football players consistently using a 15-second match video. This project combines **YOLOv8** for detection and **Deep SORT** for tracking players across frames, automatically assigning and maintaining player IDs.

---

## Folder Setup

Ensure your project folder has the following structure **before running the code**:

```
your_project/
├── main.py                        # Main tracking script
├── setup_and_run.bat              # One-click setup and execution script
├── 15sec_input_720p.mp4           # Input video (15 seconds, 720p)
├── requirements.txt               # Python dependencies (used by the bat file)
├── README.md                      # Setup and usage instructions
├── output/                        # Auto-created folder for outputs (video + CSV)
└── (best.pt will be auto-downloaded) 
```

You do **not** need to install anything manually — just run the command below.

---

## How to Run

Run this in terminal:

```bash
setup_and_run.bat
```

This will **automatically**:

1. Install Python (if needed)
2. Create and activate a virtual environment
3. Install all dependencies (via `requirements.txt`)
4. Download `best.pt` (YOLOv8 model) from Google Drive
5. Run `main.py` to:
   - Detect players frame-by-frame using YOLO
   - Track them with consistent IDs using Deep SORT
   - Save the annotated video with bounding boxes and IDs

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

## Requirements (Handled Automatically)

You do **not** need to install these manually — `setup_and_run.bat` handles them.

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

## Behind the Scenes

When you run `setup_and_run.bat`, here’s what happens:

1. **Python Installation Check**  
   If Python 3.8+ isn't installed, it prompts or installs silently.

2. **Virtual Environment Setup**  
   Creates an isolated `venv/` folder for dependencies.

3. **Dependency Installation**  
   Installs all required packages using `pip` inside `venv`.

4. **Model Download**  
   If `best.pt` is not found locally, it uses `gdown` to download from Google Drive:
   ```
   https://drive.google.com/uc?id=1CYYfDl1yZ7v6UYSligIGfRuxx0KllBW2
   ```

5. **Run the Tracker**  
   Executes `main.py`:
   - Loads video
   - Loads YOLOv8 model
   - Runs detection on each frame
   - Uses Deep SORT to assign and maintain IDs
   - Writes the result as a new video with tracking overlays

---

## Notes

- Works out-of-the-box on Windows (tested on Python 3.12)
- Make sure you are **connected to the internet** the first time (for model download)
- `output/` folder is created automatically if it doesn't exist