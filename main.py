import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
from ultralytics.nn.tasks import DetectionModel
import os
from collections import defaultdict
import math

torch.serialization.add_safe_globals({DetectionModel})
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "best.pt")
model = YOLO(model_path)

print("Model classes:", model.names)

video_path = os.path.join(base_dir, "15sec_input_720p.mp4")
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print(f"FRAME SIZE: {frame_width}x{frame_height}, FPS: {fps}")

if not cap.isOpened() or fps == 0:
    raise IOError("Cannot open video file or FPS is 0")

os.makedirs("output", exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter("output/output_tracking.mp4", fourcc, fps, (frame_width, frame_height))

tracker = DeepSort(
    max_age=70,
    n_init=3,
    nms_max_overlap=0.6,
    max_cosine_distance=0.3,
    nn_budget=150,
    override_track_class=None,
    embedder="mobilenet",
    half=True,
    bgr=True,
    embedder_gpu=True,
    embedder_model_name=None,
    embedder_wts=None,
    polygon=False,
    today=None
)

tracking_data = []
frame_count = 0

MIN_CONFIDENCE = 0.35
MIN_AREA = 600
MAX_AREA = 45000
MIN_ASPECT_RATIO = 0.6
MAX_ASPECT_RATIO = 3.5

PLAYER_CLASS_IDS = [2, 3]

track_history = defaultdict(list)
track_confidence_history = defaultdict(list)
track_quality_scores = defaultdict(float)
track_motion_history = defaultdict(list)
id_mapping = {}
next_clean_id = 1

MAX_PLAYER_SPEED = 150
MAX_ACCELERATION = 50
MIN_MOTION_CONSISTENCY = 0.3

TRACK_STATES = {
    'TENTATIVE': 0,
    'CONFIRMED': 1,
    'STABLE': 2
}

track_states = defaultdict(lambda: TRACK_STATES['TENTATIVE'])

def detect_track_collision(track1_bbox, track2_bbox, threshold=50):
    x1_center = track1_bbox[0] + track1_bbox[2] / 2
    y1_center = track1_bbox[1] + track1_bbox[3] / 2
    x2_center = track2_bbox[0] + track2_bbox[2] / 2
    y2_center = track2_bbox[1] + track2_bbox[3] / 2
    distance = math.sqrt((x1_center - x2_center)**2 + (y1_center - y2_center)**2)
    return distance < threshold

def validate_motion(track_id, current_bbox, current_frame):
    if track_id not in track_history or len(track_history[track_id]) < 2:
        return True
    current_center = [current_bbox[0] + current_bbox[2]/2, current_bbox[1] + current_bbox[3]/2]
    last_bbox = track_history[track_id][-1]
    last_center = [last_bbox[0] + last_bbox[2]/2, last_bbox[1] + last_bbox[3]/2]
    speed = math.sqrt((current_center[0] - last_center[0])**2 + (current_center[1] - last_center[1])**2)
    if speed > MAX_PLAYER_SPEED:
        return False
    if len(track_history[track_id]) >= 3:
        prev_bbox = track_history[track_id][-2]
        prev_center = [prev_bbox[0] + prev_bbox[2]/2, prev_bbox[1] + prev_bbox[3]/2]
        prev_speed = math.sqrt((last_center[0] - prev_center[0])**2 + (last_center[1] - prev_center[1])**2)
        acceleration = abs(speed - prev_speed)
        if acceleration > MAX_ACCELERATION:
            return False
    return True

def calculate_motion_consistency(track_id):
    if track_id not in track_history or len(track_history[track_id]) < 4:
        return 0.5
    recent_positions = track_history[track_id][-6:]
    if len(recent_positions) < 3:
        return 0.5
    motion_vectors = []
    for i in range(1, len(recent_positions)):
        prev_pos = recent_positions[i-1]
        curr_pos = recent_positions[i]
        prev_center = [prev_pos[0] + prev_pos[2]/2, prev_pos[1] + prev_pos[3]/2]
        curr_center = [curr_pos[0] + curr_pos[2]/2, curr_pos[1] + curr_pos[3]/2]
        motion_vector = [curr_center[0] - prev_center[0], curr_center[1] - prev_center[1]]
        motion_vectors.append(motion_vector)
    if len(motion_vectors) < 2:
        return 0.5
    consistency_scores = []
    for i in range(1, len(motion_vectors)):
        v1 = motion_vectors[i-1]
        v2 = motion_vectors[i]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        if mag1 == 0 or mag2 == 0:
            continue
        dot_product = v1[0]*v2[0] + v1[1]*v2[1]
        cos_angle = dot_product / (mag1 * mag2)
        cos_angle = max(-1, min(1, cos_angle))
        consistency = (cos_angle + 1) / 2
        consistency_scores.append(consistency)
    return np.mean(consistency_scores) if consistency_scores else 0.5

def calculate_track_quality(track_id, current_bbox, confidence):
    if track_id not in track_history:
        track_history[track_id] = []
        track_confidence_history[track_id] = []
    track_history[track_id].append(current_bbox)
    track_confidence_history[track_id].append(confidence)
    history_length = 15
    if len(track_history[track_id]) > history_length:
        track_history[track_id] = track_history[track_id][-history_length:]
        track_confidence_history[track_id] = track_confidence_history[track_id][-history_length:]
    if len(track_history[track_id]) < 2:
        return 0.3
    avg_confidence = np.mean(track_confidence_history[track_id])
    confidence_stability = 1.0 - np.std(track_confidence_history[track_id])
    confidence_score = (avg_confidence + max(0, confidence_stability)) / 2
    positions = np.array([[bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2] for bbox in track_history[track_id]])
    if len(positions) > 2:
        movement_distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        avg_movement = np.mean(movement_distances)
        stability_score = max(0, 1.0 - avg_movement / 100.0)
    else:
        stability_score = 0.5
    sizes = np.array([[bbox[2], bbox[3]] for bbox in track_history[track_id]])
    if len(sizes) > 2:
        size_consistency = 1.0 - (np.std(sizes) / np.mean(sizes))
        size_score = max(0, min(1, size_consistency))
    else:
        size_score = 0.5
    duration_score = min(1.0, len(track_history[track_id]) / 20.0)
    motion_consistency = calculate_motion_consistency(track_id)
    quality_score = (confidence_score * 0.3 + stability_score * 0.25 + size_score * 0.15 + duration_score * 0.1 + motion_consistency * 0.2)
    track_quality_scores[track_id] = quality_score
    return quality_score

def update_track_state(track_id, quality_score):
    current_state = track_states[track_id]
    if quality_score > 0.75:
        track_states[track_id] = TRACK_STATES['STABLE']
    elif quality_score > 0.55:
        track_states[track_id] = TRACK_STATES['CONFIRMED']
    else:
        if current_state == TRACK_STATES['TENTATIVE']:
            track_states[track_id] = TRACK_STATES['TENTATIVE']

def get_display_id(track_id):
    global next_clean_id
    track_state = track_states[track_id]
    if track_state >= TRACK_STATES['CONFIRMED']:
        if track_id not in id_mapping:
            id_mapping[track_id] = next_clean_id
            next_clean_id += 1
        return id_mapping[track_id], True
    return track_id, False

def filter_detections(boxes, min_conf=MIN_CONFIDENCE, min_area=MIN_AREA, max_area=MAX_AREA):
    filtered_detections = []
    
    if boxes is None or len(boxes) == 0:
        return filtered_detections
    
    for box in boxes:
        class_id = int(box.cls[0])
        conf = float(box.conf[0])
        
        x1, y1, x2, y2 = box.xyxy[0]
        w = x2 - x1
        h = y2 - y1
        area = w * h
        aspect_ratio = h / w if w > 0 else 0
        
        if class_id in PLAYER_CLASS_IDS:
            if conf >= min_conf * 0.9:
                if min_area * 0.8 <= area <= max_area * 1.1:
                    if MIN_ASPECT_RATIO * 0.9 <= aspect_ratio <= MAX_ASPECT_RATIO * 1.1:
                        if w > 20 and h > 40:
                            filtered_detections.append(([x1, y1, w, h], conf, 'player'))
                            
                            if frame_count < 5:
                                print(f"    ✓ Detection: conf={conf:.3f}, area={area:.0f}, AR={aspect_ratio:.2f}")
    
    return filtered_detections


def is_likely_false_positive(track_id, bbox, quality_score, track_info_list):
    if quality_score > 0.55:
        return False

    for other_track, other_quality, _ in track_info_list:
        other_id = other_track.track_id
        if other_id == track_id:
            continue
        if other_quality <= quality_score:
            continue

        l1, t1, r1, b1 = bbox
        l2, t2, r2, b2 = other_track.to_ltrb()

        xA = max(l1, l2)
        yA = max(t1, t2)
        xB = min(r1, r2)
        yB = min(b1, b2)
        inter_area = max(0, xB - xA) * max(0, yB - yA)
        box1_area = (r1 - l1) * (b1 - t1)
        box2_area = (r2 - l2) * (b2 - t2)
        iou = inter_area / float(box1_area + box2_area - inter_area + 1e-5)

        if iou > 0.3:
            return True

    return False

while True:
    
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, verbose=False, conf=0.15)[0]
    
    if frame_count < 5:
        print(f"\nFrame {frame_count} - Raw detections: {len(results.boxes) if results.boxes is not None else 0}")
    
    detections = filter_detections(results.boxes)
    
    if frame_count < 5:
        print(f"  After filtering: {len(detections)} detections")
    
    tracks = tracker.update_tracks(detections, frame=frame)
    
    tracks_to_display = []
    
    for track in tracks:
        if not track.is_confirmed():
            continue
            
        track_id = track.track_id
        l, t, r, b = track.to_ltrb()
        bbox = [l, t, r-l, b-t]
        
        detection_conf = 0.5
        for detection in detections:
            det_bbox, conf, _ = detection
            if abs(det_bbox[0] - l) < 20 and abs(det_bbox[1] - t) < 20:
                detection_conf = conf
                break
        
        quality_score = calculate_track_quality(track_id, bbox, detection_conf)
        update_track_state(track_id, quality_score)
        
        tracks_to_display.append((track, quality_score, detection_conf))

    final_tracks = []
    track_info_list = [(track, quality_score, detection_conf) for track, quality_score, detection_conf in tracks_to_display]
    
    for track, quality_score, detection_conf in track_info_list:
        track_id = track.track_id
        l, t, r, b = track.to_ltrb()
        bbox = [l, t, r-l, b-t]
        
        if is_likely_false_positive(track_id, bbox, quality_score, track_info_list):
            continue
        
        display_id, is_clean_id = get_display_id(track_id)
        
        final_tracks.append((track, display_id, is_clean_id, quality_score))
    
    for track, display_id, is_clean_id, quality_score in final_tracks:
        l, t, r, b = track.to_ltrb()
        l, t, r, b = int(l), int(t), int(r), int(b)
        
        if is_clean_id:
            color = (0, 255, 0)
            thickness = 3
            prefix = "Player"
        else:
            color = (0, 255, 255)
            thickness = 2
            prefix = "P?"
        
        cv2.rectangle(frame, (l, t), (r, b), color, thickness)
        
        text = f"{prefix} {display_id}"
        if not is_clean_id:
            text += f" ({quality_score:.1f})"
        
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = l + (r - l - text_size[0]) // 2
        text_y = max(t - 10, text_size[1] + 5)
        
        cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), 
                     (text_x + text_size[0] + 5, text_y + 5), color, -1)
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        if is_clean_id:
            tracking_data.append([frame_count, display_id, l, t, r, b, quality_score])
    
    output_video.write(frame)
    
    tentative_count = sum(1 for _, _, is_clean, _ in final_tracks if not is_clean)
    confirmed_count = sum(1 for _, _, is_clean, _ in final_tracks if is_clean)
    
    print(f"Frame {frame_count}: Detections: {len(detections)}, Confirmed: {confirmed_count}, Tentative: {tentative_count}")
    
    frame_count += 1

cap.release()
output_video.release()
cv2.destroyAllWindows()

df = pd.DataFrame(tracking_data, columns=["frame_id", "player_id", "x1", "y1", "x2", "y2", "quality_score"])
df.to_csv("output/tracking_log.csv", index=False)

print("Done. Output saved in 'output/' folder.")
print(f"Total frames processed: {frame_count}")
print(f"Total tracking records: {len(tracking_data)}")

if len(tracking_data) > 0:
    unique_players = df['player_id'].nunique()
    print(f"Unique confirmed players: {unique_players}")
    print(f"Player IDs: {sorted(df['player_id'].unique())}")
    
    avg_quality = df['quality_score'].mean()
    print(f"Average track quality: {avg_quality:.2f}")
    
    track_durations = df.groupby('player_id').size()
    print(f"Average track duration: {track_durations.mean():.1f} frames")
    print(f"Tracks lasting > 1 second: {sum(track_durations >= fps)}")
    
    high_quality_tracks = df[df['quality_score'] > 0.7]
    print(f"High quality tracking records: {len(high_quality_tracks)} ({len(high_quality_tracks)/len(df)*100:.1f}%)")
    
else:
    print("No confirmed tracking data recorded.")

print(f"Total tentative tracks created: {len(track_states)}")
print(f"Total clean IDs assigned: {next_clean_id - 1}")

total_motion_validations = sum(len(history) for history in track_history.values())
print(f"Total motion validations performed: {total_motion_validations}")

track_history.clear()
track_confidence_history.clear()
track_quality_scores.clear()
track_motion_history.clear()