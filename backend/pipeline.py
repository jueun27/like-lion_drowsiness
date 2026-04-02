import cv2
import numpy as np
import base64
import sys
import os
import time

sys.path.append(os.path.dirname(__file__))
from synthetic_gallery import start_gallery_thread, get_latest_frame
from capture_screen import capture_zoom, capture_window
from crop_grid import GridCrop
from extract_features import extract_features
from inference import DrowsinessPredictor

# 갤러리 백그라운드 실행
start_gallery_thread("../data/raw")

USE_ZOOM = False  # True: 실제 Zoom, False: 갤러리 테스트

# 모델 로드
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model_v2.pt")
predictor = DrowsinessPredictor(MODEL_PATH)

# 학생별 버퍼 및 상태 저장
feature_buffers = {}
prev_states = {}

# 에피소드 추적용 딕셔너리
drowsy_session_start = {}
drowsy_durations = {}

BUFFER_SIZE = 80
EAR_THR = 0.20

def frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buffer).decode('utf-8')

def capture_and_crop():
    if USE_ZOOM:
        frame = capture_zoom()
        if frame is None:
            return {}
        
        # Zoom 전용 crop
        fh, fw = frame.shape[:2]
        top_cut    = 65
        bottom_cut = 70
        left_cut   = 120
        right_cut  = 120
        frame = frame[top_cut:fh-bottom_cut, left_cut:fw-right_cut]
        frame = cv2.resize(frame, (1680, 945))
        
        # 3x3 crop
        fh, fw = frame.shape[:2]
        tile_h = fh // 3
        tile_w = fw // 3
        tiles = {}
        student_id = 1
        for row in range(3):
            for col in range(3):
                x1 = col * tile_w
                y1 = row * tile_h
                x2 = x1 + tile_w
                y2 = y1 + tile_h
                tile = frame[y1:y2, x1:x2]
                tile = cv2.resize(tile, (320, 180))
                tiles[student_id] = tile
                student_id += 1
        return tiles

    else:
        # 갤러리 테스트 (기존 방식)
        frame = get_latest_frame()
        if frame is None:
            return {}
        cropper = GridCrop()
        return cropper.crop(frame)

def update_feature_buffer(student_id, tile):
    features = extract_features(tile)
    if student_id not in feature_buffers:
        feature_buffers[student_id] = []
    feature_buffers[student_id].append({
        "ear":           features["ear"],
        "mar":           features["mar"],
        "pitch":         features["pitch"],
        "yaw":           features["yaw"],
        "roll":          features["roll"],
        "face_detected": features["face_detected"]
    })
    if len(feature_buffers[student_id]) > BUFFER_SIZE:
        feature_buffers[student_id].pop(0)

def buffer_to_model_input(buffer):
    """버퍼 → (80, 5) numpy array [ear, mar, pitch, yaw, roll]"""
    arr = []
    for f in buffer:
        arr.append([
            f["ear"]   if f["ear"]   is not None else 0.0,
            f["mar"]   if f["mar"]   is not None else 0.0,
            f["pitch"] if f["pitch"] is not None else 0.0,
            f["yaw"]   if f["yaw"]   is not None else 0.0,
            f["roll"]  if f["roll"]  is not None else 0.0,
        ])
    return np.array(arr, dtype=np.float32)

# def rule_based_judgment(buffer):
#     ear_vals = [f["ear"] for f in buffer if f["ear"] is not None]
#     if not ear_vals:
#         return "uncertain", "uncertain", 0.5
#     low_ear_ratio = sum(1 for e in ear_vals if e < EAR_THR) / len(ear_vals)
#     if low_ear_ratio > 0.3:
#         return "drowsy_sign", "eyes_closed_long", round(low_ear_ratio, 2)
#     else:
#         return "normal", "normal", round(low_ear_ratio, 2)

def build_payload(student_id, tile):
    img_base64 = frame_to_base64(tile)
    buffer = feature_buffers.get(student_id, [])

    # 버퍼 미달 (8초 미만)
    if len(buffer) < BUFFER_SIZE:
        remaining = BUFFER_SIZE - len(buffer)
        print(f"학생 {student_id} | 버퍼 쌓는 중... ({remaining}프레임 남음)")
        return {
            "student_id": f"tile_{str(student_id).zfill(2)}",
            "state": "normal",
            "score": 0.5,
            "event_label": "normal",
            "flags": {"fsm_triggered": False},
            "features": {"ear": None, "pitch": None},
            "image": img_base64,
            "drowsy_count": 0,
            "avg_drowsy_sec": 0.0
        }

    latest = buffer[-1]

    # face_absent 처리
    if not latest["face_detected"]:
        prev = prev_states.get(student_id, "normal")
        fsm = (prev == "normal")
        prev_states[student_id] = "absent_sign"
        return {
            "student_id": f"tile_{str(student_id).zfill(2)}",
            "state": "absent_sign",
            "score": 0.92,
            "event_label": "face_absent",
            "flags": {"fsm_triggered": fsm},
            "features": {"ear": None, "pitch": None},
            "image": img_base64,
            "drowsy_count": len(drowsy_durations.get(student_id, [])),
            "avg_drowsy_sec": 0.0
        }

    # 모델 추론
    model_input = buffer_to_model_input(buffer)
    model_result = predictor.predict(model_input)
    model_drowsy = model_result["class"] == "drowsy"
    model_score = model_result["probabilities"]["drowsy"]

    # # 룰베이스 판정
    # rule_state, rule_label, rule_score = rule_based_judgment(buffer)

    # 모델만으로 최종 판정
    if model_drowsy:
        state, label, score = "drowsy_sign", "eyes_closed_long", round(model_score, 2)
    else:
        state, label, score = "normal", "normal", round(1 - model_score, 2)

    prev = prev_states.get(student_id, "normal")
    now = time.time()

    # 졸음 에피소드 시작 (normal → drowsy)
    if state == "drowsy_sign" and prev == "normal":
        drowsy_session_start[student_id] = now
        fsm = True
    # 졸음 에피소드 종료 (drowsy → normal)
    elif state == "normal" and prev == "drowsy_sign":
        start = drowsy_session_start.pop(student_id, None)
        if start is not None:
            duration = round(now - start, 1)
            drowsy_durations.setdefault(student_id, []).append(duration)
        fsm = False
    else:
        fsm = False

    prev_states[student_id] = state

    # 집계 계산
    durations = drowsy_durations.get(student_id, [])
    drowsy_count = len(durations)
    avg_duration = round(sum(durations) / drowsy_count, 1) if drowsy_count > 0 else 0.0

    # 모델 로그
    # print(f"학생 {student_id} | rule: {rule_state} ({rule_score:.2f}) | model: {model_result['class']} ({model_score:.2f}) | 최종: {state} | fsm: {fsm}")
    print(f"학생 {student_id} | model: {model_result['class']} ({model_score:.2f}) | 최종: {state} | fsm: {fsm}")

    return {
        "student_id": f"tile_{str(student_id).zfill(2)}",
        "state": state,
        "score": score,
        "event_label": label,
        "flags": {"fsm_triggered": fsm},
        "features": {
            "ear": latest["ear"],
            "pitch": latest["pitch"]
        },
        "image": img_base64,
        "drowsy_count": drowsy_count,
        "avg_drowsy_sec": avg_duration
    }