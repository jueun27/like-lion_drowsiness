import cv2
import numpy as np
import base64
import sys
import os
import time

sys.path.append(os.path.dirname(__file__))
from synthetic_gallery import start_gallery_thread, get_latest_frame
from capture_screen import capture_window
from crop_grid import GridCrop
from extract_features import extract_features

# 갤러리 백그라운드 실행
start_gallery_thread("../data/raw")

USE_ZOOM = False  # True: 실제 Zoom, False: 갤러리 테스트

# 학생별 버퍼 및 상태 저장
feature_buffers = {}
prev_states = {}

# 에피소드 추적용 딕셔너리
drowsy_session_start = {}   # 학생별 현재 졸음 에피소드 시작 시각
drowsy_durations = {}       # 학생별 완료된 에피소드 지속시간 리스트 (초)

BUFFER_SIZE = 80  # 8초 * 10fps
EAR_THR = 0.20

def frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buffer).decode('utf-8')

def capture_and_crop():
    """프레임 가져와서 crop"""
    if USE_ZOOM:
        frame = capture_window("Zoom 회의")
    else:
        frame = get_latest_frame()
    if frame is None:
        return {}
    cropper = GridCrop()
    return cropper.crop(frame)

def update_feature_buffer(student_id, tile):
    """10fps로 feature 추출해서 버퍼에 쌓기"""
    features = extract_features(tile)
    
    if student_id not in feature_buffers:
        feature_buffers[student_id] = []
    
    feature_buffers[student_id].append({
        "ear": features["ear"],
        "mar": features["mar"],
        "pitch": features["pitch"],
        "face_detected": features["face_detected"]
    })
    
    # 슬라이딩 윈도우: 80프레임 유지
    if len(feature_buffers[student_id]) > BUFFER_SIZE:
        feature_buffers[student_id].pop(0)

def rule_based_judgment(buffer):
    """룰베이스 판정 - 80프레임 버퍼 기반"""
    ear_vals = [f["ear"] for f in buffer if f["ear"] is not None]
    if not ear_vals:
        return "uncertain", "uncertain", 0.5
    
    low_ear_ratio = sum(1 for e in ear_vals if e < EAR_THR) / len(ear_vals)
    
    if low_ear_ratio > 0.3:
        return "drowsy_sign", "eyes_closed_long", round(low_ear_ratio, 2)
    else:
        return "normal", "normal", round(low_ear_ratio, 2)

def build_payload(student_id, tile):
    """1fps로 이미지 + 상태 판정해서 payload 생성"""
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
    
    # 최신 feature
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
    
    # 룰베이스 판정
    state, label, score = rule_based_judgment(buffer)
    
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

    print(f"학생 {student_id} | state: {state} | fsm: {fsm}")
    
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