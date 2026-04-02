import cv2
import mediapipe as mp
import numpy as np
import os
from mediapipe.tasks.python import core, vision
from mediapipe.tasks.python.vision import (
    FaceLandmarker, FaceLandmarkerOptions, RunningMode
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

# 랜드마크 인덱스
LEFT_EYE     = [33, 160, 158, 133, 153, 144]
RIGHT_EYE    = [362, 385, 387, 263, 373, 380]
MOUTH_TOP    = [13, 82, 312]
MOUTH_BOTTOM = [14, 87, 317]
MOUTH_LEFT, MOUTH_RIGHT = 61, 291
FACE_3D = np.array([
    [  0.0,    0.0,   0.0], [  0.0, -330.0, -65.0],
    [-225.0,  170.0,-135.0], [ 225.0,  170.0,-135.0],
    [-150.0, -150.0,-125.0], [ 150.0, -150.0,-125.0],
], dtype=np.float64)
POSE_IDS = [1, 152, 33, 263, 61, 291]

def calc_ear(lm, ids, w, h):
    pts = np.array([[lm[i].x*w, lm[i].y*h] for i in ids])
    A = np.linalg.norm(pts[1]-pts[5])
    B = np.linalg.norm(pts[2]-pts[4])
    C = np.linalg.norm(pts[0]-pts[3])
    return round(float((A+B)/(2.0*C)), 4) if C > 0 else None

def calc_mar(lm, w, h):
    top   = np.array([[lm[i].x*w, lm[i].y*h] for i in MOUTH_TOP])
    bot   = np.array([[lm[i].x*w, lm[i].y*h] for i in MOUTH_BOTTOM])
    left  = np.array([lm[MOUTH_LEFT].x*w,  lm[MOUTH_LEFT].y*h])
    right = np.array([lm[MOUTH_RIGHT].x*w, lm[MOUTH_RIGHT].y*h])
    vert  = sum(np.linalg.norm(top[i]-bot[i]) for i in range(3))
    horiz = np.linalg.norm(left - right)
    return round(float(vert / (3.0 * horiz)), 4) if horiz > 0 else None

def calc_head_pose(lm, w, h):
    pts2d = np.array([[lm[i].x*w, lm[i].y*h] for i in POSE_IDS], dtype=np.float64)
    cam   = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(FACE_3D, pts2d, cam, np.zeros((4, 1)),
                                flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, None, None
    rmat, _ = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    return round(float(angles[0]), 2), round(float(angles[1]), 2), round(float(angles[2]), 2)

# FaceLandmarker 초기화 (한 번만)
opts = FaceLandmarkerOptions(
    base_options=core.base_options.BaseOptions(
        model_asset_path=MODEL_PATH,
        delegate=core.base_options.BaseOptions.Delegate.CPU,
    ),
    running_mode=RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.3,
)
detector = FaceLandmarker.create_from_options(opts)

def extract_features(tile):
    """
    타일 이미지에서 EAR, MAR, Pitch, Yaw, Roll 추출
    반환: {"ear", "mar", "pitch", "yaw", "roll", "face_detected"}
    """
    H, W = tile.shape[:2]
    rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_img)

    if not result.face_landmarks:
        return {
            "ear": None,
            "mar": None,
            "pitch": None,
            "yaw": None,
            "roll": None,
            "face_detected": False
        }

    lm = result.face_landmarks[0]
    le = calc_ear(lm, LEFT_EYE, W, H)
    re = calc_ear(lm, RIGHT_EYE, W, H)
    ear = round((le + re) / 2.0, 4) if (le and re) else None
    mar = calc_mar(lm, W, H)
    pitch, yaw, roll = calc_head_pose(lm, W, H)

    return {
        "ear": ear,
        "mar": mar,
        "pitch": pitch,
        "yaw": yaw,
        "roll": roll,
        "face_detected": True
    }