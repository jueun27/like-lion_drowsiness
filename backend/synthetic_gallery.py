import cv2
import numpy as np
import os
import threading
import time

# 공유 프레임 버퍼
latest_frame = None
frame_lock = threading.Lock()

def add_zoom_ui(tile, name, tile_w=480, tile_h=270):
    result = tile.copy()
    label = f"  {name}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
    x1, y1 = 4, tile_h - text_h - 12
    x2, y2 = x1 + text_w + 8, tile_h - 4
    overlay = result.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, result, 0.4, 0, result)
    cv2.putText(result, label, (x1 + 4, y2 - 4), font, font_scale, (255, 255, 255), thickness)
    return result

def run_gallery(video_dir, rows=4, cols=4, tile_w=480, tile_h=270, fps=10):
    global latest_frame
    
    video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")]
    print(f"영상 개수: {len(video_files)}")
    while len(video_files) < rows * cols:
        video_files = video_files * 2
    video_files = video_files[:rows * cols]

    names = [f"student {i+1}" for i in range(len(video_files))]
    caps = [cv2.VideoCapture(f) for f in video_files]
    interval = 1.0 / fps

    while True:
        gallery = np.zeros((tile_h * rows, tile_w * cols, 3), dtype=np.uint8)

        for idx, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            if not ret:
                continue

            resized = cv2.resize(frame, (tile_w, tile_h))
            resized = add_zoom_ui(resized, names[idx], tile_w, tile_h)

            row = idx // cols
            col = idx % cols
            y1, y2 = row * tile_h, (row + 1) * tile_h
            x1, x2 = col * tile_w, (col + 1) * tile_w
            gallery[y1:y2, x1:x2] = resized

        with frame_lock:
            latest_frame = gallery.copy()

        time.sleep(interval)

    for cap in caps:
        cap.release()

def get_latest_frame():
    with frame_lock:
        return latest_frame.copy() if latest_frame is not None else None

def start_gallery_thread(video_dir):
    t = threading.Thread(target=run_gallery, args=(video_dir,), daemon=True)
    t.start()
    print("갤러리 백그라운드 실행 시작!")

if __name__ == "__main__":
    cv2.namedWindow("gallery", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("gallery", 1920, 1080)
    run_gallery("../data/raw")