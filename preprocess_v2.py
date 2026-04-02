"""
전처리 스크립트 v2
JSON + new_annotation_clips.csv → windows_v2.npy / labels_v2.npy / subjects_v2.npy

실행:
    python preprocess_v2.py
출력:
    windows_v2.npy  : (N, 80, 5) float32  [ear, mar, pitch, yaw, roll]
    labels_v2.npy   : (N,)       int64    0=alert, 1=drowsy
    subjects_v2.npy : (N,)       str      subject ID (subject 단위 split용)
"""

import json
import csv
import os
import numpy as np

# ── 경로 설정 ─────────────────────────────────────────────────────────────
CSV_PATH  = './new_annotation_clips.csv'
FEAT_DIR  = './features'
OUT_DIR   = '.'

WINDOW_SIZE = 80  # 8초 × 10fps

# ── JSON 로드 ─────────────────────────────────────────────────────────────
print('JSON 로드 중...')
json_cache = {}
for fname in os.listdir(FEAT_DIR):
    if fname.endswith('.json'):
        with open(os.path.join(FEAT_DIR, fname)) as f:
            d = json.load(f)
        json_cache[d['source']] = d['frames']
print(f'  {len(json_cache)}개 파일 로드 완료')

# ── CSV 로드 ─────────────────────────────────────────────────────────────
print('CSV 로드 중...')
clips = []
with open(CSV_PATH, encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        if row['label'] not in ('alert', 'drowsy'):
            continue
        clips.append({
            'source':  row['source'],
            'subject': row['subject'],
            'start':   float(row['start_sec']),
            'end':     float(row['end_sec']),
            'label':   0 if row['label'] == 'alert' else 1,
        })
print(f'  {len(clips)}개 클립 (alert={sum(c["label"]==0 for c in clips)}, '
      f'drowsy={sum(c["label"]==1 for c in clips)})')

# ── 클립 → (80, 5) 윈도우 추출 ───────────────────────────────────────────
def extract_window(source, start, end):
    frames = json_cache.get(source, [])
    clip   = [fr for fr in frames if start <= fr['t'] <= end and fr.get('face', False)]
    if len(clip) < 20:
        return None

    arr = np.array([[fr['ear'], fr['mar'], fr['pitch'], fr['yaw'], fr['roll']]
                    for fr in clip], dtype=np.float32)

    # WINDOW_SIZE로 맞추기
    T = WINDOW_SIZE
    if len(arr) < T:
        pad = np.tile(arr[-1:], (T - len(arr), 1))
        arr = np.concatenate([arr, pad], axis=0)
    elif len(arr) > T:
        idx = np.linspace(0, len(arr)-1, T).astype(int)
        arr = arr[idx]
    return arr  # (80, 5)

print('윈도우 추출 중...')
windows, labels, subjects = [], [], []
skip = 0
for c in clips:
    w = extract_window(c['source'], c['start'], c['end'])
    if w is None:
        skip += 1
        continue
    windows.append(w)
    labels.append(c['label'])
    subjects.append(c['subject'])

windows  = np.array(windows,  dtype=np.float32)
labels   = np.array(labels,   dtype=np.int64)
subjects = np.array(subjects)

print(f'  추출 완료: {len(windows)}개 (스킵: {skip}개)')
print(f'  alert={( labels==0).sum()}, drowsy={(labels==1).sum()}')
print(f'  고유 subject: {len(set(subjects))}명')

# ── 저장 ─────────────────────────────────────────────────────────────────
np.save(os.path.join(OUT_DIR, 'windows_v2.npy'),  windows)
np.save(os.path.join(OUT_DIR, 'labels_v2.npy'),   labels)
np.save(os.path.join(OUT_DIR, 'subjects_v2.npy'), subjects)

print(f'\n저장 완료:')
print(f'  windows_v2.npy  : {windows.shape}  ({windows.nbytes/1024/1024:.1f} MB)')
print(f'  labels_v2.npy   : {labels.shape}')
print(f'  subjects_v2.npy : {subjects.shape}')
