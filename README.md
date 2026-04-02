# Model / Deep Learning — CNN+LSTM 졸음 판정 모듈

담당: 조영준

---

## 파일 구성

| 파일 | 설명 |
|------|------|
| `drowsiness_v2.ipynb` | CNN+LSTM 학습 노트북 (Google Colab T4 기준) |
| `preprocess_v2.py` | JSON + CSV → windows_v2.npy 전처리 스크립트 |
| `inference.py` | 백엔드 연동용 추론 인터페이스 |

---

## 모델 구조

```
입력: (Batch, 8, 80)  ← 8초 × 10fps, circular 변환 후 8 feature
  ↓
Conv1d(8→32) + BN + ReLU + MaxPool
Conv1d(32→64) + BN + ReLU + MaxPool
  ↓
LSTM(hidden=64, 단방향)
Dropout(0.5)
FC(64→2)
  ↓
출력: alert / drowsy
```

## 성능

| 모델 | Accuracy | F1 (Macro) | F1 (Drowsy) |
|------|----------|------------|-------------|
| Rule-based (PERCLOS) | 0.8467 | 0.8411 | 0.8115 |
| **CNN+LSTM (최종)** | **0.9200** | **0.9145** | **0.8929** |

---

## 추론 사용법

```python
from inference import DrowsinessPredictor

predictor = DrowsinessPredictor('best_model_final.pt')
result = predictor.predict(features)  # 입력: (80, 5) numpy array
# 출력: {'class': 'drowsy', 'confidence': 0.91, 'probabilities': {'alert': 0.09, 'drowsy': 0.91}}
```

### 입력 규격

- shape: `(80, 5)` — 8초 × 10fps
- feature 순서: `[EAR, MAR, pitch, yaw, roll]`
- 내부에서 자동으로 circular 변환 + 정규화 처리

---

## 학습 재현

1. Google Colab에서 `drowsiness_v2.ipynb` 열기
2. Google Drive에 데이터 업로드 (`windows_v2.npy`, `labels_v2.npy`, `subjects_v2.npy`)
3. `CFG` 클래스의 경로 설정 후 전체 실행
4. 5번 랜덤 seed 반복 학습 → best test F1 모델 자동 저장 (`best_model_final.pt`)
