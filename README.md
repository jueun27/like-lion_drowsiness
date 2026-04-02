# like-lion_drowsiness

# 백엔드 브랜치

> 담당: 권주은  
> 역할: FastAPI 백엔드 서버 구축 및 WebSocket 실시간 통신 설계

---

## 📁 파일 구조

| 파일 | 설명 |
|------|------|
| `server.py` | FastAPI + WebSocket 서버 (메인 진입점) |
| `pipeline.py` | 캡처 → crop → 특징추출 → 모델추론 파이프라인 |
| `capture_screen.py` | Zoom 창 백그라운드 캡처 |
| `crop_grid.py` | Grid 기반 학생별 타일 분할 |
| `extract_features.py` | MediaPipe 특징 추출 (EAR/MAR/Pose) |
| `synthetic_gallery.py` | 갤러리 합성 (테스트용) |
| `inference.py` | CNN+LSTM 추론 모듈  |
| `best_model_final.pt` | 학습된 모델 가중치 |
| `face_landmarker.task` | MediaPipe 모델 파일 |

---

## ⚙️ 주요 모듈 설명

### `server.py`
FastAPI 기반 서버. 두 개의 비동기 루프로 구성됨.

```python
# Feature Loop: 0.1초(10fps)마다 feature 추출 + 버퍼 쌓기
async def feature_loop():
    await asyncio.sleep(0.1)
    ...

# Broadcast Loop: 1초마다 모델 추론 + 이미지 전송
async def broadcast_loop():
    await asyncio.sleep(1)
    ...
```

| 엔드포인트 | 설명 |
|-----------|------|
| `GET /` | 서버 상태 확인 |
| `WS /ws` | WebSocket 연결 (대시보드) |
| `GET /new_dashboard.html` | 실시간 대시보드 |
| `GET /sample_detail_updated.html` | 학생 상세 페이지 |

---

### `pipeline.py`
캡처부터 판정까지 전체 파이프라인을 관리.

**핵심 설정값**

```python
USE_ZOOM = False   # True: 실제 Zoom, False: 갤러리 테스트
BUFFER_SIZE = 80   # 8초 * 10fps
EAR_THR = 0.20     # 룰베이스 임계값 (주석처리, 참고용)
```

**주요 함수**

| 함수 | 호출 주기 | 설명 |
|------|----------|------|
| `capture_and_crop()` | 매 루프 | 화면 캡처 + 타일 분할 |
| `update_feature_buffer()` | 10fps | MediaPipe 특징 추출 + 버퍼 저장 |
| `build_payload()` | 1fps | 모델 추론 + WebSocket payload 생성 |

**판정 로직 흐름**

```
캡처된 타일 이미지
  ↓
MediaPipe 특징 추출 (EAR, MAR, Pitch, Yaw, Roll)
  ↓
80프레임 슬라이딩 윈도우 버퍼 누적 (8초)
  ↓
CNN+LSTM 모델 추론
  ↓
drowsy 확률 ≥ 0.62  →  drowsy_sign
face_detected = False  →  absent_sign
그 외  →  normal
  ↓
WebSocket으로 대시보드 전송
```

**WebSocket payload 구조**

```json
{
  "student_id": "tile_01",
  "state": "normal | drowsy_sign | absent_sign",
  "score": 0.0,
  "event_label": "normal | eyes_closed_long | face_absent",
  "flags": { "fsm_triggered": false },
  "features": { "ear": 0.3, "pitch": 170.0 },
  "image": "base64...",
  "drowsy_count": 0,
  "avg_drowsy_sec": 0.0
}
```

---

### `capture_screen.py`
Zoom 앱 창을 백그라운드에서 캡처.

```python
ZOOM_KEYWORDS = ["Zoom 회의", "Zoom Meeting"]

# 창 자동 탐지: "Zoom 회의"로 시작하는 가장 큰 창 선택
def find_zoom_window(): ...

# PrintWindow API로 백그라운드 캡처
def capture_zoom(): ...
```

> ⚠️ Chrome 브라우저 Zoom은 보안 정책으로 백그라운드 캡처 불가. **Zoom 앱(설치형)** 필수.

---

### `crop_grid.py`
캡처된 전체 화면을 학생별 타일로 분할.

**Zoom 화면 기준 crop 설정 (1920×1080)**

```python
top_cut    = 65    # 상단 툴바
bottom_cut = 70    # 하단 컨트롤바
left_cut   = 120   # 좌측 여백
right_cut  = 120   # 우측 여백

# 분할 후 1680×945로 리사이즈 → 3×3 타일 (각 560×315)
frame = cv2.resize(frame, (1680, 945))
```

---

### `extract_features.py`
MediaPipe FaceLandmarker로 얼굴 랜드마크 추출.

**추출 특징**

| 특징 | 설명 | 정상 범위 |
|------|------|----------|
| `EAR` | Eye Aspect Ratio (눈 감김) | 0.25 ~ 0.40 |
| `MAR` | Mouth Aspect Ratio (하품) | 0.0 ~ 0.5 |
| `Pitch` | 고개 상하 각도 | 160 ~ 180° |
| `Yaw` | 고개 좌우 각도 | - |
| `Roll` | 고개 기울기 | - |
| `face_detected` | 얼굴 감지 여부 | True / False |

**반환 형식**

```python
{
    "ear": 0.312,
    "mar": 0.045,
    "pitch": 170.5,
    "yaw": -5.2,
    "roll": 2.1,
    "face_detected": True
}
```

---

### `synthetic_gallery.py`
`USE_ZOOM = False`일 때 사용하는 테스트용 갤러리 합성 모듈.

```
data/raw/ 폴더의 영상 파일들
  ↓
4×4 grid로 합성 (480×270 타일)
  ↓
1920×1080 가상 Zoom 갤러리 화면 생성
  ↓
백그라운드 스레드로 실시간 제공
```

---

## 🚀 실행 방법

### 1. 가상환경 활성화
```bash
drowsiness\Scripts\activate
```

### 2. 패키지 설치
```bash
pip install -r requirements.txt
```

### 3. 서버 실행
```bash
cd backend
python server.py
```

### 4. 대시보드 접속
```
http://127.0.0.1:8000/new_dashboard.html
```

---

## 🔀 모드 전환

`pipeline.py`에서 `USE_ZOOM` 값으로 전환:

```python
USE_ZOOM = False  # 갤러리 합성 영상으로 테스트
USE_ZOOM = True   # 실제 Zoom 앱 화면 캡처
```

---

## 📦 주요 의존성

```
fastapi
uvicorn
mediapipe
opencv-python
torch
scikit-learn
numpy
mss
pywin32
websockets
```
