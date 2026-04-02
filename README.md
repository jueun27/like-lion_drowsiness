# like-lion_drowsiness

## 👀 실시간 화상수업 저참여 징후 탐지 시스템

AI 심화 CV 과정 2-1조 (조영준, 권주은, 손우정, 함성민)

비대면 화상수업(Zoom) 환경에서 강사가 놓치기 쉬운 학생의 저참여 징후(졸음, 자리 이탈 등)를 실시간으로 자동 감지하여 대시보드로 제공하는 경량 AI 시스템입니다.

## 🎯 프로젝트 목표
강사의 인지적 부담을 최소화하면서, 영상에서 관측 가능한 행동 기반 이벤트(눈 감음, 얼굴 부재 등)를 자동 감지합니다. 학생의 심리를 추론하는 것이 아닌, 객관적인 데이터로 수업 참여도를 보조적으로 측정하는 시스템입니다.

## 🖥️ 시스템 데모
웹캠 모드데이터 모드실시간 학생 영상 + 졸음/부재 상태 표시EAR 기반 졸음 점수 추이 그래프

실시간 대시보드: 학생별 현재 상태(정상 / 졸음 / 부재) 표시
세션 리포트: 수업 종료 후 누적 졸음 횟수, 부재 시간, 요주의 학생 Top 3 제공
CSV 다운로드: 학생별 데이터 엑셀 저장 가능


## 🏗️ 시스템 아키텍처


## ⚙️ 핵심 기술
1. 이중 비동기 파이프라인

Feature Loop (10 FPS): 백그라운드에서 0.1초마다 특징점만 가볍게 추출하여 버퍼에 누적
Broadcast Loop (1 FPS): 1초에 한 번 누적된 8초 데이터를 모델에 전달, 이때만 이미지 인코딩 수행
효과: CPU 점유율 안정화, 대시보드 끊김 현상 제거

2. MediaPipe 기반 특징 추출

특징설명EAR (Eye Aspect Ratio)눈 감김 정도 측정MAR (Mouth Aspect Ratio)하품 감지Head Pose (Pitch/Yaw/Roll)고개 방향 측정Face Presence얼굴 부재 감지

3. CNN+LSTM 시계열 모델

입력: 80프레임(8초) × 5개 특징값 [EAR, MAR, Pitch, Yaw, Roll]
구조: 1D-CNN (시계열 패턴 추출) + LSTM (시간 흐름 학습)
무빙 윈도우: 단순 눈 깜빡임과 실제 졸음을 구분

4. Zoom 맞춤형 정밀 Crop

1920×1080 해상도 기준 Zoom UI 픽셀(상단 65px, 하단 70px, 좌우 120px) 제거
갤러리 뷰 Grid를 학생별 타일로 정밀 분할


## 📊 모델 성능
구분AccuracyF1 Score (Macro)F1 Score (Drowsy)Rule-based (EAR 임계치)0.84670.84110.8115CNN+LSTM (시계열 모델)0.92000.91450.8929

CNN+LSTM이 모든 지표에서 Rule-based 대비 우수한 성능을 보임

최종 판정 기준

졸음(drowsy_sign): CNN+LSTM 모델의 drowsy 확률 ≥ 0.62
부재(face_absent): MediaPipe 얼굴 미검출 시 즉시 판정

## 📦 주요 라이브러리
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

## 📋 데이터셋

UTA-RLDD (Real-Life Drowsiness Dataset)

총 58명, 174개 영상 (약 1,740분 분량)
클래스: Alert(정상) / Low Vigilance(저참여) / Drowsy(졸음)
