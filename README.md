# like-lion_drowsiness

## 👀 실시간 화상수업 저참여 징후 탐지 시스템

AI 심화 CV 과정 2-1조 (조영준, 권주은, 손우정, 함성민)

비대면 화상수업(Zoom) 환경에서 강사가 놓치기 쉬운 학생의 저참여 징후(졸음, 자리 이탈 등)를 실시간으로 자동 감지하여 대시보드로 제공하는 경량 AI 시스템입니다.

## 🎯 프로젝트 목표
강사의 인지적 부담을 최소화하면서, 영상에서 관측 가능한 행동 기반 이벤트(눈 감음, 얼굴 부재 등)를 자동 감지합니다. 학생의 심리를 추론하는 것이 아닌, 객관적인 데이터로 수업 참여도를 보조적으로 측정하는 시스템입니다.

## 🖥️ 시스템 데모
웹캠 모드데이터 모드실시간 학생 영상 + 졸음/부재 상태 표시EAR 기반 졸음 점수 추이 그래프

실시간 대시보드: 학생별 현재 상태(정상 / 졸음 / 부재) 표시

<img width="400" height="350" alt="image" src="https://github.com/user-attachments/assets/09eaf539-0bdf-42ef-babf-a049580522c7" />

세션 리포트: 수업 종료 후 누적 졸음 횟수, 부재 시간, 요주의 학생 Top 3 제공
CSV 다운로드: 학생별 데이터 엑셀 저장 가능

<img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/60f473d5-df45-4cc5-a927-b8ce851f978c" />


## 🏗️ 시스템 아키텍처
<img width="400" height="555" alt="image" src="https://github.com/user-attachments/assets/0f8281a2-647c-434a-9232-f0844a30282f" />


## 📦 주요 라이브러리
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

## 📋 데이터셋

UTA-RLDD (Real-Life Drowsiness Dataset)

총 58명, 174개 영상 (약 1,740분 분량)
클래스: Alert(정상) / Low Vigilance(저참여) / Drowsy(졸음)
