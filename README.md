# Model / Rule-based — 개인화 PERCLOS 졸음 판정 모듈

담당: 손우정

---

## 개요

subject별 alert 구간의 평균 EAR을 개인 기준값으로 설정하고,
8초 윈도우 내 EAR 저하 비율(PERCLOS)로 졸음 여부를 판정합니다.

눈 크기의 개인차를 보정하기 위해 고정 임계값 대신 **개인화 기준값**을 사용합니다.

---

## 파일 구성

| 파일 | 설명 |
|------|------|
| `rule_based.py` | 개인화 PERCLOS 판정 클래스 |

---

## 판정 로직

```
1. subject의 alert 클립 평균 EAR → 개인 기준값 (ear_ref)
2. ear_norm = EAR / ear_ref  (개인 기준 대비 비율)
3. PERCLOS = 80프레임 중 ear_norm < 0.8 인 프레임 비율
4. PERCLOS ≥ 0.3  →  drowsy
   PERCLOS < 0.3  →  alert
```

---

## 사용법

```python
from rule_based import RuleBasedPredictor

# subject별 alert 구간 평균 EAR (사전에 계산)
subject_alert_ear = {
    'subj_01': 0.30,
    'subj_02': 0.28,
    # ...
}

predictor = RuleBasedPredictor(subject_alert_ear)

# 단일 추론
result = predictor.predict(features, subject_id='subj_01')
# {'class': 'alert', 'perclos': 0.12, 'ear_ref': 0.30}

# 배치 추론
results = predictor.predict_batch(features_batch, subject_ids)
```

### 입력 규격

- `features`: `(80, 5)` numpy array — `[EAR, MAR, pitch, yaw, roll]`
- `subject_id`: 피험자 ID 문자열 (기준값 조회용)

---

## CNN+LSTM과의 성능 비교

| 모델 | Accuracy | F1 (Macro) | F1 (Drowsy) | 비고 |
|------|----------|------------|-------------|------|
| **Rule-based (PERCLOS)** | 0.8467 | 0.8411 | 0.8115 | 일시적 눈 감음에 취약 |
| CNN+LSTM | 0.9200 | 0.9145 | 0.8929 | 시간 흐름 패턴 인식 우수 |

> Rule-based는 특징 추출 검증 및 백업 판정 용도로 활용됩니다.
