"""
Rule-based 졸음 감지 모듈 (개인화 PERCLOS)

개요:
    subject별 alert 구간의 평균 EAR을 개인 기준값으로 설정하고,
    8초 윈도우 내 EAR 저하 비율(PERCLOS)로 졸음 여부를 판정합니다.

사용법:
    predictor = RuleBasedPredictor(subject_alert_ear)
    result = predictor.predict(features, subject_id)
    # features: (80, 5) numpy array [ear, mar, pitch, yaw, roll]
"""

import numpy as np


class RuleBasedPredictor:
    """
    개인화 PERCLOS 기반 졸음 판정기

    Args:
        subject_alert_ear (dict): subject_id → 해당 subject의 alert 구간 평균 EAR
        perclos_thresh (float): PERCLOS 임계값 (기본 0.3)
        ear_ratio_thresh (float): 눈 감김 판정 비율 (기본 0.8, 개인 기준값 대비)
    """

    def __init__(self, subject_alert_ear: dict,
                 perclos_thresh: float = 0.3,
                 ear_ratio_thresh: float = 0.8):
        self.subject_alert_ear = subject_alert_ear
        self.perclos_thresh    = perclos_thresh
        self.ear_ratio_thresh  = ear_ratio_thresh

    def predict(self, features: np.ndarray, subject_id: str) -> dict:
        """
        단일 샘플 판정

        Args:
            features:   (80, 5) numpy array [ear, mar, pitch, yaw, roll]
            subject_id: 피험자 ID (개인 기준값 조회용)

        Returns:
            {
                'class':    'alert' | 'drowsy',
                'perclos':  float,   # 0~1, 눈 감김 비율
                'ear_ref':  float,   # 개인 기준 EAR
            }
        """
        assert features.ndim == 2 and features.shape[1] >= 1, \
            f"입력 shape 오류: {features.shape}, 필요: (80, 5+)"

        ear     = features[:, 0]                                      # (80,)
        ear_ref = self.subject_alert_ear.get(subject_id, 0.25)       # 개인 기준값
        perclos = float(np.mean(ear / (ear_ref + 1e-6) < self.ear_ratio_thresh))

        return {
            'class':   'drowsy' if perclos >= self.perclos_thresh else 'alert',
            'perclos': perclos,
            'ear_ref': ear_ref,
        }

    def predict_batch(self, features_batch: np.ndarray,
                      subject_ids: list) -> list:
        """
        여러 샘플 동시 판정

        Args:
            features_batch: (N, 80, 5) numpy array
            subject_ids:    (N,) subject ID 리스트

        Returns:
            list of dict (predict 결과 N개)
        """
        assert features_batch.ndim == 3
        ear     = features_batch[:, :, 0]                             # (N, 80)
        refs    = np.array([self.subject_alert_ear.get(s, 0.25)
                            for s in subject_ids])                    # (N,)
        perclos = np.mean(ear / (refs[:, None] + 1e-6)
                          < self.ear_ratio_thresh, axis=1)            # (N,)

        return [
            {
                'class':   'drowsy' if p >= self.perclos_thresh else 'alert',
                'perclos': float(p),
                'ear_ref': float(refs[i]),
            }
            for i, p in enumerate(perclos)
        ]


# ── 단독 실행 테스트 ──────────────────────────────────────────────────────
if __name__ == '__main__':
    # 더미 subject 기준값
    subject_ear = {'subj_01': 0.30, 'subj_02': 0.28}
    predictor   = RuleBasedPredictor(subject_ear)

    # 단일 추론
    dummy  = np.random.uniform(0.1, 0.4, (80, 5)).astype(np.float32)
    result = predictor.predict(dummy, 'subj_01')
    print(f'[단일] {result}')

    # 배치 추론
    batch   = np.random.uniform(0.1, 0.4, (4, 80, 5)).astype(np.float32)
    results = predictor.predict_batch(batch, ['subj_01', 'subj_02', 'subj_01', 'subj_02'])
    for i, r in enumerate(results):
        print(f'[{i+1}] {r}')
