"""
졸음 감지 추론 모듈
사용법:
    from inference import DrowsinessPredictor
    predictor = DrowsinessPredictor('best_model_v2.pt')
    result = predictor.predict(features)  # features: (80, 5) numpy array
"""

import numpy as np
import torch
import torch.nn as nn


# ── 모델 정의 ─────────────────────────────────────────────────────────────
class CNNLSTM(nn.Module):
    """CNN + 단방향 LSTM 졸음 감지 모델"""
    def __init__(self, n_features, cnn_channels, cnn_kernel, lstm_hidden, lstm_layers, dropout, n_classes=2):
        super().__init__()
        cnn, in_ch = [], n_features
        for out_ch in cnn_channels:
            cnn += [nn.Conv1d(in_ch, out_ch, cnn_kernel, padding=cnn_kernel//2),
                    nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True), nn.MaxPool1d(2, 2)]
            in_ch = out_ch
        self.cnn  = nn.Sequential(*cnn)
        self.lstm = nn.LSTM(cnn_channels[-1], lstm_hidden, lstm_layers,
                            batch_first=True, bidirectional=False,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(lstm_hidden, n_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.fc(self.drop(out[:, -1, :]))


# ── 전처리 함수 ───────────────────────────────────────────────────────────
def add_circular_features(X):
    """
    pitch/yaw/roll을 sin/cos로 변환
    (N, 80, 5) → (N, 80, 8): ear, mar, sin_p, cos_p, sin_y, cos_y, sin_r, cos_r
    """
    rad = np.pi / 180.0
    p = X[:, :, 2:3] * rad
    y = X[:, :, 3:4] * rad
    r = X[:, :, 4:5] * rad
    return np.concatenate([
        X[:, :, :2],
        np.sin(p), np.cos(p),
        np.sin(y), np.cos(y),
        np.sin(r), np.cos(r),
    ], axis=2).astype(np.float32)


# ── 추론 클래스 ───────────────────────────────────────────────────────────
class DrowsinessPredictor:
    """
    사용 예시:
        predictor = DrowsinessPredictor('best_model_v2.pt')
        result = predictor.predict(features)  # (80, 5)
        # {'class': 'drowsy', 'confidence': 0.82, 'probabilities': {'alert': 0.18, 'drowsy': 0.82}}

        # 여러 명 동시 처리 (배치)
        results = predictor.predict_batch(features_batch)  # (N, 80, 5)
    """

    def __init__(self, model_path: str, device: str = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        ck  = torch.load(model_path, map_location=self.device, weights_only=False)
        cfg = ck['cfg']

        # 모델 복원 (저장된 state_dict 키에서 구조 추론)
        sd = ck['model_state']
        cnn_channels = self._infer_cnn_channels(sd)
        cnn_kernel   = sd[[k for k in sd if 'cnn' in k and 'weight' in k][0]].shape[2]
        lstm_hidden  = sd['lstm.weight_hh_l0'].shape[1]
        lstm_layers  = sum(1 for k in sd if k.startswith('lstm.weight_ih_l') and 'reverse' not in k)
        dropout      = 0.0  # eval 모드에서는 영향 없음

        self.model = CNNLSTM(
            n_features   = cfg['n_features'],
            cnn_channels = cnn_channels,
            cnn_kernel   = cnn_kernel,
            lstm_hidden  = lstm_hidden,
            lstm_layers  = lstm_layers,
            dropout      = dropout,
        ).to(self.device)
        self.model.load_state_dict(sd)
        self.model.eval()

        self.scaler      = ck['scaler']
        self.threshold   = ck['threshold']
        self.class_names = cfg['class_names']
        self.n_features  = cfg['n_features']
        self.window_size = cfg['window_size']

        print(f'모델 로드 완료 ({model_path})')
        print(f'  threshold: {self.threshold:.3f}  device: {self.device}')

    def _infer_cnn_channels(self, sd):
        """state_dict에서 CNN 채널 구조 추론"""
        channels = []
        i = 0
        while True:
            key = f'cnn.{i*4}.weight'  # Conv1d는 4개 레이어 단위 (Conv+BN+ReLU+Pool)
            if key not in sd:
                break
            channels.append(sd[key].shape[0])
            i += 1
        return channels

    def _preprocess(self, features: np.ndarray) -> torch.Tensor:
        """(N, 80, 5) → (N, 8, 80) tensor"""
        x = add_circular_features(features)              # (N, 80, 8)
        N, T, F = x.shape
        x = self.scaler.transform(x.reshape(-1, F)).reshape(N, T, F)
        return torch.tensor(x.transpose(0, 2, 1), dtype=torch.float32).to(self.device)

    @torch.no_grad()
    def predict(self, features: np.ndarray) -> dict:
        """
        단일 샘플 추론
        Args:
            features: (80, 5) numpy array [ear, mar, pitch, yaw, roll]
        Returns:
            {'class': 'alert'|'drowsy', 'confidence': float, 'probabilities': dict}
        """
        assert features.shape == (self.window_size, 5), \
            f'입력 shape 오류: {features.shape}, 필요: ({self.window_size}, 5)'

        x     = self._preprocess(features[np.newaxis])   # (1, 8, 80)
        probs = torch.softmax(self.model(x), dim=1).squeeze().cpu().numpy()
        pred  = 'drowsy' if probs[1] >= self.threshold else 'alert'

        return {
            'class':         pred,
            'confidence':    float(probs[1] if pred == 'drowsy' else probs[0]),
            'probabilities': {name: float(p) for name, p in zip(self.class_names, probs)},
        }

    @torch.no_grad()
    def predict_batch(self, features_batch: np.ndarray) -> list:
        """
        여러 명 동시 추론 (배치)
        Args:
            features_batch: (N, 80, 5) numpy array
        Returns:
            list of dict (predict 결과 N개)
        """
        assert features_batch.ndim == 3 and features_batch.shape[1:] == (self.window_size, 5), \
            f'입력 shape 오류: {features_batch.shape}, 필요: (N, {self.window_size}, 5)'

        x     = self._preprocess(features_batch)         # (N, 8, 80)
        probs = torch.softmax(self.model(x), dim=1).cpu().numpy()  # (N, 2)

        results = []
        for p in probs:
            pred = 'drowsy' if p[1] >= self.threshold else 'alert'
            results.append({
                'class':         pred,
                'confidence':    float(p[1] if pred == 'drowsy' else p[0]),
                'probabilities': {name: float(v) for name, v in zip(self.class_names, p)},
            })
        return results


# ── 단독 실행 테스트 ──────────────────────────────────────────────────────
if __name__ == '__main__':
    predictor = DrowsinessPredictor('best_model_v2.pt')

    # 단일 추론 테스트
    dummy = np.random.randn(80, 5).astype(np.float32)
    result = predictor.predict(dummy)
    print(f'\n[단일 추론] {result}')

    # 배치 추론 테스트 (9~16명)
    batch = np.random.randn(16, 80, 5).astype(np.float32)
    results = predictor.predict_batch(batch)
    print(f'\n[배치 추론 16명]')
    for i, r in enumerate(results):
        print(f'  [{i+1:2d}] {r["class"]:6s}  drowsy_prob={r["probabilities"]["drowsy"]:.3f}')
