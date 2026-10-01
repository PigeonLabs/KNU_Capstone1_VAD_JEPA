"""Normal calibration; anomaly-high scores. No test-set min-max normalization."""
from dataclasses import dataclass
import numpy as np


@dataclass
class NormalCalibration:
    quantile: float = .99

    def fit(self, normal_scores: np.ndarray):
        x = np.asarray(normal_scores, dtype=float)
        if x.ndim != 2 or x.shape[1] != 2 or not len(x) or not np.all(np.isfinite(x)):
            raise ValueError("Expected finite normal calibration feature/time pairs")
        if not 0 < self.quantile < 1:
            raise ValueError("Invalid calibration quantile")
        self.median = np.median(x, axis=0)
        self.scale = np.maximum(1.4826*np.median(np.abs(x-self.median), axis=0), 1e-6)
        self.threshold = float(np.quantile(self.combine(x), self.quantile))
        self.component_thresholds = np.quantile(x, self.quantile, axis=0)
        return self

    def combine(self, scores: np.ndarray):
        return ((scores-self.median)/self.scale).mean(axis=1)

    def types(self, scores: np.ndarray):
        """0 normal, 1 appearance-only, 2 time-only, 3 joint evidence; not causal diagnosis."""
        exceeds = scores > self.component_thresholds
        return exceeds[:,0].astype(int) + 2*exceeds[:,1].astype(int)

