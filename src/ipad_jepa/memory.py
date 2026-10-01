"""Normal-only PCA and phase-balanced prototype memory (no decoder)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.decomposition import PCA


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def kcenter(x: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if len(x) < count:
        raise ValueError(f"Need {count} candidates, got {len(x)}")
    chosen, minimum = [], np.full(len(x), np.inf)
    idx = int(rng.integers(len(x)))
    norm = np.sum(x*x, axis=1)
    for _ in range(count):
        chosen.append(idx)
        distance = np.maximum(norm + norm[idx] - 2*(x @ x[idx]), 0)
        minimum = np.minimum(minimum, distance)
        minimum[chosen] = -np.inf
        idx = int(np.argmax(minimum))
    return x[chosen].copy()


@dataclass
class PrototypeMemory:
    bins: int = 16
    per_bin: int = 128
    dimensions: int = 256
    seed: int = 0
    pca_samples: int = 50000
    candidate_limit: int = 10000
    temperature: float = 1.0

    def fit(self, features: np.ndarray, phase: np.ndarray) -> "PrototypeMemory":
        if features.ndim != 2 or len(features) != len(phase) or len(features) < self.dimensions:
            raise ValueError("Expected aligned feature/phase rows with enough PCA samples")
        if not np.all(np.isfinite(phase)) or np.any((phase < 0) | (phase >= 1)):
            raise ValueError("Fit phase must be in [0,1)")
        if self.candidate_limit < self.per_bin or self.pca_samples < self.dimensions:
            raise ValueError("Candidate/PCA limits smaller than requested memory")
        rng = np.random.default_rng(self.seed)
        ids = rng.choice(len(features), min(len(features), self.pca_samples), replace=False)
        sample = np.asarray(features[ids], dtype=np.float32)
        if not np.all(np.isfinite(sample)):
            raise ValueError("Nonfinite fit features")
        if self.dimensions > sample.shape[1]:
            raise ValueError("PCA dimension exceeds backbone feature dimension")
        self.pca = PCA(self.dimensions, whiten=False, svd_solver="covariance_eigh")
        self.pca.fit(sample)
        assigned = np.floor(phase*self.bins).astype(int)
        prototypes = []
        for b in range(self.bins):
            ids = np.flatnonzero(assigned == b)
            if len(ids) < self.per_bin:
                raise ValueError(f"Phase bin {b}: insufficient candidates; do not silently fill from test data")
            ids = rng.choice(ids, min(len(ids), self.candidate_limit), replace=False)
            candidates = self.transform(features[ids])
            prototypes.append(kcenter(candidates, self.per_bin, rng))
        self.prototypes = np.stack(prototypes)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        if not np.all(np.isfinite(values)):
            raise ValueError("Nonfinite query features")
        return unit(self.pca.transform(values)).astype(np.float32)

    def candidates(self, phase: float, global_search: bool = False) -> np.ndarray:
        if not np.isfinite(phase) or not 0 <= phase < 1:
            raise ValueError("Query phase must be in [0,1)")
        if global_search:
            return self.prototypes.reshape(-1, self.dimensions)
        b = int(phase*self.bins)
        bins = sorted({(b-1) % self.bins, b, (b+1) % self.bins})
        return self.prototypes[bins].reshape(-1, self.dimensions)

    def nearest(self, features: np.ndarray, phase: float, k: int = 5, global_search: bool = False):
        z = self.transform(features)
        bank = self.candidates(phase, global_search)
        if k < 1 or k > len(bank):
            raise ValueError("Invalid nearest-neighbour count")
        distance = np.maximum(np.sum(z*z,1)[:,None]+np.sum(bank*bank,1)[None,:]-2*z@bank.T, 0)
        idx = np.argpartition(distance, k-1, axis=1)[:, :k]
        distances = np.take_along_axis(distance, idx, axis=1)
        return z, bank[idx], distances

    def patch_scores(self, features: np.ndarray, phase: float, k: int = 5, global_search: bool = False):
        z, neighbours, distances = self.nearest(features, phase, k, global_search)
        if k == 1:
            return distances[:,0]
        logits = -distances/max(self.temperature, 1e-6)
        weights = np.exp(logits-logits.max(1, keepdims=True))
        weights /= weights.sum(1, keepdims=True)
        projection = (weights[:,:,None]*neighbours).sum(1)
        return np.sum((z-projection)**2, axis=1)

    def frame_score(self, features: np.ndarray, phase: float, k: int = 5, global_search: bool = False, top_fraction: float = .05):
        if not 0 < top_fraction <= 1:
            raise ValueError("top_fraction must be in (0,1]")
        scores = self.patch_scores(features, phase, k, global_search)
        count = max(1, int(np.ceil(top_fraction*len(scores))))
        return float(np.partition(scores, len(scores)-count)[-count:].mean())

    def calibrate_temperature(self, samples: list[tuple[np.ndarray, float]], k: int = 5):
        """Caller supplies NORMAL CALIBRATION patches only, bounded to <=50k total."""
        fifth = [self.nearest(x, phase, k)[2].max(axis=1) for x, phase in samples]
        if not fifth:
            raise ValueError("No normal calibration samples")
        self.temperature = max(float(np.median(np.concatenate(fifth))), 1e-6)

