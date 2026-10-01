"""Frame windows and causal, circular phase progression scores."""
from __future__ import annotations
import numpy as np


def clip_indices(target: int, length: int, mode: str, frames: int = 16, padding: bool = False) -> np.ndarray:
    if frames % 2 or frames < 2 or mode not in {"offline", "online"}:
        raise ValueError("Even frame count >=2 and offline/online mode required")
    if not 0 <= target < length:
        raise ValueError("Target is outside the sequence")
    start = target - (frames//2 if mode == "offline" else frames-1)
    result = np.arange(start, start+frames)
    if not padding and (result[0] < 0 or result[-1] >= length):
        raise ValueError("Incomplete clip; padding is only permitted when explicitly requested")
    return np.clip(result, 0, length-1)


def common_mask(length: int, clip_frames: int = 16, history: int = 5) -> np.ndarray:
    result = np.zeros(length, dtype=bool)
    start = clip_frames-1 + history-1
    stop = length - (clip_frames - clip_frames//2 - 1)
    result[start:max(start, stop)] = True
    return result


def circular_phase(probabilities: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(p.sum(1) <= 0):
        raise ValueError("Expected finite, nonnegative phase distributions")
    p = p / p.sum(1, keepdims=True)
    angle = 2*np.pi*(np.arange(p.shape[1])+.5)/p.shape[1]
    x, y = p @ np.cos(angle), p @ np.sin(angle)
    # A uniform/antipodal distribution has no circular mean. Deterministic argmax fallback.
    ambiguous = np.hypot(x, y) < 1e-8
    phi = np.mod(np.arctan2(y, x)/(2*np.pi), 1)
    phi[ambiguous] = (np.argmax(p[ambiguous], axis=1)+.5)/p.shape[1]
    return phi


def temporal_score(phases: np.ndarray, cycle_length: float, history: int = 5) -> np.ndarray:
    if cycle_length <= 0 or history < 2:
        raise ValueError("Positive normal cycle length and history>=2 required")
    phases = np.asarray(phases, dtype=np.float64)
    result = np.full(len(phases), np.nan)
    for t in range(history-1, len(phases)):
        window = phases[t-history+1:t+1]
        if not np.all(np.isfinite(window)):
            continue
        j = np.arange(1, history)
        delta = (phases[t]-phases[t-j]+.5) % 1-.5
        result[t] = np.mean(np.abs(delta-j/cycle_length))
    return result

