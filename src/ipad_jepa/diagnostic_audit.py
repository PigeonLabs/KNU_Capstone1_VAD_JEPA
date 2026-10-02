"""CPU replay and original-video cluster statistics for controlled diagnostic traces."""
from __future__ import annotations

import numpy as np

from ipad_jepa.temporal import temporal_score

FIELDS = ('frame', 'valid', 'phase', 'feature_raw', 'time_raw', 'score',
          'evidence_type', 'intervention_type')


def class_f1(matrix):
    matrix = np.asarray(matrix)
    if (matrix.shape[-2:] != (4, 4) or not np.issubdtype(matrix.dtype, np.integer)
            or np.any(matrix < 0)):
        raise ValueError('Nonnegative integer four-class confusion required')
    denominator = matrix.sum(-2) + matrix.sum(-1)
    return np.divide(2 * np.diagonal(matrix, axis1=-2, axis2=-1), denominator,
                     out=np.zeros_like(denominator, dtype=float), where=denominator > 0)


def normalized_confusion(matrix):
    """Normalize truth rows independently, leaving unsupported classes at zero."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape[-2:] != (4,4) or not np.isfinite(matrix).all() or np.any(matrix<0):
        raise ValueError('Finite nonnegative four-class confusion required')
    support = matrix.sum(-1, keepdims=True)
    return np.divide(matrix, support, out=np.zeros_like(matrix), where=support>0)


def equal_device_confusion(matrices):
    """Normalize each device before averaging; long videos cannot dominate the display."""
    matrices = np.asarray(matrices)
    if matrices.shape != (4,4,4):
        raise ValueError('Four completed device confusion matrices required')
    return normalized_confusion(matrices).mean(0)


def audit_trace(rows, length, mode, recipe, cycle, params):
    """Reconstruct truth, causal time residual, normal normalization and evidence bits."""
    if mode not in ('online', 'offline') or not rows or any(tuple(r) != FIELDS for r in rows):
        raise ValueError('Accepted mode and complete diagnostic CSV schema required')
    targets = np.arange(15, length) if mode == 'online' else np.arange(8, length - 7)
    ids = np.array([int(r['frame']) for r in rows])
    np.testing.assert_array_equal(ids, targets)
    phase = np.array([float(r['phase']) for r in rows])
    if not np.isfinite(phase).all() or np.any((phase < 0) | (phase >= 1)):
        raise ValueError('Finite circular phase required')
    dense = np.full(length, np.nan)
    dense[ids] = phase
    raw = np.array([[float(r['feature_raw']), float(r['time_raw'])] for r in rows])
    if not np.isfinite(raw[:, 0]).all():
        raise ValueError('Finite actual feature residual required')
    np.testing.assert_allclose(raw[:, 1], temporal_score(dense, cycle, 5)[ids],
                               rtol=0, atol=1e-12, equal_nan=True)
    valid = (ids >= 19) & (ids <= length - 8)
    np.testing.assert_array_equal([int(r['valid']) for r in rows], valid)
    if not valid.any() or not np.isfinite(raw[valid]).all():
        raise ValueError('Nonempty finite common diagnostic interval required')
    score = ((raw - params['median']) / params['mad_scale']).mean(1)
    np.testing.assert_allclose(score, [float(r['score']) for r in rows],
                               rtol=0, atol=1e-9, equal_nan=True)
    predicted = np.zeros(len(ids), dtype=np.int64)
    finite = np.isfinite(raw).all(1)
    predicted[finite] = (raw[finite, 0] > params['component_thresholds'][0]).astype(int)
    predicted[finite] += 2 * (raw[finite, 1] > params['component_thresholds'][1]).astype(int)
    np.testing.assert_array_equal(predicted, [int(r['evidence_type']) for r in rows])
    truth = np.zeros(length, dtype=np.int64)
    # Independently reconstruct interval union; do not call the producer's labels().
    for kind, bit in [('appearance', 1), ('temporal', 2)]:
        if recipe[kind] is not None:
            a, d = recipe[kind + '_start'], recipe[kind + '_duration']
            truth[a:a+d] |= bit
    np.testing.assert_array_equal(truth[ids], [int(r['intervention_type']) for r in rows])
    matrix = np.zeros((4, 4), dtype=np.int64)
    np.add.at(matrix, (truth[ids][valid], predicted[valid]), 1)
    return matrix


def clustered_statistics(matrices, draws=1000, seed=2026):
    """Pool all scenarios within each original video; share each draw across three seeds."""
    matrices = np.asarray(matrices)
    if (matrices.ndim != 4 or matrices.shape[0] != 3 or matrices.shape[1] < 1
            or draws < 1):
        raise ValueError('Three seeds by original video by four-class confusion required')
    class_f1(matrices)  # Validate geometry, integer counts and nonnegativity.
    total = matrices.sum(1)
    point_classes = class_f1(total).mean(0)
    rng = np.random.default_rng(seed)
    indices = rng.integers(matrices.shape[1], size=(draws, matrices.shape[1]))
    # Shape: seed, draw, selected-original-video, truth, predicted.
    pooled = matrices[:, indices].sum(2)
    class_draws = class_f1(pooled).mean(0)
    return float(point_classes.mean()), point_classes, class_draws.mean(1), class_draws


def summary_row(matrices, draws=1000, seed=2026):
    point, classes, boot, class_boot = clustered_statistics(matrices, draws, seed)
    bounds = np.quantile(boot, [.025, .975])
    class_bounds = np.quantile(class_boot, [.025, .975], axis=0)
    return {'macro_f1_mean': point, 'macro_f1_ci_low': float(bounds[0]),
            'macro_f1_ci_high': float(bounds[1]), 'per_class_f1_mean': classes.tolist(),
            'per_class_f1_ci_low': class_bounds[0].tolist(),
            'per_class_f1_ci_high': class_bounds[1].tolist(),
            'seeds': 3, 'original_videos': int(np.asarray(matrices).shape[1]),
            'bootstrap_draws': draws}, boot, class_boot
