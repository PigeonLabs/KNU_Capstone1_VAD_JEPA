import numpy as np
import pytest
from sklearn.metrics import roc_auc_score, average_precision_score
from ipad_jepa.bootstrap_metrics import VideoBootstrapMetric


@pytest.mark.parametrize('ties', [False, True])
def test_video_weights_match_duplicate_frame_resampling_for_every_draw(ties):
    rng = np.random.default_rng(41)
    run = {}
    for video, labels in [('normal', np.zeros(5, dtype=int)),
                          ('anomaly', np.ones(7, dtype=int)),
                          ('mixed', np.array([0, 1, 0, 1, 1, 0]))]:
        scores = rng.normal(size=len(labels))
        if ties: scores = np.round(scores)  # Ties across labels and videos, including zero.
        run[video] = (np.arange(len(labels)), labels, scores)
    metric = VideoBootstrapMetric(run)
    draws = np.random.default_rng(2026)
    for _ in range(1000):
        chosen = draws.choice(list(run), len(run), replace=True)
        labels = np.concatenate([run[video][1] for video in chosen])
        scores = np.concatenate([run[video][2] for video in chosen])
        expected = [roc_auc_score(labels, scores), average_precision_score(labels, scores)] \
            if len(np.unique(labels)) == 2 else [np.nan, np.nan]
        np.testing.assert_allclose(metric(chosen), expected, rtol=0, atol=1e-12, equal_nan=True)
    # Point statistic and single-video draws follow the same definition.
    for chosen in [list(run), ['normal'], ['anomaly'], ['mixed'], ['mixed', 'mixed']]:
        labels = np.concatenate([run[video][1] for video in chosen])
        scores = np.concatenate([run[video][2] for video in chosen])
        expected = [roc_auc_score(labels, scores), average_precision_score(labels, scores)] \
            if len(np.unique(labels)) == 2 else [np.nan, np.nan]
        np.testing.assert_allclose(metric(chosen), expected, rtol=0, atol=1e-12, equal_nan=True)


def test_video_weight_metrics_reject_unknown_labels_nonfinite_scores_or_empty_data():
    for labels, scores in [([-1, 1], [0., 1.]), ([0, 1], [np.nan, 1.]), ([], []),
                           ([[0], [1]], [[0.], [1.]])]:
        with pytest.raises(ValueError, match='Finite scores'):
            VideoBootstrapMetric({'video': ([], np.array(labels), np.array(scores))})
