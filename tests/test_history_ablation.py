import numpy as np
import pytest

from ipad_jepa.history_ablation import (SavedInference, alarm_flags, decode_inference,
    fit_normal, infer, pairs_for_history, primary_calibration_check)
from ipad_jepa.temporal import common_mask


def normal_sequence():
    target = np.arange(15, 180)
    phase = (target / 180 + .03 * np.sin(target / 12)) % 1
    feature = 2 + np.cos(target / 16)
    return SavedInference(180, target, phase, feature)


@pytest.mark.parametrize("history", [5, 15, 31])
def test_inference_is_prefix_invariant_and_keeps_online_tail(history):
    sequence = normal_sequence()
    calibration = fit_normal([sequence], 180, history)
    modified = SavedInference(180, sequence.targets.copy(), sequence.phase.copy(), sequence.feature.copy())
    future = modified.targets > 90
    modified.phase[future] = .8
    modified.feature[future] = 500
    before = infer(sequence, calibration, 180, history)
    after = infer(modified, calibration, 180, history)
    prefix = sequence.targets <= 90
    for original, changed in zip(before, after):
        np.testing.assert_equal(original[prefix], changed[prefix])
    ready = before[2]
    assert sequence.targets[np.flatnonzero(ready)[0]] == 15 + history - 1
    assert ready[-7:].all()


def test_history_calibration_uses_shared_mask_and_unchanged_feature_scale():
    sequence = normal_sequence()
    calibrators = [fit_normal([sequence], 180, h) for h in [5, 15, 31]]
    keep = common_mask(180, history=31)[sequence.targets]
    assert sequence.targets[keep][0] == 45 and sequence.targets[keep][-1] == 172
    for calibration in calibrators:
        assert calibration.median[0] == np.median(sequence.feature[keep])
        assert calibration.scale[0] == calibrators[0].scale[0]
    with pytest.raises(ValueError, match="shared mask"):
        fit_normal([sequence], 180, 31, shared_history=15)


def test_primary_replay_detects_tampered_source_score():
    sequence = normal_sequence()
    calibration = fit_normal([sequence], 180, 5, shared_history=5)
    pairs = pairs_for_history(sequence, 180, 5)
    score = calibration.combine(pairs)
    rows = [dict(valid=int(keep), time_raw=time, P3=value) for keep, time, value in
            zip(common_mask(180)[sequence.targets], pairs[:, 1], score)]
    meta = {"calibration": {"P3": {"median": calibration.median,
             "mad_scale": calibration.scale, "threshold": calibration.threshold}}}
    _, time_error, score_error = primary_calibration_check([sequence], [rows], meta, 180)
    assert time_error == score_error == 0
    rows[40]["P3"] += .01
    with pytest.raises(AssertionError):
        primary_calibration_check([sequence], [rows], meta, 180)


def test_annotation_columns_do_not_control_inference_or_alarm_streak():
    rows = [dict(frame=t, phase=t / 50, feature_raw=2, label=-1, valid=0) for t in range(15, 50)]
    first = decode_inference(rows, 50, "online")
    for row in rows:
        row.update(label=1, valid=1)
    second = decode_inference(rows, 50, "online")
    np.testing.assert_array_equal(first.phase, second.phase)
    np.testing.assert_array_equal(first.feature, second.feature)
    # The inference API cannot receive label-validity masks as alarm readiness.
    assert set(vars(first)) == {"frames", "targets", "phase", "feature"}
    assert alarm_flags(np.ones(5) * 2, np.ones(5, bool), 1).tolist() == [False, False, True, True, True]
    with pytest.raises(ValueError, match="target inventory"):
        decode_inference(rows[1:], 50, "online")
