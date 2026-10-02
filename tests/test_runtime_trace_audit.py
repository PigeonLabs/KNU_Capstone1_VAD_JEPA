import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from audit_runtime import audit_trace
from ipad_jepa.runtime_state import StreamingScore


def measured_rows(mode="online"):
    labels = np.zeros(40, dtype=int)
    labels[22:28] = 1
    labels[25] = -1
    known = labels != -1
    calibration = {"median": [2., .01], "mad_scale": [.1, .01], "threshold": .6}
    state = StreamingScore(calibration, 100., first_target=15 if mode == "online" else 8)
    rows = []
    for index in range(40):
        arrival = index / 30.
        start = arrival + .01 + index * .02
        emitted = start + .02
        row = {"arrival_frame": str(index), "arrival_seconds": str(arrival),
               "service_start_seconds": str(start), "emitted_seconds": str(emitted),
               "queue_wait_ms": str((start - arrival) * 1000), "service_ms": str((emitted - start) * 1000),
               "decode_resize_ms": "2", "encoder_ms": "12", "feature_cast_ms": "1", "phase_head_ms": "1",
               "projection_search_ms": "1", "score_alarm_ms": "1", "target_frame": "",
               "inference_valid": "0", "shared_metric_valid": "0", "alarm": "0"}
        if index >= 15:
            target = index if mode == "online" else index - 7
            values = state.push(target, index / 100., 2.4)
            row.update({key: str(values[key]) for key in ["phase", "feature_raw", "time_raw", "score"]})
            row.update(target_frame=str(target), label=str(labels[target]),
                       inference_valid=str(int(values["inference_valid"])), alarm=str(int(values["alarm"])),
                       shared_metric_valid=str(int(19 <= target <= 32 and known[target])),
                       target_latency_ms=str((emitted - target / 30.) * 1000),
                       lookahead_wait_ms=str(0 if mode == "online" else 7000 / 30.))
        rows.append(row)
    return rows, calibration, labels, known


@pytest.mark.parametrize("mode", ["online", "offline"])
def test_independent_trace_replay_checks_fifo_lookahead_and_keeps_unknown_tail_alarms(mode):
    rows, calibration, labels, known = measured_rows(mode)
    eligible, shared = audit_trace(rows, mode, calibration, 100., labels, known)
    assert any(row["label"] == "-1" for row in eligible)
    assert not any(row["label"] == "-1" for row in shared)
    assert eligible[-1]["alarm"] == "1"
    assert int(eligible[-1]["target_frame"]) == (39 if mode == "online" else 32)


@pytest.mark.parametrize("field,value,reason", [
    ("queue_wait_ms", "0", "AssertionError"),
    ("target_latency_ms", "0", "AssertionError"),
    ("time_raw", "1", "AssertionError"),
    ("score", "999", "AssertionError"),
    ("alarm", "0", "ValueError"),
    ("inference_valid", "0", "ValueError"),
    ("target_frame", "26", "ValueError"),
    ("label", "0", "ValueError"),
    ("encoder_ms", "1000", "ValueError"),
])
def test_corrupted_runtime_evidence_is_rejected(field, value, reason):
    rows, calibration, labels, known = measured_rows()
    changed = copy.deepcopy(rows)
    changed[25][field] = value
    with pytest.raises(AssertionError if reason == "AssertionError" else ValueError):
        audit_trace(changed, "online", calibration, 100., labels, known)


def test_runtime_trace_audit_refuses_dropped_input_frame():
    rows, calibration, labels, known = measured_rows()
    with pytest.raises(ValueError, match="All input frames"):
        audit_trace(rows[:-1], "online", calibration, 100., labels, known)
