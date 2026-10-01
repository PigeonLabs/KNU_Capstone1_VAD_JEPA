import numpy as np
import pytest
from ipad_jepa.runtime_state import StreamingScore,event_metrics
from ipad_jepa.temporal import temporal_score


def calibration():
    return {'median':[0,0],'mad_scale':[1,1],'threshold':1}


def test_streaming_temporal_score_matches_batch_and_preserves_online_tail_alarms():
    phases=(np.arange(35)*.073+.91)%1
    state=StreamingScore(calibration(),cycle_length=100)
    actual=[state.push(15+i,phase,4) for i,phase in enumerate(phases)]
    expected=temporal_score(phases,100)
    np.testing.assert_allclose([row['time_raw'] for row in actual],expected,equal_nan=True)
    assert not any(row['alarm'] for row in actual[:6])
    assert actual[6]['frame']==21 and actual[6]['alarm']
    # No future N or GT is supplied; all final seven targets retain their alarm state.
    assert all(row['alarm'] and row['inference_valid'] for row in actual[-7:])
    state2=StreamingScore(calibration(),100)
    prefix=[state2.push(15+i,phase,4) for i,phase in enumerate(phases[:12])]
    np.testing.assert_allclose([row['score'] for row in prefix],[row['score'] for row in actual[:12]],equal_nan=True)


def test_streaming_rejects_missing_target_and_invalid_calibration():
    state=StreamingScore(calibration(),100)
    with pytest.raises(ValueError,match='sequential'): state.push(16,.2,1)
    state.push(15,.2,1)
    with pytest.raises(ValueError,match='sequential'): state.push(15,.2,1)
    bad=calibration(); bad['mad_scale'][0]=0
    with pytest.raises(ValueError,match='calibration'): StreamingScore(bad,100)


def test_event_delay_includes_processing_and_offline_lookahead():
    labels=np.zeros(60,dtype=int); labels[25:31]=1; labels[40:44]=1
    frames=np.arange(19,53); alarms=np.isin(frames,[22,23,27,28])
    # Actual completion for offline targets includes seven future frame arrivals + service.
    emitted=(frames+7)/30+.02
    result=event_metrics(labels,frames,alarms,emitted,30)
    assert result['detected_events']==1 and result['missed_events']==1
    assert result['observed_end_evaluable_events']==2
    assert result['events_detected_before_observed_end']==0
    assert result['late_detected_events']==1
    assert result['events_without_alarm_before_observed_end']==2
    assert result['false_alarm_episode_starts']==1 and result['false_alarm_frames']==2
    assert result['onset_delay_p50_seconds']==pytest.approx((27+7-25)/30+.02)
    assert result['events'][0]['first_alarm_emitted_seconds']==pytest.approx(34/30+.02)
    assert result['events'][0]['observed_end_seconds']==pytest.approx(31/30)


def test_unknown_boundaries_and_outside_coverage_are_reported():
    labels=np.array([0,-1,1,1,0,0,1,1,0])
    result=event_metrics(labels,np.array([2,3,4]),np.array([1,1,0]),np.array([.2,.3,.4]),10)
    assert result['detected_events']==1 and result['events_outside_coverage']==1
    assert result['events'][0]['onset_uncertain'] and result['events'][0]['delay_seconds'] is None
    assert result['onset_delay_p50_seconds'] is None
    assert result['events_detected_before_observed_end']==1
    censored=event_metrics(np.array([0,1,1,-1]),np.array([1,2]),
                           np.array([1,1]),np.array([.1,.2]),10)
    assert censored['observed_end_evaluable_events']==0
    assert censored['events'][0]['emitted_before_observed_end'] is None
    with pytest.raises(ValueError,match='inventory'):
        event_metrics(labels,np.array([3]),np.array([1]),np.array([.2]),10)
