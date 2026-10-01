"""Streaming P3 state and post-run event accounting, independent of future length/GT."""
from collections import deque
import numpy as np


class StreamingScore:
    def __init__(self, calibration, cycle_length, first_target=15, history=5,
                 warmup_target=19, consecutive=3):
        self.median=np.asarray(calibration['median'],dtype=float)
        self.scale=np.asarray(calibration['mad_scale'],dtype=float)
        self.threshold=float(calibration['threshold'])
        if (self.median.shape!=(2,) or self.scale.shape!=(2,) or
            not np.isfinite(np.r_[self.median,self.scale,self.threshold,cycle_length]).all() or
            np.any(self.scale<=0) or cycle_length<=0 or history<2 or consecutive<1 or
            first_target<0 or warmup_target<first_target+history-1):
            raise ValueError('Invalid fixed normal calibration/streaming protocol')
        self.cycle_length=cycle_length
        self.phases=deque(maxlen=history)
        self.next_target=first_target
        self.warmup_target=warmup_target
        self.consecutive=consecutive
        self.streak=0

    def push(self,target,phase,feature_score):
        if target!=self.next_target or not np.isfinite([phase,feature_score]).all() or not 0<=phase<1:
            raise ValueError('Finite scores and sequential causal target frames required')
        self.next_target+=1; self.phases.append(float(phase))
        time_score=float('nan')
        if len(self.phases)==self.phases.maxlen:
            values=np.asarray(self.phases); j=np.arange(1,len(values))
            delta=(values[-1]-values[-1-j]+.5)%1-.5
            time_score=float(np.mean(np.abs(delta-j/self.cycle_length)))
        score=float(np.mean((np.array([feature_score,time_score])-self.median)/self.scale))
        eligible=target>=self.warmup_target and np.isfinite(score)
        self.streak=self.streak+1 if eligible and score>self.threshold else 0
        return {'frame':target,'phase':float(phase),'feature_raw':float(feature_score),
                'time_raw':time_score,'score':score,'inference_valid':bool(eligible),
                'alarm':self.streak>=self.consecutive}


def event_metrics(labels,frames,alarms,emitted_seconds,arrival_fps):
    """Read GT only after inference. Unknown boundaries cannot establish onset delay.

    Call with actual emitted targets for operational coverage, and separately with the
    shared target mask for comparisons. A GT event with no evaluated target is reported
    as outside coverage, rather than silently becoming a detected/missed event.
    """
    labels=np.asarray(labels); frames=np.asarray(frames); alarms=np.asarray(alarms,dtype=bool)
    emitted=np.asarray(emitted_seconds,dtype=float)
    if (labels.ndim!=1 or frames.ndim!=1 or alarms.shape!=frames.shape or emitted.shape!=frames.shape or
        not np.isin(labels,[-1,0,1]).all() or not np.issubdtype(frames.dtype,np.integer) or
        np.any(np.diff(frames)<=0) or np.any(frames<0) or np.any(frames>=len(labels)) or
        not np.isfinite(emitted).all() or not np.isfinite(arrival_fps) or arrival_fps<=0 or
        np.any(emitted+1e-9<frames/arrival_fps)):
        raise ValueError('Invalid post-run frame/label/emission inventory')
    events=[]; position=0
    while position<len(labels):
        if labels[position]!=1:
            position+=1; continue
        start=position
        while position<len(labels) and labels[position]==1: position+=1
        end=position-1
        selected=np.flatnonzero((frames>=start)&(frames<=end))
        hits=selected[alarms[selected]]
        onset_uncertain=start==0 or labels[start-1]==-1
        offset_uncertain=end==len(labels)-1 or labels[end+1]==-1
        first=int(hits[0]) if len(hits) else None
        first_emitted=float(emitted[first]) if first is not None else None
        observed_end=(end+1)/arrival_fps if not offset_uncertain else None
        events.append({'start_frame':start,'end_frame':end,'onset_uncertain':onset_uncertain,
                       'offset_uncertain':offset_uncertain,'covered_frames':len(selected),
                       'detected':first is not None,
                       'first_alarm_frame':int(frames[first]) if first is not None else None,
                       'first_alarm_emitted_seconds':first_emitted,
                       'observed_end_seconds':observed_end,
                       'emitted_before_observed_end':first_emitted<observed_end
                            if first_emitted is not None and observed_end is not None else None,
                       'delay_seconds':float(emitted[first]-start/arrival_fps)
                            if first is not None and not onset_uncertain else None})
    starts=alarms.copy()
    if len(starts)>1: starts[1:] &= (~alarms[:-1])|(np.diff(frames)!=1)
    normal=labels[frames]==0; unknown=labels[frames]==-1
    covered=[event for event in events if event['covered_frames']]
    bounded=[event for event in covered if not event['offset_uncertain']]
    delays=[event['delay_seconds'] for event in covered if event['delay_seconds'] is not None]
    return {'events':events,'covered_events':len(covered),
            'events_outside_coverage':sum(event['covered_frames']==0 for event in events),
            'detected_events':sum(event['detected'] for event in covered),
            'missed_events':sum(not event['detected'] for event in covered),
            'observed_end_evaluable_events':len(bounded),
            'events_detected_before_observed_end':sum(event['emitted_before_observed_end'] is True for event in bounded),
            'late_detected_events':sum(event['emitted_before_observed_end'] is False for event in bounded),
            'events_without_alarm_before_observed_end':sum(event['emitted_before_observed_end'] is not True for event in bounded),
            'false_alarm_episode_starts':int(np.sum(starts&normal)),
            'unknown_alarm_episode_starts':int(np.sum(starts&unknown)),
            'false_alarm_frames':int(np.sum(alarms&normal)),
            'evaluated_normal_frames':int(np.sum(normal)),
            'onset_delay_p50_seconds':float(np.quantile(delays,.5)) if delays else None,
            'onset_delay_p95_seconds':float(np.quantile(delays,.95)) if delays else None,
            'definition':'A false alarm episode starts on a known-normal target. Misses use GT events with at least one emitted eligible target; events touching unknown boundaries are flagged, and uncertain onset delays excluded. Before-observed-end counts require emission strictly before the first known-normal frame after the event; uncertain offsets are excluded. This is observed-event accounting, not an application SLA.'}
