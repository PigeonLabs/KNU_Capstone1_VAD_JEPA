"""Fixed held-out interventions; labels describe injected changes, not real fault causes."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib

import numpy as np

TEMPORAL = ('stall', 'reverse')
DURATIONS = (8, 16, 32)
APPEARANCE = ('occlusion', 'local_colour')
AREAS = (.01, .04, .09)
CLASSES = ('normal', 'appearance', 'temporal', 'joint')
COLOUR_SHIFT = (.25, -.15, .15)


@dataclass(frozen=True)
class Intervention:
    name: str
    frames: int
    image_size: int
    temporal: str | None
    temporal_start: int
    temporal_duration: int
    appearance: str | None
    appearance_start: int
    appearance_duration: int
    area_fraction: float
    box: tuple[int, int, int, int] | None
    seed: int = 0

    def validate(self):
        if self.frames < 64 or self.image_size < 16 or self.seed != 0:
            raise ValueError('Accepted fixed seed and adequate source dimensions required')
        if self.temporal not in (None, *TEMPORAL) or self.appearance not in (None, *APPEARANCE):
            raise ValueError('Unknown diagnostic intervention')
        lo, hi = int(np.ceil(self.frames*.25)), int(np.floor(self.frames*.75))
        for kind, start, duration in [(self.temporal, self.temporal_start, self.temporal_duration),
                                      (self.appearance, self.appearance_start, self.appearance_duration)]:
            if kind is None and duration != 0:
                raise ValueError('An absent intervention must have zero duration')
            if kind is not None and (start < lo or start+duration > hi or duration <= 0):
                raise ValueError('Intervention must remain inside the middle 50 percent')
        if self.temporal is not None and self.temporal_duration not in DURATIONS:
            raise ValueError('Unexpected temporal duration')
        if self.appearance is not None:
            if self.appearance_duration != 32 or self.area_fraction not in AREAS or self.box is None:
                raise ValueError('Unexpected appearance protocol')
            y0, y1, x0, x1 = self.box
            side = int(round(self.image_size*np.sqrt(self.area_fraction)))
            if not (0 <= y0 < y1 <= self.image_size and 0 <= x0 < x1 <= self.image_size
                    and y1-y0 == x1-x0 == side):
                raise ValueError('Invalid square intervention area')
        elif self.box is not None or self.area_fraction != 0:
            raise ValueError('Absent appearance intervention has a spatial box')
        return self

    def source_indices(self):
        """Map transformed timeline to original frames without changing frame count."""
        self.validate()
        result = np.arange(self.frames, dtype=np.int64)
        a, d = self.temporal_start, self.temporal_duration
        if self.temporal == 'stall':
            result[a:a+d] = a-1
        elif self.temporal == 'reverse':
            result[a:a+d] = result[a:a+d][::-1]
        return result

    def labels(self):
        self.validate()
        result = np.zeros(self.frames, dtype=np.int64)
        if self.temporal:
            result[self.temporal_start:self.temporal_start+self.temporal_duration] |= 2
        if self.appearance:
            result[self.appearance_start:self.appearance_start+self.appearance_duration] |= 1
        return result

    def appearance_frame(self, rgb, timeline_index):
        """Corrupt resized uint8 RGB on the transformed timeline, leaving its source intact."""
        if rgb.shape != (self.image_size, self.image_size, 3) or rgb.dtype != np.uint8:
            raise ValueError('Expected resized uint8 RGB frame')
        if not 0 <= timeline_index < self.frames:
            raise ValueError('Timeline index outside source')
        if not self.appearance or not self.appearance_start <= timeline_index < self.appearance_start+self.appearance_duration:
            return rgb
        output = rgb.copy()
        y0,y1,x0,x1 = self.box
        if self.appearance == 'occlusion':
            output[y0:y1,x0:x1] = 0
        else:
            pixels = output[y0:y1,x0:x1].astype(np.float32)/255
            shifted = np.clip(pixels+np.array(COLOUR_SHIFT,dtype=np.float32),0,1)
            output[y0:y1,x0:x1] = np.rint(shifted*255).astype(np.uint8)
        return output

    def metadata(self):
        result = asdict(self)
        result['source_mapping_sha256'] = hashlib.sha256(self.source_indices().tobytes()).hexdigest()
        result['intervention_labels_sha256'] = hashlib.sha256(self.labels().tobytes()).hexdigest()
        result['actual_area_fraction'] = 0 if self.box is None else (self.box[1]-self.box[0])*(self.box[3]-self.box[2])/self.image_size**2
        result['colour_shift_rgb_unit'] = list(COLOUR_SHIFT) if self.appearance == 'local_colour' else None
        return result


def scenarios(row, image_size=384):
    if row.get('partition') != 'training' or row.get('split') != 'diagnostic':
        raise ValueError('Only held-out normal diagnostic videos may be transformed')
    length = row['frames']
    rng = np.random.default_rng(np.random.SeedSequence([0,int(row['device'][1:]),int(row['sequence'])]))
    lo, hi = int(np.ceil(length*.25)), int(np.floor(length*.75))
    if hi-lo < 32:
        raise ValueError('Middle 50 percent cannot hold the fixed appearance window')
    anchor = int(rng.integers(lo, hi-32+1))
    margin = int(np.ceil(round(image_size*np.sqrt(.09))/2))
    cy, cx = map(int, rng.integers(margin,image_size-margin,size=2))
    specs = [('normal', None, 0, None, 0.)]
    specs += [(f'{kind}_{duration}',kind,duration,None,0.) for kind in TEMPORAL for duration in DURATIONS]
    specs += [(f'{kind}_{int(area*100)}',None,0,kind,area) for kind in APPEARANCE for area in AREAS]
    specs += [(f'{t}_{duration}__{a}_{int(area*100)}',t,duration,a,area)
              for t in TEMPORAL for duration in DURATIONS for a in APPEARANCE for area in AREAS]
    result=[]
    for name, temporal, duration, appearance, area in specs:
        side=int(round(image_size*np.sqrt(area))) if appearance else 0
        box=(cy-side//2,cy-side//2+side,cx-side//2,cx-side//2+side) if appearance else None
        result.append(Intervention(name,length,image_size,temporal,anchor+(32-duration)//2 if temporal else 0,
                                   duration,appearance,anchor if appearance else 0,32 if appearance else 0,area,box).validate())
    return result


def confusion(truth, predicted):
    truth,predicted=np.asarray(truth),np.asarray(predicted)
    if truth.shape != predicted.shape or truth.ndim != 1 or not len(truth):
        raise ValueError('Expected aligned nonempty diagnostic labels')
    if not np.issubdtype(truth.dtype,np.integer) or not np.issubdtype(predicted.dtype,np.integer):
        raise ValueError('Diagnostic classes must be integer codes')
    if np.any((truth<0)|(truth>3)|(predicted<0)|(predicted>3)):
        raise ValueError('Unknown diagnostic class')
    return np.bincount(4*truth+predicted,minlength=16).reshape(4,4)


def macro_f1(matrix):
    matrix=np.asarray(matrix)
    if matrix.shape!=(4,4) or np.any(matrix<0) or not np.isfinite(matrix).all():
        raise ValueError('Expected finite nonnegative 4-class confusion')
    tp=np.diag(matrix)
    denominator=matrix.sum(0)+matrix.sum(1)
    per_class=np.divide(2*tp,denominator,out=np.zeros(4,dtype=float),where=denominator>0)
    return float(per_class.mean()),per_class.tolist()
