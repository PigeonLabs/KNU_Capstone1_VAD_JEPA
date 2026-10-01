"""Exact whole-video resampling via integer frame weights and one score sort."""
import numpy as np


class VideoBootstrapMetric:
    def __init__(self, run):
        if not run:
            raise ValueError('Nonempty video inventory required')
        self.videos = list(run)
        self.index = {video: index for index, video in enumerate(self.videos)}
        labels = np.concatenate([run[video][1] for video in self.videos])
        scores = np.concatenate([run[video][2] for video in self.videos]).astype(np.float64)
        groups = np.concatenate([np.full(len(run[video][1]), index, dtype=np.int64)
                                 for index, video in enumerate(self.videos)])
        if (not len(scores) or scores.ndim != 1 or labels.ndim != 1 or scores.shape != labels.shape or
            not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all()):
            raise ValueError('Finite scores and binary, aligned frame labels required')
        order = np.argsort(-scores, kind='stable')
        self.positive = labels[order] == 1
        self.groups = groups[order]
        sorted_scores = scores[order]
        self.ends = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), len(scores)-1]

    def __call__(self, chosen):
        counts = np.bincount([self.index[video] for video in chosen], minlength=len(self.videos))
        weight = counts[self.groups]
        tp = np.cumsum(weight*self.positive)[self.ends]
        fp = np.cumsum(weight*~self.positive)[self.ends]
        if tp[-1] == 0 or fp[-1] == 0:
            return np.full(2, np.nan)
        recall = tp/tp[-1]
        fpr = fp/fp[-1]
        auroc = np.sum(np.diff(np.r_[0., fpr])*(recall+np.r_[0., recall[:-1]])*.5)
        precision = np.divide(tp, tp+fp, out=np.ones(len(tp), dtype=float), where=tp+fp != 0)
        ap = np.sum(np.diff(np.r_[0., recall])*precision)
        return np.array([auroc, ap])
