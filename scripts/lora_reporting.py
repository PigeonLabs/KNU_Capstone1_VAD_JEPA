"""Whole-video paired statistics for complete primary or teacher three-seed groups.

Inputs must already have passed condition audits. This module never decides whether
missing feature payloads are acceptable; the current/retained audit router owns that.
"""
import numpy as np

from ipad_jepa.lora_audit import ready_group
from summarize_experiments import check_pair, statistic, bootstrap_draws, macro_four_devices
from summarize_neighbour_ablation import difference

MODELS = ("dinov3-l", "vjepa21-l")
MODES = ("offline", "online")
DEVICES = ("R01", "R02", "R03", "R04")
SEEDS = (0, 1, 2)
VARIANTS = ("P0", "P1", "P2", "P3")


def keep_draws(values):
    keep = np.isfinite(values).all(axis=1)
    if keep.sum() < 950:
        raise ValueError("Too many degenerate whole-video resamples")
    return keep


def summarize(runs_by_condition, kind):
    if kind not in ("lora", "teacher"):
        raise ValueError("Expected primary LoRA or actual teacher pair statistics")
    treatments = ("F", "L") if kind == "lora" else ("T0", "T1")
    required = {(model, mode, device, seed) for model in MODELS for mode in MODES
                for device in DEVICES for seed in SEEDS}
    if not set(runs_by_condition) <= required:
        raise ValueError("Unexpected model/mode/device/seed in audited statistics")
    for values in runs_by_condition.values():
        if set(values) != set(treatments) or any(set(values[t]) != set(VARIANTS) for t in treatments):
            raise ValueError("Both actual treatments and all four variants required")
        reference = values[treatments[0]]["P3"]
        for treatment in treatments:
            for variant in VARIANTS:
                check_pair(reference, values[treatment][variant])
    groups = [(model, mode, device) for model in MODELS for mode in MODES for device in DEVICES
              if ready_group(runs_by_condition, model, mode, device)]
    stored, points, draws, rows, deltas = {}, {}, {}, [], []
    comparisons = ([('F_' + v, 'L_' + v, 'LoRA_minus_frozen_' + v) for v in VARIANTS]
                   + [('L_P0', 'L_P3', 'LoRA_P3_minus_P0')]) if kind == "lora" else [
                       ('T1_' + v, 'T0_' + v, 'T0_minus_T1_' + v) for v in VARIANTS]
    for model, mode, device in groups:
        for treatment in treatments:
            for variant in VARIANTS:
                runs = [runs_by_condition[model, mode, device, seed][treatment][variant] for seed in SEEDS]
                for run in runs[1:]:
                    check_pair(runs[0], run)
                values = bootstrap_draws(runs)
                keep = keep_draws(values)
                point = np.mean([statistic(run, list(run)) for run in runs], axis=0)
                low, high = np.quantile(values[keep], [.025, .975], axis=0)
                key = (model, mode, device, treatment + '_' + variant)
                stored[key], points[key], draws[key] = (runs, values[keep]), point, values
                labels = np.concatenate([v[1] for v in runs[0].values()])
                row = {'backbone': model, 'mode': mode, 'device': device, 'variant': key[3],
                       'score_variant': variant, 'seeds': 3, 'test_videos': len(runs[0]),
                       'frames': len(labels), 'anomaly_frames': int(labels.sum()),
                       'auroc_mean': float(point[0]), 'auroc_ci_low': float(low[0]), 'auroc_ci_high': float(high[0]),
                       'ap_mean': float(point[1]), 'ap_ci_low': float(low[1]), 'ap_ci_high': float(high[1]),
                       'bootstrap_draws': 1000, 'bootstrap_rejected': int((~keep).sum())}
                if kind == "teacher":
                    row['teacher_weight'] = 0. if treatment == "T0" else 1.
                else:
                    row.update(adaptation="frozen" if treatment == "F" else "lora",
                               teacher_weight=None if treatment == "F" else 1.)
                rows.append(row)
        for old, new, name in comparisons:
            a, b = (model, mode, device, old), (model, mode, device, new)
            delta = draws[b] - draws[a]
            deltas.append(difference(model, mode, name, points[b] - points[a], delta[keep_draws(delta)], device))
    macro = macro_four_devices(stored)
    for model in MODELS:
        for mode in MODES:
            if not all((model, mode, device) in groups for device in DEVICES):
                continue
            for old, new, name in comparisons:
                delta = np.mean([
                    bootstrap_draws(stored[model, mode, device, new][0], seed=np.random.SeedSequence([2026, int(device[1:])]))
                    - bootstrap_draws(stored[model, mode, device, old][0], seed=np.random.SeedSequence([2026, int(device[1:])]))
                    for device in DEVICES], axis=0)
                point = np.mean([points[model, mode, device, new] - points[model, mode, device, old] for device in DEVICES], axis=0)
                macro['paired_deltas'].append(difference(model, mode, name, point, delta[keep_draws(delta)]))
    for key in stored:
        model, mode, device, variant = key
        targets = []
        if model == 'dinov3-l':
            targets.append((('vjepa21-l', mode, device, variant), 'vjepa21-l_minus_dinov3-l', mode, variant + '_backbone'))
        if mode == 'offline':
            targets.append(((model, 'online', device, variant), model, 'online_minus_offline', variant + '_mode'))
        for target, label, result_mode, name in targets:
            if target not in stored:
                continue
            for old, new in zip(stored[key][0], stored[target][0]):
                check_pair(old, new)
            delta = draws[target] - draws[key]
            deltas.append(difference(label, result_mode, name, points[target] - points[key], delta[keep_draws(delta)], device))
    scope = ('Independently audited current or explicitly retained conditions; exact seeds0/1/2 per group; '
             'mean of per-seed metrics, paired original-video bootstrap1000; equal-device Macro4 only with all4; '
             'retained payload hashes/finite checks are historical, current metadata/tensors/traces are replayed; no runtime claim')
    return {'scope': scope, 'results': rows, 'paired_deltas': deltas}, macro, groups
