"""Render only independently audited three-seed LoRA or teacher-weight groups."""
import argparse
import json
import platform
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from plot_neighbour_ablation import digest, save, NAMES


def group_rows(summary, kind, full):
    names = ('F_', 'L_') if kind == 'lora' else ('T0_', 'T1_')
    expected = {p+v for p in names for v in ('P0','P1','P2','P3')}
    groups = {}
    for row in summary['results']:
        if row['seeds'] != 3 or full and row.get('devices') != 4:
            raise ValueError('Plots require three-seed groups and complete Macro4 when requested')
        key = (row['backbone'], row['mode'], None if full else row['device'])
        if row['variant'] in groups.setdefault(key, {}):
            raise ValueError('Duplicate audited plotted condition')
        groups[key][row['variant']] = row
    if not groups:
        raise ValueError('No audited complete three-seed groups to plot')
    if any(set(rows) != expected for rows in groups.values()):
        raise ValueError('All eight treatment/variant rows are required')
    return groups


def verify_sources(args, proof):
    if proof['status'] != 'passed':
        raise ValueError('Passed independent validation required')
    if proof['manifest_sha256'] != digest(args.manifest):
        raise ValueError('Audited manifest changed')
    kind = args.kind
    if kind == 'lora':
        count = proof['completed_seed_conditions']
        if proof['teacher_weight'] != 1. or proof['normal_thresholds_checked'] != count*8:
            raise ValueError('Current primary teacher-one LoRA audit required')
        checks = proof['condition_checks']
        entries = [(r['condition'], r, args.primary_root, 1.) for r in checks]
        verifier = 'summarize_lora_matrix.py'
        if proof['protocol_audit_sha256'] != digest('src/ipad_jepa/lora_audit.py'):
            raise ValueError('LoRA protocol verifier changed')
    else:
        count = proof['completed_seed_pairs']
        if (proof['treatment_conditions_checked'] != count*2
                or proof['normal_thresholds_checked'] != count*8):
            raise ValueError('Actual teacher pairs required')
        checks = proof['pair_checks']
        entries = []
        for row in checks:
            if len(row['treatments']) != 2 or {r['teacher_weight'] for r in row['treatments']} != {0.,1.}:
                raise ValueError('Both audited teacher treatments required')
            for treatment in row['treatments']:
                entries.append((row['condition'], treatment,
                                args.zero_root if treatment['teacher_weight'] == 0 else args.primary_root,
                                treatment['teacher_weight']))
        verifier = 'summarize_teacher_weight.py'
        if (proof['control_verifier_sha256'] != digest('src/ipad_jepa/teacher_weight_audit.py')
                or proof['condition_verifier_sha256'] != digest(Path(__file__).with_name('summarize_lora_matrix.py'))):
            raise ValueError('Teacher pair verifier changed')
    if proof['verifier_sha256'] != digest(Path(__file__).with_name(verifier)):
        raise ValueError('Independent summary verifier changed')
    keys = [tuple(r['condition']) for r in checks]
    if len(keys) != count or len(set(keys)) != count:
        raise ValueError('Audited seed inventory differs')
    complete = sum(all((m, mode, d, s) in keys for s in (0,1,2))
                   for m in NAMES for mode in ('offline','online') for d in ('R01','R02','R03','R04'))
    if complete != proof['complete_three_seed_groups'] or proof['matrix_complete'] != (count == 48):
        raise ValueError('Complete group/full matrix flags differ from actual audited conditions')
    for condition, entry, root, weight in entries:
        path = args.root/entry['condition_audit']
        if digest(path) != entry['condition_audit_sha256']:
            raise ValueError('Audited condition witness changed')
        witness = json.loads(path.read_text())
        if (witness['status'] != 'passed_completed_lora_condition'
                or witness['auditor_sha256'] != digest(Path(__file__).with_name('summarize_lora_matrix.py'))
                or witness['protocol_audit_sha256'] != digest('src/ipad_jepa/lora_audit.py')
                or witness['selected_tensor_verifier_sha256'] != digest(Path(__file__).with_name('plot_lora_evaluation.py'))
                or witness['teacher_weight'] != weight
                or tuple(witness[k] for k in ('backbone','mode','device','seed')) != tuple(condition)):
            raise ValueError('Audited witness weight/condition differs')
        source = root/condition[0]/condition[1]/condition[2]/f'seed{condition[3]}'
        for name, expected in witness['public_sources_sha256'].items():
            if digest(source/name) != expected:
                raise ValueError('Plotted public LoRA source changed after independent audit')
    for name in ('device_summary','macro_summary'):
        if proof[name+'_sha256'] != digest(args.root/(name+'.json')):
            raise ValueError('Audited summary changed')


def verify_plotted_statistics(args, summary, groups, full):
    """Rebuild every plotted mean/CI and paired difference from current CSVs."""
    from summarize_experiments import load_run, check_pair, statistic, bootstrap_draws
    selected = ('F_P3','L_P0','L_P3') if args.kind == 'lora' else ('T1_P3','T0_P3')
    roots = {'F':args.frozen_root,'L':args.primary_root,'T1':args.primary_root,'T0':args.zero_root}
    pairs = [('L_P3','F_P3','LoRA_minus_frozen_P3'),('L_P3','L_P0','LoRA_P3_minus_P0')] if args.kind == 'lora' else [('T0_P3','T1_P3','T0_minus_T1_P3')]
    manifest = json.loads(args.manifest.read_text())['sequences']
    for (model, mode, device), rows in groups.items():
        points, draws, reference = {}, {}, {}
        devices = ('R01','R02','R03','R04') if full else (device,)
        for name in selected:
            kind, variant = name.split('_'); per_device_points, per_device_draws = [], []
            for current in devices:
                runs = []
                for seed in (0,1,2):
                    folder = roots[kind]/model/mode/current/f'seed{seed}'
                    info = json.loads((folder/'metrics.json').read_text())
                    if info['status'] != 'complete_device_evaluation' or tuple(info[k] for k in ('backbone','mode','device','seed')) != (model,mode,current,seed):
                        raise ValueError('Current plotted evaluation differs')
                    if kind == 'F':
                        from plot_lora_evaluation import calibration_check
                        calibration_check(folder,info,manifest,args.data_root)
                    run = load_run(folder,variant,info)
                    if current in reference: check_pair(reference[current],run)
                    else: reference[current] = run
                    runs.append(run)
                per_device_points.append(np.mean([statistic(r,list(r)) for r in runs],axis=0))
                per_device_draws.append(bootstrap_draws(runs,seed=np.random.SeedSequence([2026,int(current[1:])]) if full else 2026))
            points[name] = np.mean(per_device_points,axis=0)
            draws[name] = np.mean(per_device_draws,axis=0)
            keep = np.isfinite(draws[name]).all(axis=1)
            if keep.sum() < 950: raise ValueError('Too many degenerate plotted video draws')
            low, high = np.quantile(draws[name][keep],[.025,.975],axis=0)
            for i, metric in enumerate(('auroc','ap')):
                np.testing.assert_allclose([rows[name][metric+s] for s in ('_mean','_ci_low','_ci_high')],
                                           [points[name][i],low[i],high[i]],rtol=0,atol=1e-9)
        for new, old, comparison in pairs:
            matches = [r for r in summary['paired_deltas'] if r['backbone']==model and r['mode']==mode
                       and r['comparison']==comparison and (full or r.get('device')==device)]
            if len(matches) != 1: raise ValueError('Exact paired plotted difference required')
            delta = draws[new]-draws[old]; keep = np.isfinite(delta).all(axis=1)
            if keep.sum() < 950: raise ValueError('Too many degenerate plotted paired draws')
            low,high = np.quantile(delta[keep],[.025,.975],axis=0)
            point = points[new]-points[old]
            for i,metric in enumerate(('auroc','ap')):
                np.testing.assert_allclose([matches[0][metric+s] for s in ('_delta','_delta_ci_low','_delta_ci_high')],
                                           [point[i],low[i],high[i]],rtol=0,atol=1e-9)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('lora','teacher'), default='lora')
    parser.add_argument('--root', type=Path)
    parser.add_argument('--primary-root', type=Path, default=Path('results/stage04'))
    parser.add_argument('--zero-root', type=Path, default=Path('results/stage05/ablations/teacher_weight/T0'))
    parser.add_argument('--frozen-root', type=Path, default=Path('results/stage02'))
    parser.add_argument('--manifest', type=Path, default=Path('results/stage00/manifest.json'))
    parser.add_argument('--data-root', type=Path, default=Path('../IPAD_dataset/IPAD_dataset'))
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.root is None:
        args.root = Path('results/stage04/matrix_audit' if args.kind == 'lora'
                         else 'results/stage05/ablations/teacher_weight/paired_audit')
    if args.out is None:
        args.out = Path('docs/figures/stage04' if args.kind == 'lora' else 'docs/figures/stage05')
    proof = json.loads((args.root/'validation.json').read_text())
    verify_sources(args, proof)
    full = proof['matrix_complete']
    summary = json.loads((args.root/('macro_summary.json' if full else 'device_summary.json')).read_text())
    groups = group_rows(summary, args.kind, full)
    verify_plotted_statistics(args,summary,groups,full)
    figures = {}
    for (model, mode, device), rows in sorted(groups.items()):
        if args.kind == 'lora':
            selected = [('Frozen P3','F_P3'), ('LoRA P0','L_P0'), ('LoRA P3','L_P3')]
            comparisons = [('LoRA P3 - Frozen P3','LoRA_minus_frozen_P3'),
                           ('LoRA P3 - LoRA P0','LoRA_P3_minus_P0')]
        else:
            selected = [('Teacher 1 / P3','T1_P3'), ('Teacher 0 / P3','T0_P3')]
            comparisons = [('Teacher 0 - Teacher 1 / P3','T0_minus_T1_P3')]
        fig, axes = plt.subplots(1,3,figsize=(13.5,4.5),layout='constrained')
        for axis, metric in zip(axes[:2], ('auroc','ap')):
            for x, (label, name), colour in zip(range(len(selected)), selected, ('#3979ae','#db8b37','#41977c')):
                point, lo, hi = [rows[name][metric+s]*100 for s in ('_mean','_ci_low','_ci_high')]
                axis.bar(x,point,width=.5,color=colour)
                axis.vlines(x,lo,hi,color='#26343f')
                axis.hlines([lo,hi],x-.07,x+.07,color='#26343f')
                axis.text(x, min(98, hi+2), f'{point:.2f}', ha='center', fontsize=9)
            axis.set(xticks=range(len(selected)),xticklabels=[r[0] for r in selected],
                     ylim=(0,103),ylabel=metric.upper()+' (%)')
            axis.tick_params(axis='x',labelsize=9)
        axis = axes[2]; ticklabels = []
        for i, (label, name) in enumerate(comparisons):
            matches = [r for r in summary['paired_deltas'] if r['backbone']==model and r['mode']==mode
                       and r['comparison']==name and (full or r.get('device')==device)]
            if len(matches) != 1:
                raise ValueError('Exact paired plotted difference required')
            for j, metric, colour in zip((0,1),('auroc','ap'),('#3979ae','#db8b37')):
                point, lo, hi = [matches[0][metric+s]*100 for s in ('_delta','_delta_ci_low','_delta_ci_high')]
                y = i*2+j; axis.plot(point,y,'o',color=colour)
                axis.hlines(y,lo,hi,color=colour,linewidth=2)
                ticklabels.append(label+'\n'+metric.upper())
        axis.axvline(0,color='#87939d',linestyle='--',linewidth=1)
        axis.set(yticks=range(len(ticklabels)),yticklabels=ticklabels,
                 ylim=(len(ticklabels)-.5,-.5),xlabel='Paired difference (pp)',title='95% whole-video CI')
        axis.tick_params(axis='y',labelsize=8)
        for axis in axes: axis.spines[['top','right']].set_visible(False)
        scope = 'equal-weight Macro4' if full else f'{device} only / partial matrix'
        title = 'LoRA feature anomaly detection' if args.kind == 'lora' else 'LoRA teacher weight 0 vs 1'
        fig.suptitle(f'{NAMES[model]} / {mode} / {title}\n{scope}',fontsize=13)
        note = ('Normal-only 20 epochs; selected joint adapter/head; refitted PCA/bank/temperature/MAD/q99; no decoder.\n'
                'Mean of seeds0/1/2 metrics; 1000 paired whole-video draws; shared GT/targets t=19..N-8. No runtime claim.\n'
                'Source: '+str(args.root/('macro_summary.json' if full else 'device_summary.json')))
        name = args.kind+'_'+model+'_'+mode+('_macro4' if full else '_'+device)
        figures.update(save(fig,args.out/name,note))
    receipt = {'status':'rendered_from_verified_three_seed_summaries','kind':args.kind,
               'matrix_complete':full,'validation_sha256':digest(args.root/'validation.json'),
               'device_summary_sha256':proof['device_summary_sha256'],
               'macro_summary_sha256':proof['macro_summary_sha256'],'plot_code_sha256':digest(__file__),
               'reporting_runtime':{'python':platform.python_version(),'numpy':np.__version__,
                                    'matplotlib':matplotlib.__version__},
               'figures':figures,'plotted_statistics_rebuilt_from_current_CSVs':True,
               'scope':'Actual audited complete three-seed groups only; plotted means/CIs/differences rebuilt; final PNG review required'}
    (args.root/'figure_sources.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__': main()
