"""Scientific plots must not promote partial seeds or incomplete device strata."""
import copy
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from plot_lora_matrix import group_rows, verify_plotted_statistics


def summary(kind='lora', full=False):
    prefixes = ['F_', 'L_'] if kind == 'lora' else ['T0_', 'T1_']
    rows = [{'backbone':'dinov3-l','mode':'offline','variant':p+v,'seeds':3,
             **({'devices':4} if full else {'device':'R01'})}
            for p in prefixes for v in ['P0','P1','P2','P3']]
    return {'results':rows}


@pytest.mark.parametrize('kind,full', [('lora',False),('lora',True),('teacher',False),('teacher',True)])
def test_exact_treatment_variant_groups_and_three_seeds_are_required(kind, full):
    assert len(group_rows(summary(kind,full),kind,full)) == 1
    partial = summary(kind,full); partial['results'][0]['seeds'] = 1
    with pytest.raises(ValueError,match='three-seed'): group_rows(partial,kind,full)


def test_teacher_one_is_not_frozen_and_missing_variant_cannot_be_hidden():
    with pytest.raises(ValueError,match='eight'): group_rows(summary(),'teacher',False)
    partial = summary(); partial['results'].pop()
    with pytest.raises(ValueError,match='eight'): group_rows(partial,'lora',False)


def test_duplicate_summary_row_and_empty_group_do_not_create_figures():
    duplicate = summary(); duplicate['results'].append(copy.deepcopy(duplicate['results'][0]))
    with pytest.raises(ValueError,match='Duplicate'): group_rows(duplicate,'lora',False)
    with pytest.raises(ValueError,match='No audited'): group_rows({'results':[]},'lora',False)


def test_partial_device_result_cannot_be_called_macro4():
    partial = summary(full=True); partial['results'][0]['devices'] = 3
    with pytest.raises(ValueError,match='Macro4'): group_rows(partial,'lora',True)


def plotted_teacher_fixture(tmp_path):
    roots = {'T0':tmp_path/'zero','T1':tmp_path/'one'}
    for kind, root in roots.items():
        for seed in (0,1,2):
            folder = root/'dinov3-l'/'offline'/'R01'/f'seed{seed}'
            (folder/'P3').mkdir(parents=True)
            for video in ('01','02'):
                with (folder/'P3'/(video+'.csv')).open('w') as stream:
                    writer = csv.DictWriter(stream,fieldnames=['frame','label','score','valid'])
                    writer.writeheader()
                    for frame,label in enumerate((0,1,0,1)):
                        writer.writerow({'frame':frame,'label':label,'score':label if kind=='T1' else 1-label,'valid':1})
            (folder/'metrics.json').write_text(json.dumps({
                'status':'complete_device_evaluation','backbone':'dinov3-l','mode':'offline','device':'R01',
                'seed':seed,'test_videos':2,'variants':{'P3':{'frames':8,'anomaly_frames':4,
                'frame_auroc':1. if kind=='T1' else 0.,'frame_ap':1. if kind=='T1' else .5}}}))
    manifest = tmp_path/'manifest.json'; manifest.write_text(json.dumps({'sequences':[]}))
    args = SimpleNamespace(kind='teacher',zero_root=roots['T0'],primary_root=roots['T1'],
                           frozen_root=tmp_path/'frozen',manifest=manifest)
    data = summary('teacher')
    for row in data['results']:
        for metric in ('auroc','ap'):
            value = 1. if row['variant'].startswith('T1_') else 0. if metric=='auroc' else .5
            row.update({metric+s:value for s in ('_mean','_ci_low','_ci_high')})
    delta = {'backbone':'dinov3-l','mode':'offline','device':'R01','comparison':'T0_minus_T1_P3'}
    for metric,value in [('auroc',-1.),('ap',-.5)]:
        delta.update({metric+s:value for s in ('_delta','_delta_ci_low','_delta_ci_high')})
    data['paired_deltas'] = [delta]
    return args,data


def test_plotted_current_csv_statistics_reject_wrong_marginal_ci(tmp_path):
    args,data = plotted_teacher_fixture(tmp_path)
    verify_plotted_statistics(args,data,group_rows(data,'teacher',False),False)
    next(r for r in data['results'] if r['variant']=='T0_P3')['auroc_ci_high'] = .1
    with pytest.raises(AssertionError):
        verify_plotted_statistics(args,data,group_rows(data,'teacher',False),False)


def test_plotted_pair_difference_cannot_reverse_the_teacher_direction(tmp_path):
    args,data = plotted_teacher_fixture(tmp_path)
    data['paired_deltas'][0]['auroc_delta'] = 1.
    with pytest.raises(AssertionError):
        verify_plotted_statistics(args,data,group_rows(data,'teacher',False),False)
