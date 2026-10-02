"""Known injections preserve timeline, source pixels and held-out role boundaries."""
from dataclasses import replace
import numpy as np
import pytest
from ipad_jepa.diagnostics import scenarios, confusion, macro_f1

ROW=dict(device='R01',sequence='02',partition='training',split='diagnostic',frames=214)


def case(name):return next(x for x in scenarios(ROW,image_size=64) if x.name==name)


def test_full_scenario_inventory_is_paired_and_inside_middle_half():
    all_cases=scenarios(ROW)
    assert len(all_cases)==len({s.name for s in all_cases})==49
    assert sum(bool(s.temporal and s.appearance) for s in all_cases)==36
    assert [s.metadata() for s in all_cases]==[s.metadata() for s in scenarios(ROW)]
    for s in all_cases:
        labels=s.labels(); assert labels.shape==(214,)
        assert np.all(labels[:54]==0) and np.all(labels[160:]==0)
        assert np.all((s.source_indices()>=0)&(s.source_indices()<214))


def test_stall_holds_preceding_frame_and_reverse_preserves_all_segment_sources():
    stall,reverse=case('stall_16'),case('reverse_16')
    a=stall.temporal_start
    expected=np.arange(214);expected[a:a+16]=a-1
    np.testing.assert_array_equal(stall.source_indices(),expected)
    expected[a:a+16]=np.arange(a,a+16)[::-1]
    np.testing.assert_array_equal(reverse.source_indices(),expected)
    np.testing.assert_array_equal(reverse.labels()[a:a+16],np.full(16,2))


@pytest.mark.parametrize('name',['occlusion_4','local_colour_4'])
def test_appearance_changes_only_declared_box_and_interval_without_mutation(name):
    s=case(name);image=np.full((64,64,3),127,np.uint8);original=image.copy()
    before=s.appearance_frame(image,s.appearance_start-1)
    np.testing.assert_array_equal(before,original)
    after=s.appearance_frame(image,s.appearance_start)
    changed=np.any(after!=original,axis=-1);y0,y1,x0,x1=s.box
    expected=np.zeros((64,64),bool);expected[y0:y1,x0:x1]=True
    np.testing.assert_array_equal(changed,expected)
    np.testing.assert_array_equal(image,original)
    assert changed.mean()==s.metadata()['actual_area_fraction']
    np.testing.assert_array_equal(s.appearance_frame(image,s.appearance_start+32),original)


def test_mixed_labels_are_union_bits_with_partial_overlap():
    s=case('stall_8__occlusion_1');labels=s.labels()
    assert np.count_nonzero(labels==3)==8 and np.count_nonzero(labels==1)==24
    assert np.count_nonzero(labels==2)==0
    assert np.count_nonzero(labels==0)==214-32


@pytest.mark.parametrize('split,partition',[('fit','training'),('calibration','training'),('test','testing')])
def test_non_diagnostic_inputs_are_rejected(split,partition):
    with pytest.raises(ValueError,match='held-out'):
        scenarios(dict(ROW,split=split,partition=partition))


def test_invalid_window_cannot_silently_clip_the_intervention():
    with pytest.raises(ValueError,match='middle'):
        replace(case('stall_8'),temporal_start=0).source_indices()


def test_confusion_and_macro_f1_match_manual_four_class_counts():
    matrix=confusion(np.array([0,0,1,1,2,2,3,3]),np.array([0,1,1,2,2,3,3,0]))
    expected=np.array([[1,1,0,0],[0,1,1,0],[0,0,1,1],[1,0,0,1]])
    np.testing.assert_array_equal(matrix,expected)
    assert macro_f1(matrix)==(.5,[.5,.5,.5,.5])
    assert macro_f1(np.zeros((4,4),int))==(0.,[0.,0.,0.,0.])
