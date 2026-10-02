"""Treatment checks prevent relabelling pilots or primary LoRA as teacher-zero runs."""
import pytest
from ipad_jepa.teacher_ablation import check_zero_training,expected_cache_bytes

def fixture():
    meta={'status':'complete_training','backbone':'dinov3-l','mode':'online','device':'R01','seed':0,
          'teacher_weight':0.,'epochs':20,'completed_epochs':20,'micro_batch':1,'accumulation':8,
          'rank':8,'alpha':16,'dropout':.05,'blocks':4,'encoder_lr':1e-4,'head_lr':1e-3,'precision':'bf16',
          'max_steps':None,'fit_clips':10,'selected_epoch':4,'best_normal_calibration_ce':2.}
    curve=[{'epoch':i+1,'train_ce':1.,'train_dense_l2':.8,'normal_calibration_ce':2.+abs(i-3),
            'normal_calibration_circular_mae':.1,'clips':10} for i in range(20)]
    return meta,curve

def test_zero_treatment_allows_nonzero_teacher_monitor_and_selects_normal_ce():
    check_zero_training(*fixture(),'dinov3-l','online','R01',0)

@pytest.mark.parametrize('field,value',[('teacher_weight',1.),('max_steps',2),('completed_epochs',19),('seed',1),('accumulation',4)])
def test_reject_primary_pilot_incomplete_or_changed_control(field,value):
    meta,curve=fixture();meta[field]=value
    with pytest.raises(ValueError,match='zero-weight'):check_zero_training(meta,curve,'dinov3-l','online','R01',0)

def test_reject_test_selected_epoch_and_nonfinite_or_partial_curve():
    meta,curve=fixture()
    with pytest.raises(ValueError,match='minimize'):check_zero_training({**meta,'selected_epoch':20},curve,'dinov3-l','online','R01',0)
    with pytest.raises(ValueError,match='20-epoch'):check_zero_training(meta,curve[:-1],'dinov3-l','online','R01',0)
    curve[1]['train_dense_l2']=float('nan')
    with pytest.raises(ValueError,match='statistics'):check_zero_training(meta,curve,'dinov3-l','online','R01',0)

def test_capacity_matches_full_clip_boundaries_and_rejects_leaked_fit():
    rows=[{'frames':32,'split':'fit','partition':'training'},{'frames':32,'split':'calibration','partition':'training'},
          {'frames':32,'partition':'testing'}]
    per_clip=2*(576*1024+1024)+8
    assert expected_cache_bytes(rows,'online')==(8+17+17)*per_clip+3*1024**2
    assert expected_cache_bytes(rows,'offline')==(5+17+17)*per_clip+3*1024**2
    rows[0]['partition']='testing'
    with pytest.raises(ValueError,match='inventory'):expected_cache_bytes(rows,'online')
