"""Full LoRA reports must reject wrong treatment, selection and bank provenance."""
import numpy as np
import pytest
from ipad_jepa.lora_audit import check_training,check_bank,ready_group

CONDITION=('vjepa21-l','offline','R01',1)

def training(weight=1.):
    meta={'status':'complete_training','teacher_weight':weight,'epochs':20,'completed_epochs':20,
          'micro_batch':1,'accumulation':8,'rank':8,'alpha':16,'dropout':.05,'blocks':4,'encoder_lr':1e-4,
          'head_lr':1e-3,'weight_decay':1e-4,'precision':'bf16','max_steps':None,'encoder_trainable_parameters':131072,
          'backbone':'vjepa21-l','mode':'offline','device':'R01','seed':1,'fit_clips':8,
          'selected_epoch':3,'best_normal_calibration_ce':2.}
    curve=[{'epoch':i+1,'train_ce':1.,'train_dense_l2':.4,'normal_calibration_ce':2.+abs(i-2),
            'normal_calibration_circular_mae':.1,'clips':8} for i in range(20)]
    return meta,curve

def test_accept_actual_zero_or_one_treatment_with_complete_normal_selection():
    for weight in [0.,1.]:check_training(*training(weight),CONDITION,weight)

@pytest.mark.parametrize('field,value',[('teacher_weight',0.),('completed_epochs',19),('micro_batch',2),('encoder_lr',.001),
                                       ('max_steps',3),('seed',0),('encoder_trainable_parameters',0)])
def test_reject_wrong_treatment_pilot_incomplete_or_changed_training_control(field,value):
    meta,curve=training();meta[field]=value
    with pytest.raises(ValueError,match='protocol'):check_training(meta,curve,CONDITION,1.)

def test_select_by_normal_ce_with_full_fit_clip_count_and_finite_monitor():
    meta,curve=training()
    with pytest.raises(ValueError,match='selection'):check_training({**meta,'selected_epoch':20},curve,CONDITION,1.)
    curve[0]['clips']=7
    with pytest.raises(ValueError,match='incomplete'):check_training(meta,curve,CONDITION,1.)
    curve[0]['clips']=8;curve[-1]['train_dense_l2']=float('nan')
    with pytest.raises(ValueError,match='Nonfinite'):check_training(meta,curve,CONDITION,1.)

def test_three_directories_do_not_substitute_for_three_matched_designated_seeds():
    keys={('dinov3-l','online','R01',s) for s in [0,1,9]}
    assert not ready_group(keys,'dinov3-l','online','R01')
    keys.add(('dinov3-l','offline','R01',2))
    assert not ready_group(keys,'dinov3-l','online','R01')
    keys.add(('dinov3-l','online','R01',2))
    assert ready_group(keys,'dinov3-l','online','R01')
    assert not ready_group(keys,'dinov3-l','online','R02')

def bank_fixture(tmp_path):
    values={'mean':np.zeros(1024,np.float32),'components':np.eye(256,1024,dtype=np.float32),
            'prototypes':np.zeros((16,128,256),np.float32),'temperature':np.array(.2),'cycle_length':np.array(160.)}
    values['prototypes'][:,:,0]=1
    normal={'bins':16,'prototypes_per_bin':128,'total_prototypes':2048,'pca_dimensions':256,'pca_samples':50000,
            'temperature_samples':50000,'temperature':.2,'cycle_length_fit_median':160.}
    path=tmp_path/'bank.npz';np.savez(path,**values)
    return path,values,normal

def test_actual_lora_bank_rejects_small_model_and_nonfinite_values(tmp_path):
    path,values,normal=bank_fixture(tmp_path)
    with np.load(path,allow_pickle=False) as bank:check_bank(bank,normal)
    values['mean']=np.zeros(768,np.float32);np.savez(path,**values)
    with np.load(path,allow_pickle=False) as bank:
        with pytest.raises(ValueError,match='geometry'):check_bank(bank,normal)
    values['mean']=np.zeros(1024,np.float32);values['prototypes'][0,0,255]=np.nan;np.savez(path,**values)
    with np.load(path,allow_pickle=False) as bank:
        with pytest.raises(ValueError,match='values'):check_bank(bank,normal)

def test_actual_bank_rejects_wrong_normal_parameters_or_unnormalized_prototypes(tmp_path):
    path,values,normal=bank_fixture(tmp_path)
    with np.load(path,allow_pickle=False) as bank:
        with pytest.raises(AssertionError):check_bank(bank,{**normal,'temperature':.3})
        with pytest.raises(ValueError,match='protocol'):check_bank(bank,{**normal,'total_prototypes':4096})
    values['prototypes']*=2;np.savez(path,**values)
    with np.load(path,allow_pickle=False) as bank:
        with pytest.raises(AssertionError):check_bank(bank,normal)
