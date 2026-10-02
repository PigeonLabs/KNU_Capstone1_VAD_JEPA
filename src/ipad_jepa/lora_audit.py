"""CPU protocol and bank invariants shared by primary and teacher-zero LoRA audits."""
import numpy as np


def check_training(meta, curves, condition, teacher_weight):
    if teacher_weight not in (0.,1.):raise ValueError('Accepted teacher weights are zero or one')
    expected={'status':'complete_training','teacher_weight':teacher_weight,'epochs':20,'completed_epochs':20,
              'micro_batch':1,'accumulation':8,'rank':8,'alpha':16,'dropout':.05,'blocks':4,
              'encoder_lr':1e-4,'head_lr':1e-3,'weight_decay':1e-4,'precision':'bf16','max_steps':None,
              'encoder_trainable_parameters':131072}
    expected.update(dict(zip(['backbone','mode','device','seed'],condition)))
    if any(meta.get(k)!=v for k,v in expected.items()):raise ValueError('Completed selected LoRA treatment/protocol differs')
    if len(curves)!=20 or [int(r['epoch']) for r in curves]!=list(range(1,21)):
        raise ValueError('Twenty complete LoRA epochs required')
    values=np.array([[float(r[k]) for k in ['train_ce','train_dense_l2','normal_calibration_ce','normal_calibration_circular_mae']] for r in curves])
    if not np.isfinite(values).all() or any(int(r['clips'])!=meta['fit_clips'] for r in curves):
        raise ValueError('Nonfinite or incomplete normal LoRA curve')
    chosen=int(values[:,2].argmin())
    if meta.get('selected_epoch')!=chosen+1 or meta.get('best_normal_calibration_ce')!=float(values[chosen,2]):
        raise ValueError('LoRA selection differs from minimum normal CE')


def check_bank(bank, normal):
    expected={'mean':(1024,),'components':(256,1024),'prototypes':(16,128,256),'temperature':(),'cycle_length':()}
    if set(bank.files)!=set(expected):raise ValueError('Selected LoRA bank inventory differs')
    for name,shape in expected.items():
        value=bank[name]
        if value.shape!=shape or not np.isfinite(value).all():raise ValueError('Invalid selected LoRA bank geometry/values')
        if name in {'mean','components','prototypes'} and value.dtype!=np.float32:
            raise ValueError('LoRA bank PCA/prototypes must be FP32')
    geometry={'bins':16,'prototypes_per_bin':128,'total_prototypes':2048,'pca_dimensions':256,'pca_samples':50000,'temperature_samples':50000}
    if any(normal.get(k)!=v for k,v in geometry.items()) or normal['temperature']<1e-6:
        raise ValueError('LoRA normal memory protocol differs')
    np.testing.assert_allclose([float(bank['temperature']),float(bank['cycle_length'])],
                               [normal['temperature'],normal['cycle_length_fit_median']],rtol=0,atol=0)
    np.testing.assert_allclose(np.linalg.norm(bank['prototypes'],axis=-1),1,rtol=0,atol=2e-5)
    np.testing.assert_allclose(bank['components']@bank['components'].T,np.eye(256),rtol=0,atol=2e-4)


def ready_group(completed, model, mode, device):
    """A mean/CI requires the three exact seeds, regardless of directory count."""
    return all((model,mode,device,seed) in completed for seed in (0,1,2))
