import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from ipad_jepa.runtime_adaptation import selected_runtime_adapter
from ipad_jepa.features import file_hash


def selected_condition(tmp_path):
    source = Path(__file__).parents[1]/'src/ipad_jepa'
    teacher = {'backbone': 'dinov3-l', 'mode': 'online', 'weights_sha256': 'base-weights',
        'upstream_commit': 'upstream', 'adapter_sha256': 'base-adapter',
        'reader_sha256': 'reader', 'image_size': 384, 'clip_frames': 16, 'fit_stride': 4,
        'preprocessing': 'RGB full-frame PIL bilinear resize; ImageNet mean/std',
        'feature_dtype': 'float16 from BF16 inference'}
    protocol = {'backbone': 'dinov3-l', 'mode': 'online', 'device': 'R01', 'seed': 0,
        'epochs': 20, 'max_steps': None, 'teacher_identity': teacher, 'teacher_weight': 1.,
        'blocks': 4, 'rank': 8, 'alpha': 16, 'dropout': .05,
        'trainer_sha256': file_hash(source/'train_lora.py'),
        'adaptation_sha256': file_hash(source/'adaptation.py')}
    head = {'weight': torch.arange(4.).reshape(2, 2)}
    torch.save({'protocol': protocol, 'selected_epoch': 4, 'head': head,
                'adapter': {'last4.q_A': torch.ones(2, 2)}}, tmp_path/'selected_adapter.pt')
    metadata = {**protocol, 'status': 'complete_training', 'completed_epochs': 20,
                'selected_epoch': 4, 'best_normal_calibration_ce': .5}
    (tmp_path/'lora_training.json').write_text(json.dumps(metadata))
    with (tmp_path/'lora_training.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['epoch', 'normal_calibration_ce'])
        writer.writeheader()
        writer.writerows({'epoch': epoch, 'normal_calibration_ce': .5 if epoch == 4 else 1.+epoch/100}
                         for epoch in range(1, 21))
    selected_hash = file_hash(tmp_path/'selected_adapter.pt')
    identity = {**teacher, 'adapter_sha256': hashlib.sha256(json.dumps({
        'base': teacher['adapter_sha256'], 'source': protocol['adaptation_sha256'],
        'selected': selected_hash}, sort_keys=True).encode()).hexdigest()}
    phase = {'selected_adapter_sha256': selected_hash, 'selected_epoch': 4,
             'training': 'Joint LoRA + phase head; no post-adaptation head retraining'}
    args = SimpleNamespace(model='dinov3-l', mode='online', device='R01', seed=0,
                          adaptation='lora', lora_run=tmp_path)
    return args, identity, phase, head, teacher


def test_runtime_selected_joint_head_and_rebuilt_memory_are_bound(tmp_path):
    args, identity, phase, head, teacher = selected_condition(tmp_path)
    adapter, receipt = selected_runtime_adapter(args, identity, phase, head, teacher)
    assert torch.equal(adapter['last4.q_A'], torch.ones(2, 2))
    assert receipt['selected_adapter_sha256'] == file_hash(tmp_path/'selected_adapter.pt')
    assert receipt['teacher_weight'] == 1.
    with pytest.raises(ValueError, match='rebuilt normal memory'):
        selected_runtime_adapter(args, teacher, phase, head, teacher)
    with pytest.raises(ValueError, match='phase-head tensors'):
        selected_runtime_adapter(args, identity, phase, {'weight': head['weight']+1}, teacher)


@pytest.mark.parametrize('change,reason', [('seed', 'condition differs'),
    ('epoch', 'selected joint training'), ('teacher', 'selected LoRA teacher'),
    ('partial', 'completed 20-epoch')])
def test_runtime_refuses_another_seed_epoch_teacher_or_partial_training(tmp_path, change, reason):
    args, identity, phase, head, teacher = selected_condition(tmp_path)
    if change == 'seed': args.seed = 1
    elif change == 'epoch': phase['selected_epoch'] = 20
    elif change == 'teacher': teacher = {**teacher, 'weights_sha256': 'other-base'}
    else:
        metadata = json.loads((tmp_path/'lora_training.json').read_text())
        metadata['completed_epochs'] = 19
        (tmp_path/'lora_training.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=reason):
        selected_runtime_adapter(args, identity, phase, head, teacher)


def test_runtime_frozen_choice_rejects_adapted_head_and_missing_lora_directory(tmp_path):
    args, identity, phase, head, teacher = selected_condition(tmp_path)
    args.lora_run = None
    with pytest.raises(ValueError, match='selected training directory'):
        selected_runtime_adapter(args, identity, phase, head, teacher)
    args.adaptation = 'frozen'
    with pytest.raises(ValueError, match='selected LoRA head'):
        selected_runtime_adapter(args, identity, phase, head, teacher)
    adapter, receipt = selected_runtime_adapter(args, teacher, {}, head, teacher)
    assert adapter is None and receipt['adaptation'] == 'frozen'
