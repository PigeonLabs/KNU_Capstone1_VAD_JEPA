"""Bind a runtime encoder and joint phase head to the selected normal-only adapter."""
import hashlib
import json
from pathlib import Path
import torch
from ipad_jepa.adapted_features import selected_protocol
from ipad_jepa.features import file_hash


def selected_runtime_adapter(args, identity, phase_meta, phase_state, teacher_identity):
    kind = getattr(args, 'adaptation', 'frozen')
    training = getattr(args, 'lora_run', None)
    if kind not in {'frozen', 'lora'}:
        raise ValueError('Unsupported runtime adaptation')
    if kind == 'frozen':
        if training is not None or 'selected_adapter_sha256' in phase_meta:
            raise ValueError('Frozen runtime cannot use a selected LoRA head')
        if identity != teacher_identity:
            raise ValueError('Runtime encoder/reader differs from the normal training protocol')
        return None, {'adaptation': 'frozen', 'selected_adapter_sha256': None,
                      'lora_training_sha256': None, 'teacher_weight': None}
    if training is None:
        raise ValueError('LoRA runtime requires the selected training directory')
    training = Path(training)
    metadata, checkpoint = selected_protocol(training)
    condition = (args.model, args.mode, args.device, args.seed)
    if tuple(metadata.get(key) for key in ['backbone', 'mode', 'device', 'seed']) != condition:
        raise ValueError('Selected LoRA condition differs from runtime')
    if metadata['teacher_identity'] != teacher_identity:
        raise ValueError('Runtime base encoder/reader differs from selected LoRA teacher')
    if any(metadata.get(key) != value for key, value in
           {'blocks': 4, 'rank': 8, 'alpha': 16, 'dropout': .05}.items()):
        raise ValueError('Selected LoRA architecture differs from the accepted experiment')
    if metadata.get('teacher_weight') not in {0., 1.}:
        raise ValueError('Unsupported selected teacher-weight condition')
    selected_hash = file_hash(training/'selected_adapter.pt')
    expected = dict(teacher_identity)
    expected['adapter_sha256'] = hashlib.sha256(json.dumps({
        'base': teacher_identity['adapter_sha256'], 'source': metadata['adaptation_sha256'],
        'selected': selected_hash}, sort_keys=True).encode()).hexdigest()
    if identity != expected:
        raise ValueError('Runtime adapter differs from the rebuilt normal memory')
    if (phase_meta.get('selected_adapter_sha256') != selected_hash or
        phase_meta.get('selected_epoch') != metadata['selected_epoch'] or
        phase_meta.get('training') != 'Joint LoRA + phase head; no post-adaptation head retraining'):
        raise ValueError('Runtime phase head is not bound to the selected joint training')
    selected_head = checkpoint['head']
    if (phase_state.keys() != selected_head.keys() or
        any(not torch.equal(phase_state[key], selected_head[key]) for key in phase_state)):
        raise ValueError('Runtime phase-head tensors differ from selected joint training')
    return checkpoint['adapter'], {'adaptation': 'lora', 'selected_adapter_sha256': selected_hash,
        'lora_training_sha256': file_hash(training/'lora_training.json'),
        'teacher_weight': metadata['teacher_weight']}
