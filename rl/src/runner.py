"""Training/checkpoint runner for the high-frequency RGB-D policy."""

import rexnavdp  # noqa: F401  (sys.path bootstrap)
from dataclasses import asdict
from pathlib import Path
from typing import Mapping
import random
import numpy as np
import torch
from .policy import PolicyConfig, ReactiveActorCritic
from .ppo import ReactivePPO
from .observation import STATE_VERSION, DIRECT_STATE_VERSION
from .encoder import DAV2_TYPE, PREPROCESS_VERSION

CONTROL_MODE_RESIDUAL = 'residual'
CONTROL_MODE_DIRECT = 'direct'
ACTION_MAPPING_RESIDUAL = 'residual_v1'
ACTION_MAPPING_DIRECT = 'direct_v1'

VISUAL_METADATA_KEYS = (
    'visual_encoder_type', 'visual_encoder_model', 'visual_feature_layer',
    'visual_token_dim', 'visual_grid_size', 'visual_encoder_frozen',
    'preprocess_version', 'depth_preprocess_version', 'fingerprint',
)


def _mode_metadata(control_mode):
    if control_mode == CONTROL_MODE_RESIDUAL:
        return STATE_VERSION, ACTION_MAPPING_RESIDUAL
    if control_mode == CONTROL_MODE_DIRECT:
        return DIRECT_STATE_VERSION, ACTION_MAPPING_DIRECT
    raise ValueError(f'Unknown control mode {control_mode!r}')


def _encoder_identity(encoder_fingerprint='unverified', encoder_metadata=None):
    """Accept legacy fingerprint strings and the new serialized metadata dict."""
    if isinstance(encoder_fingerprint, Mapping):
        if encoder_metadata is not None:
            raise ValueError('Pass encoder metadata only once')
        encoder_metadata = dict(encoder_fingerprint)
        encoder_fingerprint = encoder_metadata.get('fingerprint', 'unverified')
    metadata = None if encoder_metadata is None else dict(encoder_metadata)
    if metadata is not None and metadata.get('fingerprint') != encoder_fingerprint:
        raise ValueError('Encoder metadata fingerprint is inconsistent')
    return str(encoder_fingerprint), metadata


class ReactiveRunner:
    def __init__(self, policy, num_envs, steps=64, ppo_config=None,
                 encoder_fingerprint='unverified', control_mode=CONTROL_MODE_RESIDUAL,
                 encoder_metadata=None):
        self.policy, self.num_envs, self.steps = policy, num_envs, steps
        self.device = next(policy.parameters()).device
        self.encoder_fingerprint, self.encoder_metadata = _encoder_identity(
            encoder_fingerprint, encoder_metadata)
        self.control_mode = control_mode
        self.state_version, self.action_mapping_version = _mode_metadata(control_mode)
        options = dict(num_learning_epochs=4, num_mini_batches=min(4, num_envs),
            learning_rate=3e-4, entropy_coef=.005, gamma=.99, lam=.95,
            clip_param=.2, max_grad_norm=1., schedule='fixed')
        options.update(ppo_config or {})
        if num_envs % options['num_mini_batches']:
            raise ValueError('Recurrent PPO requires num_envs divisible by mini_batches')
        self.ppo_config = options
        self.algorithm = ReactivePPO(policy, device=str(self.device), **options)
        self.iteration = 0
        self.obs = None

    def collect_and_update(self, env):
        if self.obs is None:
            self.obs = env.reset().to(self.device)
            self.policy.initialize_hidden(self.num_envs)
            self.algorithm.init_storage('rl', self.num_envs, self.steps, self.obs, [2])
        self.policy.train()
        for _ in range(self.steps):
            with torch.no_grad():
                latent = self.algorithm.act(self.obs)
                next_obs, rewards, terminated, truncated, extras = env.step(latent)
                next_obs = next_obs.to(self.device)
                terminal_obs = extras.get('terminal_observation')
                if terminal_obs is not None:
                    terminal_obs = terminal_obs.to(self.device)
                self.algorithm.process_transition(next_obs, rewards.to(self.device),
                    terminated.to(self.device), truncated.to(self.device), terminal_obs)
                self.obs = next_obs
        self.algorithm.compute_returns(self.obs)
        metrics = self.algorithm.update()
        self.iteration += 1
        if not all(np.isfinite(v) for v in metrics.values()):
            raise FloatingPointError(f'Nonfinite PPO metrics: {metrics}')
        return dict(iteration=self.iteration, **metrics)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name+'.tmp')
        preprocess = (self.encoder_metadata or {}).get('preprocess_version', PREPROCESS_VERSION)
        payload = {'format_version': 1, 'policy_config': asdict(self.policy.config),
            'policy': self.policy.state_dict(), 'optimizer': self.algorithm.optimizer.state_dict(),
            'iteration': self.iteration, 'ppo_config': self.ppo_config,
            'control_mode': self.control_mode, 'state_version': self.state_version,
            'action_mapping_version': self.action_mapping_version,
            'preprocess_version': preprocess,
            'encoder_fingerprint': self.encoder_fingerprint,
            'torch_rng': torch.get_rng_state(), 'numpy_rng': np.random.get_state(),
            'python_rng': random.getstate(),
            'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        if self.encoder_metadata is not None:
            payload['visual_encoder_metadata'] = self.encoder_metadata
            # Flat fields make experiment manifests/checkpoints easy to inspect.
            payload.update({key: self.encoder_metadata[key] for key in VISUAL_METADATA_KEYS[:-1]})
        torch.save(payload, temp)
        temp.replace(path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        validate_checkpoint(checkpoint, self.policy.config, self.encoder_fingerprint,
            self.control_mode, self.encoder_metadata)
        stored_ppo, current_ppo = dict(checkpoint['ppo_config']), dict(self.ppo_config)
        stored_ppo.pop('num_mini_batches', None); current_ppo.pop('num_mini_batches', None)
        if stored_ppo != current_ppo:
            raise ValueError('PPO configuration differs from checkpoint')
        self.policy.load_state_dict(checkpoint['policy'], strict=True)
        self.algorithm.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iteration = checkpoint['iteration']
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        np.random.set_state(checkpoint['numpy_rng']); random.setstate(checkpoint['python_rng'])
        if checkpoint.get('cuda_rng') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint['cuda_rng']])
        self.obs = None
        self.policy.reset()


def validate_checkpoint(checkpoint, config, encoder_fingerprint,
                        control_mode=CONTROL_MODE_RESIDUAL, encoder_metadata=None):
    fingerprint, runtime_meta = _encoder_identity(encoder_fingerprint, encoder_metadata)
    if checkpoint.get('format_version') != 1:
        raise ValueError('Reactive checkpoint format mismatch')
    stored_cfg = checkpoint.get('policy_config', {})
    for key, value in asdict(config).items():
        if key in stored_cfg and stored_cfg[key] != value:
            raise ValueError('Reactive checkpoint configuration mismatch')
    state_version, action_mapping_version = _mode_metadata(control_mode)
    if checkpoint.get('state_version') != state_version:
        raise ValueError('Reactive observation version mismatch')
    if checkpoint.get('action_mapping_version', action_mapping_version) != action_mapping_version:
        raise ValueError('Action mapping version mismatch')

    stored_meta = checkpoint.get('visual_encoder_metadata')
    if stored_meta is not None:
        if runtime_meta is None:
            # Backward-compatible callers may still pass just the exact hash.
            if stored_meta.get('fingerprint') != fingerprint:
                raise ValueError('Visual encoder fingerprint mismatch')
        else:
            mismatch = [key for key in VISUAL_METADATA_KEYS
                        if stored_meta.get(key) != runtime_meta.get(key)]
            if mismatch:
                raise ValueError(f'Visual encoder mismatch in fields: {mismatch}')
        expected_preprocess = stored_meta.get('preprocess_version')
    else:
        # Metadata-free checkpoints predate YOLO support and are DA-V2 only.
        if runtime_meta is not None and runtime_meta.get('visual_encoder_type') != DAV2_TYPE:
            raise ValueError('Legacy DA-V2 checkpoint cannot be loaded with a YOLO visual encoder')
        expected_preprocess = PREPROCESS_VERSION
    if checkpoint.get('preprocess_version') != expected_preprocess:
        raise ValueError('Reactive preprocessing version mismatch')
    if checkpoint.get('encoder_fingerprint') != fingerprint:
        raise ValueError('Visual encoder weights differ from the training encoder')


def load_policy(path, encoder_fingerprint, device='cpu',
                control_mode=CONTROL_MODE_RESIDUAL, encoder_metadata=None):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = PolicyConfig(**checkpoint['policy_config'])
    validate_checkpoint(checkpoint, config, encoder_fingerprint, control_mode, encoder_metadata)
    policy = ReactiveActorCritic(config).to(device)
    policy.load_state_dict(checkpoint['policy'], strict=True)
    return policy.eval()


def warm_start_policy(path, config, encoder_fingerprint, device='cpu',
                      control_mode=CONTROL_MODE_DIRECT, encoder_metadata=None):
    """Load deterministic BC weights while keeping target exploration config."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    source_config = PolicyConfig(**checkpoint['policy_config'])
    validate_checkpoint(checkpoint, source_config, encoder_fingerprint,
        control_mode, encoder_metadata)
    policy = ReactiveActorCritic(config).to(device)
    state, target_state = checkpoint['policy'].copy(), policy.state_dict()
    for key in ('log_std', 'log_std_min', 'log_std_max'):
        state[key] = target_state[key]
    policy.load_state_dict(state, strict=True)
    return policy
