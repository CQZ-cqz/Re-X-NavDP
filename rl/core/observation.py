"""Versioned physical state vector; RGB and depth must be a synchronized pair."""

import rexnavdp  # noqa: F401  (sys.path bootstrap)
import numpy as np
import torch
from tensordict import TensorDict

STATE_FIELDS = (
    ('body_velocity', 6, 1.), ('projected_gravity', 3, 1.), ('pointgoal_xy', 2, 5.),
    ('nominal', 2, .5), ('previous_command', 2, .5), ('previous_residual', 2, .5),
    ('plan_age', 1, 3.), ('rgbd_age', 1, .5), ('depth_valid_fraction', 1, 1.),
    ('path_valid', 1, 1.), ('rgbd_valid', 1, 1.), ('rgbd_pose_delta', 3, 1.),
    ('new_rgbd', 1, 1.), ('dt', 1, .04), ('danger', 1, 1.),
)
STATE_VERSION = 'current_rgbd_residual_v1'

# Direct (MPC-free) tracker state: 32 dims. Scale may be scalar or a per-axis tuple.
DIRECT_STATE_FIELDS = (
    ('body_velocity', 6, 1.),
    ('projected_gravity', 3, 1.),
    ('pointgoal_xy', 2, 5.),
    ('previous_command', 2, .5),
    ('previous_command_delta', 2, (.04, .08)),
    ('command_velocity_error', 2, .5),
    ('plan_age', 1, 3.),
    ('rgbd_age', 1, .5),
    ('plan_pose_delta', 3, (1., 1., np.pi)),
    ('rgbd_pose_delta', 3, (1., 1., np.pi)),
    ('depth_valid_fraction', 1, 1.),
    ('path_valid', 1, 1.),
    ('rgbd_valid', 1, 1.),
    ('new_plan', 1, 1.),
    ('new_rgbd', 1, 1.),
    ('dt', 1, .04),
    ('remaining_path_arc', 1, 2.),
)
DIRECT_STATE_VERSION = 'current_rgbd_direct_tracker_v1'
DIRECT_STATE_DIM = sum(size for _, size, _ in DIRECT_STATE_FIELDS)  # 32


def make_observation(features, path, path_mask, fields):
    rgb = features['rgb_tokens']
    batch, device = rgb.shape[0], rgb.device
    pieces = []
    for name, size, scale in STATE_FIELDS:
        value = torch.as_tensor(fields[name], dtype=torch.float32, device=device)
        if size == 1 and value.shape == (batch,):
            value = value[:, None]
        if value.shape != (batch, size):
            raise ValueError(f'{name} must have shape {(batch, size)}, got {value.shape}')
        if not torch.isfinite(value).all():
            raise ValueError(f'{name} contains nonfinite data')
        pieces.append((value/scale).clamp(-10, 10))
    return TensorDict({'rgb_tokens': rgb.detach(),
        'depth_tokens': features['depth_tokens'].detach(),
        'path': torch.as_tensor(path, device=device, dtype=torch.float32),
        'path_mask': torch.as_tensor(path_mask, device=device, dtype=torch.float32),
        'state': torch.cat(pieces, -1)}, batch_size=[batch], device=device)


def make_direct_observation(features, path, path_mask, fields):
    """Assemble the 32-dim direct tracker observation; scales may be per-axis."""
    rgb = features['rgb_tokens']
    batch, device = rgb.shape[0], rgb.device
    pieces = []
    for name, size, scale in DIRECT_STATE_FIELDS:
        value = torch.as_tensor(fields[name], dtype=torch.float32, device=device)
        if size == 1 and value.shape == (batch,):
            value = value[:, None]
        if value.shape != (batch, size):
            raise ValueError(f'{name} must have shape {(batch, size)}, got {value.shape}')
        if not torch.isfinite(value).all():
            raise ValueError(f'{name} contains nonfinite data')
        scale_t = torch.as_tensor(scale, dtype=torch.float32, device=device)
        pieces.append((value/scale_t).clamp(-10, 10))
    state = torch.cat(pieces, -1)
    if state.shape != (batch, DIRECT_STATE_DIM):
        raise ValueError(f'Direct state must be {DIRECT_STATE_DIM} dims, got {state.shape[-1]}')
    return TensorDict({'rgb_tokens': rgb.detach(),
        'depth_tokens': features['depth_tokens'].detach(),
        'path': torch.as_tensor(path, device=device, dtype=torch.float32),
        'path_mask': torch.as_tensor(path_mask, device=device, dtype=torch.float32),
        'state': state}, batch_size=[batch], device=device)
