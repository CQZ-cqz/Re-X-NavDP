"""Latent residual -> bounded base velocity. This is not a collision guarantee."""

import rexnavdp  # noqa: F401  (sys.path bootstrap)
from dataclasses import dataclass, asdict
import math
import torch


@dataclass(frozen=True)
class ActionLimits:
    max_v: float = .5
    max_w: float = .5
    normal_delta_v: float = .15
    normal_delta_w: float = .25
    acceleration_v: float = 1.
    acceleration_w: float = 2.

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError('All action limits must be finite and positive')


@dataclass(frozen=True)
class DirectLimits:
    max_v: float = .5
    max_w: float = .5
    acceleration_v: float = 1.
    acceleration_w: float = 2.
    coupled_radius: float = 1.1

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError('All direct action limits must be finite and positive')
        if self.coupled_radius < 1.:
            raise ValueError('coupled_radius must be at least 1 to preserve per-axis limits')


class DirectActionMapper:
    """Latent z -> bounded base velocity. No nominal/residual; PPO optimizes z directly.

    Pipeline per plan 2.6:
      latent z -> tanh -> desired [v*, w*] -> slew-rate limiter -> safety -> applied [v, w]
    """
    def __init__(self, limits=None):
        self.limits = limits or DirectLimits()

    def __call__(self, latent, previous, dt, emergency=None):
        if latent.ndim != 2 or latent.shape[-1] != 2 or previous.shape != latent.shape:
            raise ValueError('Expected matching B x 2 latent and previous controls')
        if not torch.isfinite(previous).all():
            raise ValueError('Previous control must be finite')
        cfg = self.limits
        bound = latent.new_tensor([cfg.max_v, cfg.max_w])
        rate = latent.new_tensor([cfg.acceleration_v, cfg.acceleration_w])
        batch = latent.shape[0]
        emergency = torch.zeros(batch, dtype=torch.bool, device=latent.device) if emergency is None else emergency.bool()
        if emergency.shape != (batch,):
            raise ValueError('Emergency mask must have shape B')
        delta_t = torch.as_tensor(dt, device=latent.device, dtype=latent.dtype)
        if delta_t.ndim == 0:
            delta_t = delta_t.expand(batch)
        if delta_t.shape != (batch,) or not torch.isfinite(delta_t).all() or (delta_t <= 0).any():
            raise ValueError('dt must be positive scalar or B vector')
        invalid = ~torch.isfinite(latent).all(-1)
        a = torch.nan_to_num(latent).tanh()
        raw_desired = bound*a  # independent per-axis tanh bounds
        # A data-calibrated convex shield prevents high forward/reverse speed
        # and high yaw rate from being commanded simultaneously. Radius 1.1
        # preserves each axis maximum while allowing a small corner margin.
        normalized_radius = torch.linalg.vector_norm(raw_desired/bound, dim=-1)
        joint_scale = torch.clamp(cfg.coupled_radius/normalized_radius.clamp_min(1e-6), max=1.)
        desired = raw_desired*joint_scale[:, None]
        joint_limited = normalized_radius > cfg.coupled_radius+1e-6
        step_limit = rate*delta_t[:, None]
        applied = torch.maximum(torch.minimum(desired, previous+step_limit), previous-step_limit).clamp(-bound, bound)
        # Independent slew clipping can trace a short L-shaped path between two
        # feasible points. If that path leaves the convex joint envelope, move
        # back along the previous->candidate segment; this preserves both slew
        # bounds and the coupled constraint.
        outside = torch.linalg.vector_norm(applied/bound, dim=-1) > cfg.coupled_radius+1e-6
        if outside.any():
            delta = applied-previous
            lower = torch.zeros(batch, device=latent.device, dtype=latent.dtype)
            upper = torch.ones_like(lower)
            for _ in range(12):
                fraction = (lower+upper)*.5
                point = previous+fraction[:, None]*delta
                inside = torch.linalg.vector_norm(point/bound, dim=-1) <= cfg.coupled_radius
                lower = torch.where(inside, fraction, lower)
                upper = torch.where(inside, upper, fraction)
            projected = previous+lower[:, None]*delta
            applied = torch.where(outside[:, None], projected, applied)
        stop = emergency | invalid
        applied = torch.where(stop[:, None], torch.zeros_like(applied), applied)
        return {'command': applied, 'desired': desired, 'raw_desired': raw_desired,
                'joint_limited': joint_limited, 'emergency': stop,
                'limited': (applied-desired).abs().amax(-1) > 1e-6}


class ActionComposer:
    def __init__(self, limits=None):
        self.limits = limits or ActionLimits()

    def __call__(self, latent, nominal, previous, dt, danger=None, emergency=None):
        if latent.ndim != 2 or latent.shape[-1] != 2 or nominal.shape != latent.shape or previous.shape != latent.shape:
            raise ValueError('Expected matching B x 2 latent, nominal and previous controls')
        if not torch.isfinite(nominal).all() or not torch.isfinite(previous).all():
            raise ValueError('Nominal/previous control must be finite')
        cfg = self.limits
        bound = latent.new_tensor([cfg.max_v, cfg.max_w])
        small = latent.new_tensor([cfg.normal_delta_v, cfg.normal_delta_w])
        rate = latent.new_tensor([cfg.acceleration_v, cfg.acceleration_w])
        batch = latent.shape[0]
        danger = torch.zeros(batch, dtype=torch.bool, device=latent.device) if danger is None else danger.bool()
        emergency = torch.zeros_like(danger) if emergency is None else emergency.bool()
        if danger.shape != (batch,) or emergency.shape != (batch,):
            raise ValueError('Danger and emergency masks must have shape B')
        delta_t = torch.as_tensor(dt, device=latent.device, dtype=latent.dtype)
        if delta_t.ndim == 0:
            delta_t = delta_t.expand(batch)
        if delta_t.shape != (batch,) or not torch.isfinite(delta_t).all() or (delta_t <= 0).any():
            raise ValueError('dt must be positive scalar or B vector')
        nominal = nominal.clamp(-bound, bound)
        invalid = ~torch.isfinite(latent).all(-1)
        a = torch.nan_to_num(latent).tanh()
        lower, upper = -bound-nominal, bound-nominal
        lower = torch.where(danger[:, None], lower, torch.maximum(lower, -small))
        upper = torch.where(danger[:, None], upper, torch.minimum(upper, small))
        # a=0 gives zero residual even when the feasible interval is asymmetric.
        delta = torch.where(a >= 0, a*upper, -a*lower)
        desired = (nominal+delta).clamp(-bound, bound)
        step_limit = rate*delta_t[:, None]
        applied = torch.maximum(torch.minimum(desired, previous+step_limit), previous-step_limit).clamp(-bound, bound)
        stop = emergency | invalid
        # Explicit emergency command takes precedence over comfort slew limits.
        applied = torch.where(stop[:, None], torch.zeros_like(applied), applied)
        return {'command': applied, 'raw_residual': delta,
                'applied_residual': applied-nominal, 'emergency': stop,
                'limited': (applied-desired).abs().amax(-1) > 1e-6}
