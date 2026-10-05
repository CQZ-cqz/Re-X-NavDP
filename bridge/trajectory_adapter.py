"""Teacher trajectory conventions: raw deltas, Q path and execution path."""
import torch


def teacher_paths(teacher, raw):
    """raw [N,H,3]; preserve distinct original smoothing paths and /4 scaling."""
    if raw.ndim != 3 or raw.shape[-1] != 3 or not torch.isfinite(raw).all():
        raise ValueError("expected finite [N,H,3] raw action deltas")
    options = dict(num_points=raw.shape[1]+1, smooth_factor=.5, weight=teacher.weight)
    actions = teacher.smooth_trajectory(raw, **options)
    q_path = teacher.smooth_cumulative_trajectory(torch.cumsum(raw/4, 1), **options)
    return torch.cumsum(actions/4, 1), q_path
