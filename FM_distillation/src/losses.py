"""FM training loss objectives: latent-aligned CFM and distribution matching.

Kept separate from ``training.py`` so the pipeline stays readable. All functions
take already-loaded batch dicts / tensors; none import the training pipeline.
"""

import math

import torch


def candidate_errors(model, inputs, noise, t):
    """Unreduced CFM objective, one scalar per flattened candidate."""
    target = inputs["action_deltas"].detach()
    x = (1 - t[:, None, None]) * noise + t[:, None, None] * target
    predicted = model(x, t, inputs["goal_embed"].detach(),
                      inputs["rgbd_embed"].detach(), inputs["embodiment"])
    return (predicted - (target - noise)).square().mean(dim=(-1, -2))


def all_candidate_loss(model, data, device, rng, chunk_size, *, backward=False):
    """Exact mean over B*K, including a short final chunk; one optimizer step outside.

    All random draws happen before chunking so changing chunk size preserves pairs.
    Only each chunk's repeated conditions are moved to the GPU.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    b, k, h, d = data["action_deltas"].shape
    n = b * k
    targets = data["action_deltas"].reshape(n, h, d)
    if "initial_noise" in data:
        # Latent alignment: pair each teacher trajectory with the exact initial
        # noise that generated it (stored in the label), preserving mode identity.
        noise = data["initial_noise"].reshape(n, h, d)
    else:
        noise = torch.randn(n, h, d, generator=rng)
    t = torch.rand(n, generator=rng)
    errors = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        owners = torch.arange(start, end) // k
        inputs = {key: value[owners].to(device) for key, value in data.items()
                  if key != "action_deltas"}
        inputs["action_deltas"] = targets[start:end].to(device)
        per_candidate = candidate_errors(model, inputs, noise[start:end].to(device),
                                         t[start:end].to(device))
        if not torch.isfinite(per_candidate).all():
            raise ValueError("nonfinite candidate loss")
        if backward:
            (per_candidate.sum() / n).backward()
        errors.append(per_candidate.detach().cpu())
    return torch.cat(errors).reshape(b, k), t.reshape(b, k)


def pairwise_diversity(deltas):
    """Mean pairwise cumulative-XY distance across candidates. [B,K,24,3] -> [B]."""
    path = deltas.cumsum(-2) / 4
    matrix = (path[:, :, None, :, :2] - path[:, None, :, :, :2]).norm(dim=-1).mean(-1)
    k = deltas.shape[1]
    mask = torch.triu(torch.ones(k, k, dtype=torch.bool, device=deltas.device), diagonal=1)
    return matrix[:, mask].mean(-1)


def distribution_loss(model, data, device, *, candidates=8, steps=4,
                      lambda_mu=0.1, lambda_sigma=0.5):
    """Symmetric 2nd-order distribution matching on the cumulative path.

    Matches the per-candidate mean (location) and per-waypoint standard deviation
    (spread), both symmetric so the student is pulled back from over-spreading as
    well as from mode collapse. Differentiable through ``model.sample_with_grad``.
    """
    samples = model.sample_with_grad(data["goal_embed"].to(device), data["rgbd_embed"].to(device),
                                     data["embodiment"].to(device), candidates=candidates, steps=steps)
    p_s = samples.cumsum(-2) / 4
    p_t = data["action_deltas"].to(device).cumsum(-2) / 4
    mu_s, mu_t = p_s.mean(1), p_t.mean(1)
    std_s, std_t = p_s.std(1, unbiased=False), p_t.std(1, unbiased=False)
    return lambda_mu * (mu_s - mu_t).square().mean() + lambda_sigma * (std_s - std_t).square().mean()


def entropic_ot(C, eps, n_iter=20):
    """Full entropy-regularized OT value for [B, Ks, Kt] cost matrices, uniform marginals.

    Returns ``<P, C> + eps * <P, log P>`` per batch — the complete objective of the
    entropic OT problem, not just the transport term. Differentiable through ``C``; at
    the Sinkhorn fixed point the gradient w.r.t. ``C`` is the optimal plan ``P``
    (envelope theorem), so keeping the entropy term makes the gradient the standard
    entropic-OT one. ``eps`` shares the units of ``C`` and sets the matching softness.
    """
    B, Ks, Kt = C.shape
    if eps <= 0 or n_iter < 1:
        raise ValueError("eps must be positive and n_iter positive")
    log_K = -C / eps
    log_a = C.new_full((B, Ks), -math.log(Ks))
    log_b = C.new_full((B, Kt), -math.log(Kt))
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    for _ in range(n_iter):
        log_u = log_a - torch.logsumexp(log_K + log_v[:, None, :], dim=2)
        log_v = log_b - torch.logsumexp(log_K + log_u[:, :, None], dim=1)
    log_P = log_u[:, :, None] + log_K + log_v[:, None, :]
    P = torch.exp(log_P)
    transport = (P * C).sum(dim=(1, 2))
    entropy = (P * log_P).sum(dim=(1, 2))
    return transport + eps * entropy


def sinkhorn_divergence(s, t, eps, n_iter=20):
    """Debiased Sinkhorn divergence between trajectories ``s`` [B,Ks,W,D] and ``t`` [B,Kt,W,D].

    ``OT_eps(s,t) - 0.5 OT_eps(s,s) - 0.5 OT_eps(t,t)`` — zero when the two sets
    coincide and non-negative otherwise, so it matches the full multimodal
    distribution instead of doing soft assignment with an entropic bias. Gradient
    flows to ``s`` (student) via both the cross- and self-transport plans; ``t`` is fixed.
    """
    C_st = (s[:, :, None] - t[:, None]).norm(dim=-1).mean(dim=-1)
    C_ss = (s[:, :, None] - s[:, None]).norm(dim=-1).mean(dim=-1)
    C_tt = (t[:, :, None] - t[:, None]).norm(dim=-1).mean(dim=-1)
    return (entropic_ot(C_st, eps, n_iter)
            - 0.5 * entropic_ot(C_ss, eps, n_iter)
            - 0.5 * entropic_ot(C_tt, eps, n_iter))


def sinkhorn_loss(model, data, device, *, candidates=8, steps=4, eps=0.1, n_iter=20):
    """Sinkhorn divergence between on-policy student samples and teacher candidates.

    Unlike mean+std matching this sees the whole multimodal structure, and the
    debiased divergence is minimized exactly at ``p_student = p_teacher`` (no
    entropic bias toward spread). Training-only; inference unchanged. Differentiable
    through ``model.sample_with_grad``.
    """
    samples = model.sample_with_grad(data["goal_embed"].to(device), data["rgbd_embed"].to(device),
                                     data["embodiment"].to(device), candidates=candidates, steps=steps)
    s = samples.cumsum(-2)[..., :2] / 4
    t = data["action_deltas"].to(device).cumsum(-2)[..., :2] / 4
    return sinkhorn_divergence(s, t, eps, n_iter).mean()


def loss_diagnostics(errors, times, actions):
    """Distribution and overlapping motion/time groups, NOT fixed-slot modes.

    Motion groups use teacher cumulative XY endpoint, not collision/recovery labels.
    Empty groups carry count=0 and mean=null, never an artificial zero loss.
    """
    values = errors.detach().float().cpu()
    endpoint = actions.detach().cpu().sum(dim=-2)[..., :2] / 4
    angle = torch.atan2(endpoint[..., 1], endpoint[..., 0]).abs()
    masks = {"backward_endpoint": endpoint[..., 0] < 0,
             "nonbackward_endpoint": endpoint[..., 0] >= 0,
             "large_endpoint_angle_gt60deg": (angle > math.pi / 3) & (endpoint.norm(dim=-1) > .1)}
    for i in range(4):
        masks[f"time_quartile_{i}"] = (times >= i / 4) & (times < (i + 1) / 4)
    groups = {}
    for name, mask in masks.items():
        subset = values[mask]
        groups[name] = dict(count=subset.numel(), mean=subset.mean().item() if subset.numel() else None)
    return dict(count=values.numel(), mean=values.mean().item(),
                p50=values.quantile(.5).item(), p90=values.quantile(.9).item(),
                maximum=values.max().item(), mean_observation_max=values.max(dim=1).values.mean().item(),
                groups=groups)
