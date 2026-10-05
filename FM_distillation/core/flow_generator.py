"""Experimental conditional FM generator; not enabled in the policy server.

Noise-to-data convention: x(t)=(1-t)*noise+t*raw_action_deltas, v=target-noise.
Targets are BEFORE teacher smoothing and BEFORE cumsum(action / 4). They are
not the public policy's returned cumulative trajectories. No RTC is applied.
"""

import rexnavdp  # noqa: F401  (sys.path bootstrap)
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F


class CompactFlowGenerator(nn.Module):
    """Keep teacher conditioning width/injection, shrink decoder depth only.

    The caller supplies frozen teacher RGB-D and point-encoder features. Small
    conditioning modules are copied (not shared) and frozen. The velocity
    decoder/head are newly initialized; this is NOT a trained distilled model.
    Continuous FM time uses the teacher sinusoidal function with scale 9.
    """

    def __init__(self, teacher, depth=4, time_scale=9.0):
        super().__init__()
        if depth < 1 or time_scale <= 0:
            raise ValueError("positive depth and time_scale required")
        self.token_dim = teacher.token_dim
        self.predict_size = teacher.predict_size
        self.memory_tokens = teacher.memory_size * 16
        self.distinguish_embodiment = teacher.distinguish_embodiment
        self.time_scale = float(time_scale)
        self.depth = depth
        self.condition_names = (
            "cond_pos_embed", "out_pos_embed", "time_emb",
            "embodiment_embedding", "embody_tgt_delta", "embody_out_film")
        for name in self.condition_names:
            module = deepcopy(getattr(teacher, name))
            module.requires_grad_(False)
            setattr(self, name, module)
        self.input_embed = nn.Linear(3, self.token_dim)
        layer = nn.TransformerDecoderLayer(
            self.token_dim, teacher.attention_heads, 4*self.token_dim,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, depth)
        # PyTorch clones layers with identical initial weights: randomize each.
        for block in self.decoder.layers:
            for parameter in block.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
        self.layernorm = nn.LayerNorm(self.token_dim)
        self.velocity_head = nn.Linear(self.token_dim, 3)
        self.register_buffer("tgt_mask", teacher.tgt_mask.detach().clone())
        reference = next(teacher.point_encoder.parameters())
        self.to(device=reference.device, dtype=reference.dtype)

    def _embodiment(self, embodiment, batch, device):
        idx = torch.as_tensor(embodiment, device=device)
        if idx.dtype.is_floating_point or idx.dtype == torch.bool:
            raise ValueError("embodiment must contain integer IDs")
        idx = idx.long()
        if idx.ndim == 0:
            idx = idx.expand(batch)
        if idx.shape != (batch,):
            raise ValueError("embodiment must be scalar or [B]")
        if ((idx < 0) | (idx >= self.embodiment_embedding.num_embeddings)).any():
            raise ValueError("embodiment ID out of range")
        return idx

    def forward(self, x, t, goal_embed, rgbd_embed, embodiment=0):
        """[B,24,3], [B], [B,1,384], [B,128,384] -> velocity [B,24,3]."""
        batch = x.shape[0]
        if x.shape != (batch, self.predict_size, 3):
            raise ValueError("invalid action-delta shape")
        if goal_embed.shape != (batch, 1, self.token_dim):
            raise ValueError("invalid goal feature shape")
        if rgbd_embed.shape != (batch, self.memory_tokens, self.token_dim):
            raise ValueError("invalid RGB-D feature shape")
        t = torch.as_tensor(t, device=x.device, dtype=x.dtype)
        if t.ndim == 0:
            t = t.expand(batch)
        if t.shape != (batch,):
            raise ValueError("time must be scalar or [B]")
        time = self.time_emb(t * self.time_scale).to(x.dtype).unsqueeze(1)
        memory = torch.cat([time, goal_embed, goal_embed, goal_embed, rgbd_embed], 1)
        memory = memory + self.cond_pos_embed(memory)
        action = self.input_embed(x)
        action = action + self.out_pos_embed(action)
        e = None
        if self.distinguish_embodiment:
            e = self.embodiment_embedding(self._embodiment(embodiment, batch, x.device))
            memory = torch.cat([memory, e.unsqueeze(1)], 1)
            action = action + self.embody_tgt_delta(e).unsqueeze(1)
        output = self.layernorm(self.decoder(action, memory, tgt_mask=self.tgt_mask))
        if e is not None:
            gamma, beta = self.embody_out_film(e).chunk(2, -1)
            output = (1 + gamma.unsqueeze(1)) * output + beta.unsqueeze(1)
        return self.velocity_head(output)

    def flow_loss(self, action_deltas, goal_embed, rgbd_embed, embodiment=0,
                  *, noise=None, t=None):
        """Conditional flow matching MSE; teacher targets/features detached.

        Each row is one candidate, never the average of incompatible paths.
        Candidate weighting/filtering belongs to the future data pipeline.
        """
        target = action_deltas.detach()
        noise = torch.randn_like(target) if noise is None else noise.detach()
        if noise.shape != target.shape:
            raise ValueError("noise shape must match target")
        if t is None:
            t = torch.rand(target.shape[0], device=target.device, dtype=target.dtype)
        t = torch.as_tensor(t, device=target.device, dtype=target.dtype)
        if t.shape != (target.shape[0],) or not torch.isfinite(t).all() or ((t < 0) | (t > 1)).any():
            raise ValueError("training time must be finite [B] in [0,1]")
        x = (1-t[:, None, None])*noise + t[:, None, None]*target
        prediction = self(x, t, goal_embed.detach(), rgbd_embed.detach(), embodiment)
        return F.mse_loss(prediction, target-noise)

    @torch.no_grad()
    def sample(self, goal_embed, rgbd_embed, embodiment=0, *, candidates=8,
               steps=4, initial_noise=None):
        """Euler integrate 0 -> 1. Returns RAW deltas [B,K,24,3], no clamp/RTC."""
        if self.training:
            raise RuntimeError("call eval() before sampling")
        if not isinstance(steps, int) or not isinstance(candidates, int) or min(steps, candidates) < 1:
            raise ValueError("steps/candidates must be positive integers")
        batch = goal_embed.shape[0]
        shape = (batch, candidates, self.predict_size, 3)
        if initial_noise is None:
            x = torch.randn(shape, device=goal_embed.device, dtype=goal_embed.dtype)
        else:
            if initial_noise.shape != shape:
                raise ValueError(f"initial_noise must have shape {shape}")
            x = initial_noise.to(goal_embed).clone()
        x = x.reshape(batch*candidates, self.predict_size, 3)
        goal = goal_embed.repeat_interleave(candidates, 0)
        rgbd = rgbd_embed.repeat_interleave(candidates, 0)
        idx = self._embodiment(embodiment, batch, x.device).repeat_interleave(candidates)
        for i in range(steps):
            x = x + self(x, i/steps, goal, rgbd, idx)/steps
        return x.reshape(shape)


@torch.no_grad()
def generate_and_rank(teacher, student, goal_embed, rgbd_embed, embodiment=0,
                      *, candidates=8, steps=4, initial_noise=None):
    """Experimental adapter preserving teacher smoothing, dual-Q mean, top-2.

    Does not invoke teacher diffusion actors, encode images, or run recovery/MPC.
    Teacher must be the same checkpoint whose conditions initialized student.
    Keeping the whole teacher resident here is a prototype, not deployment export.
    """
    if teacher.training or student.training:
        raise RuntimeError("teacher and student must both be in eval mode")
    if candidates < 2:
        raise ValueError("top-2 ranking requires at least two candidates")
    raw = student.sample(goal_embed, rgbd_embed, embodiment, candidates=candidates,
                         steps=steps, initial_noise=initial_noise)
    batch, count, horizon, _ = raw.shape
    flat = raw.reshape(batch*count, horizon, 3)
    if not torch.isfinite(flat).all():
        raise RuntimeError("nonfinite FM output")
    from bridge.trajectory_adapter import teacher_paths
    from bridge.q_evaluator import mean_q
    execution_path, path_for_q = teacher_paths(teacher, flat)
    goal = goal_embed.repeat_interleave(count, 0)
    rgbd = rgbd_embed.repeat_interleave(count, 0)
    idx = student._embodiment(embodiment, batch, flat.device).repeat_interleave(count)
    scores = mean_q(teacher, path_for_q, rgbd, goal, idx).reshape(batch, count)
    trajectories = execution_path.reshape(batch, count, horizon, 3)
    if not torch.isfinite(scores).all() or not torch.isfinite(trajectories).all():
        raise RuntimeError("nonfinite ranked output")
    indices = (-scores).argsort(1)[:, :2]
    best = trajectories[torch.arange(batch, device=flat.device)[:, None], indices]
    return dict(raw_action_deltas=raw, trajectories=trajectories, scores=scores,
                top_indices=indices, top_trajectories=best)
