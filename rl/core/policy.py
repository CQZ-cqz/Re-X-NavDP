"""Current RGB-D token fusion + recurrent actor/critic with RSL-RL PPO API.

The DA backbone is deliberately outside this trainable module. Rollouts store
frozen tokens, never trainable fusion embeddings. Actions are latent Gaussian
z; tanh and physical residual mapping live in ActionComposer.
"""

import rexnavdp  # noqa: F401  (sys.path bootstrap)
from dataclasses import dataclass, asdict
import math
from typing import Optional
import torch
from torch import nn
from torch.distributions import Normal
from rsl_rl.networks import Memory


@dataclass(frozen=True)
class PolicyConfig:
    token_dim: int = 384
    grid_size: int = 8
    model_dim: int = 128
    path_points: int = 8
    state_dim: int = 28
    queries: int = 8
    heads: int = 4
    hidden_dim: int = 128
    init_std: float = 0.1
    # Per-axis initial Gaussian std and clamp (direct mode currently uses init_std_v=.15/w=.20,
    # clamp [.05,.8]; residual defaults reproduce the legacy exp(-5)..exp(1) clamp).
    init_std_v: float = 0.1
    init_std_w: float = 0.1
    std_min: float = math.exp(-5.)
    std_max: float = math.exp(1.)
    # None = unbounded actor mean (residual); direct mode sets 3.0 for mu=3*tanh(mu/3).
    mean_limit: Optional[float] = None

    def __post_init__(self):
        for key, value in asdict(self).items():
            if value is None:
                continue
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be positive')
        if self.model_dim % self.heads:
            raise ValueError('model_dim must be divisible by heads')


class RGBDFusion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.model_dim
        self.rgb_project = nn.Sequential(nn.LayerNorm(cfg.token_dim), nn.Linear(cfg.token_dim, d))
        self.depth_project = nn.Sequential(nn.LayerNorm(cfg.token_dim), nn.Linear(cfg.token_dim, d))
        # Modality identity plus explicit 2D patch position preserve spatial information.
        self.modality = nn.Parameter(torch.zeros(2, 1, d))
        y, x = torch.meshgrid(torch.linspace(-1, 1, cfg.grid_size), torch.linspace(-1, 1, cfg.grid_size), indexing='ij')
        self.register_buffer('positions', torch.stack((x.flatten(), y.flatten()), -1))
        self.position_project = nn.Linear(2, d, bias=False)
        self.path_net = nn.Sequential(nn.Linear(cfg.path_points*5, 128), nn.ELU(), nn.Linear(128, 64))
        self.state_net = nn.Sequential(nn.Linear(cfg.state_dim, 128), nn.ELU(), nn.Linear(128, 64))
        self.queries = nn.Parameter(torch.randn(cfg.queries, d)*.02)
        self.condition = nn.Linear(128, d)
        self.attention = nn.MultiheadAttention(d, cfg.heads, dropout=0., batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d*2), nn.GELU(), nn.Linear(d*2, d))
        self.norm2 = nn.LayerNorm(d)
        self.readout = nn.Linear(cfg.queries*d, 128)

    def forward(self, obs):
        cfg = self.cfg
        leading = obs['state'].shape[:-1]
        expected = (cfg.grid_size**2, cfg.token_dim)
        if obs['state'].shape[-1] != cfg.state_dim:
            raise ValueError('Incorrect state vector dimension')
        for name in ('rgb_tokens', 'depth_tokens'):
            if obs[name].shape != (*leading, *expected):
                raise ValueError(f'{name} shape does not match state batch/sequence')
        if obs['path'].shape != (*leading, cfg.path_points, 4) or obs['path_mask'].shape != (*leading, cfg.path_points):
            raise ValueError('Incorrect path/mask shape')
        rgb = obs['rgb_tokens'].reshape(-1, *expected).float()
        dep = obs['depth_tokens'].reshape(-1, *expected).float()
        position = self.position_project(self.positions)
        rgb = self.rgb_project(rgb) + position + self.modality[0]
        dep = self.depth_project(dep) + position + self.modality[1]
        tokens = torch.cat((rgb, dep), dim=1)  # B x 128 x 128
        mask = obs['path_mask'].reshape(-1, cfg.path_points, 1).float()
        path = obs['path'].reshape(-1, cfg.path_points, 4).float()*mask
        path_embed = self.path_net(torch.cat((path, mask), -1).flatten(1))
        state_embed = self.state_net(obs['state'].reshape(-1, cfg.state_dim).float())
        query = self.queries.unsqueeze(0) + self.condition(torch.cat((path_embed, state_embed), -1)).unsqueeze(1)
        attended, _ = self.attention(query, tokens, tokens, need_weights=False)
        fused = self.norm1(query + attended)
        fused = self.norm2(fused + self.ffn(fused))
        features = torch.cat((self.readout(fused.flatten(1)), path_embed, state_embed), -1)
        return features.reshape(*leading, 256)


class ReactiveActorCritic(nn.Module):
    is_recurrent = True
    def __init__(self, config=None):
        super().__init__()
        self.config = config or PolicyConfig()
        self.fusion = RGBDFusion(self.config)
        self.memory_a = Memory(256, self.config.hidden_dim, 1, 'gru')
        self.memory_c = Memory(256, self.config.hidden_dim, 1, 'gru')
        self.actor = nn.Sequential(nn.Linear(self.config.hidden_dim, 64), nn.ELU(), nn.Linear(64, 2))
        self.critic = nn.Sequential(nn.Linear(self.config.hidden_dim, 64), nn.ELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        self.log_std = nn.Parameter(torch.tensor([math.log(self.config.init_std_v), math.log(self.config.init_std_w)]))
        self.register_buffer('log_std_min', torch.tensor(math.log(self.config.std_min)))
        self.register_buffer('log_std_max', torch.tensor(math.log(self.config.std_max)))
        self.distribution = None

    def initialize_hidden(self, batch_size):
        device = next(self.parameters()).device
        for memory in (self.memory_a, self.memory_c):
            memory.hidden_state = torch.zeros(1, batch_size, self.config.hidden_dim, device=device)

    def _features(self, obs, memory, masks=None, hidden_state=None):
        result = memory(self.fusion(obs), masks, hidden_state)
        # Memory returns [1,B,H] online, [T,B,H] for PPO padded sequences.
        return result.squeeze(0) if masks is None else result

    def _mean(self, raw):
        if self.config.mean_limit is None:
            return raw
        return self.config.mean_limit*torch.tanh(raw/self.config.mean_limit)

    def act(self, obs, masks=None, hidden_state=None):
        features = self._features(obs, self.memory_a, masks, hidden_state)
        mean = self._mean(self.actor(features))
        std = self.log_std.clamp(self.log_std_min, self.log_std_max).exp().expand_as(mean)
        self.distribution = Normal(mean, std)
        return self.distribution.sample()

    def act_inference(self, obs):
        return self._mean(self.actor(self._features(obs, self.memory_a)))

    def bc_forward(self, obs, hidden_state):
        """Batched sequence forward for behavior cloning (BPTT).

        Returns ``(actor_mean [K,B,2], new_hidden [1,B,H])`` without touching the
        policy's stored memory, so the caller can carry the hidden state across
        truncated-BPTT chunks.
        """
        fused = self.fusion(obs)  # (K, B, 256)
        out, new_hidden = self.memory_a.rnn(fused, hidden_state)
        return self._mean(self.actor(out)), new_hidden

    def evaluate(self, obs, masks=None, hidden_state=None):
        return self.critic(self._features(obs, self.memory_c, masks, hidden_state))

    @torch.no_grad()
    def peek_value(self, obs):
        """Bootstrap V(next_obs) without advancing the rollout critic's memory twice."""
        previous = self.memory_c.hidden_state
        try:
            return self.evaluate(obs)
        finally:
            self.memory_c.hidden_state = previous

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(-1)

    def get_actions_log_prob(self, actions):
        log_prob = self.distribution.log_prob(actions).sum(-1)
        # RSL-RL 3.1.2 PPO squeezes ALL singleton axes of stored log-probs.
        # For recurrent B=1 batches, returning [T,1] would broadcast against
        # old [T] into [T,T], coupling unrelated transitions in the PPO ratio.
        return log_prob.squeeze() if actions.ndim == 3 else log_prob

    def get_hidden_states(self):
        return self.memory_a.hidden_state, self.memory_c.hidden_state

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def update_normalization(self, obs):
        # Physical scaling is deterministic and versioned in observation.py.
        pass
