"""RSL-RL PPO adapter with explicit terminal-observation bootstrap semantics."""

import rexnavdp  # noqa: F401  (sys.path bootstrap)
import torch
from rsl_rl.algorithms import PPO


class ReactivePPO(PPO):
    def process_transition(self, next_obs, rewards, terminated, truncated, terminal_obs=None):
        terminated, truncated = terminated.bool(), truncated.bool()
        timeout = truncated & ~terminated
        adjusted = rewards.clone()
        if timeout.any():
            if terminal_obs is None:
                raise ValueError('Timeout bootstrap requires pre-reset terminal observations')
            with torch.no_grad():
                terminal_value = self.policy.peek_value(terminal_obs).squeeze(-1)
            adjusted += self.gamma*terminal_value*timeout.float()
        # Upstream process_env_step bootstraps V(current) for time_outs.
        # We supply the correct V(terminal next_obs) above and omit that flag.
        super().process_env_step(next_obs, adjusted, terminated | truncated, {})

    def compute_returns(self, obs):
        with torch.no_grad():
            last_values = self.policy.peek_value(obs)
        self.storage.compute_returns(last_values, self.gamma, self.lam,
            normalize_advantage=not self.normalize_advantage_per_mini_batch)
