"""Behavior-cloning warm-start for the direct tracker.

Regresses the bounded physical command onto the MPC teacher (plan P4), then
PPO fine-tunes. The teacher is only used offline to build the dataset; runtime
and PPO never construct an MPC solver.
"""

import rexnavdp  # noqa: F401  (sys.path bootstrap)
import torch
import torch.nn.functional as F


def teacher_to_latent(teacher, max_v=0.5, max_w=0.5, eps=1e-3):
    """Invert the action map: teacher [v, w] -> latent z = atanh(u/bound)."""
    teacher = torch.as_tensor(teacher, dtype=torch.float32)
    bound = teacher.new_tensor([max_v, max_w])
    ratio = torch.clamp(teacher/bound, -1.+eps, 1.-eps)
    return torch.atanh(ratio)


def latent_to_command(latent, max_v=0.5, max_w=0.5):
    """Apply the direct policy's bounded command map without the runtime slew limit."""
    latent = torch.as_tensor(latent, dtype=torch.float32)
    bound = latent.new_tensor([max_v, max_w])
    return bound*torch.tanh(latent)


def _chunk_obs(episode, start, end, device):
    """Batched observation for a [start, end) chunk of an episode (K, 1, ...)."""
    return {'rgb_tokens': episode['rgb_tokens'][start:end].unsqueeze(1).to(device),
            'depth_tokens': episode['depth_tokens'][start:end].unsqueeze(1).to(device),
            'path': episode['path'][start:end].unsqueeze(1).to(device),
            'path_mask': episode['path_mask'][start:end].unsqueeze(1).to(device),
            'state': episode['state'][start:end].unsqueeze(1).to(device)}


def _zero_hidden(policy):
    device = next(policy.parameters()).device
    return torch.zeros(1, 1, policy.config.hidden_dim, device=device)


def _batch_chunks(chunks, device):
    """Stack equal-length episode chunks into the policy's [T,B,...] layout."""
    keys = ('rgb_tokens', 'depth_tokens', 'path', 'path_mask', 'state')
    obs = {key: torch.stack([episode[key][start:end] for episode, start, end in chunks], 1).to(device)
           for key in keys}
    teacher = torch.stack([episode['teacher'][start:end] for episode, start, end in chunks], 1).to(device)
    return obs, teacher


def train_bc(policy, dataset, optimizer, chunk=32, batch_size=8, device='cpu',
             clip=1.0, rng=None, loss_space='physical'):
    """One shuffled, batched truncated-BPTT pass; return command-space MSE.

    The MPC label is a bounded physical command. Regressing its inverse-tanh
    latent overweights labels close to the actuator limits and made the clutter
    policy collapse to its global mean. Command-space loss matches deployment.
    """
    if chunk < 1 or batch_size < 1 or loss_space not in ('physical', 'latent'):
        raise ValueError('Invalid BC chunk, batch size, or loss space')
    policy.train()
    rng = rng if rng is not None else __import__('numpy').random.RandomState()
    by_length = {}
    for episode in dataset:
        t_len = int(episode['teacher'].shape[0])
        for start in range(0, t_len, chunk):
            end = min(start+chunk, t_len)
            by_length.setdefault(end-start, []).append((episode, start, end))
    groups = list(by_length.values())
    rng.shuffle(groups)
    batches = []
    for group in groups:
        rng.shuffle(group)
        batches.extend(group[start:start+batch_size] for start in range(0, len(group), batch_size))
    rng.shuffle(batches)

    squared_error = 0.
    elements = 0
    for batch in batches:
        obs, teacher = _batch_chunks(batch, device)
        hidden = torch.zeros(1, len(batch), policy.config.hidden_dim, device=device)
        mean, _ = policy.bc_forward(obs, hidden)
        prediction = latent_to_command(mean)
        if loss_space == 'physical':
            loss = F.mse_loss(prediction, teacher)
        else:
            loss = F.mse_loss(mean, teacher_to_latent(teacher))
        optimizer.zero_grad()
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), clip)
        optimizer.step()
        squared_error += float(((prediction.detach()-teacher)**2).sum())
        elements += teacher.numel()
    return squared_error/max(1, elements)


@torch.no_grad()
def evaluate_bc(policy, dataset, device='cpu'):
    """Physical-command MSE over complete recurrent held-out episodes."""
    return evaluate_bc_metrics(policy, dataset, device)['physical_mse']


@torch.no_grad()
def evaluate_bc_metrics(policy, dataset, device='cpu'):
    """Return physical/latent MSE and collapse diagnostics."""
    policy.eval()
    physical_error = latent_error = elements = 0.
    predictions, targets = [], []
    for episode in dataset:
        t_len = int(episode['teacher'].shape[0])
        hidden = _zero_hidden(policy)
        for start in range(0, t_len, 64):
            end = min(start+64, t_len)
            mean, new_hidden = policy.bc_forward(_chunk_obs(episode, start, end, device), hidden)
            teacher = episode['teacher'][start:end].to(device).unsqueeze(1)
            command = latent_to_command(mean)
            physical_error += float(((command-teacher)**2).sum())
            latent_error += float(((mean-teacher_to_latent(teacher))**2).sum())
            elements += teacher.numel()
            predictions.append(command[:, 0].cpu())
            targets.append(teacher[:, 0].cpu())
            hidden = new_hidden
    prediction = torch.cat(predictions) if predictions else torch.empty(0, 2)
    target = torch.cat(targets) if targets else torch.empty(0, 2)
    correlation = []
    for axis in range(2):
        if len(prediction) < 2 or prediction[:, axis].std() < 1e-8 or target[:, axis].std() < 1e-8:
            correlation.append(0.)
        else:
            correlation.append(float(torch.corrcoef(torch.stack(
                (prediction[:, axis], target[:, axis])))[0, 1]))
    return {
        'physical_mse': physical_error/max(1, elements),
        'latent_mse': latent_error/max(1, elements),
        'prediction_std_v': float(prediction[:, 0].std()) if len(prediction) else 0.,
        'prediction_std_w': float(prediction[:, 1].std()) if len(prediction) else 0.,
        'correlation_v': correlation[0], 'correlation_w': correlation[1],
    }
