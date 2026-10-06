"""Frozen teacher backbone copied into FM so distilled inference is self-contained.

The distilled ``CompactFlowGenerator`` replaces only the diffusion decoder. Full
inference still needs the teacher's RGB-D encoder, goal encoder, dual-Q critic
and B-spline smoothing. These are copied from ``baselines/x-navdp/eval/src`` so a single
deploy checkpoint (student + this backbone) runs without the posttrain weights.
"""

import numpy as np
import torch
from torch import nn
from scipy.interpolate import splprep, splev

from rexnavdp import ROOT

import sys
_THIRD_PARTY_ROOT = ROOT / "third_party"
_DEPTH_ANYTHING_ROOT = _THIRD_PARTY_ROOT / "depth_anything"
for _p in (_THIRD_PARTY_ROOT, _DEPTH_ANYTHING_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional encoding for time embeddings (copied from policy_backbone)."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding (copied from policy_backbone)."""

    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.position_embedding = nn.Embedding(max_len, embed_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        return self.position_embedding(position_ids)


class NavDP_RGBD_Backbone(nn.Module):
    """RGB-D encoder using Depth Anything V2 (copied from policy_backbone)."""

    def __init__(self, image_size=224, embed_size=512, memory_size=8, device="cuda:0"):
        super().__init__()
        self.device = device
        self.memory_size = memory_size
        self.image_size = image_size
        self.embed_size = embed_size

        from depth_anything.depth_anything_v2.dpt import DepthAnythingV2
        model_configs = {"vits": {"encoder": "vits", "features": 64,
                                  "out_channels": [48, 96, 192, 384]}}
        self.rgb_model = DepthAnythingV2(**model_configs["vits"]).pretrained.float().eval()
        self.preprocess_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        self.preprocess_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

        self.depth_model = DepthAnythingV2(**model_configs["vits"]).pretrained.float().train()

        self.former_query = LearnablePositionalEncoding(384, self.memory_size * 16)
        self.former_pe = LearnablePositionalEncoding(384, (self.memory_size + 1) * 256)
        self.former_net = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(384, 8, batch_first=True), 2)
        self.project_layer = nn.Linear(384, embed_size)

    def forward(self, images, depths):
        with torch.no_grad():
            if len(images.shape) == 4:
                tensor_images = torch.as_tensor(images, dtype=torch.float32,
                    device=self.device).permute(0, 3, 1, 2)
                tensor_images = tensor_images.reshape(-1, 3, self.image_size, self.image_size)
                tensor_norm_images = (tensor_images - self.preprocess_mean.reshape(1, 3, 1, 1).to(self.device)) / \
                    self.preprocess_std.reshape(1, 3, 1, 1).to(self.device)
                image_token = self.rgb_model.get_intermediate_layers(tensor_norm_images)[0]
            else:
                tensor_images = torch.as_tensor(images, dtype=torch.float32,
                    device=self.device).permute(0, 1, 4, 2, 3)
                B, T, C, H, W = tensor_images.shape
                tensor_images = tensor_images.reshape(-1, 3, self.image_size, self.image_size)
                tensor_norm_images = (tensor_images - self.preprocess_mean.reshape(1, 3, 1, 1).to(self.device)) / \
                    self.preprocess_std.reshape(1, 3, 1, 1).to(self.device)
                image_token = self.rgb_model.get_intermediate_layers(tensor_norm_images)[0].reshape(B, T * 256, -1)

            if len(depths.shape) == 4:
                tensor_depths = torch.as_tensor(depths, dtype=torch.float32,
                    device=self.device).permute(0, 3, 1, 2)
                tensor_depths = tensor_depths.reshape(-1, 1, self.image_size, self.image_size)
                tensor_depths = torch.concat([tensor_depths, tensor_depths, tensor_depths], dim=1)
                depth_token = self.depth_model.get_intermediate_layers(tensor_depths)[0]
            else:
                tensor_depths = torch.as_tensor(depths, dtype=torch.float32,
                    device=self.device).permute(0, 1, 4, 2, 3)
                B, T, C, H, W = tensor_depths.shape
                tensor_depths = tensor_depths.reshape(-1, 1, self.image_size, self.image_size)
                tensor_depths = torch.concat([tensor_depths, tensor_depths, tensor_depths], dim=1)
                depth_token = self.depth_model.get_intermediate_layers(tensor_depths)[0].reshape(B, T * 256, -1)

            former_token = torch.concat((image_token, depth_token), dim=1) + \
                self.former_pe(torch.concat((image_token, depth_token), dim=1))
            former_query = self.former_query(
                torch.zeros((image_token.shape[0], self.memory_size * 16, 384), device=self.device))
            memory_token = self.former_net(former_query, former_token)
            return self.project_layer(memory_token)


class FrozenBackbone(nn.Module):
    """Frozen teacher subset: encoders + dual-Q critic + smoothing.

    Module names match ``NavDP_Policy_Embodiment`` so ``load_state_dict(strict=False)``
    can restore it from the teacher's own checkpoint. The diffusion decoder is
    intentionally omitted (the distilled student replaces it).
    """

    def __init__(self, image_size=224, memory_size=8, predict_size=24, temporal_depth=16,
                 heads=8, token_dim=384, device="cuda:0", distinguish_embodiment=True):
        super().__init__()
        self.device = device
        self.image_size = image_size
        self.memory_size = memory_size
        self.predict_size = predict_size
        self.temporal_depth = temporal_depth
        self.attention_heads = heads
        self.token_dim = token_dim
        self.distinguish_embodiment = distinguish_embodiment

        self.rgbd_encoder = NavDP_RGBD_Backbone(image_size, token_dim, memory_size=memory_size, device=device)
        self.point_encoder = nn.Linear(3, token_dim)

        # Conditioning modules copied into the student; kept here too for the Q critic.
        self.cond_pos_embed = LearnablePositionalEncoding(token_dim, memory_size * 16 + 4)
        self.out_pos_embed = LearnablePositionalEncoding(token_dim, predict_size)
        self.time_emb = SinusoidalPosEmb(token_dim)
        self.tgt_mask = (torch.triu(torch.ones(predict_size, predict_size)) == 1).transpose(0, 1)
        self.tgt_mask = self.tgt_mask.float().masked_fill(
            self.tgt_mask == 0, float("-inf")).masked_fill(self.tgt_mask == 1, float(0.0))

        self.embodiment_embedding = nn.Embedding(3, token_dim)
        self.embody_tgt_delta = nn.Sequential(
            nn.Linear(token_dim, token_dim), nn.GELU(), nn.Linear(token_dim, token_dim))
        nn.init.zeros_(self.embody_tgt_delta[2].weight)
        nn.init.zeros_(self.embody_tgt_delta[2].bias)
        self.embody_out_film = nn.Linear(token_dim, 2 * token_dim)
        nn.init.zeros_(self.embody_out_film.weight)
        nn.init.zeros_(self.embody_out_film.bias)

        self.cond_critic_mask_no_embod = torch.zeros((predict_size, 4 + memory_size * 16))
        self.cond_critic_mask_no_embod[:, 0:1] = float("-inf")
        self.cond_critic_mask_one_embod = torch.zeros((predict_size, 5 + memory_size * 16))
        self.cond_critic_mask_one_embod[:, 0:1] = float("-inf")

        self.decoder_q_layer = nn.TransformerDecoderLayer(
            d_model=token_dim, nhead=heads, dim_feedforward=4 * token_dim,
            activation="gelu", dropout=0.0, batch_first=True, norm_first=True)
        self.decoder_q = nn.TransformerDecoder(decoder_layer=self.decoder_q_layer, num_layers=temporal_depth)
        self.layernorm_q = nn.LayerNorm(token_dim)
        self.q_pool_mlp = nn.Sequential(
            nn.Linear(predict_size * token_dim, token_dim), nn.GELU(), nn.Linear(token_dim, token_dim))
        self.q1_heads = nn.Linear(token_dim, 1)
        self.q2_heads = nn.Linear(token_dim, 1)
        self.input_embed_q = nn.Linear(3, token_dim)

        self.weight = np.ones(self.predict_size + 1)
        self.weight[:9] = 5.0
        self.weight[:5] = 50.0
        self.weight[0] = 100.0

    def _align_embody_idx(self, embodiment, batch_size):
        embody_idx = torch.as_tensor(embodiment, device=self.device).long()
        if embody_idx.dim() == 0:
            embody_idx = embody_idx.expand(batch_size)
        elif embody_idx.shape[0] != batch_size:
            repeat = batch_size // embody_idx.shape[0]
            embody_idx = embody_idx.repeat_interleave(repeat)
        return embody_idx.clamp(0, self.embodiment_embedding.num_embeddings - 1)

    def _pool_critic_output(self, critic_output):
        return self.q_pool_mlp(critic_output.flatten(start_dim=1))

    def _q_heads_forward(self, pooled, is_target=False):
        heads_q1 = self.q1_heads
        heads_q2 = self.q2_heads
        return heads_q1(pooled).squeeze(-1), heads_q2(pooled).squeeze(-1)

    def predict_pointgoal_q(self, predict_trajectory, rgbd_embed, point_embed, is_target=False, embodiment=0):
        action_embeddings = self.input_embed_q(predict_trajectory)
        action_embeddings = action_embeddings + self.out_pos_embed(action_embeddings)
        cond_embeddings = torch.cat([point_embed, point_embed, point_embed, point_embed, rgbd_embed], dim=1) + \
            self.cond_pos_embed(torch.cat([point_embed, point_embed, point_embed, point_embed, rgbd_embed], dim=1))
        cond_critic_mask = self.cond_critic_mask_no_embod
        e = None
        if self.distinguish_embodiment:
            embody_idx = self._align_embody_idx(embodiment, cond_embeddings.shape[0])
            e = self.embodiment_embedding(embody_idx)
            cond_embeddings = torch.cat([cond_embeddings, e.unsqueeze(1)], dim=1)
            cond_critic_mask = self.cond_critic_mask_one_embod
            action_embeddings = action_embeddings + self.embody_tgt_delta(e).unsqueeze(1)
        critic_output = self.decoder_q(tgt=action_embeddings, memory=cond_embeddings,
                                       memory_mask=cond_critic_mask.to(self.device))
        critic_output = self.layernorm_q(critic_output)
        if e is not None:
            dgamma, dbeta = self.embody_out_film(e).chunk(2, dim=-1)
            critic_output = (1.0 + dgamma.unsqueeze(1)) * critic_output + dbeta.unsqueeze(1)
        pooled = self._pool_critic_output(critic_output)
        return self._q_heads_forward(pooled, is_target=is_target)

    def smooth_trajectory(self, points, num_points=24, smooth_factor=0.5, weight=None):
        batch = points.shape[0]
        points = torch.cumsum(points / 4.0, dim=1)
        points = torch.concat((torch.zeros_like(points[:, 0:1]), points), dim=1)
        points = points.cpu().numpy()
        data = []
        for i in range(batch):
            points_t = points[i, :, :2].T
            n_pts = points_t.shape[0]
            k = min(3, n_pts - 1)
            tck, u = splprep(points_t, w=weight, s=smooth_factor, k=k)
            u_new = np.linspace(0, 1, num_points)
            x_new, y_new = splev(u_new, tck)
            res = np.column_stack((x_new, y_new, points[i, :, 2:]))
            data.append((res[1:, :] - res[:-1, :]) * 4.0)
        return torch.tensor(np.stack(data, axis=0)).to(self.device).float()

    def smooth_cumulative_trajectory(self, points, num_points=24, smooth_factor=0.5, weight=None):
        batch = points.shape[0]
        points_with_zero = torch.concat((torch.zeros_like(points[:, 0:1]), points), dim=1)
        points_np = points_with_zero.cpu().numpy()
        data = []
        for i in range(batch):
            points_t = points_np[i, :, :2].T
            n_pts = points_t.shape[1]
            k = min(3, n_pts - 1)
            try:
                tck, u = splprep(points_t, w=weight, s=smooth_factor, k=k)
                u_new = np.linspace(0, 1, num_points)
                x_new, y_new = splev(u_new, tck)
                res = np.column_stack((x_new, y_new, points_np[i, :, 2:]))[1:]
            except ValueError:
                res = points_np[i, 1:, :]
            data.append(res)
        return torch.tensor(np.stack(data, axis=0), dtype=torch.float32, device=self.device)


def backbone_meta(teacher):
    """Capture the metadata needed to rebuild a FrozenBackbone from a teacher."""
    return dict(image_size=getattr(teacher, "image_size", 224),
                memory_size=teacher.memory_size,
                predict_size=teacher.predict_size,
                temporal_depth=getattr(teacher, "temporal_depth", 16),
                heads=getattr(teacher, "attention_heads", 8),
                token_dim=teacher.token_dim,
                distinguish_embodiment=teacher.distinguish_embodiment)


class FMPolicy(nn.Module):
    """Self-contained distilled policy: load one checkpoint, run full inference.

    Bundles the frozen teacher backbone (encoder + Q + smoothing) with the
    distilled ``CompactFlowGenerator`` so no separate posttrain weights are
    needed at inference time.
    """

    def __init__(self, checkpoint, device="cuda:0", depth=4, time_scale=9.0):
        super().__init__()
        from FM_distillation.src.flow_generator import CompactFlowGenerator, generate_and_rank
        self._generate_and_rank = generate_and_rank
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        meta = state["backbone_meta"]
        signature = state.get("signature") or {}
        depth = int(signature.get("depth", depth))
        time_scale = float(signature.get("time_scale", time_scale))
        backbone = FrozenBackbone(device=device, **meta)
        missing, unexpected = backbone.load_state_dict(state["teacher"], strict=False)
        if missing:
            raise ValueError(f"missing backbone weights: {missing}")
        self.backbone = backbone.eval().requires_grad_(False)
        student = CompactFlowGenerator(backbone, depth=depth, time_scale=time_scale)
        student.load_state_dict(state["student"], strict=True)
        self.student = student.eval().requires_grad_(False)
        self.to(device)

    @torch.no_grad()
    def predict(self, goal, rgb, depth, embodiment=0, *, candidates=8, steps=4,
                initial_noise=None):
        """Raw observation -> ranked trajectories/scores/top (numpy)."""
        device = self.backbone.device
        goal_t = torch.as_tensor(goal, dtype=torch.float32, device=device)
        if goal_t.dim() == 1:
            goal_t = goal_t[None]
        goal_embed = self.backbone.point_encoder(goal_t).unsqueeze(1)
        rgbd_embed = self.backbone.rgbd_encoder(rgb, depth)
        result = self._generate_and_rank(self.backbone, self.student, goal_embed, rgbd_embed,
                                         embodiment, candidates=candidates, steps=steps,
                                         initial_noise=initial_noise)
        return {k: v.cpu().numpy() for k, v in result.items() if torch.is_tensor(v)}


class DeployPolicy(FMPolicy):
    """FMPolicy exposed through the ``NavDP_Agent`` ``navi_former`` interface.

    The agent calls ``smooth_trajectory`` (guidance history) and
    ``predict_pointgoal_action_with_guidance`` (per-step planning) on
    ``navi_former``. This forwards those to the bundled frozen backbone, so a
    single ``deploy.pt`` drives the closed-loop eval without the posttrain
    weights.
    """

    def __init__(self, checkpoint, device="cuda:0", depth=4, time_scale=9.0,
                 rtc_enabled=False, beta=5.0):
        super().__init__(checkpoint, device=device, depth=depth, time_scale=time_scale)
        self.rtc_enabled = rtc_enabled
        self.beta = beta

    def smooth_trajectory(self, points, num_points=24, smooth_factor=0.5, weight=None):
        return self.backbone.smooth_trajectory(points, num_points=num_points,
                                               smooth_factor=smooth_factor, weight=weight)

    def smooth_cumulative_trajectory(self, points, num_points=24, smooth_factor=0.5, weight=None):
        return self.backbone.smooth_cumulative_trajectory(points, num_points=num_points,
                                                          smooth_factor=smooth_factor, weight=weight)

    @torch.no_grad()
    def predict_pointgoal_action_with_guidance(self, goal_point, input_images, input_depths,
                                               sample_num=8, **kwargs):
        """Mirror ``NavDP_Policy_Embodiment``'s ranked-planning return signature.

        Returns ``(all_trajectory, all_values, top_trajectory, None)`` numpy
        arrays; the fourth slot (visualization mask) is produced by the agent.
        """
        if sample_num != 8:
            raise ValueError("expected eight candidates")
        device = self.backbone.device
        goal_t = torch.as_tensor(goal_point, dtype=torch.float32, device=device)
        if goal_t.dim() == 1:
            goal_t = goal_t[None]
        goal = self.backbone.point_encoder(goal_t).unsqueeze(1)
        rgbd = self.backbone.rgbd_encoder(input_images, input_depths)
        embodiment = kwargs.get("embodiment", 0)
        sampler = self.student
        stats = {}
        if self.rtc_enabled:
            from FM_distillation.src.rtc import sample_with_rtc

            class GuidedView:
                training = False
                _embodiment = self.student._embodiment

                def sample(self, *args, **sampling):
                    guidance = {name: kwargs[name] for name in ("prev_action", "valid_segment_len",
                        "guidance_factor", "start_index", "end_index", "guidance_step",
                        "prefix_attention_schedule") if name in kwargs}
                    return sample_with_rtc(self.student, *args, **sampling, **guidance,
                                           beta=self.beta, stats=stats)

            sampler = GuidedView()
        result = self._generate_and_rank(self.backbone, sampler, goal, rgbd, embodiment,
                                         candidates=8, steps=4)
        if self.rtc_enabled:
            import json
            print("FM_RTC " + json.dumps(stats), flush=True)
        return (result["trajectories"].cpu().numpy(), result["scores"].cpu().numpy(),
                result["top_trajectories"].cpu().numpy(), None)


def fm_deploy_agent_class(deploy_path, rtc_enabled=False, beta=5.0):
    """Build a ``NavDP_Agent`` subclass that loads one self-contained deploy.pt."""
    from eval.src.policy_agent import NavDP_Agent

    class FMDeployAgent(NavDP_Agent):
        def _build_navi_former(self, navi_model, **kwargs):
            return DeployPolicy(deploy_path, device=self.device,
                                rtc_enabled=rtc_enabled, beta=beta)

    return FMDeployAgent


def export_deploy_checkpoint(student, teacher_state, meta, signature, path):
    """Write a single self-contained deploy checkpoint (student + frozen backbone).

    ``teacher_state`` and ``meta`` are captured at training time (the full teacher
    object is deleted afterwards to free memory); pass them here explicitly.
    """
    import os
    import torch
    payload = dict(student=student.state_dict(),
                   teacher=teacher_state,
                   backbone_meta=meta,
                   signature=signature)
    temporary = str(path) + ".pending"
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path
