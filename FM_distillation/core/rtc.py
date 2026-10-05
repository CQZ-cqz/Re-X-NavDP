"""Inference-only trajectory-prefix RTC adaptation for the existing FM student.

Flow guidance: https://arxiv.org/html/2506.07339v1 equations (2)-(4).
Uses X-NavDP's spatially aligned remaining raw deltas, XY prefix weights,
six strong/two weak candidates and stuck bypass; not fixed-duration action chunks.
No training/model/source fingerprint changes. beta is an experimental hyperparameter.
"""

import math


def sample_with_rtc(student, goal, rgbd, embodiment=0, *, candidates=8, steps=4,
                    initial_noise=None, prev_action, valid_segment_len,
                    guidance_factor, start_index=0, end_index=23,
                    guidance_step=5, prefix_attention_schedule="exp", beta=5., stats=None):
    import torch
    from eval.src.policy_network_embodiment import get_prefix_weights
    if student.training or steps < 1 or candidates < 1 or not math.isfinite(beta) or beta <= 0:
        raise ValueError("RTC requires eval mode, positive steps/candidates/beta")
    b, h = goal.shape[0], student.predict_size
    device, dtype = goal.device, goal.dtype
    y = torch.as_tensor(prev_action, device=device, dtype=dtype)
    valid = torch.as_tensor(valid_segment_len, device=device)
    factor = torch.as_tensor(guidance_factor, device=device, dtype=dtype)
    if y.shape != (b,h,3) or valid.shape != (b,):
        raise ValueError("RTC history must be [B,H,3], valid lengths [B]")
    if (not torch.isfinite(y).all() or not torch.isfinite(valid).all()
            or (valid < 0).any() or (valid > h).any() or (valid != valid.floor()).any()):
        raise ValueError("invalid RTC history/lengths")
    if factor.ndim == 0:
        factor = factor.expand(b,candidates)
    elif factor.shape == (candidates,):
        factor = factor[None].expand(b,-1)
    if factor.shape != (b,candidates) or not torch.isfinite(factor).all() or (factor < 0).any():
        raise ValueError("invalid RTC candidate factors")
    # Exact X-NavDP prefix function (end_index is unused in its pinv routine too).
    weights = get_prefix_weights(start_index, valid.repeat_interleave(candidates), h,
                                 prefix_attention_schedule, device=device).to(dtype)
    factor = factor.reshape(b*candidates,1,1)
    active = bool((weights*factor).ne(0).any()) and guidance_step >= 0
    if stats is not None:
        stats.update(active=active, guided_steps=0, beta=beta,
                     valid_lengths=valid.tolist(), candidate_factors=factor.flatten().tolist())
    if not active:
        return student.sample(goal,rgbd,embodiment,candidates=candidates,steps=steps,
                              initial_noise=initial_noise)
    shape = (b,candidates,h,3)
    if initial_noise is None:
        x = torch.randn(shape,device=device,dtype=dtype)
    else:
        if initial_noise.shape != shape:
            raise ValueError(f"initial_noise must have shape {shape}")
        x = initial_noise.to(goal).clone()
    x = x.reshape(b*candidates,h,3)
    y = y.repeat_interleave(candidates,0).detach()
    goal = goal.repeat_interleave(candidates,0).detach()
    rgbd = rgbd.repeat_interleave(candidates,0).detach()
    idx = student._embodiment(embodiment,b,device).repeat_interleave(candidates)
    corrections = []
    for i in range(steps):
        t = i/steps
        # Preserve late-guidance gating: legacy DDPM k<=5 becomes 9*(1-t)<=5.
        guided = 9*(1-t) <= guidance_step
        with torch.enable_grad() if guided else torch.no_grad():
            x = x.detach().requires_grad_(guided)
            velocity = student(x,t,goal,rgbd,idx)
            if guided:
                clean = x+(1-t)*velocity
                residual = ((y-clean)*weights).detach()
                correction = torch.autograd.grad(clean,x,grad_outputs=residual)[0].detach()
                # min(beta,(1-t)/(t*r^2)), r^2=(1-t)^2/(t^2+(1-t)^2).
                coefficient = beta if t == 0 else min(beta,(t*t+(1-t)**2)/(t*(1-t)))
                correction = coefficient*factor*correction
                velocity = velocity.detach()+correction
                corrections.append(correction.square().mean().sqrt().item())
            x = (x+velocity/steps).detach()
        if not torch.isfinite(x).all():
            raise RuntimeError("nonfinite RTC flow trajectory")
    if stats is not None:
        stats.update(guided_steps=len(corrections), correction_rms=corrections)
    return x.reshape(shape)


def fm_rtc_agent_class(student_path, teacher_sha256, beta=5.):
    import json
    import types
    import torch
    from eval.src.policy_agent import NavDP_Agent
    from FM_distillation.core.flow_generator import CompactFlowGenerator, generate_and_rank
    class FMAgent(NavDP_Agent):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            teacher = self.navi_former
            state = torch.load(student_path,map_location="cpu",weights_only=True)
            if state["signature"]["teacher_sha256"] != teacher_sha256:
                raise ValueError("student/teacher identity mismatch")
            student = CompactFlowGenerator(teacher,depth=4).to(self.device)
            student.load_state_dict(state["student"],strict=True)
            teacher.eval().requires_grad_(False)
            student.eval().requires_grad_(False)
            self.fm_student = student
            @torch.no_grad()
            def predict(network,goal_point,input_images,input_depths,sample_num=8,**kwargs):
                if sample_num != 8:
                    raise ValueError("expected eight candidates")
                goal = network.point_encoder(torch.as_tensor(goal_point,dtype=torch.float32,
                                                             device=network.device)).unsqueeze(1)
                rgbd = network.rgbd_encoder(input_images,input_depths)
                stats = {}
                class GuidedView:
                    training = False
                    _embodiment = student._embodiment
                    def sample(self,*args,**sampling):
                        guidance = {name: kwargs[name] for name in ("prev_action","valid_segment_len",
                            "guidance_factor","start_index","end_index","guidance_step",
                            "prefix_attention_schedule") if name in kwargs}
                        return sample_with_rtc(student,*args,**sampling,**guidance,beta=beta,stats=stats)
                result = generate_and_rank(network,GuidedView() if network.rtc_enabled else student,
                    goal,rgbd,kwargs.get("embodiment",0),steps=4,candidates=8)
                if network.rtc_enabled:
                    print("FM_RTC "+json.dumps(stats),flush=True)
                return tuple(result[k].cpu().numpy() for k in ("trajectories","scores","top_trajectories"))+(None,)
            teacher.predict_pointgoal_action_with_guidance = types.MethodType(predict,teacher)
    return FMAgent
