"""Joint RTC-on capture / RTC-off labels with shared encoder features.

New labels explicitly distinguish full audits from structural validation.
This opt-in module does not change the existing teacher or offline trainer.
"""

from contextlib import contextmanager
import hashlib
import random
import time

import numpy as np
import torch

from FM_distillation.core.fm_data import save_record, sha256, teacher_tap, validate_label


@contextmanager
def cached_encoder_outputs(model, rgbd, goal):
    """Replace only forward calls; restore instance attributes even on failure."""
    saved = []
    try:
        for module, output in ((model.rgbd_encoder, rgbd), (model.point_encoder, goal)):
            saved.append((module, "forward" in module.__dict__, module.__dict__.get("forward")))
            module.forward = lambda *args, _output=output, **kwargs: _output
        yield
    finally:
        for module, existed, original in reversed(saved):
            if existed:
                module.forward = original
            else:
                del module.forward


def validate_joint_label(arrays, meta):
    if meta.get("joint_label_version") != 1:
        raise ValueError("not a joint label")
    audited = meta.get("verification") == "full"
    if meta.get("verification") not in ("full", "structural"):
        raise ValueError("unknown verification scope")
    if meta.get("tap_equivalence") is not audited or meta.get("q_recompute_verified") is not audited:
        raise ValueError("verification flags must describe actual per-record checks")
    # Reuse v1 shape/algebra/provenance checks, then apply the honest v2 audit
    # policy above. Do not persist synthetic verification flags to the record.
    validate_label(arrays, {**meta, "tap_equivalence": True, "q_recompute_verified": True})


def shared_label(model, obs, src, rgbd, goal, teacher_hash, observation_hash, seed, audit):
    if not model.rtc_enabled or model.training:
        raise ValueError("online teacher must be RTC-on and in eval mode")
    device = torch.device(model.device)
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    numpy_rng, python_rng = np.random.get_state(), random.getstate()
    try:
        with torch.random.fork_rng(devices=cuda_devices), cached_encoder_outputs(model, rgbd, goal):
            model.rtc_enabled = False
            torch.manual_seed(seed)
            initial = torch.randn(8,24,3,device=device)
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state(device) if cuda_devices else None
            def predict():
                torch.set_rng_state(cpu_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state(cuda_rng,device)
                with torch.no_grad():
                    return model.predict_pointgoal_action_with_guidance(
                        obs["pointgoal"][None],obs["rgb"][None],obs["depth"][None],8,
                        np.array([src["valid_segment_len"]]),obs["prev_action"][None],0,23,
                        np.array(src["guidance_factor"]),guidance_step=5,
                        embodiment=src["embodiment"],initial_noise=initial)
            reference = predict() if audit else None
            with teacher_tap(model) as tapped:
                actual = predict()
            if audit:
                for a,b in zip(reference[:3],actual[:3]):
                    np.testing.assert_allclose(a,b,atol=1e-6,rtol=1e-5)
                with torch.no_grad():
                    kwargs = dict(num_points=25,smooth_factor=.5,weight=model.weight)
                    torch.testing.assert_close(model.smooth_trajectory(tapped["raw"],**kwargs),tapped["smoothed_actions"])
                    path = model.smooth_cumulative_trajectory(tapped["raw"].div(4).cumsum(1),**kwargs)
                    torch.testing.assert_close(path,tapped["q_path"])
                    q1,q2 = model.predict_pointgoal_q(path,tapped["rgbd"],tapped["goal"],
                                                     is_target=False,embodiment=src["embodiment"])
                    torch.testing.assert_close(q1,tapped["q1"])
                    torch.testing.assert_close(q2,tapped["q2"])
            def cpu(x):
                return x.detach().cpu().numpy()
            q1,q2 = tapped["q1"],tapped["q2"]
            arrays = dict(raw_action_deltas=cpu(tapped["raw"]),smoothed_actions=cpu(tapped["smoothed_actions"]),
                q_path=cpu(tapped["q_path"]),trajectories=actual[0][0],scores=actual[1][0],
                q1=cpu(q1),q2=cpu(q2),top_indices=cpu((-(q1+q2)/2).argsort()[:2]),
                top_trajectories=actual[2][0],rgbd_embed=cpu(tapped["rgbd"])[0],goal_embed=cpu(tapped["goal"])[0],
                initial_noise=cpu(initial),sampler_cpu_rng_state=cpu(cpu_rng))
            if cuda_rng is not None:
                arrays["sampler_cuda_rng_state"] = cpu(cuda_rng)
    finally:
        model.rtc_enabled = True
        np.random.set_state(numpy_rng)
        random.setstate(python_rng)
    meta = {**src,"kind":"label","rtc_enabled":False,"candidates":8,"teacher_sha256":teacher_hash,
            "observation_sha256":observation_hash,"seed":seed,"target_space":"raw_pre_smoothing_action_deltas",
            "joint_label_version":1,"verification":"full" if audit else "structural",
            "tap_equivalence":bool(audit),"q_recompute_verified":bool(audit),"shared_online_encoding":True}
    validate_joint_label(arrays,meta)
    return arrays,meta


def joint_agent_class(run, manifest, audit_every=100):
    from FM_distillation.core.fm_capture_agent import recording_agent_class
    recording = recording_agent_class(run,manifest)
    class JointAgent(recording):
        def _save_observation(self,*args,**kwargs):
            super()._save_observation(*args,**kwargs)
            self._joint_observation = self._fm_pending

        def step_pointgoal_with_guidance(self,*args,**kwargs):
            encoded = {}
            def capture(name):
                def hook(module,inputs,output):
                    if name in encoded:
                        raise RuntimeError("encoder called more than once in online request")
                    encoded[name] = output.detach().clone()
                return hook
            model = self.navi_former
            handles = [model.rgbd_encoder.register_forward_hook(capture("rgbd")),
                       model.point_encoder.register_forward_hook(capture("goal"))]
            try:
                result = super().step_pointgoal_with_guidance(*args,**kwargs)
            finally:
                for handle in handles:
                    handle.remove()
            obs,src = self._joint_observation
            name = f'obs_{self._fm_episode:04d}_{src["step"]:06d}.npz'
            source = run/"observations"/name
            seed = int(hashlib.sha256(f'{manifest["seed"]}:{src["scene"]}:{src["episode_id"]}:{src["step"]}'.encode()).hexdigest()[:8],16)
            audit = src["step"] % audit_every == 0
            started = time.perf_counter()
            arrays,meta = shared_label(model,obs,src,encoded["rgbd"],encoded["goal"],
                                      manifest["teacher_sha256"],sha256(source),seed,audit)
            meta.update(observation=name,label_wall_s=time.perf_counter()-started,
                        audit_every=audit_every,online_result_unchanged=True)
            path = run/"labels"/name
            temporary = path.with_suffix(".pending")
            save_record(temporary,arrays,meta)
            temporary.replace(path)
            self._joint_observation = None
            return result
    return JointAgent
