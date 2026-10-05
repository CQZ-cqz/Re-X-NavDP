"""Load the original teacher without importing Isaac or changing its weights."""


def load_checkpoint(checkpoint, device="cpu", *, rtc_enabled=False):
    import torch
    from eval.src.policy_network_embodiment import NavDP_Policy_Embodiment

    teacher = NavDP_Policy_Embodiment(temporal_depth=16, device=device, rtc_enabled=rtc_enabled)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    keys = teacher.load_state_dict(state, strict=False)
    if keys.missing_keys:
        raise ValueError(f"missing teacher weights: {keys.missing_keys}")
    # Training-only checkpoint keys may be absent from the inference model.
    return teacher.to(device).eval().requires_grad_(False)
