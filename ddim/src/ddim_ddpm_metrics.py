"""Offline metrics for comparing diffusion samplers on identical observations.

Each sampled candidate is a 24-waypoint trajectory ``(x, y)`` in the robot's
local frame (the z column is always ~0 and is dropped). Q-values are the frozen
critic scores returned by the network, computed with the exact same critic and
trajectory preprocessing for every sampler.

Every metric is computed per observation; the caller (``compare_ddim_ddpm.py``)
aggregates them across observations and seeds.

Terminology
-----------
* ``matched`` -- Hungarian minimum-cost pairing between the two 8-candidate
  sets, then mean ADE/FDE over the matched pairs. Measures whether the candidate
  *sets* are similar.
* ``preferred`` -- ADE/FDE between the Q-argmax trajectory of each set. Measures
  whether the *final chosen behavior* is similar.
* ``diversity`` -- mean pairwise ADE within a set and dispersion of endpoints.
  Guards against "all 8 candidates collapsed into one".
"""

import rexnavdp  # noqa: F401  (sys.path bootstrap)

import numpy as np
from scipy.optimize import linear_sum_assignment

N_WAYPOINTS = 24


def as_2d(paths):
    """Drop the (always ~0) z column: [..., 24, 3] -> [..., 24, 2]."""
    paths = np.asarray(paths, dtype=np.float64)
    return paths[..., :2]


def _waypoint_dist(a, b):
    return np.linalg.norm(a - b, axis=-1)  # [24]


def ade(a, b):
    """Average Displacement Error: mean L2 over the 24 waypoints, in metres."""
    return float(np.mean(_waypoint_dist(a, b)))


def fde(a, b):
    """Final Displacement Error: L2 distance at the last waypoint, in metres."""
    return float(_waypoint_dist(a, b)[-1])


def pairwise_ade_matrix(A, B):
    """8x8 ADE matrix between two candidate sets (rows=A, cols=B)."""
    A, B = as_2d(A), as_2d(B)
    M = np.zeros((len(A), len(B)), dtype=np.float64)
    for i in range(len(A)):
        for j in range(len(B)):
            M[i, j] = ade(A[i], B[j])
    return M


def hungarian_pairing(A, B):
    """Min-cost perfect matching between two candidate sets.

    Returns ``(matched_ade, matched_fde)``: mean ADE and mean endpoint distance
    over the matched pairs.
    """
    A, B = as_2d(A), as_2d(B)
    M = pairwise_ade_matrix(A, B)
    rows, cols = linear_sum_assignment(M)
    matched_ade = float(np.mean(M[rows, cols]))
    matched_fde = float(np.mean([fde(A[i], B[j]) for i, j in zip(rows, cols)]))
    return matched_ade, matched_fde


def preferred_pairing(A, B, qa, qb):
    """ADE/FDE between the Q-argmax trajectories of each set.

    Returns ``(ade, fde, argmax_a, argmax_b)``.
    """
    A, B = as_2d(A), as_2d(B)
    ia, ib = int(np.argmax(qa)), int(np.argmax(qb))
    return ade(A[ia], B[ib]), fde(A[ia], B[ib]), ia, ib


def diversity(A):
    """Within-set candidate diversity.

    Returns ``(mean_pairwise_ade, endpoint_dispersion)``. Diversity is not "more
    is better"; a large drop for equal Q is evidence of distribution collapse.
    """
    A = as_2d(A)
    n = len(A)
    if n < 2:
        return 0.0, 0.0
    endpoints = A[:, -1, :]
    centroid = endpoints.mean(axis=0)
    disp = float(np.mean(np.linalg.norm(endpoints - centroid, axis=-1)))
    pairwise = [ade(A[i], A[j]) for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(pairwise)), disp


def q_stats(q):
    """Summary statistics of a Q-value vector."""
    q = np.asarray(q, dtype=np.float64)
    return {
        "mean": float(q.mean()),
        "max": float(q.max()),
        "min": float(q.min()),
        "q25": float(np.percentile(q, 25)),
        "argmax": int(np.argmax(q)),
    }


def compare_sets(ref, cand, q_ref, q_cand):
    """Compare a candidate set against a reference set for one observation.

    ``ref``/``cand`` are ``[8, 24, 3]`` arrays; ``q_*`` are ``[8]`` frozen critic
    values. Returns a flat dict of scalar metrics.
    """
    matched_ade, matched_fde = hungarian_pairing(ref, cand)
    pref_ade, pref_fde, _, _ = preferred_pairing(ref, cand, q_ref, q_cand)
    qr = q_stats(q_ref)
    qc = q_stats(q_cand)
    div_r = diversity(ref)
    div_c = diversity(cand)
    return {
        "matched_ade": matched_ade,
        "matched_fde": matched_fde,
        "preferred_ade": pref_ade,
        "preferred_fde": pref_fde,
        "q_ref_mean": qr["mean"],
        "q_cand_mean": qc["mean"],
        "q_ref_max": qr["max"],
        "q_cand_max": qc["max"],
        "q_ref_min": qr["min"],
        "q_cand_min": qc["min"],
        "q_ref_q25": qr["q25"],
        "q_cand_q25": qc["q25"],
        "q_max_diff": qc["max"] - qr["max"],
        "q_mean_diff": qc["mean"] - qr["mean"],
        "q_min_diff": qc["min"] - qr["min"],
        "q_win": float(qc["max"] > qr["max"]),
        "diversity_ref_pairwise_ade": div_r[0],
        "diversity_ref_endpoint_disp": div_r[1],
        "diversity_cand_pairwise_ade": div_c[0],
        "diversity_cand_endpoint_disp": div_c[1],
    }
