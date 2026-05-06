"""CLS-structured reward with PCDR operators replacing 3 of 7 gates.

Drop-in replacement for nuplan_reward_cls_nocomfort.compute_cls_reward.

PCDR-replaced operators (forward-equivalent + closed-form gradients):
  NC  — OBB SAT smooth-min vs the previous center-distance > 3 m proxy
  EP  — Frenet projection with implicit-function-theorem gradient
        vs the previous path-length-ratio proxy
  DDC — sliding-window negative-progress threshold
        vs the previous heading-vs-expert proxy

Inherited proxies (unchanged from nocomfort):
  DAC, MP, TTC, SC, C
"""
from __future__ import annotations

import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from pcdr_operators import (
    pcdr_nc_gate,
    pcdr_ep_gate,
    pcdr_ddc_gate,
    pcdr_sc_gate,
    pcdr_dac_corridor_gate,
    build_route_baseline,
    heading_from_motion,
)


# Pacifica vehicle dims (matches diffusion_planner/model/guidance/collision.py).
PACIFICA_LENGTH = 4.768
PACIFICA_WIDTH = 1.951
DT = 0.1

# PCDR smoothing temperatures.
PCDR_NC_BETA = 0.05    # meters; bias bound on smooth signed distance ≤ β log 8 = 0.104 m
PCDR_PROJ_BETA = 0.5   # meters; segment-selection softmin temperature for EP/DDC


def ste_gate(value: torch.Tensor, threshold: float, steepness: float = 20.0,
              pass_high: bool = True) -> torch.Tensor:
    """Straight-through estimator gate (copy of nocomfort.ste_gate)."""
    soft = (torch.sigmoid((value - threshold) * steepness) if pass_high
            else torch.sigmoid((threshold - value) * steepness))
    hard = (soft > 0.5).float()
    return hard + (soft - soft.detach())


def compute_cls_reward(ego_traj: torch.Tensor, data: dict) -> torch.Tensor:
    """PCDR-augmented CLS reward.

    ego_traj: (T, >=2) trajectory with xy in cols 0:2.
    data:     npz dict with ego_agent_future, neighbor_agents_future,
              neighbor_agents_past, route_lanes.

    Returns scalar reward in approximately [0, 1].
    """
    device = ego_traj.device
    dtype = ego_traj.dtype

    def _zero():
        return torch.tensor(0.0, device=device, dtype=dtype)

    expert = data.get('ego_agent_future')
    nbrs = data.get('neighbor_agents_future')
    nbrs_past = data.get('neighbor_agents_past')
    route = data.get('route_lanes')
    if expert is None or nbrs is None or route is None:
        return _zero()

    if expert.dim() == 3: expert = expert[0]
    if nbrs.dim() == 4: nbrs = nbrs[0]
    if nbrs_past is not None and nbrs_past.dim() == 4: nbrs_past = nbrs_past[0]
    if route.dim() == 4: route = route[0]

    ego_xy = ego_traj[:, :2]
    T = ego_xy.shape[0]
    expert_xy = expert[:, :2]

    ego_d = torch.diff(ego_xy, dim=0)
    ego_v = torch.norm(ego_d, dim=-1) / DT
    ego_a = torch.diff(ego_v, dim=0) / DT
    ego_jerk = torch.diff(ego_a, dim=0) / DT

    # ── L1: MP (proxy: progress ratio ≥ 30%) ──
    exp_d = torch.diff(expert_xy, dim=0)
    expert_progress = torch.norm(exp_d, dim=-1).sum().clamp(min=2.0)
    ego_progress_path = torch.norm(ego_d, dim=-1).sum()
    progress_ratio = ego_progress_path / expert_progress
    MP = ste_gate(progress_ratio, threshold=0.30, steepness=5.0, pass_high=True)

    # ── L2: NC — PCDR (OBB SAT smooth-min) ──
    nbr_xy = nbrs[:, :T, :2]
    nbr_valid = (nbr_xy.abs().sum(dim=-1) > 1e-6)
    if nbrs.shape[-1] >= 3:
        nbr_h = nbrs[:, :T, 2]
    else:
        nbr_h = torch.zeros_like(nbr_xy[..., 0])
    nbr_cos_h = torch.cos(nbr_h)
    nbr_sin_h = torch.sin(nbr_h)
    if nbrs_past is not None and nbrs_past.shape[-1] >= 8:
        # neighbor_agents_past layout: cols 6=W, 7=L (last past frame is index -1)
        nbr_W = nbrs_past[:, -1, 6]
        nbr_L = nbrs_past[:, -1, 7]
        # Filter out absent neighbors (zero W/L).
        nbr_size_ok = (nbr_W > 1e-3) & (nbr_L > 1e-3)
        nbr_W = torch.where(nbr_size_ok, nbr_W,
                            nbr_W.new_full((), 0.5))                       # tiny default
        nbr_L = torch.where(nbr_size_ok, nbr_L,
                            nbr_L.new_full((), 0.5))
    else:
        # Fallback: assume Pacifica-sized neighbors.
        N = nbr_xy.shape[0]
        nbr_W = torch.full((N,), PACIFICA_WIDTH, device=device, dtype=dtype)
        nbr_L = torch.full((N,), PACIFICA_LENGTH, device=device, dtype=dtype)

    # Ego heading from motion (matches collision.py guidance choice).
    ego_cos_h, ego_sin_h = heading_from_motion(ego_xy)
    NC = pcdr_nc_gate(
        ego_xy, ego_cos_h, ego_sin_h, PACIFICA_LENGTH, PACIFICA_WIDTH,
        nbr_xy, nbr_cos_h, nbr_sin_h, nbr_L, nbr_W, nbr_valid,
        beta=PCDR_NC_BETA, ste=True,
    )

    # ── L3: DAC — PCDR corridor (lane-edge based, replaces lane-distance proxy) ──
    if route.shape[-1] >= 8:
        DAC = pcdr_dac_corridor_gate(
            ego_xy, ego_cos_h, ego_sin_h, PACIFICA_LENGTH, PACIFICA_WIDTH,
            route[..., :8], beta=PCDR_PROJ_BETA, steepness=5.0,
        )
    else:
        # Fallback to proxy if route_lanes lacks edge channels.
        route_xy = route[:, :, :2].reshape(-1, 2)
        route_valid = (route_xy.abs().sum(dim=-1) > 1e-6)
        route_xy_flat = route_xy[route_valid]
        if route_xy_flat.shape[0] == 0:
            DAC = torch.tensor(1.0, device=device, dtype=dtype)
        else:
            d_ego = torch.cdist(ego_xy, route_xy_flat).min(dim=-1).values
            d_exp = torch.cdist(expert_xy, route_xy_flat).min(dim=-1).values
            DAC = ste_gate(d_ego.max(),
                            threshold=d_exp.max().detach() + 0.3,
                            steepness=3.0, pass_high=False)

    # ── L4 & EP/DDC: PCDR Frenet projection ──
    polyline_xy = route[:, :, :2]                                         # (L, P, 2)
    polyline, arc_length = build_route_baseline(polyline_xy, expert_xy=expert_xy)
    if polyline is not None and polyline.shape[0] >= 2:
        ep_gate, _, _ = pcdr_ep_gate(
            ego_xy, expert_xy, polyline, arc_length,
            beta=PCDR_PROJ_BETA, ste=True,
        )
        EP = ep_gate
        DDC = pcdr_ddc_gate(
            ego_xy, polyline, arc_length, dt=DT,
            beta=PCDR_PROJ_BETA,
        )
    else:
        # Fallback: degenerate route — keep both gates open, proxy-EP from path.
        EP = torch.clamp(progress_ratio, max=1.0)
        DDC = torch.tensor(1.0, device=device, dtype=dtype)

    # ── L5: TTC (proxy, unchanged) ──
    if nbr_xy.shape[0] > 0 and ego_d.shape[0] > 0:
        nbr_d = torch.diff(nbr_xy, dim=1)
        nbr_v = nbr_d / DT
        ego_v_vec = ego_d / DT
        rel_v = ego_v_vec.unsqueeze(0) - nbr_v
        rel_pos = nbr_xy[:, :T - 1, :] - ego_xy[:T - 1].unsqueeze(0)
        rel_pos_norm = rel_pos / (torch.norm(rel_pos, dim=-1, keepdim=True) + 1e-6)
        closing_speed = -(rel_v * rel_pos_norm).sum(dim=-1)
        closing_speed = torch.clamp(closing_speed, min=1e-3)
        ttc_n = torch.norm(rel_pos, dim=-1) / closing_speed
        ttc_n = ttc_n + (~nbr_valid[:, :T - 1]).float() * 1e6
        min_ttc = ttc_n.min()
        TTC = ste_gate(min_ttc, threshold=0.95, steepness=5.0, pass_high=True)
    else:
        TTC = torch.tensor(1.0, device=device, dtype=dtype)

    # ── L6: SC — PCDR per-lane lookup (replaces global 30 m/s cap) ──
    rl_speed = data.get('route_lanes_speed_limit')
    rl_has = data.get('route_lanes_has_speed_limit')
    if (rl_speed is not None and rl_has is not None
            and polyline is not None and polyline.shape[0] >= 2):
        if rl_speed.dim() >= 2:
            rl_speed_t = rl_speed.squeeze(-1) if rl_speed.shape[-1] == 1 else rl_speed[..., 0]
            rl_has_t = rl_has.squeeze(-1) if rl_has.shape[-1] == 1 else rl_has[..., 0]
        else:
            rl_speed_t = rl_speed
            rl_has_t = rl_has
        if rl_speed_t.dim() == 2:                                              # (B, L) → (L,)
            rl_speed_t = rl_speed_t[0]
            rl_has_t = rl_has_t[0]
        SC = pcdr_sc_gate(
            ego_xy, polyline, arc_length, rl_speed_t, rl_has_t,
            fallback_limit=13.4, dt=DT,
            beta=PCDR_PROJ_BETA, steepness=5.0, margin=2.5,
        )
    else:
        max_speed = ego_v.max() if ego_v.numel() > 0 else torch.tensor(
            0.0, device=device, dtype=dtype)
        SC = ste_gate(max_speed, threshold=30.0, steepness=3.0, pass_high=False)

    # ── Combine (no-comfort variant: drop comfort term) ──
    multiplier = NC * DAC * DDC * MP
    weighted_avg = (5.0 * TTC + 5.0 * EP + 4.0 * SC) / 14.0
    return multiplier * weighted_avg


if __name__ == '__main__':
    import numpy as np
    torch.manual_seed(0)

    T = 80
    ego = torch.stack([torch.linspace(0, 40, T), torch.zeros(T)], dim=-1)
    ego_var = ego.clone().requires_grad_(True)

    expert = torch.stack([
        torch.linspace(0, 40, T), torch.zeros(T), torch.zeros(T)
    ], dim=-1).unsqueeze(0)

    # Build a "real-ish" route_lanes: 3 lanes, 30 pts each, 12 features.
    rl = torch.zeros(1, 3, 30, 12)
    rl[:, 0, :, 0] = torch.linspace(0, 40, 30)                            # x
    rl[:, 0, :, 1] = 0.0                                                   # y

    nbrs_future = torch.zeros(1, 5, T, 3)
    nbrs_past = torch.zeros(1, 5, 21, 8)
    nbrs_past[..., 6] = 1.95
    nbrs_past[..., 7] = 4.77

    data = {
        'ego_agent_future': expert,
        'neighbor_agents_future': nbrs_future,
        'neighbor_agents_past': nbrs_past,
        'route_lanes': rl,
    }
    reward = compute_cls_reward(ego_var, data)
    print(f'PCDR-CLS on straight trajectory: {reward.item():.4f}')
    reward.backward()
    print(f'  ‖∂reward/∂ego‖ = {ego_var.grad.norm().item():.4e}')
