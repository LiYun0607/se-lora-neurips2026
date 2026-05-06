"""PCDR + explicit margin objective (Direction-1, v3).

Empirically supported by ChatGPT deep research (2026-05-01) and the v1/v2
perturbation-augmentation failures. The mechanism we are addressing:

  PCDR-NC's smooth-min gradient at perturbed position d=ρ from boundary
  decays as exp(-d/β), with β=0.05m the forward-fidelity smoothing scale.
  At ρ=0.30m, gradient ≈ exp(-6) ≈ 0.0025 — saturated. Perturbation-based
  augmentation (D-2) cannot push margin past β log(K) ≈ 10cm.

This script keeps PCDR's *forward* bit-exactness (compute_cls_reward
unchanged) and ADDS an auxiliary margin penalty on top:

  L_train = -R_PCDR(τ)
            + λ_NC  · β_back · softplus( (ρ_NC  - m_NC(τ))  / β_back )
            + λ_DAC · β_back · softplus( (ρ_DAC - m_DAC(τ)) / β_back )

  where
    m_NC(τ)  = min over (neighbor, timestep) of smooth SAT signed distance,
               computed with β_back ≫ 0.05m so gradient reaches |d| ~ ρ
    m_DAC(τ) = min over (corner, timestep) of corridor lateral signed distance
    softplus(x) ≈ relu(x), smooth near 0
    multiplying by β_back keeps the penalty in metres-units

Audit (forward) is unchanged: PCDR-NC / EP / DDC remain bit-exact to the
nuPlan operators. The margin term is *training-only* and never enters the
forward audit. This is the "PCDR-Audit + PCDR-Train decomposition" framing
recommended by the deep-research report.

Initial values (from the report):
  ρ_NC = 0.40m, ρ_DAC = 0.30m, β_back = 0.12m, λ = 0.5
"""
from __future__ import annotations

import glob
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Output dir BEFORE importing base trainer.
MARGIN_OUT_DIR = os.environ.get(
    'MARGIN_OUT_DIR',
    './checkpoints/pcdr_margin',
)
os.environ['PHASE5_OUT_DIR'] = MARGIN_OUT_DIR
os.makedirs(MARGIN_OUT_DIR, exist_ok=True)

MARGIN_SEED  = int(os.environ.get('MARGIN_SEED', '42'))
RHO_NC       = float(os.environ.get('MARGIN_RHO_NC',   '0.40'))
RHO_DAC      = float(os.environ.get('MARGIN_RHO_DAC',  '0.30'))
BETA_BACK    = float(os.environ.get('MARGIN_BETA',     '0.12'))
LAMBDA       = float(os.environ.get('MARGIN_LAMBDA',   '0.5'))

import train_lora_for_closedloop as base                                       # noqa: E402
from nuplan_reward_cls_pcdr import compute_cls_reward                          # noqa: E402
from pcdr_operators import (                                                   # noqa: E402
    _box_corners, smooth_obb_signed_distance,
    heading_from_motion, pcdr_dac_corridor_gate,
)


# Pacifica vehicle dims (matches PCDR convention).
PACIFICA_LENGTH = 4.768
PACIFICA_WIDTH  = 1.951

# Default neighbor box dims when explicit dims absent in npz.
DEFAULT_NBR_LENGTH = 4.5
DEFAULT_NBR_WIDTH  = 2.0


def _compute_nc_smooth_margin(ego_traj: torch.Tensor, data: dict,
                               beta_back: float) -> torch.Tensor:
    """Min over (N, T) of smooth-SAT signed distance with β=β_back.

    Returns scalar; positive = separated, negative = penetrating.

    NOTE on parsing (matches compute_cls_reward in nuplan_reward_cls_pcdr.py):
      * neighbor_agents_future has 3 channels (x, y, heading_angle in radians).
        cos/sin must be computed via cos(angle), sin(angle); channels 3+ do
        NOT exist and previous v3 code fell to fallback cos=1, sin=0.
      * neighbor box dimensions live in neighbor_agents_PAST channels [6=W, 7=L]
        (last past frame); future has no W/L channels.
    """
    device = ego_traj.device
    dtype  = ego_traj.dtype

    nbrs = data.get('neighbor_agents_future')
    nbrs_past = data.get('neighbor_agents_past')
    if nbrs is None:
        return torch.tensor(1e3, device=device, dtype=dtype)
    if nbrs.dim() == 4:
        nbrs = nbrs[0]                                # (N, T, F)
    if nbrs_past is not None and nbrs_past.dim() == 4:
        nbrs_past = nbrs_past[0]

    if nbrs.shape[0] == 0:
        return torch.tensor(1e3, device=device, dtype=dtype)

    # Ego-frame heading from motion (as in pcdr_nc_gate).
    ego_xy = ego_traj[:, :2]
    cos_e, sin_e = heading_from_motion(ego_xy)

    T = ego_xy.shape[0]
    N = nbrs.shape[0]

    nbr_xy = nbrs[..., :2].to(dtype)                  # (N, T, 2)

    # neighbor heading: channel 2 of future is heading angle (radians).
    if nbrs.shape[-1] >= 3:
        nbr_h = nbrs[..., 2].to(dtype)                # (N, T) heading in radians
        cos_n = torch.cos(nbr_h)
        sin_n = torch.sin(nbr_h)
    else:
        cos_n = torch.ones((N, T), device=device, dtype=dtype)
        sin_n = torch.zeros((N, T), device=device, dtype=dtype)

    # Validity: nonzero xy.
    nbr_valid = (nbr_xy.abs().sum(dim=-1) > 1e-6)     # (N, T)

    if not nbr_valid.any():
        return torch.tensor(1e3, device=device, dtype=dtype)

    # Box dims: from neighbor_agents_PAST last frame, channels [6=W, 7=L].
    # Filter out absent agents (zero W/L) and use small default.
    if nbrs_past is not None and nbrs_past.shape[-1] >= 8:
        nbr_W_raw = nbrs_past[:, -1, 6].to(dtype)
        nbr_L_raw = nbrs_past[:, -1, 7].to(dtype)
        size_ok = (nbr_W_raw > 1e-3) & (nbr_L_raw > 1e-3)
        nbr_wid = torch.where(size_ok, nbr_W_raw, nbr_W_raw.new_full((), 0.5))
        nbr_len = torch.where(size_ok, nbr_L_raw, nbr_L_raw.new_full((), 0.5))
    else:
        nbr_wid = torch.full((N,), DEFAULT_NBR_WIDTH,  device=device, dtype=dtype)
        nbr_len = torch.full((N,), DEFAULT_NBR_LENGTH, device=device, dtype=dtype)

    # Build OBB corners.
    ego_len_t = torch.full((T,), PACIFICA_LENGTH, device=device, dtype=dtype)
    ego_wid_t = torch.full((T,), PACIFICA_WIDTH,  device=device, dtype=dtype)
    ego_corners = _box_corners(ego_xy, cos_e, sin_e, ego_len_t, ego_wid_t)        # (T, 4, 2)
    ego_b = ego_corners.unsqueeze(0).expand(N, -1, -1, -1)                        # (N, T, 4, 2)

    nbr_len_t = nbr_len.unsqueeze(-1).expand(-1, T)
    nbr_wid_t = nbr_wid.unsqueeze(-1).expand(-1, T)
    nbr_corners = _box_corners(nbr_xy, cos_n, sin_n, nbr_len_t, nbr_wid_t)        # (N, T, 4, 2)

    # Smooth signed distance with WIDE β so gradient reaches ρ.
    sd_smooth = smooth_obb_signed_distance(ego_b, nbr_corners, beta=beta_back)    # (N, T)
    sd_smooth = torch.where(nbr_valid, sd_smooth, sd_smooth.new_full((), 1e3))

    # Smooth-min over (N, T) using same β_back (gradient flows).
    min_sd_smooth = -beta_back * torch.logsumexp(
        -sd_smooth.reshape(-1) / beta_back, dim=0)

    # STE: forward = HARD geometric min signed distance (no smoothing bias),
    # backward = smooth gradient. This way ρ is interpretable directly as
    # "minimum required true SAT clearance in metres".
    with torch.no_grad():
        from pcdr_operators import _sat_axis_gaps
        hard_gaps = _sat_axis_gaps(ego_b, nbr_corners)                             # (N, T, 8)
        hard_max_gap = hard_gaps.max(dim=-1).values                                # (N, T)
        hard_max_gap = torch.where(nbr_valid, hard_max_gap,
                                   hard_max_gap.new_full((), 1e3))
        hard_min_sd = hard_max_gap.min()                                           # scalar

    return hard_min_sd + (min_sd_smooth - min_sd_smooth.detach())


def _compute_dac_smooth_margin(ego_traj: torch.Tensor, data: dict,
                                beta_back: float) -> torch.Tensor:
    """Min over (corner, timestep) of corridor lateral signed distance.

    Reuses pcdr_dac_corridor_gate's internal `best_signed` computation but
    bypasses the STE wrap to expose the raw signed distance.
    """
    device = ego_traj.device
    dtype  = ego_traj.dtype

    route = data.get('route_lanes')
    if route is None:
        return torch.tensor(1e3, device=device, dtype=dtype)
    if route.dim() == 4:
        route = route[0]
    if route.shape[-1] < 8:
        return torch.tensor(1e3, device=device, dtype=dtype)

    ego_xy = ego_traj[:, :2]
    cos_e, sin_e = heading_from_motion(ego_xy)
    T = ego_xy.shape[0]

    cl = route[..., :2].to(dtype)
    le = cl + route[..., 4:6].to(dtype)
    re = cl + route[..., 6:8].to(dtype)
    L, P, _ = cl.shape

    lane_valid = (cl.abs().sum(dim=(1, 2)) > 1e-6)
    if not lane_valid.any():
        return torch.tensor(1e3, device=device, dtype=dtype)

    # Build ego corners.
    ego_len_t = torch.full((T,), PACIFICA_LENGTH, device=device, dtype=dtype)
    ego_wid_t = torch.full((T,), PACIFICA_WIDTH,  device=device, dtype=dtype)
    ego_corners = _box_corners(ego_xy, cos_e, sin_e, ego_len_t, ego_wid_t)        # (T, 4, 2)
    qpts = ego_corners.reshape(-1, 2)                                             # (Q=T*4, 2)

    best_signed = ego_xy.new_full((qpts.shape[0],), -1e6)

    for li in range(L):
        if not lane_valid[li]:
            continue
        cl_i = cl[li]
        le_i = le[li]
        re_i = re[li]
        valid_pts = (cl_i.abs().sum(dim=-1) > 1e-6)
        n_valid = int(valid_pts.sum().item())
        if n_valid < 2:
            continue
        cl_v = cl_i[valid_pts]; le_v = le_i[valid_pts]; re_v = re_i[valid_pts]
        a = cl_v[:-1]; b = cl_v[1:]
        seg_dir = b - a
        seg_len = torch.norm(seg_dir, dim=-1).clamp(min=1e-6)
        unit_tan = seg_dir / seg_len.unsqueeze(-1)
        unit_n   = torch.stack([-unit_tan[..., 1], unit_tan[..., 0]], dim=-1)
        left_dot  =  ((le_v[:-1] - a) * unit_n).sum(dim=-1)
        right_dot = -((re_v[:-1] - a) * unit_n).sum(dim=-1)
        left_w  = left_dot.clamp(min=0.5)
        right_w = right_dot.clamp(min=0.5)

        diff = qpts.unsqueeze(1) - a.unsqueeze(0)
        seg_len_e = seg_len.unsqueeze(0)
        t_unclip = (diff * seg_dir.unsqueeze(0)).sum(dim=-1) / seg_len_e.pow(2)
        t = t_unclip.clamp(0.0, 1.0)
        q_proj = a.unsqueeze(0) + t.unsqueeze(-1) * seg_dir.unsqueeze(0)
        diff_to_q = qpts.unsqueeze(1) - q_proj
        s_lat = (diff_to_q * unit_n.unsqueeze(0)).sum(dim=-1)
        m_left  = left_w.unsqueeze(0)  - s_lat
        m_right = s_lat + right_w.unsqueeze(0)
        signed = torch.minimum(m_left, m_right)
        # STE on inner smooth-max over segments: forward = hard max (no β·log(P)
        # bias), backward = smooth gradient over all segments.
        signed_best_lane_smooth = beta_back * torch.logsumexp(signed / beta_back, dim=-1)
        with torch.no_grad():
            signed_best_lane_hard = signed.max(dim=-1).values
        signed_best_lane = (signed_best_lane_hard
                            + (signed_best_lane_smooth - signed_best_lane_smooth.detach()))
        best_signed = torch.maximum(best_signed, signed_best_lane)

    # Smooth-min over corners*T.
    smooth_min_dac = -beta_back * torch.logsumexp(-best_signed / beta_back, dim=0)
    # STE: forward = hard min (no smooth bias), backward = smooth gradient.
    with torch.no_grad():
        hard_min_dac = best_signed.min()
    return hard_min_dac + (smooth_min_dac - smooth_min_dac.detach())


def _compute_reward_margin(ego_traj: torch.Tensor, data: dict) -> torch.Tensor:
    """PCDR-exact main reward minus softplus margin penalties (NC + DAC)."""
    r_pcdr = compute_cls_reward(ego_traj, data)

    m_nc  = _compute_nc_smooth_margin(ego_traj, data, BETA_BACK)
    m_dac = _compute_dac_smooth_margin(ego_traj, data, BETA_BACK)

    p_nc  = BETA_BACK * F.softplus((RHO_NC  - m_nc)  / BETA_BACK)
    p_dac = BETA_BACK * F.softplus((RHO_DAC - m_dac) / BETA_BACK)

    return r_pcdr - LAMBDA * (p_nc + p_dac)


base.compute_reward_nuplan = _compute_reward_margin
base.OUT_DIR = MARGIN_OUT_DIR
base.SAVE_TAG = (f'margin_v3_rho_nc{int(RHO_NC*100):02d}_dac{int(RHO_DAC*100):02d}'
                 f'_b{int(BETA_BACK*100):02d}_l{int(LAMBDA*100):02d}_s{MARGIN_SEED}')

random.seed(MARGIN_SEED)
np.random.seed(MARGIN_SEED)
torch.manual_seed(MARGIN_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(MARGIN_SEED)


def main():
    base.log("=" * 60)
    base.log("PCDR + Margin-Objective (Direction-1, v3) LoRA Training")
    base.log(f"  forward reward: PCDR (compute_cls_reward, bit-exact)")
    base.log(f"  margin penalty: softplus((ρ - m) / β_back),  NC + DAC")
    base.log(f"  ρ_NC={RHO_NC:.2f}m  ρ_DAC={RHO_DAC:.2f}m  β_back={BETA_BACK:.2f}m  λ={LAMBDA:.2f}")
    base.log(f"  seed: {MARGIN_SEED}")
    base.log(f"  out:  {MARGIN_OUT_DIR}")
    base.log("=" * 60)

    cfg = base.Config(base.ARGS_PATH, guidance_fn=None)
    all_npz = sorted(glob.glob(os.path.join(base.NPZ_DIR, '*.npz')))
    random.shuffle(all_npz)
    train_paths = all_npz[:base.MAX_SCENES]

    solvers = [(f"DPM++ 2nd (MARGIN-V3 ρ_NC={RHO_NC:.2f} ρ_DAC={RHO_DAC:.2f} β={BETA_BACK:.2f})",
                base.dpm_sample,
                f'dpm_margin_v3_rho_nc{int(RHO_NC*100):02d}_dac{int(RHO_DAC*100):02d}'
                f'_b{int(BETA_BACK*100):02d}_l{int(LAMBDA*100):02d}')]

    for name, fn, tag in solvers:
        try:
            r, _ = base.train_solver(name, fn, cfg, train_paths, tag)
            base.log(f"DONE: {name}")
        except Exception as e:
            import traceback
            base.log(f"FAILED: {name} — {e}")
            base.log(traceback.format_exc())
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
