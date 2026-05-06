"""CLS-structured reward with straight-through estimator gates.

Mimics the nuPlan devkit's closed-loop score formula:
  CLS = NC * DAC * DDC * MP * (5*TTC + 5*EP + 4*SC + 2*C) / 16

Each gate/factor is computed with a forward-hard / backward-soft split:
- Forward: steep sigmoid approximating devkit's discrete threshold, giving
  CLS values very close to what the evaluator measures.
- Backward: the underlying soft sigmoid's gradient, so the policy can still
  learn from gate-saturated signals (which previously produced zero gradient
  in our B2-A/B3-A/B3-B falsification branches).

Applied on Phase 4's rolled-out trajectory (not the raw planner output),
so BOTH the measurement-point mismatch (state space) and the scoring-function
mismatch (reward structure) are addressed.
"""
from __future__ import annotations

import torch


def ste_gate(value: torch.Tensor, threshold: float, steepness: float = 20.0, pass_high: bool = True) -> torch.Tensor:
    """Straight-through estimator gate.

    Forward: 1.0 if value passes devkit threshold (binary), 0.0 else.
    Backward: gradient of sigmoid((value - threshold) * steepness), so gradient flows
              even when the forward hard gate is saturated at 0 or 1.

    Args:
        value: scalar or tensor to gate.
        threshold: devkit's discrete threshold (pass if value >= threshold when pass_high).
        steepness: sigmoid steepness for backward gradient.
        pass_high: True means gate = 1 if value >= threshold; False means gate = 1 if value <= threshold.
    """
    soft = torch.sigmoid((value - threshold) * steepness) if pass_high else torch.sigmoid((threshold - value) * steepness)
    hard = (soft > 0.5).float()
    # ST trick: value = hard on forward, gradient comes from soft
    return hard + (soft - soft.detach())


def compute_cls_reward(ego_traj: torch.Tensor, data: dict) -> torch.Tensor:
    """Compute CLS-structured reward on ego_traj.

    ego_traj: (T, >=2) trajectory with xy in first 2 cols. Typically T=80 steps @ 10Hz.
    data: npz dict with ego_agent_future, neighbor_agents_future, route_lanes, and optionally
          lane speed limits.

    Returns scalar reward in approximately [0, 1] matching devkit CLS scale.
    """
    device = ego_traj.device
    dtype = ego_traj.dtype

    def _zero():
        return torch.tensor(0.0, device=device, dtype=dtype)

    expert = data.get('ego_agent_future')
    nbrs = data.get('neighbor_agents_future')
    route = data.get('route_lanes')
    if expert is None or nbrs is None or route is None:
        return _zero()

    if expert.dim() == 3: expert = expert[0]
    if nbrs.dim() == 4: nbrs = nbrs[0]
    if route.dim() == 4: route = route[0]

    ego_xy = ego_traj[:, :2]  # (T, 2)
    T = ego_xy.shape[0]
    dt = 0.1

    # ── L0: speed profile + derived kinematics ──
    ego_d = torch.diff(ego_xy, dim=0)
    ego_v = torch.norm(ego_d, dim=-1) / dt  # (T-1,)
    ego_a = torch.diff(ego_v, dim=0) / dt  # (T-2,)
    ego_jerk = torch.diff(ego_a, dim=0) / dt  # (T-3,)

    # ── L1: MP (making progress) — ego progress must be >= 30% of expert ──
    expert_xy = expert[:, :2]  # (T_e, 2)
    exp_d = torch.diff(expert_xy, dim=0)
    expert_steps = torch.norm(exp_d, dim=-1)
    expert_progress = expert_steps.sum().clamp(min=2.0)
    ego_steps = torch.norm(ego_d, dim=-1)
    ego_progress = ego_steps.sum()
    progress_ratio = ego_progress / expert_progress  # ~[0, 1+]

    MP = ste_gate(progress_ratio, threshold=0.30, steepness=5.0, pass_high=True)

    # ── L2: NC (no ego at-fault collisions) — min distance to any neighbor ──
    # nbrs: (N, T_n, D) with xy in cols [0,1]. T_n should match or exceed T.
    nbr_xy = nbrs[:, :T, :2]  # (N, T, 2)
    # Ego at each step: (T, 2). Compute distances for each (ego_t, nbr_n, t) triple at same t.
    ego_exp = ego_xy.unsqueeze(0)  # (1, T, 2)
    diff_n = ego_exp - nbr_xy  # (N, T, 2)
    dist_n = torch.norm(diff_n, dim=-1)  # (N, T)
    # Only consider valid neighbors (nonzero xy is proxy for presence)
    nbr_valid = (nbrs[:, :T, :2].abs().sum(dim=-1) > 1e-6)  # (N, T)
    dist_n = dist_n + (~nbr_valid).float() * 1e6
    # min over (N, T): the overall closest approach
    min_dist = dist_n.min()  # scalar
    # NC gate: pass if min_dist > 3.0m (rough collision threshold)
    NC = ste_gate(min_dist, threshold=3.0, steepness=5.0, pass_high=True)

    # ── L3: DAC (drivable area compliance) — max deviation from route-lane proxy ──
    route_xy = route[:, :, :2].reshape(-1, 2)  # (L*P, 2)
    valid = (route_xy.abs().sum(dim=-1) > 1e-6)
    route_xy = route_xy[valid]
    if route_xy.shape[0] == 0:
        DAC = torch.tensor(1.0, device=device, dtype=dtype)
        EP = torch.tensor(1.0, device=device, dtype=dtype)
    else:
        d_ego_to_route = torch.cdist(ego_xy, route_xy).min(dim=-1).values  # (T,)
        # Expert distance for reference (measures route lane coverage)
        d_exp_to_route = torch.cdist(expert_xy, route_xy).min(dim=-1).values
        ego_max_dev = d_ego_to_route.max()
        exp_max_dev = d_exp_to_route.max()
        # DAC gate: pass if ego_max_dev <= exp_max_dev + 0.3m
        DAC = ste_gate(ego_max_dev, threshold=exp_max_dev.detach() + 0.3, steepness=3.0, pass_high=False)
        # EP (expert progress, 0-1): smoother, progress_ratio capped at 1.0 is OK
        EP = torch.clamp(progress_ratio, max=1.0)

    # ── L4: DDC (driving direction compliance) ──
    # Ego heading = direction of motion. Expert heading likewise. Compare at each step.
    if ego_d.shape[0] > 0 and exp_d.shape[0] > 0:
        ego_heading = torch.atan2(ego_d[:, 1], ego_d[:, 0])
        exp_d_resampled = exp_d[:ego_d.shape[0]]
        exp_heading = torch.atan2(exp_d_resampled[:, 1], exp_d_resampled[:, 0])
        heading_diff = torch.abs(torch.atan2(torch.sin(ego_heading - exp_heading),
                                              torch.cos(ego_heading - exp_heading)))
        # Max heading diff (devkit: progress-in-opposite-direction is the hard failure mode)
        max_heading_diff = heading_diff.max()
        # Pass if max diff <= π/2 (90 degrees — same general direction)
        DDC = ste_gate(max_heading_diff, threshold=1.57, steepness=5.0, pass_high=False)
    else:
        DDC = torch.tensor(1.0, device=device, dtype=dtype)

    # ── L5: TTC (time-to-collision > 0.95s at every step) ──
    # Approximate TTC via relative velocity to nearest neighbor.
    if nbr_xy.shape[0] > 0 and ego_d.shape[0] > 0:
        # Use nearest-neighbor in-plane speed approximation
        # TTC_t = distance_t / closing_speed_t  (capped)
        nbr_d = torch.diff(nbr_xy, dim=1)  # (N, T-1, 2)
        nbr_v = nbr_d / dt  # (N, T-1, 2)
        ego_v_vec = ego_d / dt  # (T-1, 2)
        rel_v = ego_v_vec.unsqueeze(0) - nbr_v  # (N, T-1, 2)
        # Closing speed = magnitude of relative velocity component toward nbr
        rel_pos = nbr_xy[:, :T-1, :] - ego_xy[:T-1].unsqueeze(0)
        rel_pos_norm = rel_pos / (torch.norm(rel_pos, dim=-1, keepdim=True) + 1e-6)
        closing_speed = -(rel_v * rel_pos_norm).sum(dim=-1)  # (N, T-1)
        closing_speed = torch.clamp(closing_speed, min=1e-3)
        ttc_n = torch.norm(rel_pos, dim=-1) / closing_speed  # (N, T-1)
        ttc_n = ttc_n + (~nbr_valid[:, :T-1]).float() * 1e6
        min_ttc = ttc_n.min()
        TTC = ste_gate(min_ttc, threshold=0.95, steepness=5.0, pass_high=True)
    else:
        TTC = torch.tensor(1.0, device=device, dtype=dtype)

    # ── L6: SC (speed limit compliance) ──
    # Not all npz have lane speed limits accessible. Use 30 m/s (≈108 km/h) as universal cap.
    speed_limit = 30.0
    max_speed = ego_v.max() if ego_v.numel() > 0 else torch.tensor(0.0, device=device, dtype=dtype)
    SC = ste_gate(max_speed, threshold=speed_limit, steepness=3.0, pass_high=False)

    # ── L7: C (comfort) — AND of: jerk<8.37, lon_jerk<4.13, lat_acc<4.89, lon_acc<2.4 ──
    # Combined as product so any failure → Comfort = 0
    if ego_jerk.numel() > 0:
        max_jerk = torch.abs(ego_jerk).max()
        c_jerk = ste_gate(max_jerk, threshold=8.37, steepness=1.0, pass_high=False)
    else:
        c_jerk = torch.tensor(1.0, device=device, dtype=dtype)

    if ego_a.numel() > 0:
        max_acc = torch.abs(ego_a).max()
        c_acc = ste_gate(max_acc, threshold=2.4, steepness=2.0, pass_high=False)
    else:
        c_acc = torch.tensor(1.0, device=device, dtype=dtype)

    C = c_jerk * c_acc

    # ── Combine into CLS ──
    multiplier = NC * DAC * DDC * MP
    weighted_avg = (5.0 * TTC + 5.0 * EP + 4.0 * SC + 2.0 * C) / 16.0
    cls = multiplier * weighted_avg
    return cls


if __name__ == '__main__':
    # Minimal self-test
    torch.manual_seed(0)
    T = 80
    ego = torch.stack([torch.linspace(0, 40, T), torch.zeros(T)], dim=-1)
    ego_var = ego.clone().requires_grad_(True)

    # Fake data
    data = {
        'ego_agent_future': torch.stack([torch.linspace(0, 40, T), torch.zeros(T), torch.zeros(T)], dim=-1).unsqueeze(0),
        'neighbor_agents_future': torch.zeros(1, 5, T, 4),  # no neighbors (zeros → filtered)
        'route_lanes': torch.stack([torch.linspace(0, 40, 30), torch.zeros(30)], dim=-1).unsqueeze(0).unsqueeze(0).expand(1, 3, 30, 2),
    }
    # route_lanes expected (L, P, 12) — make it (3, 30, 12) by padding
    rl = torch.zeros(1, 3, 30, 12)
    rl[:, :, :, :2] = data['route_lanes']
    data['route_lanes'] = rl

    reward = compute_cls_reward(ego_var, data)
    print(f'CLS reward on straight trajectory: {reward.item():.4f}')
    reward.backward()
    print(f'grad norm: {ego_var.grad.norm().item():.4f}')
    print(f'grad has NaN: {torch.isnan(ego_var.grad).any().item()}')
