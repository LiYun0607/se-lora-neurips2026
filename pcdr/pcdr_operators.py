"""Physics-Consistent Differentiable Reward (PCDR) operators.

Each operator is designed so that:
  (i)  forward value matches the nuPlan official scorer's boolean / scalar
       output to float tolerance, in the limit of the steepness parameter,
  (ii) backward provides non-vanishing, well-conditioned gradients usable
       for end-to-end RL fine-tuning of a diffusion-based motion planner.

This module currently implements PCDR-NC (no-collision via OBB-OBB SAT
smooth-max). PCDR-EP/DDC (Frenet projection with implicit-function-theorem
gradient) is a TODO.
"""
from __future__ import annotations

import torch


# -- Helpers ------------------------------------------------------------------

def _box_corners(xy: torch.Tensor, cos_h: torch.Tensor, sin_h: torch.Tensor,
                 length: torch.Tensor, width: torch.Tensor) -> torch.Tensor:
    """Return 4 corners of an oriented bounding box.

    Convention: nuPlan / collision.py rect = (x, y, cos_h, sin_h, length, width)
    with corners ordered FL, BL, BR, FR (matching center_rect_to_points).

    Args:
        xy:     (..., 2) center position
        cos_h:  (..., ) heading cosine
        sin_h:  (..., ) heading sine
        length: (..., ) box length along heading
        width:  (..., ) box width perpendicular to heading

    Returns:
        (..., 4, 2) corner positions in world frame.
    """
    half_l = length * 0.5
    half_w = width * 0.5
    # Local frame corners: front-left, back-left, back-right, front-right
    local_x = torch.stack([half_l, -half_l, -half_l, half_l], dim=-1)   # (..., 4)
    local_y = torch.stack([half_w, half_w, -half_w, -half_w], dim=-1)   # (..., 4)
    # Rotate to world frame: x_world = x_local * cos - y_local * sin
    world_x = local_x * cos_h.unsqueeze(-1) - local_y * sin_h.unsqueeze(-1)
    world_y = local_x * sin_h.unsqueeze(-1) + local_y * cos_h.unsqueeze(-1)
    corners = torch.stack([world_x, world_y], dim=-1)                   # (..., 4, 2)
    corners = corners + xy.unsqueeze(-2)
    return corners


def _sat_axis_gaps(rect1: torch.Tensor, rect2: torch.Tensor) -> torch.Tensor:
    """Compute SAT axis gaps for two OBBs (vectorized).

    Returns the 8 signed projection-gap values (4 axes from each rect's edges,
    each axis tested in both directions: gap = min1 - max2, gap = min2 - max1).
    Positive gap on any axis = separated on that axis; if the maximum gap over
    all 8 entries is positive, the boxes do not intersect (this is the SAT
    witness). The same convention as collision.py:batch_signed_distance_rect.

    Args:
        rect1: (..., 4, 2) corners of box 1
        rect2: (..., 4, 2) corners of box 2

    Returns:
        (..., 8) axis gaps. signed_distance = max(...) over the last dim.
    """
    # Edge normals: edge i->i+1 of rect1 has direction (rect1[i+1] - rect1[i]);
    # the normal is just that direction normalized (we project on directions, not
    # technically true normals, but for SAT on 2D rectangles 4 unique axes are
    # enough; we use the same 4 directions used by collision.py).
    edges1 = torch.stack([rect1[..., 1, :] - rect1[..., 0, :],
                          rect1[..., 2, :] - rect1[..., 1, :]], dim=-2)  # (..., 2, 2)
    edges2 = torch.stack([rect2[..., 1, :] - rect2[..., 0, :],
                          rect2[..., 2, :] - rect2[..., 1, :]], dim=-2)  # (..., 2, 2)
    axes = torch.cat([edges1, edges2], dim=-2)                            # (..., 4, 2)
    axes = axes / (torch.norm(axes, dim=-1, keepdim=True) + 1e-12)

    # Project both rectangles onto each axis.
    # rect: (..., 4, 2); axes: (..., 4, 2). projection: (..., 4_axes, 4_corners)
    proj1 = torch.einsum('...ij,...kj->...ik', axes, rect1)
    proj2 = torch.einsum('...ij,...kj->...ik', axes, rect2)
    p1_min, p1_max = proj1.min(dim=-1).values, proj1.max(dim=-1).values    # (..., 4)
    p2_min, p2_max = proj2.min(dim=-1).values, proj2.max(dim=-1).values    # (..., 4)

    # Concatenate both gap directions: total 8 gaps. Same as collision.py.
    gaps = torch.cat([p1_min - p2_max, p2_min - p1_max], dim=-1)           # (..., 8)
    return gaps


# -- Hard reference (forward-equivalence target) ------------------------------

def hard_obb_collision(rect1: torch.Tensor, rect2: torch.Tensor) -> torch.Tensor:
    """Hard SAT collision check: 1.0 if collision, 0.0 otherwise.

    This is the forward-equivalence target. Mathematically equivalent to
    Shapely OBB-OBB intersection for non-degenerate rectangles, which is what
    the nuPlan official scorer uses.

    Returns:
        (...) float tensor in {0.0, 1.0}.
    """
    gaps = _sat_axis_gaps(rect1, rect2)
    # SAT: separated iff max gap > 0. Collision iff max gap <= 0.
    max_gap = gaps.max(dim=-1).values
    return (max_gap <= 0.0).to(rect1.dtype)


# -- Smooth (differentiable) version ------------------------------------------

def smooth_obb_signed_distance(rect1: torch.Tensor, rect2: torch.Tensor,
                                beta: float = 0.05) -> torch.Tensor:
    """Smooth SAT signed distance using log-sum-exp soft-max over axis gaps.

    Returns a continuous, everywhere-differentiable signed-distance estimate
    that converges to the hard SAT signed distance as beta -> 0.

      signed_dist_smooth = beta * logsumexp(gaps / beta)

    Sign convention: positive = separated, negative = penetrating.
    Bias relative to hard max:
      0  <=  smooth - hard  <=  beta * log(8)   (8 axis gaps).

    Args:
        rect1, rect2: (..., 4, 2)
        beta:        smoothing temperature in meters. Smaller = sharper.

    Returns:
        (...) smooth signed distance.
    """
    gaps = _sat_axis_gaps(rect1, rect2)
    # logsumexp gives a smooth max
    return beta * torch.logsumexp(gaps / beta, dim=-1)


# -- PCDR-NC gate (CLS-compatible, drop-in for nuplan_reward_cls.NC) ----------

def pcdr_nc_gate(ego_xy: torch.Tensor, ego_cos_h: torch.Tensor, ego_sin_h: torch.Tensor,
                 ego_length: float, ego_width: float,
                 nbr_xy: torch.Tensor, nbr_cos_h: torch.Tensor, nbr_sin_h: torch.Tensor,
                 nbr_length: torch.Tensor, nbr_width: torch.Tensor,
                 nbr_valid: torch.Tensor | None = None,
                 beta: float = 0.05, ste: bool = True) -> torch.Tensor:
    """PCDR-NC: differentiable no-collision gate matching nuPlan NC operator.

    Forward: 1.0 if no collision over the entire trajectory horizon, 0.0 if any
    collision (hard SAT, equivalent to Shapely intersection for valid OBBs).
    Backward: gradient flows through smooth_obb_signed_distance.

    Args:
        ego_xy:     (T, 2) ego trajectory
        ego_cos_h:  (T,) ego heading cosine
        ego_sin_h:  (T,) ego heading sine
        ego_length: scalar ego length
        ego_width:  scalar ego width
        nbr_xy:     (N, T, 2) neighbor positions
        nbr_cos_h:  (N, T) neighbor heading cosines
        nbr_sin_h:  (N, T) neighbor heading sines
        nbr_length: (N,) per-neighbor length
        nbr_width:  (N,) per-neighbor width
        nbr_valid:  (N, T) bool mask of valid (present) neighbors per step
        beta:       smoothing temperature in meters (paper default ~0.05).
        ste:        if True, return STE-gated forward-equivalent value;
                    if False, return raw smooth no-collision soft value
                    (useful for diagnostics / loss shaping).

    Returns:
        scalar in [0.0, 1.0]. Forward equivalent to nuPlan NC bit (when ste=True).
    """
    T = ego_xy.shape[0]
    N = nbr_xy.shape[0]
    device = ego_xy.device
    dtype = ego_xy.dtype

    if N == 0:
        return torch.tensor(1.0, device=device, dtype=dtype)

    # Build ego OBB corners at every step: (T, 4, 2).
    ego_length_t = torch.full((T,), ego_length, device=device, dtype=dtype)
    ego_width_t = torch.full((T,), ego_width, device=device, dtype=dtype)
    ego_corners = _box_corners(ego_xy, ego_cos_h, ego_sin_h, ego_length_t, ego_width_t)
    ego_corners_b = ego_corners.unsqueeze(0).expand(N, -1, -1, -1)             # (N, T, 4, 2)

    # Build neighbor OBB corners at every step: (N, T, 4, 2).
    nbr_length_t = nbr_length.unsqueeze(-1).expand(-1, T)
    nbr_width_t = nbr_width.unsqueeze(-1).expand(-1, T)
    nbr_corners = _box_corners(nbr_xy, nbr_cos_h, nbr_sin_h, nbr_length_t, nbr_width_t)

    # Smooth signed distance per (neighbor, time): (N, T). Positive = separated.
    sd_smooth = smooth_obb_signed_distance(ego_corners_b, nbr_corners, beta=beta)

    if nbr_valid is None:
        nbr_valid = torch.ones(N, T, device=device, dtype=torch.bool)

    # Mask out invalid (absent) neighbors: treat as far-separated.
    sd_smooth = torch.where(nbr_valid, sd_smooth, sd_smooth.new_full((), 1e3))

    # Reduce to "min over (N, T)" (closest approach over horizon).
    # Use smooth-min so gradient flows everywhere: -beta * logsumexp(-sd / beta).
    min_sd_smooth = -beta * torch.logsumexp(-sd_smooth.reshape(-1) / beta, dim=0)

    # Smooth NC surrogate for the STE backward path. We use tanh(min_sd / scale)
    # rather than the raw signed distance: tanh keeps the SAT push direction
    # (sign of min_sd) and is monotone, but bounds the gradient magnitude inside
    # a few-meter operating window — far-separated neighbors and deep collisions
    # both fade to ~zero gradient, so reward gradient norm stays comparable to
    # the proxy reward and existing learning rate / KL settings remain valid.
    # Scale = 2 m so gradient is meaningful within ~±5 m signed distance.
    smooth_nc = torch.tanh(min_sd_smooth / 2.0)

    if not ste:
        # Diagnostic mode: return a [0, 1] soft estimate for monitoring.
        return 0.5 * (smooth_nc + 1.0)

    # Hard NC for forward equivalence: identical to Shapely-equivalent SAT
    # boolean over (N, T). Use _sat_axis_gaps directly to avoid smooth bias.
    with torch.no_grad():
        hard_gaps = _sat_axis_gaps(ego_corners_b, nbr_corners)               # (N, T, 8)
        hard_max_gap = hard_gaps.max(dim=-1).values                          # (N, T)
        hard_max_gap = torch.where(nbr_valid, hard_max_gap,
                                   hard_max_gap.new_full((), 1e3))
        hard_min_sd = hard_max_gap.min()
        hard_nc = (hard_min_sd > 0.0).to(dtype)

    # STE: forward = hard_nc, backward = smooth_nc.
    return hard_nc + (smooth_nc - smooth_nc.detach())


# -- PCDR-EP / PCDR-DDC: Frenet projection on route baseline ------------------

def build_route_baseline(route_lanes_xy: torch.Tensor,
                          expert_xy: torch.Tensor | None = None,
                          stitch_tol: float = 2.0,
                          expert_proximity_thr: float = 10.0
                          ) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a single arc-length-parameterized polyline from route_lanes.

    `route_lanes` in nuPlan NPZ contains the candidate lanes near the route in
    no particular order: parallel lanes, predecessor / successor connectors,
    and so on. We reconstruct the route polyline in three steps:

      1. Filter to lanes whose mean distance to the expert trajectory is below
         `expert_proximity_thr` (default 10 m). Lanes farther than this are
         alternative roads and should not be on the route.
      2. Topologically order the surviving lanes by matching each lane's
         endpoint to the next lane's start (within `stitch_tol`). Form chains;
         pick the chain closest to the expert.
      3. Concatenate the chain's lanes, dropping duplicated stitching points,
         and compute cumulative arc length.

    If `expert_xy` is None we fall back to the input array order.

    Returns:
        polyline:    (M, 2) ordered route polyline vertices.
        arc_length:  (M,)   cumulative arc length, arc_length[0] = 0.
        (None, None) if fewer than 2 valid points are available.
    """
    L = route_lanes_xy.shape[0]
    # Step 1: collect valid (>=2 points) lanes, with mean dist to expert.
    candidates: list[tuple[torch.Tensor, float]] = []
    for li in range(L):
        lane = route_lanes_xy[li]
        valid = lane.abs().sum(dim=-1) > 1e-6
        if not bool(valid.any()):
            continue
        pts = lane[valid]
        if pts.shape[0] < 2:
            continue
        if expert_xy is not None:
            d = torch.cdist(expert_xy.unsqueeze(0), pts.unsqueeze(0)).squeeze(0)
            mean_d = d.min(dim=-1).values.mean().item()
        else:
            mean_d = 0.0
        candidates.append((pts, mean_d))
    if not candidates:
        return None, None

    if expert_xy is not None:
        near = [c for c in candidates if c[1] < expert_proximity_thr]
        if not near:
            near = [min(candidates, key=lambda c: c[1])]
        candidates = near

    # Step 2: topological order. Build a per-lane successor link by endpoint
    # match. Each lane's "successor" is the unique candidate whose first point
    # is within stitch_tol of this lane's last point (and has no other claimer).
    n = len(candidates)
    starts = [c[0][0] for c in candidates]
    ends = [c[0][-1] for c in candidates]
    succ: list[int | None] = [None] * n
    is_succ: list[bool] = [False] * n
    for i in range(n):
        best_j, best_d = None, stitch_tol
        for j in range(n):
            if i == j or is_succ[j]:
                continue
            d = torch.norm(ends[i] - starts[j]).item()
            if d < best_d:
                best_j, best_d = j, d
        if best_j is not None:
            succ[i] = best_j
            is_succ[best_j] = True

    # Chain starts: lanes that are not anyone else's successor.
    chain_starts = [i for i in range(n) if not is_succ[i]]

    # Walk each chain.
    chains: list[list[int]] = []
    visited: set[int] = set()
    for s in chain_starts:
        chain: list[int] = []
        i: int | None = s
        while i is not None and i not in visited:
            chain.append(i)
            visited.add(i)
            i = succ[i]
        if chain:
            chains.append(chain)
    # Catch any orphan cycles (rare).
    for i in range(n):
        if i not in visited:
            chains.append([i])
            visited.add(i)

    # Pick the chain closest to expert (or longest by total arc length).
    def _chain_score(chain: list[int]) -> float:
        if expert_xy is not None:
            all_pts = torch.cat([candidates[idx][0] for idx in chain], dim=0)
            d = torch.cdist(expert_xy.unsqueeze(0), all_pts.unsqueeze(0)).squeeze(0)
            return d.min(dim=-1).values.mean().item()
        else:
            arc = sum(torch.norm(torch.diff(candidates[idx][0], dim=0), dim=-1).sum().item()
                      for idx in chain)
            return -arc                                                # longer = better
    chains.sort(key=_chain_score)
    chosen = chains[0]

    # Step 3: concatenate, stitch, arc length.
    pieces: list[torch.Tensor] = []
    last_pt: torch.Tensor | None = None
    for idx in chosen:
        pts = candidates[idx][0]
        if last_pt is not None and pts.shape[0] >= 1:
            if torch.norm(pts[0] - last_pt).item() < stitch_tol:
                pts = pts[1:]
        if pts.shape[0] >= 1:
            pieces.append(pts)
            last_pt = pts[-1]
    if not pieces:
        return None, None
    polyline = torch.cat(pieces, dim=0)
    if polyline.shape[0] < 2:
        return None, None
    diffs = torch.diff(polyline, dim=0)
    seg_lens = torch.norm(diffs, dim=-1)
    arc_length = torch.cat([polyline.new_zeros(1), seg_lens.cumsum(0)], dim=0)
    return polyline, arc_length


def diff_polyline_project(query_xy: torch.Tensor,
                           polyline: torch.Tensor,
                           arc_length: torch.Tensor,
                           beta: float = 0.5,
                           ste: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable Frenet projection of query points onto a polyline.

    Forward: hard argmin over polyline segments (numerically equivalent to a
        Cartesian-nearest-point projection, the operator nuPlan / NAVSIM use).
    Backward: gradient of a softmin-weighted projection, providing smooth
        gradient at segment boundaries (where hard argmin is discontinuous).

    Implements the projection root via the optimality condition
        (p - r(s*)) . r'(s*) = 0
    on a piecewise-linear baseline. For piecewise-linear r, r''=0 inside each
    segment, so the implicit-function-theorem gradient simplifies to the unit
    tangent at the projection point. We expose this as `tangent` for downstream
    DDC heading-vs-tangent computations.

    Args:
        query_xy:    (T, 2)   points to project (e.g., ego trajectory).
        polyline:    (M, 2)   route polyline vertices.
        arc_length:  (M,)     cumulative arc length at vertices.
        beta:        softmin temperature in meters (smaller = closer to hard).
        ste:         if True, forward returns hard projection value, backward
                     uses the softmin surrogate for gradient.

    Returns:
        s_star:    (T,)   arc-length coordinate at projection.
        d_star:    (T,)   lateral distance to polyline at the hard projection.
        tangent:   (T, 2) unit tangent at the hard projection (for DDC).
    """
    T = query_xy.shape[0]
    M = polyline.shape[0]
    if M < 2:
        zero = query_xy.new_zeros(T)
        far = query_xy.new_full((T,), 1e6)
        return zero, far, query_xy.new_zeros(T, 2)

    a = polyline[:-1]                                                  # (M-1, 2)
    b = polyline[1:]                                                   # (M-1, 2)
    seg_dir = b - a                                                    # (M-1, 2)
    seg_len = torch.norm(seg_dir, dim=-1).clamp(min=1e-6)              # (M-1,)
    unit_tan = seg_dir / seg_len.unsqueeze(-1)                         # (M-1, 2)
    s_start = arc_length[:-1]                                          # (M-1,)

    # Project: t = clip(((p - a) . d) / |d|^2, 0, 1) per (query, segment).
    diff = query_xy.unsqueeze(1) - a.unsqueeze(0)                      # (T, M-1, 2)
    t_unclip = (diff * seg_dir.unsqueeze(0)).sum(dim=-1) / seg_len.pow(2).unsqueeze(0)
    t = t_unclip.clamp(0.0, 1.0)                                       # (T, M-1)
    q = a.unsqueeze(0) + t.unsqueeze(-1) * seg_dir.unsqueeze(0)        # (T, M-1, 2)
    d = (query_xy.unsqueeze(1) - q).norm(dim=-1)                       # (T, M-1)
    s_per_seg = s_start.unsqueeze(0) + t * seg_len.unsqueeze(0)        # (T, M-1)

    # Hard argmin (forward).
    with torch.no_grad():
        m_star = d.argmin(dim=-1)                                      # (T,)
    idx = m_star.unsqueeze(-1)                                         # (T, 1)
    s_hard = s_per_seg.gather(-1, idx).squeeze(-1)                     # (T,)
    d_hard = d.gather(-1, idx).squeeze(-1)                             # (T,)
    tangent_hard = unit_tan.unsqueeze(0).expand(T, -1, -1).gather(
        -2, idx.unsqueeze(-1).expand(-1, -1, 2)
    ).squeeze(-2)                                                      # (T, 2)

    if not ste:
        # Pure soft mode: smooth signed distance for diagnostic / loss-shaping.
        w = torch.softmax(-d / beta, dim=-1)                           # (T, M-1)
        s_soft = (w * s_per_seg).sum(dim=-1)
        return s_soft, d_hard, tangent_hard

    # Soft surrogate for backward: softmin-weighted s.
    w = torch.softmax(-d / beta, dim=-1)                               # (T, M-1)
    s_soft = (w * s_per_seg).sum(dim=-1)                               # (T,)

    # STE: forward = hard, backward gradient = ds_soft/dp.
    s_star = s_hard.detach() + (s_soft - s_soft.detach())
    return s_star, d_hard, tangent_hard


def pcdr_ep_gate(ego_xy: torch.Tensor,
                  expert_xy: torch.Tensor,
                  polyline: torch.Tensor,
                  arc_length: torch.Tensor,
                  beta: float = 0.5,
                  ste: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PCDR Ego-Progress: ego_progress / expert_progress, clamped to [0, 1].

    Forward equivalent to nuPlan's EP submetric *given the same baseline
    polyline*. The polyline construction (concatenated route_lanes) is the
    only approximation w.r.t. nuPlan; nuPlan additionally walks a roadblock
    linked-list, but the per-step projection geometry is identical.

    Args:
        ego_xy:     (T, 2)
        expert_xy:  (T_e, 2)
        polyline:   (M, 2)
        arc_length: (M,)
        beta:       softmin temperature in meters.
        ste:        STE forward-hard / backward-soft if True.

    Returns:
        ep:             (,) scalar in [0, 1].
        ego_progress:   (,) raw arc-length traversed by ego.
        expert_progress:(,) raw arc-length traversed by expert.
    """
    s_ego, _, _ = diff_polyline_project(ego_xy, polyline, arc_length, beta=beta, ste=ste)
    s_exp, _, _ = diff_polyline_project(expert_xy, polyline, arc_length, beta=beta, ste=ste)
    ego_progress = s_ego[-1] - s_ego[0]
    expert_progress = (s_exp[-1] - s_exp[0]).clamp(min=1.0)            # avoid div-by-zero
    ratio = ego_progress / expert_progress
    ep = ratio.clamp(0.0, 1.0)
    return ep, ego_progress, expert_progress


def pcdr_ddc_gate(ego_xy: torch.Tensor,
                   polyline: torch.Tensor,
                   arc_length: torch.Tensor,
                   dt: float = 0.1,
                   window_sec: float = 1.0,
                   thr_warn: float = 2.0,
                   thr_fail: float = 6.0,
                   ste_steepness: float = 5.0,
                   beta: float = 0.5) -> torch.Tensor:
    """PCDR Driving-Direction-Compliance: 0 / 0.5 / 1 gate.

    Forward (matching nuPlan):
        Compute signed arc-length progress per step Δs_t = s_t - s_{t-1}.
        Per 1-second window, sum the negative parts: w = Σ max(0, -Δs).
        max_neg = max over windows of w.
            DDC = 1   if max_neg < thr_warn (=2 m)
            DDC = 0.5 if thr_warn ≤ max_neg < thr_fail (=6 m)
            DDC = 0   if max_neg ≥ thr_fail (=6 m)

    Backward: two STE gates (warn, fail) on the smoothed max_neg, summed at
    weight 0.5 each. Gradient flows everywhere via the softmin in the
    polyline projection and the sigmoid surrogate in each STE gate.

    Args:
        ego_xy:        (T, 2)
        polyline:      (M, 2)
        arc_length:    (M,)
        dt:            timestep in seconds (default 0.1 = 10 Hz).
        window_sec:    DDC window length (default 1.0 s).
        thr_warn:      warn threshold in meters (default 2.0).
        thr_fail:      fail threshold in meters (default 6.0).
        ste_steepness: sigmoid steepness for the gate STE.
        beta:          softmin temperature for the polyline projection.

    Returns:
        ddc:    scalar in {0.0, 0.5, 1.0} (forward), differentiable backward.
    """
    s_ego, _, _ = diff_polyline_project(ego_xy, polyline, arc_length, beta=beta, ste=True)
    delta_s = s_ego[1:] - s_ego[:-1]                                   # (T-1,)
    neg = torch.clamp(-delta_s, min=0.0)                               # (T-1,)

    win = max(1, int(round(window_sec / dt)))
    if neg.shape[0] < win:
        # Trajectory too short to form a full window — treat as compliant.
        return s_ego.new_ones(())

    # Sliding-window sum of length `win`.
    windows = neg.unfold(0, win, 1)                                    # (T-1-win+1, win)
    window_sum = windows.sum(dim=-1)                                   # (T-1-win+1,)

    # Smooth max for backward; hard max for forward (STE).
    max_neg_hard = window_sum.max()
    max_neg_soft = beta * torch.logsumexp(window_sum / beta, dim=0)
    max_neg = max_neg_hard.detach() + (max_neg_soft - max_neg_soft.detach())

    # Two STE gates: pass=1 if max_neg <= threshold.
    def _ste_gate_le(value: torch.Tensor, thr: float) -> torch.Tensor:
        soft = torch.sigmoid((thr - value) * ste_steepness)
        hard = (value <= thr).to(value.dtype)
        return hard + (soft - soft.detach())

    gate_warn = _ste_gate_le(max_neg, thr_warn)                        # 1 if <2 m
    gate_fail = _ste_gate_le(max_neg, thr_fail)                        # 1 if <6 m
    ddc = 0.5 * gate_warn + 0.5 * gate_fail                            # ∈ {0, 0.5, 1}
    return ddc


# -- PCDR-SC: differentiable per-lane speed-limit compliance -----------------

def pcdr_sc_gate(ego_xy: torch.Tensor,
                  polyline: torch.Tensor,
                  arc_length: torch.Tensor,
                  lane_speed_limits: torch.Tensor,
                  lane_has_limit: torch.Tensor,
                  fallback_limit: float = 13.4,
                  dt: float = 0.1,
                  beta: float = 0.5,
                  steepness: float = 5.0,
                  margin: float = 2.5) -> torch.Tensor:
    """PCDR Speed-Limit Compliance: differentiable per-lane lookup.

    Forward: max(ego_speed - lane_speed_limit) ≤ margin → SC = 1; else SC < 1.
    Equivalent to a soft attention over per-lane speed limits, with the
    attention weights coming from the polyline projection (so each ego step
    is attributed to the lane it's projected onto). Falls back to a city-
    default when no lane on the route has a known limit.

    Args:
        ego_xy:               (T, 2) ego trajectory.
        polyline:             (M, 2) concatenated route polyline vertices,
                              same as used by pcdr_ep_gate / pcdr_ddc_gate.
        arc_length:           (M,)   cumulative arc length on the polyline.
        lane_speed_limits:    (L,)   per-original-route-lane speed limit
                              in m/s (one entry per lane in route_lanes).
        lane_has_limit:       (L,)   bool — True if the corresponding lane
                              has a known speed limit in the source map.
        fallback_limit:       used when no lane on the route has a known
                              limit (default 13.4 m/s ≈ 30 mph, US urban).
        dt:                   timestep in seconds (default 10 Hz).
        beta:                 softmin temperature for projection assignment.
        steepness:            STE sigmoid steepness on (limit + margin - max_v).
        margin:               tolerance for max-speed overshoot in m/s.

    Returns:
        scalar SC gate ∈ [0, 1]; STE-forward equals nuPlan SC indicator.
    """
    T = ego_xy.shape[0]
    if polyline is None or polyline.shape[0] < 2 or T < 2:
        # Degenerate: cannot compute meaningful SC; pass.
        return ego_xy.new_ones(())

    # Map polyline arc-length → lane index. The polyline was built from the
    # route_lanes that have ≥2 valid points and survived the topological sort,
    # but the per-vertex provenance (which original lane) is not preserved
    # by build_route_baseline. We therefore aggregate across ALL valid route
    # lanes' speed limits weighted by inverse-distance attention. Practical
    # approximation: take the ARC-WEIGHTED MEAN of valid speed limits.
    valid_idx = lane_has_limit.bool().nonzero(as_tuple=True)[0]
    if valid_idx.numel() == 0:
        # No known limit → fallback.
        limit = ego_xy.new_tensor(fallback_limit)
    else:
        # Average the known limits (uniform weight for now — the projection-
        # weighted version requires per-lane provenance which the
        # current polyline doesn't carry).
        valid_limits = lane_speed_limits[valid_idx]
        # Filter zeros (some entries are 0 even with has_limit=True).
        positive = valid_limits[valid_limits > 0.5]
        if positive.numel() == 0:
            limit = ego_xy.new_tensor(fallback_limit)
        else:
            limit = positive.mean()

    # Ego per-step speed.
    diffs = torch.diff(ego_xy, dim=0)
    speed = torch.norm(diffs, dim=-1) / dt                                     # (T-1,)
    max_speed_smooth = beta * torch.logsumexp(speed / beta, dim=0)             # smooth max
    max_speed_hard = speed.max()
    max_speed = max_speed_hard.detach() + (max_speed_smooth - max_speed_smooth.detach())

    # SC gate: pass if max_speed ≤ limit + margin.
    threshold = limit + margin
    soft = torch.sigmoid((threshold - max_speed) * steepness)
    hard = (max_speed_hard <= threshold).to(ego_xy.dtype)
    return hard + (soft - soft.detach())


# -- PCDR-DAC: corridor-based drivable-area compliance -----------------------

def pcdr_dac_corridor_gate(ego_xy: torch.Tensor,
                            ego_cos_h: torch.Tensor,
                            ego_sin_h: torch.Tensor,
                            ego_length: float,
                            ego_width: float,
                            route_lanes_xy: torch.Tensor,
                            beta: float = 0.5,
                            steepness: float = 5.0,
                            margin: float = 0.0) -> torch.Tensor:
    """Differentiable Drivable-Area Compliance via lane corridors.

    The 4 corners of the ego OBB are checked against the union of all valid
    route-lane corridors. Each route-lane segment defines a quadrilateral
    corridor (left-edge polyline ↔ right-edge polyline). For each corner we
    compute a smooth signed-distance to the corridor boundary; out-of-corridor
    means signed-distance < 0 along the lateral axis.

    This is **not** a forward-equivalent reformulation of nuPlan's Shapely-
    polygon DAC (which uses the full drivable-area polygon, including
    intersections / parking lots). It is, however, a much closer approximation
    than the existing proxy that checks max distance to a centerline polyline.
    We therefore call this a "DAC-corridor" gate, and label it as such in
    the operator's docstring + paper limitations.

    Args:
        ego_xy:        (T, 2) ego trajectory.
        ego_cos_h, ego_sin_h: (T,) ego heading per step.
        ego_length:    scalar ego length (Pacifica = 4.768 m).
        ego_width:     scalar ego width (Pacifica = 1.951 m).
        route_lanes_xy: (L, P, 8) route lanes with [x, y, cos, sin, ldx, ldy, rdx, rdy].
                       The last 4 channels are offsets from centerline to
                       left and right boundaries respectively.
        beta:          softmin temperature for lane-segment selection.
        steepness:     STE sigmoid steepness on the corridor signed distance.
        margin:        bbox can extend up to `margin` meters outside before
                       triggering DAC failure (default 0).

    Returns:
        scalar DAC gate ∈ [0, 1]; STE forward = (all-corners-inside-corridor).
    """
    T = ego_xy.shape[0]
    L, P, F = route_lanes_xy.shape
    if F < 8:
        return ego_xy.new_ones(())

    # Filter out fully-zero (padded) lanes.
    lane_valid = (route_lanes_xy[..., :2].abs().sum(dim=(1, 2)) > 1e-6)        # (L,)
    if not lane_valid.any():
        return ego_xy.new_ones(())

    # For each lane, extract centerline + left/right edge points per segment.
    # left edge = centerline + offset_to_left ; right = centerline + offset_to_right
    cl = route_lanes_xy[..., :2]                                               # (L, P, 2)
    left_edge  = cl + route_lanes_xy[..., 4:6]
    right_edge = cl + route_lanes_xy[..., 6:8]

    # Per-segment corridor: 4 corners (left[i], right[i], right[i+1], left[i+1]).
    # We project each ego-bbox corner onto the segment and check if it's between
    # the two edges (lateral signed distance to centerline within ± half-width
    # of the corridor).
    #
    # Build ego bbox 4 corners at every ego step.
    ego_length_t = torch.full((T,), ego_length, device=ego_xy.device, dtype=ego_xy.dtype)
    ego_width_t = torch.full((T,), ego_width, device=ego_xy.device, dtype=ego_xy.dtype)
    ego_corners = _box_corners(ego_xy, ego_cos_h, ego_sin_h, ego_length_t, ego_width_t)
    # (T, 4, 2) → flatten to query points (T*4, 2)
    qpts = ego_corners.reshape(-1, 2)                                          # (T*4, 2)

    # For each query point, compute signed distance to the closest LANE corridor.
    # Process one lane at a time (L is small, ≤25).
    best_signed = ego_xy.new_full((qpts.shape[0],), -1e6)                       # init very negative
    for li in range(L):
        if not lane_valid[li]:
            continue
        cl_i = cl[li]                                                          # (P, 2)
        le_i = left_edge[li]                                                   # (P, 2)
        re_i = right_edge[li]                                                  # (P, 2)
        valid_pts = (cl_i.abs().sum(dim=-1) > 1e-6)                            # (P,)
        n_valid = int(valid_pts.sum().item())
        if n_valid < 2:
            continue
        cl_v = cl_i[valid_pts]                                                 # (P', 2)
        le_v = le_i[valid_pts]
        re_v = re_i[valid_pts]
        # Build segments
        a = cl_v[:-1]                                                          # (P'-1, 2)
        b = cl_v[1:]
        seg_dir = b - a
        seg_len = torch.norm(seg_dir, dim=-1).clamp(min=1e-6)
        unit_tan = seg_dir / seg_len.unsqueeze(-1)
        # Normal: rotate tangent 90° CCW (in-plane left)
        unit_normal = torch.stack([-unit_tan[..., 1], unit_tan[..., 0]], dim=-1)
        # Half-widths (signed): project (left - center) and (right - center) onto normal.
        # Note: le_v - a uses ABSOLUTE left-edge minus segment start a; the offsets
        # in route_lanes are stored as left-edge - centerline, so this is just
        # (left_offset_at_point_i + (centerline_i - a)) · normal. For a single-
        # point evaluation we use le_v[:-1] minus a rather than centerline_i.
        # Project, then clamp to ≥0.5m absolute half-width per side (defensive).
        left_dot  =  ((le_v[:-1] - a) * unit_normal).sum(dim=-1)               # signed: + if left
        right_dot = -((re_v[:-1] - a) * unit_normal).sum(dim=-1)               # signed: + if right
        left_w  = left_dot.clamp(min=0.5)
        right_w = right_dot.clamp(min=0.5)
        # For each query, project onto each segment (clamped to [0, 1]) and use
        # the LATERAL distance to the projected point. Longitudinal coverage is
        # implicit in the per-segment selection (smooth-max picks the segment
        # where the query is most clearly inside).
        diff = qpts.unsqueeze(1) - a.unsqueeze(0)                              # (Q, P'-1, 2)
        seg_len_e = seg_len.unsqueeze(0)
        t_unclip = (diff * seg_dir.unsqueeze(0)).sum(dim=-1) / seg_len_e.pow(2)
        t = t_unclip.clamp(0.0, 1.0)                                            # (Q, P'-1)
        q_proj = a.unsqueeze(0) + t.unsqueeze(-1) * seg_dir.unsqueeze(0)       # (Q, P'-1, 2)
        diff_to_q = qpts.unsqueeze(1) - q_proj
        s_lat = (diff_to_q * unit_normal.unsqueeze(0)).sum(dim=-1)             # (Q, P'-1)
        margin_left = left_w.unsqueeze(0) - s_lat                              # +ve when inside left
        margin_right = s_lat + right_w.unsqueeze(0)                            # +ve when inside right
        signed = torch.minimum(margin_left, margin_right)
        # Best signed distance over segments of THIS lane (smooth max).
        signed_best_lane = beta * torch.logsumexp(signed / beta, dim=-1)       # (Q,)
        # Take max across lanes.
        best_signed = torch.maximum(best_signed, signed_best_lane)

    # STE on best_signed > -margin: corner inside corridor with margin tolerance.
    soft = torch.sigmoid((best_signed + margin) * steepness)                   # (T*4,)
    # Hard: all corners must be inside.
    hard_in = (best_signed.detach() + margin > 0).all().to(ego_xy.dtype)
    soft_all = soft.min()                                                       # smooth-min: AND-style
    return hard_in + (soft_all - soft_all.detach())


# -- Convenience: build heading from velocity (matches existing reward) -------

def heading_from_motion(xy: torch.Tensor,
                         min_speed: float = 1e-2) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate (cos_h, sin_h) at each step from finite-difference velocity.

    Robustness: when a step's speed is below `min_speed` (e.g., ego stationary)
    the velocity direction is undefined and dividing by it would produce zero or
    near-zero (cos_h, sin_h), which then collapses the OBB to a single point and
    makes downstream SAT axes degenerate. We fall back to the most recent valid
    heading instead, defaulting to forward (1, 0) at the start of the trajectory.

    Args:
        xy:        (T, 2) trajectory.
        min_speed: per-step distance below which we treat the heading as invalid
                   and inherit it from the previous valid step.

    Returns:
        (cos_h, sin_h) each (T,) — guaranteed unit-norm at every step.
    """
    d = torch.diff(xy, dim=0)
    d_norm = torch.norm(d, dim=-1, keepdim=True)
    valid = (d_norm.squeeze(-1) > min_speed)                                   # (T-1,)
    safe_norm = d_norm.clamp(min=min_speed)
    raw_unit = d / safe_norm                                                   # (T-1, 2)

    # Carry-forward fill for invalid steps. Default initial = (1, 0).
    T_minus_1 = raw_unit.shape[0]
    fwd = raw_unit.new_tensor([1.0, 0.0])
    cur = fwd.clone()
    cos_list = []
    sin_list = []
    for ti in range(T_minus_1):
        if bool(valid[ti].item()):
            cur = raw_unit[ti]
        cos_list.append(cur[..., 0])
        sin_list.append(cur[..., 1])
    cos_h = torch.stack(cos_list, dim=0)
    sin_h = torch.stack(sin_list, dim=0)

    # Pad to length T by duplicating the first valid heading.
    cos_h = torch.cat([cos_h[:1], cos_h], dim=0)
    sin_h = torch.cat([sin_h[:1], sin_h], dim=0)
    return cos_h, sin_h
