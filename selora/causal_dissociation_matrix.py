#!/usr/bin/env python3
"""
4x4 causal-dissociation matrix experiment (nuPlan SE-LoRA).

For each scene from city c, run the planner with each of the 4 experts
active (patching paradigm: own expert = baseline; foreign experts = patched).
Compute *raw physical* per-trajectory statistics rather than saturated
composite scores, so the dissociation signal is not washed out:

  - speed_mean (m/s)   — mean ego velocity magnitude
  - jerk_mean  (m/s^3) — mean ego jerk magnitude
  - route_dev  (m)     — mean lateral distance to nearest route lane
  - traj_div   (m)     — mean L2 distance from the *own-expert* baseline trajectory
                         (measures how much patching in expert e changes the trajectory)

Outputs four 4x4 matrices M_metric[c, e].  Dissociation signature:
  - traj_div diagonal = 0 by construction; off-diagonal entries show how strongly
    expert e perturbs ODD c's trajectory -> specialization magnitude per expert
  - speed / jerk / route_dev show *which physical channel* each expert affects.
"""
import os, sys, json, glob, random, time
import numpy as np
import torch

SCRIPT_DIR = './'
sys.path.insert(0, SCRIPT_DIR)

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
import diffusion_planner.model.diffusion_utils.dpm_solver_pytorch as dpm
from train_nuplan_molora import (
    MoLoRALinear, apply_molora, set_active_expert, load_npz, get_city, CITY_NAMES,
)
from train_lora_for_closedloop import _get_decoder_internals

DEVICE = torch.device('cuda')
MODEL_PATH = f'{SCRIPT_DIR}/checkpoints/model.pth'
ARGS_PATH = f'{SCRIPT_DIR}/checkpoints/args.json'
MOLORA_PATH = './checkpoints/se_lora/molora_all_weights.pth'
NPZ_DIR = './data/processed_npz'
OUT = './data/nuplan/causal_dissociation_matrix.json'

SCENES_PER_CITY = 30
SAMPLE_STEPS = 10
DT = 0.1  # nuPlan time resolution


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def dpm_sample_once(model, cfg, data_norm):
    B, P, future_len = 1, 1 + cfg.predicted_neighbor_num, cfg.future_len
    dit, ns, enc_out, current_states, neighbor_mask, route_lanes = \
        _get_decoder_internals(model, data_norm, cfg)
    model_fn = dpm.model_wrapper(
        dit, ns, model_type=dit.model_type,
        model_kwargs={'cross_c': enc_out, 'route_lanes': route_lanes,
                      'neighbor_current_mask': neighbor_mask},
        guidance_type='uncond')

    def correct(xt, t, step):
        xt_r = xt.reshape(B, P, -1, 4)
        return torch.cat([current_states.unsqueeze(2), xt_r[:, :, 1:, :]],
                         dim=2).reshape(B, P, -1)

    solver = dpm.DPM_Solver(model_fn, ns, correcting_xt_fn=correct)
    torch.manual_seed(42)
    xT = torch.cat([current_states[:, :, None],
                    torch.randn(B, P, future_len, 4, device=DEVICE) * 0.5],
                   dim=2).reshape(B, P, -1)
    with torch.no_grad():
        x0 = solver.sample(xT.clone(), steps=SAMPLE_STEPS, t_start=1.0, t_end=1e-3,
                           order=2, skip_type='logSNR', method='multistep',
                           denoise_to_zero=True)
    traj = cfg.state_normalizer.inverse(x0.reshape(B, P, -1, 4))[0, 0, 1:]
    return traj  # (T, 4) = (T, [x, y, cos_h, sin_h]) after normalization


def physical_stats(traj, route_lanes):
    """Compute speed_mean, jerk_mean, route_dev from a single trajectory."""
    xy = traj[:, :2]
    vel = (xy[1:] - xy[:-1]) / DT
    speed = vel.norm(dim=1)
    acc = (vel[1:] - vel[:-1]) / DT
    jerk = (acc[1:] - acc[:-1]) / DT
    speed_mean = speed.mean().item()
    jerk_mean = jerk.norm(dim=1).mean().item()

    # Route deviation: min distance from each ego point to any route-lane point
    if route_lanes is not None and route_lanes.numel() > 0:
        rl = route_lanes.reshape(-1, route_lanes.shape[-1])[:, :2]
        # Filter zero rows (padding)
        rl = rl[rl.abs().sum(dim=1) > 1e-6]
        if rl.shape[0] > 0:
            d = torch.cdist(xy, rl)  # (T, N_route)
            route_dev = d.min(dim=1).values.mean().item()
        else:
            route_dev = float('nan')
    else:
        route_dev = float('nan')

    return {'speed_mean': speed_mean, 'jerk_mean': jerk_mean,
            'route_dev': route_dev}


def main():
    log('Loading config + base model')
    cfg = Config(ARGS_PATH, guidance_fn=None)

    model = Diffusion_Planner(cfg)
    ckpt = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    state = ckpt.get('ema_state_dict', ckpt.get('model', ckpt))
    if any(k.startswith('module.') for k in state):
        state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)

    log('Applying MoLoRA + loading trained LoRA weights')
    apply_molora(model, shared_rank=4, expert_rank=8, n_experts=4, alpha=32.0)
    lora_ckpt = torch.load(MOLORA_PATH, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(lora_ckpt, strict=False)
    log(f'  missing_lora={sum(1 for k in missing if "shared_" in k or "expert_" in k)}'
        f' unexpected={len(unexpected)}')
    model = model.to(DEVICE).eval()

    log('Sampling scene files per city')
    all_npz = glob.glob(os.path.join(NPZ_DIR, '*.npz'))
    by_city = {name: [] for name in CITY_NAMES}
    for f in all_npz:
        c = get_city(f)
        if c in by_city:
            by_city[c].append(f)
    random.seed(123)
    for c in CITY_NAMES:
        random.shuffle(by_city[c])
        by_city[c] = by_city[c][:SCENES_PER_CITY]
        log(f'  {c}: {len(by_city[c])}')

    # scene_results[city] = list of dict:
    #   { 'trajs': {e: Tensor(T,4)}, 'stats': {e: {...}}, 'data': raw data }
    scene_results = {c: [] for c in CITY_NAMES}

    total_forward = sum(SCENES_PER_CITY for _ in CITY_NAMES) * 4
    done = 0
    t0 = time.time()
    for c in CITY_NAMES:
        for path in by_city[c]:
            try:
                data = load_npz(path, DEVICE)
                data_norm = cfg.observation_normalizer(
                    {k: v.clone() for k, v in data.items()})
            except Exception:
                continue

            trajs_by_expert = {}
            stats_by_expert = {}
            for e_idx in range(4):
                set_active_expert(model, e_idx)
                try:
                    traj = dpm_sample_once(model, cfg, data_norm)
                    stats = physical_stats(traj, data.get('route_lanes', None))
                    trajs_by_expert[e_idx] = traj.detach().cpu()
                    stats_by_expert[e_idx] = stats
                except Exception as ex:
                    trajs_by_expert[e_idx] = None
                    stats_by_expert[e_idx] = {'error': str(ex)}
                done += 1
            scene_results[c].append({
                'trajs': trajs_by_expert, 'stats': stats_by_expert})

            if done % 40 == 0:
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1)
                eta = (total_forward - done) / max(rate, 1e-6)
                log(f'  [{done}/{total_forward}] rate={rate:.2f}/s eta={eta/60:.1f}min')

    # Build 4x4 matrices for each metric
    METRIC_KEYS = ['speed_mean', 'jerk_mean', 'route_dev']
    log('\nBuilding metric matrices (rows=scene city, cols=active expert)')
    summary = {}
    for metric in METRIC_KEYS:
        mat = np.full((4, 4), np.nan)
        for c_i, c in enumerate(CITY_NAMES):
            for e in range(4):
                vals = [s['stats'][e].get(metric) for s in scene_results[c]
                        if 'error' not in s['stats'][e] and
                        s['stats'][e].get(metric) is not None and
                        not np.isnan(s['stats'][e].get(metric, np.nan))]
                if vals:
                    mat[c_i, e] = float(np.mean(vals))
        summary[metric] = {'matrix': mat.tolist()}
        log(f'  {metric}:')
        for c_i, c in enumerate(CITY_NAMES):
            row = '  '.join(f'{mat[c_i, e]:.4f}' for e in range(4))
            log(f'    {c:12s} {row}')

    # Trajectory divergence matrix: mean ||traj(c, e) - traj(c, c)||
    log('\n  traj_div (vs. own-expert baseline; diagonal = 0):')
    div_mat = np.full((4, 4), np.nan)
    for c_i, c in enumerate(CITY_NAMES):
        for e in range(4):
            divs = []
            for s in scene_results[c]:
                own = s['trajs'][c_i]
                alt = s['trajs'][e]
                if own is None or alt is None:
                    continue
                d = (alt[:, :2] - own[:, :2]).norm(dim=1).mean().item()
                divs.append(d)
            if divs:
                div_mat[c_i, e] = float(np.mean(divs))
        row = '  '.join(f'{div_mat[c_i, e]:.4f}' for e in range(4))
        log(f'    {c:12s} {row}')
    summary['traj_div'] = {'matrix': div_mat.tolist()}

    # Dissociation index: for each metric, | own_diag - off_mean |
    log('\nPer-metric dissociation (|own_diag - mean_off_diag|, averaged over ODDs):')
    for metric in METRIC_KEYS + ['traj_div']:
        mat = np.array(summary[metric]['matrix'])
        diag = np.diag(mat)
        off = np.array([
            np.nanmean([mat[c, e] for e in range(4) if e != c])
            for c in range(4)
        ])
        delta = diag - off
        mean_abs_delta = float(np.nanmean(np.abs(delta)))
        log(f'  {metric:12s}  own={diag.round(3).tolist()}  off={off.round(3).tolist()}')
        log(f'  {"":12s}  delta={delta.round(3).tolist()}  |mean_abs_delta|={mean_abs_delta:.3f}')
        summary[metric]['diag'] = diag.tolist()
        summary[metric]['off_mean'] = off.tolist()
        summary[metric]['delta'] = delta.tolist()
        summary[metric]['mean_abs_delta'] = mean_abs_delta

    # Save
    out_raw = {}
    for c in CITY_NAMES:
        out_raw[c] = [{'stats': s['stats']} for s in scene_results[c]]
    with open(OUT, 'w') as f:
        json.dump({'cities': CITY_NAMES, 'scenes_per_city': SCENES_PER_CITY,
                   'summary': summary, 'raw_stats': out_raw}, f, indent=2,
                  default=lambda o: None)
    log(f'\nSaved: {OUT}')


if __name__ == '__main__':
    main()
