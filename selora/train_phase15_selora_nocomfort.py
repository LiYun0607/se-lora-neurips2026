#!/usr/bin/env python3
"""Phase 11: SE-LoRA (shared + per-city expert) with CLS-STE reward.

Combines:
- MoLoRA architecture from train_nuplan_molora.py (shared_rank=4 + expert_rank=8)
- CLS-STE reward + KL + differentiable rollout from train_phase10_fresh.py
- Balanced per-city sampling (NO turn oversampling — Phase 10 oversampling biased
  LoRA toward low-speed maneuvering and away from high-speed scenarios)

Hypothesis: per-city expert can specialize for ODD characteristics (Boston dense
intersections vs Vegas highway), while shared captures generic dynamics. This
matches the paper's SE-LoRA contribution and was demonstrated on Autoware.
"""
import sys, os, time, json, random, glob, collections, argparse
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
import diffusion_planner.model.diffusion_utils.dpm_solver_pytorch as dpm

from train_nuplan_molora import (
    MoLoRALinear, apply_molora, set_active_expert, freeze_non_lora,
    CITY_MAP, CITY_NAMES, get_city,
)
from train_lora_for_closedloop import load_npz, dpm_sample
from differentiable_rollout import rollout_from_planner_output, RolloutConfig
from nuplan_reward_cls_nocomfort import compute_cls_reward

DEVICE = torch.device('cuda')
MODEL_PATH = f'{SCRIPT_DIR}/checkpoints/model.pth'
ARGS_PATH = f'{SCRIPT_DIR}/checkpoints/args.json'
NPZ_DIR = os.environ.get('PHASE15_NPZ_DIR', './data/processed_npz')

SHARED_RANK = 4
EXPERT_RANK = 8
LORA_ALPHA = 32.0
LR = 1e-6  # Phase 10 best
KL_BETA = 0.3
GRAD_ACCUM = 4
EPOCHS = 3
SCENES_PER_CITY = 2500  # 2500 × 4 = 10000 scenes/epoch (matches Phase 10 effort)
CHECKPOINT_EVERY = 1000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def extract_initial_state(data, device):
    ecs = data.get('ego_current_state')
    s = torch.zeros(6, device=device, dtype=torch.float32)
    if ecs is None:
        return s
    if ecs.dim() >= 2:
        ecs = ecs[0]
    if len(ecs) > 5:
        s[3] = ecs[4].item()
    if len(ecs) > 7:
        s[4] = ecs[6].item()
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--out_dir', type=str, required=True)
    ap.add_argument('--scenes_per_city', type=int, default=SCENES_PER_CITY)
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    args = ap.parse_args()

    OUT_DIR = args.out_dir
    os.makedirs(OUT_DIR, exist_ok=True)
    SEED = args.seed

    log("=" * 60)
    log(f"PHASE 15: SE-LoRA + nocomfort (no VDGRPO)  (seed={SEED})")
    log(f"  shared_rank={SHARED_RANK}, expert_rank={EXPERT_RANK}, n_experts={len(CITY_NAMES)}")
    log(f"  cities: {CITY_NAMES}")
    log(f"  scenes/city/epoch={args.scenes_per_city}  epochs={args.epochs}")
    log(f"  LR={LR} KL={KL_BETA}  reward=CLS-STE")
    log("=" * 60)

    random.seed(SEED); torch.manual_seed(SEED); np.random.seed(SEED)

    cfg = Config(ARGS_PATH, guidance_fn=None)

    # Group npz by city
    all_npz = glob.glob(os.path.join(NPZ_DIR, '*.npz'))
    city_files = collections.defaultdict(list)
    for f in all_npz:
        city = get_city(f)
        if city != 'unknown':
            city_files[city].append(f)
    for c in CITY_NAMES:
        log(f"  {c}: {len(city_files[c])} scenes available")

    # Build SE-LoRA model
    model = Diffusion_Planner(cfg)
    ckpt = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    state = ckpt.get('ema_state_dict', ckpt.get('model', ckpt))
    if any(k.startswith('module.') for k in state.keys()):
        state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    apply_molora(model, SHARED_RANK, EXPERT_RANK, len(CITY_NAMES), LORA_ALPHA)
    freeze_non_lora(model)
    model = model.to(DEVICE)

    # Build base model for KL reference
    base_model = Diffusion_Planner(cfg)
    base_model.load_state_dict(state, strict=False)
    base_model = base_model.to(DEVICE).eval()
    for p in base_model.parameters():
        p.requires_grad = False

    noise_schedule = dpm.NoiseScheduleVP('linear', continuous_beta_0=0.1, continuous_beta_1=20.)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
    rollout_cfg = RolloutConfig(dt=0.1, n_steps=80)

    results = {"name": "phase15_selora_nocomfort", "seed": SEED, "lr": LR,
               "shared_rank": SHARED_RANK, "expert_rank": EXPERT_RANK,
               "cities": CITY_NAMES, "epochs": []}

    for epoch in range(1, args.epochs + 1):
        log(f"\nEpoch {epoch}/{args.epochs}")
        model.train()

        # Build per-city iterators
        city_iters = {}
        for c in CITY_NAMES:
            random.shuffle(city_files[c])
            city_iters[c] = iter(city_files[c][:args.scenes_per_city])

        city_rewards = {c: [] for c in CITY_NAMES}
        city_kls = {c: [] for c in CITY_NAMES}
        total_steps = 0

        # Round-robin through cities
        for step_i in range(args.scenes_per_city):
            for city_idx, city in enumerate(CITY_NAMES):
                try:
                    path = next(city_iters[city])
                except StopIteration:
                    continue
                try:
                    set_active_expert(model, city_idx)
                    data = load_npz(path, DEVICE)
                    data_norm = cfg.observation_normalizer(
                        {k: v.clone() if isinstance(v, torch.Tensor) else v
                         for k, v in data.items()})
                    ego_traj = dpm_sample(model, cfg, data_norm, noise_schedule)
                    init_state = extract_initial_state(data, DEVICE)
                    planner_xy = torch.cat(
                        [torch.zeros(1, 2, device=DEVICE, dtype=ego_traj.dtype),
                         ego_traj[:, :2]], dim=0)
                    rolled = rollout_from_planner_output(planner_xy, init_state, rollout_cfg)
                    rolled_traj = ego_traj.clone()
                    rolled_traj[:, :2] = rolled[1:, :2]
                    reward = compute_cls_reward(rolled_traj, data)

                    with torch.no_grad():
                        _, base_dec = base_model(data_norm)
                        base_traj = base_dec['prediction'][0, 0]
                    kl_penalty = torch.mean((ego_traj - base_traj.detach()) ** 2)
                    loss = -reward + KL_BETA * kl_penalty
                    if torch.isnan(loss) or torch.isinf(loss) or torch.isnan(reward):
                        optimizer.zero_grad(); continue
                    (loss / GRAD_ACCUM).backward()

                    total_steps += 1
                    if total_steps % GRAD_ACCUM == 0:
                        grad_nan = any(torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
                                       for p in model.parameters() if p.grad is not None)
                        if grad_nan:
                            optimizer.zero_grad(); continue
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                        optimizer.step(); optimizer.zero_grad()

                    city_rewards[city].append(reward.item())
                    city_kls[city].append(kl_penalty.item())
                except Exception:
                    optimizer.zero_grad(); continue

                if total_steps % CHECKPOINT_EVERY == 0:
                    w_nan = any(torch.isnan(p).any()
                                for n, p in model.named_parameters()
                                if 'shared_' in n or 'expert_' in n)
                    if not w_nan:
                        ckpt_state = {k: v.detach().clone()
                                      for k, v in model.state_dict().items()
                                      if 'shared_' in k or 'expert_' in k}
                        torch.save(ckpt_state,
                                   os.path.join(OUT_DIR, f'selora_snapshot_epoch{epoch}_step{total_steps}.pth'))

            if (step_i + 1) % 250 == 0:
                avg_r = {c: np.mean(rs[-250:]) if rs else 0 for c, rs in city_rewards.items()}
                overall = np.mean([v for v in avg_r.values()])
                log(f"  [{step_i+1}/{args.scenes_per_city}] " +
                    " ".join(f"{c[:3]}={avg_r[c]:.3f}" for c in CITY_NAMES) +
                    f" | overall={overall:.3f}")

        # Epoch summary
        epoch_result = {"epoch": epoch}
        for c in CITY_NAMES:
            epoch_result[c] = float(np.mean(city_rewards[c])) if city_rewards[c] else 0.0
            epoch_result[f"{c}_n"] = len(city_rewards[c])
        epoch_result["overall"] = float(np.mean([epoch_result[c] for c in CITY_NAMES]))
        results["epochs"].append(epoch_result)
        log(f"  Epoch {epoch}: " +
            " ".join(f"{c[:3]}={epoch_result[c]:.3f}" for c in CITY_NAMES) +
            f" | overall={epoch_result['overall']:.3f}")
        with open(os.path.join(OUT_DIR, 'training_result.json'), 'w') as f:
            json.dump(results, f, indent=2)

    # Save final SE-LoRA state
    selora_state = {k: v for k, v in model.state_dict().items()
                    if 'shared_' in k or 'expert_' in k}
    torch.save(selora_state, os.path.join(OUT_DIR, f'selora_phase15_seed{SEED}.pth'))
    log(f"\nDONE. seed={SEED}, final overall reward={results['epochs'][-1]['overall']:.3f}")


if __name__ == '__main__':
    main()
