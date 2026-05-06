#!/usr/bin/env python3
"""Phase 13: SE-LoRA + VD-GRPO + nocomfort reward (best-of-all).

Combines:
- MoLoRA architecture (shared + per-city expert)
- Differentiable DPM-Solver++ rollout
- nocomfort reward (drops C term from weighted_avg, validated by nocomfort_600 = 0.8994)
- VD-GRPO advantage estimation (Plan-R1 / Autoware RA-L paper):
    1. K trajectories per scene with different initial noise
    2. Advantage = (R_k - mean_R) / C  (NO std division — preserves rare-event grad)
    3. Positive-only filter: only update on A_k > 0
    4. Loss = -(1/|K+|) Σ A_k · R_k  (gradient flows through differentiable solver)

Hypothesis: group baseline + positive filter prevents reward hacking
(Phase 10 seed 313 issue) and stabilizes against reward-proxy / metric mismatch.
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
    CITY_NAMES, get_city,
)
from train_lora_for_closedloop import load_npz, dpm_sample
from differentiable_rollout import rollout_from_planner_output, RolloutConfig
from nuplan_reward_cls_pcdr import compute_cls_reward  # SE-LoRA + VD-GRPO + PCDR (5 ops)

DEVICE = torch.device('cuda')
MODEL_PATH = f'{SCRIPT_DIR}/checkpoints/model.pth'
ARGS_PATH = f'{SCRIPT_DIR}/checkpoints/args.json'
NPZ_DIR = os.environ.get('PHASE13_NPZ_DIR', './data/processed_npz')

SHARED_RANK = 4
EXPERT_RANK = 8
LORA_ALPHA = 32.0
LR = 1e-6
KL_BETA = 0.1   # lower than Phase 11 (0.3) — group baseline already stabilizes
GRAD_ACCUM = 4
EPOCHS = 3
SCENES_PER_CITY = 1000
K_SAMPLES = 2   # group size (paper used 8, we use 2 for memory + speed)
ADV_C = 1.0     # variance-decoupled scaling constant
CHECKPOINT_EVERY = 500


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


def vd_grpo_step(model, cfg, data_norm, data, noise_schedule, rollout_cfg, K):
    """Sample K trajectories, compute VD-GRPO advantages and positive-filtered loss.

    Returns: (loss, mean_reward, n_positive) — loss is differentiable, others are floats.
    """
    rewards = []
    egos = []
    for k in range(K):
        # Each call to dpm_sample generates a fresh torch.randn → different noise init
        ego_traj = dpm_sample(model, cfg, data_norm, noise_schedule)
        init_state = extract_initial_state(data, DEVICE)
        planner_xy = torch.cat(
            [torch.zeros(1, 2, device=DEVICE, dtype=ego_traj.dtype),
             ego_traj[:, :2]], dim=0)
        rolled = rollout_from_planner_output(planner_xy, init_state, rollout_cfg)
        rolled_traj = ego_traj.clone()
        rolled_traj[:, :2] = rolled[1:, :2]
        r = compute_cls_reward(rolled_traj, data)
        rewards.append(r)
        egos.append(ego_traj)
    R = torch.stack(rewards)  # (K,)
    mean_R = R.mean()
    advantages = (R - mean_R.detach()) / ADV_C  # (K,)
    pos_mask = advantages > 0  # (K,) bool
    n_pos = int(pos_mask.sum().item())
    if n_pos == 0:
        # All trajectories tied — no gradient signal from VD-GRPO
        return None, mean_R.item(), 0
    # Differentiate through R_k weighted by A_k (advantage acts as weight; gradient flows through R_k)
    loss = -(advantages[pos_mask].detach() * R[pos_mask]).mean()
    return loss, mean_R.item(), n_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--out_dir', type=str, required=True)
    ap.add_argument('--scenes_per_city', type=int, default=SCENES_PER_CITY)
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--K', type=int, default=K_SAMPLES)
    args = ap.parse_args()

    OUT_DIR = args.out_dir
    os.makedirs(OUT_DIR, exist_ok=True)
    SEED = args.seed
    K = args.K

    log("=" * 60)
    log(f"3-IN-1: SE-LoRA + VD-GRPO + PCDR-5ops reward (NC+EP+DDC+DAC+SC)  (seed={SEED})")
    log(f"  shared_rank={SHARED_RANK}, expert_rank={EXPERT_RANK}, n_experts={len(CITY_NAMES)}")
    log(f"  K_samples={K}, scenes/city/epoch={args.scenes_per_city}, epochs={args.epochs}")
    log(f"  LR={LR} KL={KL_BETA} ADV_C={ADV_C}  reward=CLS-STE")
    log("=" * 60)

    random.seed(SEED); torch.manual_seed(SEED); np.random.seed(SEED)

    cfg = Config(ARGS_PATH, guidance_fn=None)

    all_npz = glob.glob(os.path.join(NPZ_DIR, '*.npz'))
    city_files = collections.defaultdict(list)
    for f in all_npz:
        c = get_city(f)
        if c != 'unknown':
            city_files[c].append(f)
    for c in CITY_NAMES:
        log(f"  {c}: {len(city_files[c])} scenes available")

    model = Diffusion_Planner(cfg)
    ckpt = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    state = ckpt.get('ema_state_dict', ckpt.get('model', ckpt))
    if any(k.startswith('module.') for k in state.keys()):
        state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    apply_molora(model, SHARED_RANK, EXPERT_RANK, len(CITY_NAMES), LORA_ALPHA)
    freeze_non_lora(model)
    model = model.to(DEVICE)

    base_model = Diffusion_Planner(cfg)
    base_model.load_state_dict(state, strict=False)
    base_model = base_model.to(DEVICE).eval()
    for p in base_model.parameters():
        p.requires_grad = False

    noise_schedule = dpm.NoiseScheduleVP('linear', continuous_beta_0=0.1, continuous_beta_1=20.)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
    rollout_cfg = RolloutConfig(dt=0.1, n_steps=80)

    results = {"name": "phase13_combined", "seed": SEED, "K": K, "lr": LR,
               "shared_rank": SHARED_RANK, "expert_rank": EXPERT_RANK,
               "cities": CITY_NAMES, "epochs": []}

    for epoch in range(1, args.epochs + 1):
        log(f"\nEpoch {epoch}/{args.epochs}")
        model.train()

        city_iters = {}
        for c in CITY_NAMES:
            random.shuffle(city_files[c])
            city_iters[c] = iter(city_files[c][:args.scenes_per_city])

        city_rewards = {c: [] for c in CITY_NAMES}
        city_n_pos = {c: [] for c in CITY_NAMES}
        total_steps = 0

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

                    # ── VD-GRPO: K samples + advantage filter ──
                    grpo_loss, mean_r, n_pos = vd_grpo_step(
                        model, cfg, data_norm, data, noise_schedule, rollout_cfg, K)
                    if grpo_loss is None:
                        # all K rewards tied — no signal, skip without backprop
                        city_rewards[city].append(mean_r)
                        city_n_pos[city].append(0)
                        continue

                    # KL anchor (computed on a single sample for cost)
                    with torch.no_grad():
                        ego_kl_sample = dpm_sample(model, cfg, data_norm, noise_schedule)
                        _, base_dec = base_model(data_norm)
                        base_traj = base_dec['prediction'][0, 0]
                    kl_pen = torch.mean((ego_kl_sample - base_traj.detach()) ** 2)

                    loss = grpo_loss + KL_BETA * kl_pen
                    if torch.isnan(loss) or torch.isinf(loss):
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

                    city_rewards[city].append(mean_r)
                    city_n_pos[city].append(n_pos)
                except Exception:
                    optimizer.zero_grad(); continue

                if total_steps > 0 and total_steps % CHECKPOINT_EVERY == 0:
                    w_nan = any(torch.isnan(p).any()
                                for n, p in model.named_parameters()
                                if 'shared_' in n or 'expert_' in n)
                    if not w_nan:
                        ckpt_state = {k: v.detach().clone()
                                      for k, v in model.state_dict().items()
                                      if 'shared_' in k or 'expert_' in k}
                        torch.save(ckpt_state,
                                   os.path.join(OUT_DIR, f'selora_vdgrpo_pcdr5_snapshot_epoch{epoch}_step{total_steps}.pth'))

            if (step_i + 1) % 100 == 0:
                avg_r = {c: np.mean(rs[-100:]) if rs else 0 for c, rs in city_rewards.items()}
                avg_pos = {c: np.mean(ps[-100:]) if ps else 0 for c, ps in city_n_pos.items()}
                overall = np.mean([v for v in avg_r.values()])
                log(f"  [{step_i+1}/{args.scenes_per_city}] " +
                    " ".join(f"{c[:3]}={avg_r[c]:.3f}(p={avg_pos[c]:.1f})"
                             for c in CITY_NAMES) +
                    f" | overall={overall:.3f}")

        epoch_result = {"epoch": epoch}
        for c in CITY_NAMES:
            epoch_result[c] = float(np.mean(city_rewards[c])) if city_rewards[c] else 0.0
            epoch_result[f"{c}_n"] = len(city_rewards[c])
            epoch_result[f"{c}_avg_pos"] = float(np.mean(city_n_pos[c])) if city_n_pos[c] else 0.0
        epoch_result["overall"] = float(np.mean([epoch_result[c] for c in CITY_NAMES]))
        results["epochs"].append(epoch_result)
        log(f"  Epoch {epoch}: overall={epoch_result['overall']:.3f}  " +
            " ".join(f"{c[:3]}={epoch_result[c]:.3f}" for c in CITY_NAMES))
        with open(os.path.join(OUT_DIR, 'training_result.json'), 'w') as f:
            json.dump(results, f, indent=2)

    selora_state = {k: v for k, v in model.state_dict().items()
                    if 'shared_' in k or 'expert_' in k}
    torch.save(selora_state, os.path.join(OUT_DIR, f'selora_vdgrpo_pcdr5_seed{SEED}.pth'))
    log(f"\nDONE. seed={SEED}, final overall reward={results['epochs'][-1]['overall']:.3f}")


if __name__ == '__main__':
    main()
