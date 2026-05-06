#!/usr/bin/env python3
"""
Train LoRA on nuPlan data with 4 solvers: DPM++, DDIM η=0, DDIM η=0.3, Euler.
Saves checkpoints for closed-loop evaluation.
"""
import sys, os, time, json, math, random, glob
import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
import diffusion_planner.model.diffusion_utils.dpm_solver_pytorch as dpm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = os.path.join(SCRIPT_DIR, 'checkpoints/model.pth')
ARGS_PATH = os.path.join(SCRIPT_DIR, 'checkpoints/args.json')
NPZ_DIR = './data/processed_npz'
OUT_DIR = './checkpoints/lora'
os.makedirs(OUT_DIR, exist_ok=True)

EPOCHS = 3  # best from hyperparam sweep
LR = 1e-6  # best from hyperparam sweep
LORA_RANK = 16
LORA_ALPHA = 32.0
GRAD_ACCUM = 4
MAX_SCENES = 20000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class LoRALinear(nn.Module):
    def __init__(self, original, rank=16, alpha=32.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        in_f = original.in_features
        out_f = original.out_features
        self.lora_A = nn.Parameter(torch.randn(rank, in_f) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))

    def forward(self, x):
        return self.original(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale


def apply_lora(model, rank=16, alpha=32.0):
    dit = model.decoder.decoder.dit
    dit.preproj.fc1 = LoRALinear(dit.preproj.fc1, rank, alpha)
    dit.preproj.fc2 = LoRALinear(dit.preproj.fc2, rank, alpha)
    for block in dit.blocks:
        block.mlp1.fc1 = LoRALinear(block.mlp1.fc1, rank, alpha)
        block.mlp1.fc2 = LoRALinear(block.mlp1.fc2, rank, alpha)
        block.mlp2.fc1 = LoRALinear(block.mlp2.fc1, rank, alpha)
        block.mlp2.fc2 = LoRALinear(block.mlp2.fc2, rank, alpha)
    dit.final_layer.proj[1] = LoRALinear(dit.final_layer.proj[1], rank, alpha)
    dit.final_layer.proj[4] = LoRALinear(dit.final_layer.proj[4], rank, alpha)


def freeze_non_lora(model):
    for name, param in model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False


def save_lora_checkpoint(model, path):
    """Save only LoRA parameters."""
    lora_state = {k: v for k, v in model.state_dict().items() if 'lora_' in k}
    torch.save(lora_state, path)
    log(f"  Saved LoRA checkpoint ({len(lora_state)} params) to {path}")


def save_full_merged_checkpoint(model, path):
    """Save full model state dict (for closed-loop eval)."""
    torch.save({'model': model.state_dict()}, path)
    log(f"  Saved full checkpoint to {path}")


def load_npz(path, device):
    loaded = np.load(path)
    data = {}
    for key, value in loaded.items():
        if key in {"map_name", "token"}:
            continue
        arr = np.expand_dims(value, axis=0)
        if arr.dtype == np.bool_:
            data[key] = torch.tensor(arr, dtype=torch.bool, device=device)
        elif arr.dtype in (np.int32, np.int64):
            data[key] = torch.tensor(arr, dtype=torch.long, device=device)
        else:
            data[key] = torch.tensor(arr, dtype=torch.float32, device=device)
    # Do NOT truncate neighbors here — encoder needs all 32, decoder handles truncation internally
    return data


from nuplan_reward import compute_nuplan_reward


def compute_reward_nuplan(ego_traj, data):
    """nuPlan-aligned reward using expert, neighbors, route lanes."""
    expert = data.get('ego_agent_future')
    nbrs = data.get('neighbor_agents_future')
    route = data.get('route_lanes')

    if expert is None or nbrs is None or route is None:
        # Fallback to simple reward
        xy = ego_traj[:, :2]
        step_d = torch.norm(xy[1:] - xy[:-1], dim=1)
        return torch.clamp(step_d.sum(), max=200.0) * 0.005

    # Remove batch dim if present
    if expert.dim() == 3:
        expert = expert[0]
    if nbrs.dim() == 4:
        nbrs = nbrs[0]
    if route.dim() == 4:
        route = route[0]

    reward, _ = compute_nuplan_reward(ego_traj, expert, nbrs, route)
    return reward


def _get_decoder_internals(model, data_norm, cfg):
    """Extract encoder output and decoder internals for custom sampling."""
    P = 1 + cfg.predicted_neighbor_num
    with torch.no_grad():
        enc_out = model.encoder(data_norm)

    decoder = model.decoder.decoder
    dit = decoder.dit

    ego_cur = data_norm["ego_current_state"][:, :4]
    nbr_cur = data_norm["neighbor_agents_past"][:, :P-1, -1, :4]
    current_states = torch.cat([ego_cur[:, None], nbr_cur], dim=1)

    nbr_past = data_norm["neighbor_agents_past"]
    # neighbor_mask for decoder: only first P-1 neighbors
    neighbor_mask = (nbr_past[:, :P-1, -1, :].abs().sum(-1) < 1e-6)
    route_lanes = data_norm.get("route_lanes")

    ns = dpm.NoiseScheduleVP(schedule='linear')

    # enc_out is a dict; pass the tensor
    encoding = enc_out['encoding'] if isinstance(enc_out, dict) else enc_out

    return dit, ns, encoding, current_states, neighbor_mask, route_lanes


def _differentiable_dpm_sampler(dit, xT, noise_schedule, other_model_params, correcting_xt_fn, steps=10):
    """DPM-Solver++ without torch.no_grad() — allows gradient flow for LoRA training.
    Mirrors the library's dpm_sampler but differentiable."""
    model_fn = dpm.model_wrapper(
        dit, noise_schedule,
        model_type=dit.model_type,
        model_kwargs=other_model_params,
        guidance_type="uncond")

    solver = dpm.DPM_Solver(model_fn, noise_schedule, correcting_xt_fn=correcting_xt_fn)

    # Replicate DPM_Solver.sample() logic WITHOUT no_grad
    device = xT.device
    timesteps = solver.get_time_steps(skip_type='logSNR', t_T=1.0, t_0=1e-3, N=steps, device=device)
    order = 2

    step = 0
    t = timesteps[step]
    t_prev_list = [t]
    model_prev_list = [solver.model_fn(xT, t)]
    if correcting_xt_fn is not None:
        xT = correcting_xt_fn(xT, t, step)

    x = xT
    for step in range(1, steps + 1):
        t_cur = timesteps[step]
        if step < order:
            x = solver.dpm_solver_first_update(x, t_prev_list[-1], t_cur, model_s=model_prev_list[-1])
        else:
            x = solver.multistep_dpm_solver_second_update(x, model_prev_list, t_prev_list, t_cur)

        if correcting_xt_fn is not None:
            x = correcting_xt_fn(x, t_cur, step)

        # Update history
        model_prev_list.append(solver.model_fn(x, t_cur))
        t_prev_list.append(t_cur)
        if len(model_prev_list) > order:
            model_prev_list.pop(0)
            t_prev_list.pop(0)

    # denoise_to_zero
    t_0 = torch.tensor(1e-3, device=device)
    if timesteps[-1] != t_0:
        x = solver.denoise_to_zero_fn(x, timesteps[-1])
        if correcting_xt_fn is not None:
            x = correcting_xt_fn(x, t_0, steps)

    return x


def dpm_sample(model, cfg, data_norm, noise_schedule_unused):
    """Differentiable DPM-Solver++ — uses library internals but without no_grad."""
    B = 1
    P = 1 + cfg.predicted_neighbor_num
    future_len = cfg.future_len

    dit, ns, enc_out, current_states, neighbor_mask, route_lanes = \
        _get_decoder_internals(model, data_norm, cfg)

    def correcting_xt_fn(xt, t, step):
        xt_r = xt.reshape(B, P, -1, 4)
        xt_out = torch.cat([current_states.unsqueeze(2), xt_r[:, :, 1:, :]], dim=2)
        return xt_out.reshape(B, P, -1)

    xT = torch.cat([current_states[:, :, None],
                     torch.randn(B, P, future_len, 4, device=DEVICE) * 0.5],
                    dim=2).reshape(B, P, -1)

    x0 = _differentiable_dpm_sampler(
        dit, xT, ns,
        other_model_params={"cross_c": enc_out, "route_lanes": route_lanes,
                            "neighbor_current_mask": neighbor_mask},
        correcting_xt_fn=correcting_xt_fn, steps=10)

    x0_denorm = cfg.state_normalizer.inverse(x0.reshape(B, P, -1, 4))
    return x0_denorm[0, 0, 1:]


def _make_solver_and_init(model, cfg, data_norm):
    """Common setup for all samplers: encoder, noise schedule, initial state."""
    B = 1
    P = 1 + cfg.predicted_neighbor_num
    future_len = cfg.future_len

    dit, ns, enc_out, current_states, neighbor_mask, route_lanes = \
        _get_decoder_internals(model, data_norm, cfg)

    model_fn = dpm.model_wrapper(
        dit, ns, model_type=dit.model_type,
        model_kwargs={"cross_c": enc_out, "route_lanes": route_lanes,
                      "neighbor_current_mask": neighbor_mask},
        guidance_type="uncond")

    def correcting_xt_fn(xt, t, step):
        xt_r = xt.reshape(B, P, -1, 4)
        xt_out = torch.cat([current_states.unsqueeze(2), xt_r[:, :, 1:, :]], dim=2)
        return xt_out.reshape(B, P, -1)

    solver = dpm.DPM_Solver(model_fn, ns, correcting_xt_fn=correcting_xt_fn)

    xT = torch.cat([current_states[:, :, None],
                     torch.randn(B, P, future_len, 4, device=DEVICE) * 0.5],
                    dim=2).reshape(B, P, -1)

    return solver, ns, xT, correcting_xt_fn, cfg.state_normalizer, B, P


def ddim_sample(model, cfg, data_norm, noise_schedule_unused):
    """DDIM = DPM-Solver 1st order only. Uses library internals for correct math."""
    solver, ns, xT, correct_fn, state_norm, B, P = _make_solver_and_init(model, cfg, data_norm)

    steps = 10
    timesteps = solver.get_time_steps(skip_type='logSNR', t_T=1.0, t_0=1e-3, N=steps, device=DEVICE)

    x = xT
    if correct_fn:
        x = correct_fn(x, timesteps[0], 0)

    # All 1st-order steps (= DDIM)
    for i in range(steps):
        model_s = solver.model_fn(x, timesteps[i])
        x = solver.dpm_solver_first_update(x, timesteps[i], timesteps[i+1], model_s=model_s)
        if correct_fn:
            x = correct_fn(x, timesteps[i+1], i+1)

    x = solver.denoise_to_zero_fn(x, timesteps[-1])
    if correct_fn:
        x = correct_fn(x, torch.tensor(1e-3, device=DEVICE), steps)

    x0_denorm = state_norm.inverse(x.reshape(B, P, -1, 4))
    return x0_denorm[0, 0, 1:]


def ddim_stochastic_sample(model, cfg, data_norm, noise_schedule_unused):
    """DDIM with η=0.3: add noise at each step. Expected to cause gradient explosion."""
    solver, ns, xT, correct_fn, state_norm, B, P = _make_solver_and_init(model, cfg, data_norm)
    eta = 0.3

    steps = 10
    timesteps = solver.get_time_steps(skip_type='logSNR', t_T=1.0, t_0=1e-3, N=steps, device=DEVICE)

    x = xT
    if correct_fn:
        x = correct_fn(x, timesteps[0], 0)

    # 1st order steps with injected stochastic noise (η=0.3)
    for i in range(steps):
        # Deterministic DDIM step
        model_s = solver.model_fn(x, timesteps[i])
        x_det = solver.dpm_solver_first_update(x, timesteps[i], timesteps[i+1], model_s=model_s)

        # Add stochastic noise scaled by eta
        sigma_now = ns.marginal_std(timesteps[i])
        sigma_next = ns.marginal_std(timesteps[i+1])
        noise_scale = eta * sigma_next
        x = x_det + noise_scale * torch.randn_like(x_det)

        if correct_fn:
            x = correct_fn(x, timesteps[i+1], i+1)

    x = solver.denoise_to_zero_fn(x, timesteps[-1])
    if correct_fn:
        x = correct_fn(x, torch.tensor(1e-3, device=DEVICE), steps)

    x0_denorm = state_norm.inverse(x.reshape(B, P, -1, 4))
    return x0_denorm[0, 0, 1:]


def euler_sample(model, cfg, data_norm, noise_schedule_unused):
    """Euler 1st order with uniform timesteps (not logSNR). Shortest gradient path."""
    solver, ns, xT, correct_fn, state_norm, B, P = _make_solver_and_init(model, cfg, data_norm)

    steps = 10
    # Euler uses uniform timesteps instead of logSNR
    timesteps = torch.linspace(1.0, 1e-3, steps + 1).to(DEVICE)

    x = xT
    if correct_fn:
        x = correct_fn(x, timesteps[0], 0)

    # All 1st-order steps with uniform schedule
    for i in range(steps):
        model_s = solver.model_fn(x, timesteps[i])
        x = solver.dpm_solver_first_update(x, timesteps[i], timesteps[i+1], model_s=model_s)
        if correct_fn:
            x = correct_fn(x, timesteps[i+1], i+1)

    x = solver.denoise_to_zero_fn(x, timesteps[-1])
    if correct_fn:
        x = correct_fn(x, torch.tensor(1e-3, device=DEVICE), steps)

    x0_denorm = state_norm.inverse(x.reshape(B, P, -1, 4))
    return x0_denorm[0, 0, 1:]


KL_BETA = 0.3  # best from hyperparam sweep


def train_solver(name, solver_fn, cfg, train_paths, tag):
    log(f"\n{'='*60}")
    log(f"Training: {name} (KL_BETA={KL_BETA})")
    log(f"{'='*60}")

    # Create on CPU, load weights, apply LoRA, THEN move to DEVICE
    model = Diffusion_Planner(cfg)
    ckpt = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    state = ckpt.get('ema_state_dict', ckpt.get('model', ckpt))
    if any(k.startswith('module.') for k in state.keys()):
        state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    apply_lora(model, LORA_RANK, LORA_ALPHA)
    freeze_non_lora(model)
    model = model.to(DEVICE)

    # Base model (frozen, no LoRA) for KL reference
    base_model = Diffusion_Planner(cfg)
    base_model.load_state_dict(state, strict=False)
    base_model = base_model.to(DEVICE).eval()
    for p in base_model.parameters():
        p.requires_grad = False

    noise_schedule = dpm.NoiseScheduleVP('linear', continuous_beta_0=0.1, continuous_beta_1=20.)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)

    results = {"solver": name, "epochs": []}

    for epoch in range(1, EPOCHS + 1):
        log(f"  Epoch {epoch}/{EPOCHS}")
        model.train()
        random.shuffle(train_paths)
        total_r = 0; n = 0

        for i, path in enumerate(train_paths):
            try:
                data = load_npz(path, DEVICE)
                data_norm = cfg.observation_normalizer(
                    {k: v.clone() if isinstance(v, torch.Tensor) else v
                     for k, v in data.items()})

                ego_traj = solver_fn(model, cfg, data_norm, noise_schedule)
                reward = compute_reward_nuplan(ego_traj, data)

                # KL regularization: penalize deviation from base model output
                with torch.no_grad():
                    _, base_dec = base_model(data_norm)
                    base_traj = base_dec['prediction'][0, 0]  # (T, 4)
                kl_penalty = torch.mean((ego_traj - base_traj.detach()) ** 2)

                loss = -reward + KL_BETA * kl_penalty
                (loss / GRAD_ACCUM).backward()

                if (i + 1) % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                total_r += reward.item()
                n += 1
            except Exception as e:
                if i < 3:
                    log(f"    Error: {e}")
                continue

            if (i + 1) % 500 == 0:
                log(f"    [{i+1}/{len(train_paths)}] reward={total_r/n:.2f}")

        mean_r = total_r / max(n, 1)
        log(f"  Reward: {mean_r:.2f} ({n} steps)")
        results["epochs"].append({"epoch": epoch, "reward": mean_r})

    # Save checkpoints
    save_lora_checkpoint(model, os.path.join(OUT_DIR, f'lora_{tag}.pth'))
    save_full_merged_checkpoint(model, os.path.join(OUT_DIR, f'model_{tag}.pth'))

    return results, model


def main():
    log("="*60)
    log("nuPlan LoRA Training for Closed-Loop Evaluation")
    log(f"Data: {NPZ_DIR}")
    log(f"Scenes: {MAX_SCENES}, Epochs: {EPOCHS}")
    log("="*60)

    cfg = Config(ARGS_PATH, guidance_fn=None)

    all_npz = sorted(glob.glob(os.path.join(NPZ_DIR, '*.npz')))
    random.seed(42)
    random.shuffle(all_npz)
    train_paths = all_npz[:MAX_SCENES]
    log(f"Using {len(train_paths)} scenes")

    all_results = []

    # Fresh training with nuPlan-aligned reward — retrain both solvers
    solvers = [
        ("DPM++ 2nd (MATCHED)",   dpm_sample,              "dpm"),
        ("DDIM eta=0",             ddim_sample,             "ddim"),
    ]

    for name, fn, tag in solvers:
        try:
            r, _ = train_solver(name, fn, cfg, train_paths, tag)
            all_results.append(r)
        except Exception as e:
            log(f"FAILED: {name} — {e}")
            all_results.append({"solver": name, "epochs": [], "error": str(e)})
        torch.cuda.empty_cache()

        # Save intermediate results after each solver
        with open(os.path.join(OUT_DIR, 'training_results.json'), 'w') as f:
            json.dump(all_results, f, indent=2)

    # Final summary
    log("\n" + "="*60)
    log("TRAINING RESULTS (4 solvers)")
    log("="*60)
    for r in all_results:
        eps = r.get("epochs", [])
        if eps:
            log(f"{r['solver']:<25} Ep1={eps[0]['reward']:.1f}  Ep5={eps[-1]['reward']:.1f}")
        else:
            log(f"{r['solver']:<25} DIVERGED/FAILED")

    with open(os.path.join(OUT_DIR, 'training_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    log(f"Saved to {OUT_DIR}")
    log("DONE — now run closed-loop eval with sim_lora_runner.sh")


if __name__ == "__main__":
    main()
