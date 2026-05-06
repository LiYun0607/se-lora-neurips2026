"""
Non-AD sanity check: solver-mismatch divergence on a pretrained image diffusion model.

Replicates §4.3's trajectory-divergence experiment on CIFAR-10 DDPM:
- For the same initial noise, generate samples with DDIM (matched-stochastic) vs DPM-Solver++ (matched-deterministic) at N=5, 10, 20 steps.
- Quantify divergence L2 between the produced images.
- Show: divergence shrinks monotonically with step count, consistent with O(h^p) prediction of backward error analysis.
- Implication: the train-deploy solver-mismatch claim of §3.1 generalizes beyond AD diffusion to standard image diffusion, supporting the Gronwall amplification's universality.

Pretrained model: google/ddpm-cifar10-32 (HuggingFace, ~32MB, unconditional CIFAR-10).
"""

import time
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from diffusers import DDPMPipeline, DDIMScheduler, DPMSolverMultistepScheduler

torch.manual_seed(42)
np.random.seed(42)
sns.set_theme(style="whitegrid", context="paper", rc={"font.family": "serif", "font.size": 9})

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "google/ddpm-cifar10-32"
N_SEEDS = 30   # CPU-friendly; gives reasonable error bars
STEPS = [5, 10, 20]

print(f"loading {MODEL_ID} on {DEVICE}...")
pipe = DDPMPipeline.from_pretrained(MODEL_ID)
unet = pipe.unet.to(DEVICE).eval()
img_shape = (3, 32, 32)
print(f"model loaded. UNet params: {sum(p.numel() for p in unet.parameters())/1e6:.1f}M")

def make_schedulers(num_steps):
    """Return DDIM and DPM-Solver++ schedulers configured for the given step count.
    Use DDPM defaults (T=1000, linear betas) matching the google/ddpm-cifar10-32 training."""
    ddim = DDIMScheduler(num_train_timesteps=1000, beta_start=1e-4, beta_end=0.02,
                         beta_schedule="linear", clip_sample=True, set_alpha_to_one=False)
    ddim.set_timesteps(num_steps)
    dpm = DPMSolverMultistepScheduler(num_train_timesteps=1000, beta_start=1e-4, beta_end=0.02,
                                       beta_schedule="linear", solver_order=2,
                                       algorithm_type="dpmsolver++")
    dpm.set_timesteps(num_steps)
    return ddim, dpm

def sample_with_scheduler(noise, scheduler):
    """Generate one image with the given scheduler from given initial noise."""
    x = noise.clone()
    with torch.no_grad():
        for t in scheduler.timesteps:
            t_tensor = t.to(DEVICE) if torch.is_tensor(t) else torch.tensor(t, device=DEVICE)
            model_output = unet(x, t_tensor).sample
            x = scheduler.step(model_output, t, x).prev_sample
    return x

# Generate same-noise pairs at each step budget
print(f"\nGenerating {N_SEEDS} same-noise pairs at each step count {STEPS}...")
results = {}
for n_steps in STEPS:
    ddim_sched, dpm_sched = make_schedulers(n_steps)
    divergences = []
    t0 = time.time()
    for seed in range(N_SEEDS):
        torch.manual_seed(seed)
        noise = torch.randn(1, *img_shape, device=DEVICE)

        x_ddim = sample_with_scheduler(noise, ddim_sched)
        x_dpm = sample_with_scheduler(noise, dpm_sched)

        # L2 divergence per pixel (averaged over channels and pixels)
        diff = (x_ddim - x_dpm).flatten()
        l2_per_pixel = torch.sqrt((diff ** 2).mean()).item()
        # Endpoint divergence (full L2 norm)
        l2_norm = torch.norm(diff).item()
        divergences.append({"seed": seed, "l2_per_pixel": l2_per_pixel, "l2_norm": l2_norm})

        if seed % 10 == 0:
            elapsed = time.time() - t0
            print(f"  N={n_steps:>2} steps, seed {seed}: L2/pix={l2_per_pixel:.4f}  ({elapsed:.0f}s)")

    arr_per_pix = np.array([d["l2_per_pixel"] for d in divergences])
    arr_norm = np.array([d["l2_norm"] for d in divergences])
    results[n_steps] = {
        "mean_l2_per_pixel": float(arr_per_pix.mean()),
        "std_l2_per_pixel":  float(arr_per_pix.std()),
        "mean_l2_norm":      float(arr_norm.mean()),
        "std_l2_norm":       float(arr_norm.std()),
        "max_l2_norm":       float(arr_norm.max()),
        "n_samples": N_SEEDS,
        "all_l2_per_pix": arr_per_pix.tolist(),
    }
    print(f"  N={n_steps}: mean L2/pix = {arr_per_pix.mean():.4f} ± {arr_per_pix.std():.4f}")

print("\n=== KEY RESULT ===")
for n_steps in STEPS:
    r = results[n_steps]
    print(f"  N={n_steps:>2}:  L2/pix = {r['mean_l2_per_pixel']:.4f} ± {r['std_l2_per_pixel']:.4f}   "
          f"L2-norm = {r['mean_l2_norm']:.2f} ± {r['std_l2_norm']:.2f}")

# Save
with open("../data/sanity_image_diffusion_divergence.json", "w") as f:
    json.dump(results, f, indent=2)
print("saved ../data/sanity_image_diffusion_divergence.json")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))

# Panel (a): bar chart of L2/pix vs N steps
ax = axes[0]
xs = np.array(STEPS)
means = np.array([results[n]["mean_l2_per_pixel"] for n in STEPS])
stds = np.array([results[n]["std_l2_per_pixel"] for n in STEPS])
ax.bar(np.arange(len(xs)), means, yerr=stds, capsize=4,
       color=["#d62728", "#ff7f0e", "#2ca02c"], edgecolor="black", lw=0.6)
for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(i, m + s + 0.005, f"{m:.4f}", ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(np.arange(len(xs)))
ax.set_xticklabels([f"N={n}" for n in xs])
ax.set_ylabel(r"per-pixel $L_2$ divergence")
ax.set_title("(a) DDIM vs DPM-Solver++ divergence (CIFAR-10 DDPM)")

# Panel (b): log-log plot to demonstrate O(h^p) scaling
ax = axes[1]
ax.errorbar(xs, means, yerr=stds, fmt="o-", color="#1f77b4", capsize=4, lw=1.6, label="empirical")
# Fit O(1/N) and O(1/N^2) reference lines
ref1 = means[0] * (xs[0] / xs)
ref2 = means[0] * (xs[0] / xs) ** 2
ax.plot(xs, ref1, color="gray", ls="--", lw=0.8, label=r"$\propto 1/N$ (1st order)")
ax.plot(xs, ref2, color="gray", ls=":", lw=0.8, label=r"$\propto 1/N^2$ (2nd order)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"sampling steps $N$")
ax.set_ylabel(r"per-pixel $L_2$ divergence")
ax.set_title(r"(b) $\mathcal{O}(h^p)$ scaling (log-log)")
ax.legend(fontsize=8, loc="upper right")

plt.tight_layout()
out = "../figures/fig_sanity_image_diffusion_divergence.pdf"
plt.savefig(out, bbox_inches="tight", pad_inches=0.05)
plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", pad_inches=0.05, dpi=160)
print(f"saved {out}")
