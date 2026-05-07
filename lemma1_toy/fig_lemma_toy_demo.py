"""1D toy demonstration of Lemma 1 (K=1).

Three panels: reward landscape, gradient magnitude (log), and gradient
ascent from theta_0=0.7. Compares bit-exact 0/1 indicator, two log-sum-exp
soft-min surrogates (beta=0.05 / 0.12), and PCDR-CR (proxy + bit-exact margin
auxiliary). NeurIPS-style: shared style helper, no hardcoded fontsizes.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from _neurips_style import apply

apply(style="ticks")

# Set2-derived palette: distinct hues for the four parameterizations.
palette = sns.color_palette("Set2", n_colors=4)
PALETTE = {
    "indicator": palette[2],   # blue-grey
    "soft005":   palette[1],   # orange
    "soft012":   palette[0],   # green
    "v4a":       palette[3],   # red
}

RHO = 0.40
theta = np.linspace(-0.5, 1.0, 1500)


def indicator(d):
    return (d > 0).astype(float)


def soft_min(d, beta):
    return 1.0 / (1.0 + np.exp(-d / beta))


def grad_indicator(d):
    return np.gradient(indicator(d), d[1] - d[0])


def grad_soft(d, beta):
    s = soft_min(d, beta)
    return s * (1 - s) / beta


def v4a_reward(d, beta_back, lam, rho):
    soft_main = 1.0 / (1.0 + np.exp(-d / 0.5))
    margin_pen = lam * beta_back * np.log(1 + np.exp((rho - d) / beta_back))
    return soft_main - margin_pen


def grad_v4a(d, beta_back, lam, rho):
    soft_main = 1.0 / (1.0 + np.exp(-d / 0.5))
    g_main = soft_main * (1 - soft_main) / 0.5
    g_pen = -lam / (1.0 + np.exp(-(rho - d) / beta_back))
    return g_main - g_pen


R_ind = indicator(theta)
R_005 = soft_min(theta, 0.05)
R_012 = soft_min(theta, 0.12)
R_v4a = v4a_reward(theta, beta_back=0.12, lam=0.20, rho=RHO)

EPS = 1e-4
G_ind = np.maximum(np.abs(grad_indicator(theta)), EPS)
G_005 = np.maximum(np.abs(grad_soft(theta, 0.05)), EPS)
G_012 = np.maximum(np.abs(grad_soft(theta, 0.12)), EPS)
G_v4a = np.maximum(np.abs(grad_v4a(theta, 0.12, 0.20, RHO)), EPS)


def gd_trajectory(grad_fn, theta_0=0.7, lr=0.005, n_iter=300):
    th = theta_0
    history = [th]
    for _ in range(n_iter):
        d = th
        if grad_fn == "indicator":
            g = 0.0
        elif grad_fn == "soft005":
            g = grad_soft(np.array([d]), 0.05)[0]
        elif grad_fn == "soft012":
            g = grad_soft(np.array([d]), 0.12)[0]
        else:  # v4a
            g = grad_v4a(np.array([d]), 0.12, 0.20, RHO)[0]
        th = th + lr * g
        history.append(th)
    return np.array(history)


traj = {
    "indicator": gd_trajectory("indicator"),
    "soft005":   gd_trajectory("soft005"),
    "soft012":   gd_trajectory("soft012"),
    "v4a":       gd_trajectory("v4a"),
}

fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.7))
LW = 1.0
LS = {"indicator": "-", "soft005": "--", "soft012": ":", "v4a": "-."}
LBL = {
    "indicator": r"bit-exact $\mathbf{1}[d{>}0]$",
    "soft005":   r"soft, $\beta{=}0.05$",
    "soft012":   r"soft, $\beta{=}0.12$",
    "v4a":       r"PCDR-CR",
}


def _plot(ax, x, ys, key_order):
    for k in key_order:
        ax.plot(x, ys[k], color=PALETTE[k], lw=LW, ls=LS[k], label=LBL[k])


keys = ["indicator", "soft005", "soft012", "v4a"]

# Panel (a): reward landscape
ax = axes[0]
_plot(ax, theta, {"indicator": R_ind, "soft005": R_005,
                  "soft012": R_012, "v4a": R_v4a}, keys)
ax.axvline(RHO, color="0.55", lw=0.4, ls=":")
ax.set_xlabel(r"signed margin $d$ (m)")
ax.set_ylabel(r"reward $R(d)$")
ax.set_title("(a) Reward landscape", loc="left")
ax.set_xlim(-0.3, 1.0)

# Panel (b): gradient magnitude (log scale)
ax = axes[1]
_plot(ax, theta, {"indicator": G_ind, "soft005": G_005,
                  "soft012": G_012, "v4a": G_v4a}, keys)
ax.axhline(0.05, color="0.55", lw=0.4, ls="-.")
ax.axvline(RHO, color="0.55", lw=0.4, ls=":")
ax.set_xlabel(r"signed margin $d$ (m)")
ax.set_ylabel(r"$|\nabla_{\!\theta} R|$")
ax.set_yscale("log")
ax.set_ylim(EPS * 0.5, 30)
ax.set_xlim(-0.3, 1.0)
ax.set_title("(b) Gradient magnitude (log)", loc="left")

# Panel (c): GD trajectory
ax = axes[2]
iters = np.arange(len(traj["indicator"]))
for k in keys:
    ax.plot(iters, traj[k], color=PALETTE[k], lw=LW, ls=LS[k], label=LBL[k])
ax.axhline(RHO, color="0.55", lw=0.4, ls=":")
ax.set_xlabel("gradient-ascent iteration")
ax.set_ylabel(r"policy param $\theta$")
ax.set_title(r"(c) GD from $\theta_0{=}0.7$", loc="left")
ax.set_xlim(0, 300)

# Single shared legend below all three panels
handles, labels = axes[2].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="lower center", bbox_to_anchor=(0.5, -0.04),
    ncol=4, handlelength=1.4, handletextpad=0.4,
    columnspacing=1.2, borderpad=0.3, frameon=False,
)
for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout(w_pad=1.2)
out = "../figures/fig_lemma_toy_demo.pdf"
plt.savefig(out, bbox_inches="tight", pad_inches=0.02)
print(f"saved {out}")
