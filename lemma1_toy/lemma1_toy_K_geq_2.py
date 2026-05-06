"""
Lemma 1 toy demonstration with K>=2 active constraints.

Reviewer M8 noted that the original toy uses d=1, K=1 (so log K = 0 makes
part 3 vacuous). This script tests Lemma 1 part 3 (forward-vs-gradient-reach
trade-off feasibility) with K in {2, 4, 8} active constraints.

For each (K, beta) point we sweep:
  - Sample K signed margins d_i drawn near a target distance d_star = 0.40m
    (some above and some below the boundary, mimicking real PCDR active set)
  - Compute the *softmax-weighted-average* surrogate
        \\tilde R = sum_i w_i d_i, w_i = e^{-d_i/beta} / Z
  - Compute true gradient norm at d_star (using the correct product-rule formula
    that includes the (1 + (\\tilde R - d_i)/beta) factor reviewer M8 flagged)
  - Compute forward error |\\tilde R - min_i d_i|
  - Plot empirical (epsilon, eta) achievable region vs predicted Pareto frontier
        d_star * log K <= epsilon * log(1/eta)

Output:
  data/lemma1_K_geq_2.json
  figures/fig_lemma1_K_geq_2.pdf
"""
import json
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

K_VALUES = [2, 4, 8]
BETAS = np.geomspace(0.005, 1.0, 25)  # smoothing scale sweep
D_STAR = 0.40  # target distance (metres) at which we want gradient reach

# For each K, sample margins: K-1 active near or below boundary, 1 well above
def sample_margins(K, rng):
    """K margins: 1 at d_star, others spread between -0.1 and 0.3 (mostly violating)."""
    d = rng.uniform(low=-0.1, high=0.3, size=K)
    # Force one margin to be exactly d_star (the one we want gradient reach to)
    d[0] = D_STAR
    return d


def softmax(d, beta):
    z = -d / beta
    z = z - z.max()
    w = np.exp(z); w /= w.sum()
    return w


def evaluate(d, beta):
    """Compute forward error and gradient norm at d[0] (= d_star) for the
    softmax-weighted-average surrogate \\tilde R = sum_i w_i d_i."""
    w = softmax(d, beta)
    R = (w * d).sum()
    fwd_err = R - d.min()  # should be <= beta * log K (Lemma 1 part 1)
    # CORRECT gradient including the (1 + (R - d_i)/beta) factor (M8 fix)
    grad_i = w * (1 + (R - d) / beta)  # gradient at component i
    grad_at_dstar = abs(grad_i[0])  # we want reach to d[0] = d_star
    # Bound from Lemma 1 part 2 (incorrect / conservative form)
    bound_partial = np.exp(-(d - d.min()) / beta)
    return fwd_err, grad_at_dstar, R, w, bound_partial[0]


# Sweep: for each K, sweep beta, record (epsilon, eta)
results = {}
for K in K_VALUES:
    eps_arr, eta_arr, beta_arr, fwd_bound, grad_bound = [], [], [], [], []
    R_arr = []
    rng = np.random.RandomState(123 + K)
    d = sample_margins(K, rng)
    for beta in BETAS:
        eps, eta, R, w, gb = evaluate(d, beta)
        eps_arr.append(eps)
        eta_arr.append(eta)
        beta_arr.append(beta)
        R_arr.append(R)
        fwd_bound.append(beta * np.log(K))
        grad_bound.append(gb)
    results[K] = {
        'd': d.tolist(),
        'd_star': D_STAR,
        'beta': beta_arr,
        'epsilon': eps_arr,
        'eta': eta_arr,
        'R': R_arr,
        'fwd_bound': fwd_bound,
        'grad_bound': grad_bound,
    }
    print(f"K={K}:")
    print(f"  margins d_i = {[f'{x:.3f}' for x in d]}")
    print(f"  beta=0.05 (audit): eps={results[K]['epsilon'][np.argmin(np.abs(np.array(BETAS)-0.05))]:.4f}  "
          f"eta_grad={results[K]['eta'][np.argmin(np.abs(np.array(BETAS)-0.05))]:.4e}")
    print(f"  beta=0.12 (relaxed): eps={results[K]['epsilon'][np.argmin(np.abs(np.array(BETAS)-0.12))]:.4f}  "
          f"eta_grad={results[K]['eta'][np.argmin(np.abs(np.array(BETAS)-0.12))]:.4e}")

# Save
out = 'data/lemma1_K_geq_2.json'
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nsaved {out}")

# Plot
plt.rcParams.update({'font.family': 'serif', 'font.size': 10})
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))

for idx, K in enumerate(K_VALUES):
    r = results[K]
    eps = np.array(r['epsilon'])
    eta = np.array(r['eta'])
    beta = np.array(r['beta'])
    fwd_b = np.array(r['fwd_bound'])
    grad_b = np.array(r['grad_bound'])

    ax = axes[idx]
    # Empirical achievable region
    ax.plot(eps, eta, 'o-', color='#2b6cb0', lw=1.5, markersize=4, label='empirical')
    # Lemma 1 predicted bound: eps_max(beta) = beta log K, eta_min(beta) = exp(-d_star/beta)
    eps_lemma = beta * np.log(K)
    eta_lemma = np.exp(-D_STAR / beta)
    ax.plot(eps_lemma, eta_lemma, '--', color='#dd6b20', lw=1.2, label=f'Lemma 1 frontier ($d^*={D_STAR}$)')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'forward error $\varepsilon$')
    if idx == 0:
        ax.set_ylabel(r'gradient strength $\eta$ at $d^*$')
    ax.set_title(f'K = {K} active constraints')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('figures/fig_lemma1_K_geq_2.pdf', bbox_inches='tight')
plt.savefig('figures/fig_lemma1_K_geq_2.png', bbox_inches='tight', dpi=160)
print(f"saved figures/fig_lemma1_K_geq_2.pdf")
