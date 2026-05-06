"""
TOST equivalence test on paired DPM++ vs DDIM val14 closed-loop scores.
Uses local desktop aggregator_metric parquets.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

DPM_RUN = "./data/val14_eval/lora_dpm_2026-04-17-11-00-52"
DDIM_RUN = "./data/val14_eval/lora_ddim_2026-04-17-15-04-24"

def load_per_scene(run_dir):
    f = list(Path(run_dir).glob("aggregator_metric/closed_loop_nonreactive_agents_weighted_average*.parquet"))[0]
    df = pd.read_parquet(f)
    # filter to per-scenario rows (drop the global aggregator row)
    df = df[df["aggregator_type"].str.contains("scenario", case=False, na=False)] if "aggregator" in str(df["aggregator_type"].dtype) else df
    df = df[~df["scenario"].isna()][["scenario", "log_name", "scenario_type", "score"]]
    return df

dpm = load_per_scene(DPM_RUN)
ddim = load_per_scene(DDIM_RUN)

print(f"DPM++ rows after filter: {len(dpm)}")
print(f"DDIM  rows after filter: {len(ddim)}")

# Match by (scenario, log_name) — paired
dpm_idx = dpm.set_index(["scenario", "log_name"])["score"]
ddim_idx = ddim.set_index(["scenario", "log_name"])["score"]

common = dpm_idx.index.intersection(ddim_idx.index)
print(f"matched rows: {len(common)}")

dpm_paired = dpm_idx.loc[common].astype(float).values
ddim_paired = ddim_idx.loc[common].astype(float).values

# Drop NaNs
mask = ~(np.isnan(dpm_paired) | np.isnan(ddim_paired))
dpm_paired = dpm_paired[mask]
ddim_paired = ddim_paired[mask]
n = len(dpm_paired)
print(f"non-NaN paired n = {n}")

mean_dpm = dpm_paired.mean()
mean_ddim = ddim_paired.mean()
diff = dpm_paired - ddim_paired
mean_diff = diff.mean()
sd_diff = diff.std(ddof=1)
se_diff = sd_diff / np.sqrt(n)

print()
print("=" * 60)
print("PAIRED TWO-SIDED t-TEST")
print("=" * 60)
print(f"DPM++  mean CLS = {mean_dpm:.4f}")
print(f"DDIM   mean CLS = {mean_ddim:.4f}")
print(f"Δ (DPM++ - DDIM) = {mean_diff:+.4f}")
print(f"SD(diff)         = {sd_diff:.4f}")
print(f"SE(diff)         = {se_diff:.6f}")

t_stat, p_two = stats.ttest_rel(dpm_paired, ddim_paired)
print(f"t-stat = {t_stat:+.3f},  p (two-sided) = {p_two:.4f}")

# Wilcoxon signed-rank
nonzero_diff = diff[diff != 0]
W, p_w = stats.wilcoxon(nonzero_diff)
print(f"Wilcoxon W = {W:.0f},  p = {p_w:.4f}  (n_nonzero = {len(nonzero_diff)})")

# Catastrophic contingency
DPM_cat = (dpm_paired == 0).astype(int)
DDIM_cat = (ddim_paired == 0).astype(int)
A = int(((DPM_cat == 0) & (DDIM_cat == 0)).sum())
B = int(((DPM_cat == 0) & (DDIM_cat == 1)).sum())  # DPM saves DDIM
C = int(((DPM_cat == 1) & (DDIM_cat == 0)).sum())  # DPM introduces
D = int(((DPM_cat == 1) & (DDIM_cat == 1)).sum())  # both fail
print()
print("=" * 60)
print("CATASTROPHIC CONTINGENCY (score == 0)")
print("=" * 60)
print(f"  A (both pass)        = {A}")
print(f"  B (DPM++ saves DDIM) = {B}")
print(f"  C (DPM++ introduces) = {C}")
print(f"  D (both fail)        = {D}")
print(f"  DPM++ catastrophic   = {C + D}")
print(f"  DDIM  catastrophic   = {B + D}")
print(f"  catastrophic overlap = {D}/{max(B+D, C+D)} = {100 * D / max(B + D, C + D):.1f}%")

# McNemar continuity-corrected
mc_chi2 = (max(0, abs(B - C) - 1)) ** 2 / (B + C) if (B + C) > 0 else 0
mc_p = 1 - stats.chi2.cdf(mc_chi2, df=1)
print(f"McNemar χ² (cc) = {mc_chi2:.3f},  p = {mc_p:.4f}")

# TOST equivalence test: paired
# H0_lower: μ_diff ≤ -Δ_eq;  H0_upper: μ_diff ≥ +Δ_eq
# Reject both at α=0.05 means equivalence demonstrated
print()
print("=" * 60)
print("TOST EQUIVALENCE TEST (paired)")
print("=" * 60)
for delta_eq in [0.003, 0.005, 0.010, 0.015, 0.020]:
    t_lower = (mean_diff - (-delta_eq)) / se_diff   # >0 favours equiv
    t_upper = (mean_diff - delta_eq) / se_diff      # <0 favours equiv
    p_lower = 1 - stats.t.cdf(t_lower, df=n - 1)    # one-sided upper
    p_upper = stats.t.cdf(t_upper, df=n - 1)         # one-sided lower
    p_tost = max(p_lower, p_upper)
    decision = "EQUIVALENT" if p_tost < 0.05 else "INCONCLUSIVE"
    print(f"  Δ_eq = ±{delta_eq:.3f}:  t_lower={t_lower:+.3f} (p={p_lower:.4f}), "
          f"t_upper={t_upper:+.3f} (p={p_upper:.4f})  →  TOST p = {p_tost:.4f}  [{decision}]")

# 95% CI on mean diff
ci_low, ci_high = stats.t.interval(0.95, df=n - 1, loc=mean_diff, scale=se_diff)
print(f"\n95% CI on Δ: [{ci_low:+.4f}, {ci_high:+.4f}]")

# Bootstrap CI
print()
print("=" * 60)
print("BOOTSTRAP 95% CI (10000 resamples)")
print("=" * 60)
rng = np.random.default_rng(42)
boot = []
for _ in range(10000):
    idx = rng.integers(0, n, n)
    boot.append(diff[idx].mean())
boot = np.array(boot)
print(f"bootstrap mean: {boot.mean():+.5f}, 95% CI: [{np.percentile(boot, 2.5):+.4f}, {np.percentile(boot, 97.5):+.4f}]")

# Save derived numbers
out_summary = {
    "n_matched": n,
    "dpm_mean": mean_dpm,
    "ddim_mean": mean_ddim,
    "delta": mean_diff,
    "sd_diff": sd_diff,
    "se_diff": se_diff,
    "t_paired": float(t_stat),
    "p_paired": float(p_two),
    "wilcoxon_W": float(W),
    "wilcoxon_p": float(p_w),
    "contingency_A": A,
    "contingency_B": B,
    "contingency_C": C,
    "contingency_D": D,
    "catastrophic_overlap_pct": 100 * D / max(B + D, C + D),
    "mcnemar_chi2": mc_chi2,
    "mcnemar_p": float(mc_p),
}
import json
with open("../data/tost_results.json", "w") as f:
    json.dump(out_summary, f, indent=2)
print(f"\nsaved ../data/tost_results.json")
