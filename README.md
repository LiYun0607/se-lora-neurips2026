# SE-LoRA: Reproducibility Release

This repository contains the code and data referenced in Appendix S6 (Code, Data, and Reproducibility) of the NeurIPS 2026 submission *"Reward Fine-Tuning Diffusion Planners Reveals an Intervention-Supported Shared/Expert LoRA Decomposition"*.

**OpenReview Submission Number**: `26402`.

## Layout

```
.
├── pcdr/                 # Bit-exact PCDR audit code (NC, DAC, DDC, EP, TTC primitives)
├── diffusion_solver/     # Differentiable DPM-Solver++ unrolling + V4-A' margin reward
├── selora/               # SE-LoRA training pipeline + 4×4 causal activation patching
├── lemma1_toy/           # 1-D and K≥2 toy of Lemma 1 (no-go frontier)
├── tost/                 # TOST equivalence test pre-specified analysis script
├── data/                 # Per-scenario CSVs reproducing every numerical claim
│   ├── bc_baseline_val14.csv          # BC pretrained, val14 nonreactive (n=1118)
│   ├── dpm_lora_val14.csv             # Matched DPM-Solver++ LoRA, val14 (n=1118)
│   ├── ddim_lora_val14.csv            # Mismatched DDIM LoRA, val14 (n=1118)
│   ├── v4a_seed{42,43,44}_val14.csv   # V4-A' multi-seed (n=1118 each)
│   ├── dppo_val14.csv                 # DPPO matched-compute baseline (Table 5)
│   ├── tost_results.json              # TOST equivalence test outputs
│   ├── lemma1_K_geq_2.json            # Multi-constraint Lemma 1 frontier data
│   ├── sanity_solver_divergence_multidim.json   # Dim-robustness scaling
│   ├── b1_grad_cosine_s90{2,4,5,6}.json         # 4-seed gradient cosine ablation
│   ├── o_matched_open_loop.json                 # Sim-to-real KS=0.016 matched comparison
│   └── autoware_obstacle/             # Per-route CSVs from Autoware 4-ODD evaluation
├── requirements.txt
└── LICENSE               # Apache 2.0
```

## Reproducing the main numerical claims

| Paper claim | Source files |
|---|---|
| Table 5 BC 0.8973 / Cat 61 / Perfect 548 | `data/bc_baseline_val14.csv`, threshold `score≥0.99` for Perfect |
| Table 5 DPM++ LoRA 0.8849 / DDIM 0.8836 | `data/dpm_lora_val14.csv`, `data/ddim_lora_val14.csv` |
| Table 5 V4-A' (3-seed) 0.8964±0.0013 | `data/v4a_seed{42,43,44}_val14.csv`, mean across files |
| Table tab:tost TOST p=0.041 | `tost/run_tost_dpm_ddim.py` on `dpm_lora_val14.csv` + `ddim_lora_val14.csv` |
| §4.5 9.7×±4.0 gradient cosine ratio | `data/b1_grad_cosine_s90{2,4,5}.json` |
| Appendix S3 Lemma 1 K≥2 frontier | `lemma1_toy/lemma1_toy_K_geq_2.py` reproduces `data/lemma1_K_geq_2.json` |
| §4.7 KS=0.016 matched open-loop | `data/o_matched_open_loop.json` |
| Table 5 DPPO matched-compute 0.8960 / Cat 63 / Perfect 547 | `data/dppo_val14.csv` |

## Dependencies

The code targets PyTorch ≥ 2.0 with CUDA 11.8+. Install via:

```bash
pip install -r requirements.txt
```

The `nuplan-devkit` is required to re-run the val14 closed-loop simulation but is NOT required for re-deriving the per-scenario CSVs from the released parquet aggregates (see `data/`). Install nuplan-devkit separately from <https://github.com/motional/nuplan-devkit>.

## Running each module

### PCDR bit-exact audit (`pcdr/`)

`pcdr_operators.py` provides differentiable Python implementations of:
- `no_collision(ego, neighbors)` — bit-exact 0/1 vs `Shapely` on 52,696 cells
- `drivable_area_compliance(ego, drivable_mask)` — 4.4×10⁻¹⁶ vs numpy reference
- `driving_direction_compliance(ego, route_lanes)` — same precision
- `ego_progress(ego, route)` and `time_to_collision(ego, neighbors)` — at machine precision

Use `nuplan_reward_cls_pcdr.py` to compose them into the soft-CLS-structured reward used in V4-A'.

### Differentiable DPM-Solver++ + V4-A' (`diffusion_solver/`)

`differentiable_rollout.py` implements the kinematic-bicycle rollout with autograd. `train_selora_vdgrpo_pcdr5ops.py` is the V4-A' training script (noc soft-CLS proxy + PCDR margin auxiliary). `run_pcdr_margin_training.py` is the launcher script (3 seeds × 3 epochs).

### SE-LoRA + activation patching (`selora/`)

`train_phase15_selora_nocomfort.py` trains shared rank-4 + per-ODD expert rank-8 SE-LoRA. `causal_dissociation_matrix.py` runs the 4×4 cross-ODD activation patching that produces the ρ=−0.96 Singapore-Pittsburgh mirror finding.

### Lemma 1 toys (`lemma1_toy/`)

`lemma1_toy_K_geq_2.py` reproduces the multi-constraint Pareto frontier (Appendix S3 Figure). `sanity_image_diffusion_divergence.py` runs the multi-dimensional sanity check (Appendix S2 Table S6).

### TOST equivalence (`tost/`)

`run_tost_dpm_ddim.py` reproduces Table tab:tost using `dpm_lora_val14.csv` and `ddim_lora_val14.csv`. Pre-specified equivalence margin: ±0.005 CLS units.

## Path conventions

All scripts use relative paths (e.g., `./data/processed_npz/`, `./checkpoints/lora/`) with optional environment-variable overrides for nuPlan-devkit-processed NPZ inputs (e.g., `PHASE15_NPZ_DIR`). The processed NPZ inputs themselves are NOT included due to size (~150 GB); they are derivable from public nuPlan releases via `nuplan-devkit` preprocessing scripts.

## License

All source code in this repository is released under the Apache License 2.0. See `LICENSE`. The released CSV data is derived from public nuPlan val14 evaluations and inherits the upstream nuPlan license.

## Note on anonymization

This repository is anonymized for double-blind review. Author names, affiliations, and machine identifiers have been removed from source comments and paths. The repository will be de-anonymized at acceptance time and migrated to a permanent location.
