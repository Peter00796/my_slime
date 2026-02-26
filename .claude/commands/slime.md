**Purpose**: SLIME Offlineness Research — full project context, cluster ops, and codebase knowledge

---

## Research Overview

**Topic**: Quantifying Offlineness in GRPO Training Without KL Penalty
**Model**: Qwen2.5-Math-1.5B (primary), extensible to DeepSeek-R1-1.5B, Qwen3-4B
**Framework**: SLIME (LLM post-training framework by THUDM/Tsinghua)
**Core Insight**: When kl_coef=0.0 for pure reward optimization in math/reasoning RL, ESS collapses (~0.2), indicating extreme off-policy drift and risk of policy collapse.

### Research Goals
1. **Quantify Offlineness**: Measure data staleness, IS degradation, KL removal impact on GRPO stability
2. **Identify Danger Zone**: Find ESS/reward collapse thresholds
3. **Propose Offlineness Index**: Composite metric to predict policy collapse early

### Key Metrics to Monitor
- `offlineness/ess_ratio` — Effective Sample Size ratio (target: >0.8 safe, <0.3 danger)
- `offlineness/mismatch_kl` — Chi-squared divergence proxy between policies
- `offlineness/clip_frac_02` — Fraction of importance weights outside [0.8, 1.2]
- `offlineness/max_ratio` — Maximum importance weight (extreme off-policy indicator)
- `pg_clipfrac` — PPO clip trigger rate
- `ppo_kl` — Per-token KL between old and current policy
- `train_rollout_logprob_abs_diff` — Absolute diff between train-start and rollout log-probs
- `entropy_loss` — Policy entropy (collapse = entropy dropping)
- `grad_norm` — Gradient health indicator

### Ablation Plan
- **Exp 1**: Reduce PPO epochs from 3 to 1 (or increase batch size) → expect ESS recovery
- **Exp 2**: Tighten eps-clip from 0.2 to 0.05 (hard brake) → expect slower but stable reward growth

---

## Infrastructure: USC CARC Endeavour Cluster

### SSH Access
```
Host: usc-endeavour
HostName: endeavour2.usc.edu
User: muruzhan
```
**Requirement**: Mac VPN must be ON, or use internal IP 10.72.0.16.

### Git Workflow (Local ↔ Cluster)
- Cluster remote name is `Offline` (not `origin`; `origin` = upstream THUDM/slime)
- Cluster remote URL: `git@github.com:Peter00796/my_slime.git` (SSH)
- **Workflow**: local edit → `git commit` → `git push origin Offline/Working` → SSH to cluster → `cd slime_env/slime && git pull Offline Offline/Working`
- Cluster SSH key (`~/.ssh/id_ed25519`, passphrase removed) is registered on GitHub
- Cluster `~/.ssh/config` has a `Host github.com` entry pointing to `id_ed25519`

### Working Directory
```
/project2/swabhas_1625/muruzhan/yanxinpeng
```

### Directory Layout (Cluster)
```
yanxinpeng/
├── slime_env/
│   ├── slime/          # SLIME source (editable install, branch: Offline/Working)
│   ├── Megatron-LM/    # Megatron backend
│   └── sglang/         # SGLang rollout engine
├── models/
│   ├── Qwen2.5-Math-1.5B/       # HF checkpoint
│   └── Qwen2.5-Math-1.5B-mcore/ # Megatron-Core converted checkpoint
├── checkpoints/
│   └── Qwen2.5-Math-1.5B_slime/ # Training saves
├── data/
│   ├── dapo-math-17k/   # Training data (DAPO)
│   ├── gsm8k/           # GSM8K eval
│   └── aime24/          # AIME 2024 eval
├── configs/              # Run scripts (run-qwen2.5-1.5B.sh etc.)
├── hf_cache/             # Isolated HF cache (MUST set HF_HOME)
└── wandb/                # WandB logs
```

### Resource Request
**Interactive (srun)**:
```bash
srun --partition=nlp --gres=gpu:4 --cpus-per-task=8 --mem=64G --time=02:00:00 --pty bash
```

**Batch (sbatch)** — preferred for autonomous runs:
```bash
sbatch /project2/swabhas_1625/muruzhan/yanxinpeng/submit_fixed_ess.sh
```
**NOTE**: Node c05-05 has broken CUDA drivers (Error 802). Always `--exclude=c05-05` in sbatch scripts.

Always run `squeue -u muruzhan` first. If a node is already allocated, `ssh <node>` directly.

### Environment Init (MUST run on every compute node)
```bash
eval "$(micromamba shell hook --shell bash)"
micromamba activate slime
export HF_HOME=/project2/swabhas_1625/muruzhan/yanxinpeng/hf_cache
```

### Code Modification Rule
After editing ANY file under `slime_env/slime/slime/`, you MUST run:
```bash
cd /project2/swabhas_1625/muruzhan/yanxinpeng/slime_env/slime
pip install -e .
```
Otherwise Ray workers will use stale code.

### Running Experiments
```bash
cd /project2/swabhas_1625/muruzhan/yanxinpeng
source configs/run-qwen2.5-1.5B.sh
```

---

## Codebase Architecture

### Critical Source Files

| File | Purpose | Key Lines |
|------|---------|-----------|
| `slime/backends/megatron_utils/loss.py` | Policy loss, offlineness metrics, TIS | `compute_offlineness_metrics()` L391-437, `policy_loss_function()` L510-730 |
| `slime/utils/ppo_utils.py` | KL estimators, PPO clip, GAE, GRPO returns | `compute_approx_kl()` L11-51, `compute_policy_loss()` L124-148 |
| `slime/utils/arguments.py` | All CLI args (RL, clipping, offlineness flags) | `--enable-offlineness-metrics` custom addition |
| `slime/backends/megatron_utils/model.py` | Training loop, metric logging, checkpointing | L643 assertion: ppo_kl==0 → pg_clipfrac==0 |
| `slime/rollout/sglang_rollout.py` | Data generation with SGLang | Reward computation integration |
| `patches/offlineness_monitor.py` | Standalone monkey-patch (superseded) | Full ESS/KL computation |
| `train.py` | Entry point orchestrator | Ray job submission |

### Key Training Arguments (GRPO Config)
```bash
--advantage-estimator grpo
--kl-coef 0.0              # No KL penalty (pure RL)
--kl-loss-type low_var_kl   # KL estimator type
--entropy-coef 0.0
--eps-clip 0.2              # Lower PPO clip bound
--eps-clip-high 0.28        # Upper PPO clip bound (asymmetric)
--enable-offlineness-metrics # Our custom flag
```

### Current Run Config (run-qwen2.5-1.5B.sh)
- **GPUs**: 4 (2 actor training + 2 rollout/SGLang)
- **TP**: 2 (tensor parallel)
- **Batch**: global_batch_size=128, rollout_batch_size=16, n_samples_per_prompt=8
- **Rollout**: num_rollout=10000, max_response_len=2048, temperature=1.0
- **Optimizer**: Adam, lr=1e-6, constant schedule, weight_decay=0.1
- **Data**: DAPO-math-17k (train), AIME24 (eval every 100 steps)
- **WandB**: project=slime-dev-qwen2.5-MATH-1.5B-ess-metric

---

## Known Issues & Bugs

### BUG 1: ESS Metric Compares Wrong Log-Probs — FIXED
**Location**: `loss.py` L688-696
**Problem**: Was using `batch["log_probs"]` (start of current PPO epoch) instead of `batch["rollout_log_probs"]` (data generation time). Measured within-epoch drift, NOT true offlineness.
**Evidence**: mismatch_kl=0.0 and clip_frac=0.0 in ALL steps (ratio ≈ 1.0 within epoch).
**Fix Applied**: Now uses `batch["rollout_log_probs"]` when available, falling back to `old_log_probs`. Both `loss.py` and `patches/offlineness_monitor.py` updated.

### BUG 2: `max_ratio == ess_ratio` in Every Step (SUSPICIOUS)
**Symptom**: Both values are identical (e.g., 0.1953125). Mathematically impossible if ratio tensor has meaningful variance.
**Likely Cause**: Shape mismatch or broadcasting error in `compute_offlineness_metrics()`. The ratio tensor may have unexpected structure after mask filtering.
**Debug**: Add `print(ratio.shape, ratio.min(), ratio.max(), ratio.mean())` to verify.

### BUG 3: `pg_loss ≈ 0` (values ±1e-8)
**Impact**: Model is barely learning. Either advantages are zeroed by normalization, or the clip + ratio combination squashes all signal.
**Note**: This may be expected behavior at step 300+ if the model has already converged on easy problems.

### NOT A BUG: `pg_clipfrac = 0` When `kl_coef = 0`
**Reason**: `ppo_kl = old_log_probs - log_probs ≈ 0` within same PPO epoch → ratio = exp(0) = 1.0 → nothing clips. This is correct behavior but makes pg_clipfrac useless as an offlineness indicator. The REAL clip fraction should use rollout log-probs.

---

## Custom Code Modifications (Branch: Offline/Working)

3 files changed vs upstream SLIME (commit c959517):

### 1. `slime/backends/megatron_utils/loss.py` (+63 lines)
- Added `compute_offlineness_metrics()` function (L391-437)
- Integrated into `policy_loss_function()` return dict (L688-724)
- Gated by `--enable-offlineness-metrics` flag

### 2. `slime/utils/arguments.py` (+6 lines)
- Added `--enable-offlineness-metrics` boolean flag

### 3. `patches/offlineness_monitor.py` (+142 lines)
- Monkey-patch approach (apply_patch/remove_patch)
- Superseded by direct loss.py integration but kept for reference

---

## Operational Checklists

### Before Running an Experiment
1. `squeue -u muruzhan` — check existing allocations
2. SSH to compute node or `srun` for new allocation
3. Run environment init (micromamba + HF_HOME)
4. If code was modified: `cd slime_env/slime && pip install -e .`
5. Verify WandB project/group name in config script
6. `source configs/run-qwen2.5-1.5B.sh`

### After a Run Completes
1. Check `slime_env/slime/run.log` for final metrics
2. Check WandB dashboard for curves
3. Extract key metrics: ESS trend, reward curve, pg_loss, grad_norm
4. Save checkpoint if promising

### Debugging a Failed Run
1. Check `run.log` for Python tracebacks
2. Check Ray dashboard: `http://<node>:8265`
3. Common issues:
   - OOM → reduce `max-tokens-per-gpu` or `rollout-batch-size`
   - SGLang crash → `pkill -u "$USER" sglang` and retry
   - Stale code → `pip install -e .` in slime dir
   - CUDA errors → check `CUDA_VISIBLE_DEVICES` and GPU availability
