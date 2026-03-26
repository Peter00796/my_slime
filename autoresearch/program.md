# SLIME Autoresearch

Autonomous 24/7 research pipeline for SLIME offlineness experiments on USC CARC Endeavour cluster.

## How It Works

You are an autonomous research agent. You run SLIME training experiments on a remote GPU cluster by:
1. Creating config scripts and sbatch scripts
2. Pushing code + configs to the cluster via SSH
3. Submitting jobs via `sbatch`
4. Polling `squeue` until completion
5. Extracting metrics from Ray logs
6. Analyzing results and deciding the next experiment
7. Logging everything to `autoresearch/results.tsv` and `autoresearch/experiment_notes.md`

## Setup (Once Per Session)

1. **Load context**: Run `/slime` to load full project context.
2. **Check cluster access**: `ssh usc-endeavour "squeue -u muruzhan"` — verify VPN is up.
3. **Read state**: Read `autoresearch/results.tsv` and `autoresearch/experiment_notes.md` to understand where the last session left off.
4. **Resume or start fresh**: Pick up from the last experiment's conclusions, or start a new research direction if the previous one is exhausted.

## Cluster Details

- **SSH**: `ssh usc-endeavour`
- **Workdir**: `/project2/swabhas_1625/muruzhan/yanxinpeng`
- **SLIME source**: `slime_env/slime/` (branch: `Offline/Working`)
- **Bad nodes**: Always `--exclude=c05-05,c06-05`
- **Partition**: `nlp` (has preemption — resubmit if killed)
- **GPUs**: 4x per job (2 actor + 2 rollout)
- **Env init**: `eval "$(micromamba shell hook --shell bash)" && micromamba activate slime`
- **After code changes**: `cd slime_env/slime && pip install -e .`

## The Experiment Loop

LOOP FOREVER:

### Step 1: Decide What to Run

Read `autoresearch/results.tsv` and `autoresearch/experiment_notes.md`. Based on prior results, decide the next experiment. Consider:
- What research questions remain open?
- What parameter combinations haven't been tested?
- Did the last experiment suggest a follow-up?
- Is there a surprising result that needs replication or deeper investigation?

Write your reasoning to `autoresearch/experiment_notes.md` BEFORE running. Include:
- **Hypothesis**: What you expect to happen and why
- **Config changes**: Exact parameters being varied
- **Success criteria**: What would confirm/reject the hypothesis

### Step 2: Create the Config

Create or modify a config script at: `/project2/swabhas_1625/muruzhan/yanxinpeng/configs/run-qwen2.5-1.5B-auto-<N>.sh`

Where `<N>` is the experiment number from results.tsv.

The config is a shell script that sets environment variables and calls the SLIME training command. Base your config on the standard SLIME parameters:
```bash
# Standard base parameters (modify as needed for your experiment)
--advantage-estimator grpo
--kl-coef 0.0
--kl-loss-type low_var_kl
--entropy-coef 0.0
--enable-offlineness-metrics
--rollout-batch-size 16
--n-samples-per-prompt 8
--max-response-length 2048
--temperature 1.0
--weight-decay 0.1
```

**Parameters you CAN vary** (the research levers):
| Parameter | Range | Notes |
|-----------|-------|-------|
| `--lr` | 1e-6 to 1e-4 | Dominant lever for offlineness |
| `--eps-clip` | 0.01 to 1e6 | PPO clip bound; 1e6 = effectively no clip |
| `--num-steps-per-rollout` | 1, 2, 4, 8, 16 | Data reuse within rollout |
| `--use-rollout-logprobs` | flag on/off | Stale vs fresh log-probs in PPO ratio |
| `--ppo-epochs` | 1, 3, 5 | PPO epoch count |
| `--num-rollout` | 100-500 | Duration of experiment (more = longer) |
| `--normalize-advantages` | flag on/off | Advantage normalization |
| `--global-batch-size` | varies | Cannot coexist with num-steps-per-rollout |

**Parameters you must NOT change** (fixed infrastructure):
- Model: Qwen2.5-Math-1.5B (mcore checkpoint)
- TP: 2
- GPU count: 4
- Data: DAPO-math-17k
- Eval: AIME24

**Checkpoint LR contamination**: The checkpoint `Qwen2.5-Math-1.5B_slime` (iter 799) has optimizer state with lr=0.0001. Any experiment with a different LR needs `--no-load-optim`. When in doubt, always use `--no-load-optim`.

Always set a unique `--wandb-group` like `auto-exp-<N>-<short-description>`.

### Step 3: Create the sbatch Script

Write to the cluster via SSH:
```bash
ssh usc-endeavour "cat > /project2/swabhas_1625/muruzhan/yanxinpeng/submit_auto.sh << 'SBATCH_EOF'
#!/bin/bash
#SBATCH --partition=nlp
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=slime-auto-<N>
#SBATCH --exclude=c05-05,c06-05
#SBATCH --output=/project2/swabhas_1625/muruzhan/yanxinpeng/slurm_auto_%j.log

eval \"\$(micromamba shell hook --shell bash)\"
micromamba activate slime
export HF_HOME=/project2/swabhas_1625/muruzhan/yanxinpeng/hf_cache
cd /project2/swabhas_1625/muruzhan/yanxinpeng/slime_env/slime
pip install -e . 2>&1 | tail -3
source /project2/swabhas_1625/muruzhan/yanxinpeng/configs/run-qwen2.5-1.5B-auto-<N>.sh
SBATCH_EOF"
```

### Step 4: Push Code Changes (If Any)

If you modified SLIME source code (not just configs):
```bash
# Local: commit and push
git add -A && git commit -m "auto-exp-<N>: <description>"
git push origin Offline/Working

# Remote: pull
ssh usc-endeavour "cd /project2/swabhas_1625/muruzhan/yanxinpeng/slime_env/slime && git pull Offline Offline/Working"
```

If only config changes, write configs directly via SSH (no git needed).

### Step 5: Submit

```bash
ssh usc-endeavour "sbatch /project2/swabhas_1625/muruzhan/yanxinpeng/submit_auto.sh"
```

Extract the SLURM job ID from the output (e.g., "Submitted batch job 2812345").

### Step 6: Monitor Until Completion

Poll every 2-3 minutes:
```bash
ssh usc-endeavour "squeue -u muruzhan -j <jobid> -h"
```

- If output is non-empty: job still running. Wait and poll again.
- If output is empty: job finished. Proceed to Step 7.
- If job disappeared quickly (< 5 min): likely preemption or crash. Check the SLURM log.

**Do NOT flood the cluster with polls.** 2-3 minute intervals are sufficient for 1-2 hour jobs.

### Step 7: Extract Results

```bash
# Get SLURM log
ssh usc-endeavour "tail -100 /project2/swabhas_1625/muruzhan/yanxinpeng/slurm_auto_<jobid>.log"

# Find the node and Ray job ID from the SLURM log, then get training metrics:
ssh usc-endeavour "ssh <node> 'eval \"\$(micromamba shell hook --shell bash)\" && micromamba activate slime && RAY_ADDRESS=http://127.0.0.1:8265 ray job logs <ray_job_id>' 2>/dev/null | grep 'step [0-9]' | tail -50"
```

Extract key metrics from the final rollouts:
- `ess_ratio` — effective sample size (>0.99 = on-policy, <0.8 = off-policy)
- `entropy_loss` — policy entropy (increasing = good exploration, decreasing = collapse)
- `pg_clipfrac` — PPO clip activation rate
- `max_ratio` — maximum importance weight
- `raw_reward` — solve rate
- `grad_norm` — training stability

### Step 8: Analyze and Record

**Record to `autoresearch/results.tsv`:**
```
exp_id	hypothesis	config_changes	num_rollouts	final_ess	final_entropy	final_reward	final_pg_clipfrac	max_ratio	status	finding
```

- `status`: `confirmed` (hypothesis correct), `rejected` (hypothesis wrong), `surprising` (unexpected result), `crash`, `preempted`
- `finding`: One-line summary of what was learned

**Update `autoresearch/experiment_notes.md`** with:
- Full analysis of the result
- Whether hypothesis was confirmed or rejected
- What this implies for the next experiment
- Any new research questions raised

### Step 9: Debate (Self-Review)

Before deciding the next experiment, argue AGAINST your planned next step:
- "Is this the most informative experiment I could run?"
- "Am I just doing a grid search, or am I testing a real hypothesis?"
- "Does this follow from the results, or am I just guessing?"
- "Is there a simpler experiment that would answer the same question?"
- "Am I exploring enough, or am I stuck in a local optimum of ideas?"

Write the debate to `experiment_notes.md`, then make your final decision.

### Step 10: Loop Back to Step 1

**NEVER STOP.** The human may be asleep or away. You are autonomous. If you run out of ideas:
- Re-read the 12 findings in MEMORY.md
- Look for gaps in the parameter space coverage
- Try replication of surprising results
- Try combinations that haven't been tested
- Try pushing known boundaries further (e.g., N=16 steps, lr=5e-5)
- Read the experiment log LaTeX for inspiration

---

## Research Agenda (Starting Points)

These are open questions derived from the 12 findings. Prioritize based on information value.

### Tier 1: High Priority (Direct Extensions)

1. **Prescriptive fix at higher LR**: The stale-logprobs + tight-clip fix was tested at lr=1e-5. Does it hold at lr=5e-5? lr=1e-4? This tests whether the fix is robust or LR-dependent.

2. **N=16 rollout steps**: N=8 was the max tested. Does N=16 break the fix? Does it reveal new stability boundaries?

3. **Prescriptive fix durability**: Run the fix for 500+ rollouts instead of 186. Does entropy keep rising or plateau? Does reward continue improving?

### Tier 2: Medium Priority (New Dimensions)

4. **PPO epochs WITH stale logprobs**: Epochs were tested without stale logprobs (finding: no effect). With stale logprobs, do epochs finally matter?

5. **Asymmetric clipping**: The fix uses eps-clip=0.05 (symmetric). What about asymmetric: tight lower (0.05), loose upper (0.2)?

6. **Entropy coefficient**: All runs used entropy-coef=0.0. Does adding small entropy bonus (0.01) interact with the prescriptive fix?

### Tier 3: Lower Priority (Exploration)

7. **Advantage normalization interaction**: --normalize-advantages defaults to False. Does enabling it change the stale-logprobs dynamics?

8. **Different eps-clip values with stale logprobs**: 0.05 worked, but is 0.1 or 0.02 better?

9. **Warm-starting from different checkpoints**: Does the fix depend on the specific checkpoint (iter 799)?

---

## Failure Handling

| Failure | Action |
|---------|--------|
| Job preempted | Resubmit immediately. Log as `preempted` in results.tsv |
| CUDA Error 802 | Node is bad. Already excluded. If new bad node, add to exclude list and resubmit |
| Ray OOM | Reduce rollout-batch-size or max-tokens-per-gpu. Log crash and try smaller config |
| SSH timeout | VPN may be down. Wait 5 min and retry 3 times. If still failing, write status to experiment_notes.md and stop gracefully |
| Stale code on cluster | Always `pip install -e .` in sbatch script (already in template) |
| WandB auth failure | Set WANDB_API_KEY in sbatch script or disable with --wandb-mode disabled |
| Job stuck > 3 hours | `scancel <jobid>`, log as crash, investigate |

---

## Session Handoff Protocol

When a Claude Code session ends (user closes laptop, context limit, etc.), the next session picks up by:

1. Running `/slime` to load project context
2. Reading `autoresearch/results.tsv` — full experiment history
3. Reading `autoresearch/experiment_notes.md` — analysis, reasoning, next planned experiment
4. The last entry in experiment_notes.md should always end with "NEXT PLANNED EXPERIMENT:" describing what to run next

**Always leave the state clean for the next session.**
