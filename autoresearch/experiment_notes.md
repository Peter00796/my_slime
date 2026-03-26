# SLIME Autoresearch — Experiment Notes

This file tracks the reasoning, analysis, and decisions made by the autonomous research agent.
Each session appends to this file. Read from the bottom to find the latest state.

---

## Session 0 — Initialization

### Prior Work Summary (from manual ablations)
12 findings established. Core thesis: GRPO stability is a coincidence of conservative LR + fresh log-probs in PPO ratio. Prescriptive fix: `--use-rollout-logprobs` + `--eps-clip 0.05` produces entropy increase (4x) and reward improvement.

### Open Questions (ranked by information value)
1. Does the prescriptive fix hold at higher LR (5e-5, 1e-4)?
2. What happens at N=16 rollout steps?
3. Does the fix sustain over 500+ rollouts?
4. Do PPO epochs matter when using stale logprobs?
5. Does asymmetric clipping help?

---

## Experiment 001 — Prescriptive Fix at lr=5e-5

**Submitted**: 2026-03-26, SLURM job 3044652
**Config**: `run-qwen2.5-1.5B-auto-001.sh`
**Key changes vs proven fix (lr=1e-5)**:
- `--lr 5e-5` (5x higher than proven fix)
- `--use-rollout-logprobs` (stale log-probs)
- `--eps-clip 0.05 --eps-clip-high 0.05` (tight symmetric clip)
- `--num-steps-per-rollout 8`
- `--num-rollout 200`
- `--no-load-optim` (avoid checkpoint LR contamination)

**Hypothesis**: The prescriptive fix (stale logprobs + tight clip) is robust to LR increases. At lr=5e-5, entropy should still increase and reward should improve, but clipping will be more active (higher LR → larger policy updates → more ratios outside clip bounds).

**Success criteria**:
- Entropy increases (>0.5 by rollout 100)
- ESS stays above 0.95
- Reward improves or stays stable
- No collapse/NaN

**Failure would mean**: The fix only works in a narrow LR band around 1e-5, weakening its generality.

**Status**: COMPLETED — REJECTED

### Results (SLURM 3044657, node c06-01)
Note: First two submissions (3044652, 3044656) failed due to `num_rollout=200 < start_rollout_id=800` causing empty training loop. Fixed by setting `num_rollout=1000`.

**Trajectory** (rollouts 800-850, ~50 rollouts over 1 hour):
- **Rollout 800 (steps 6506-6515)**: ESS 0.986-0.999, entropy 0.13-0.31, pg_clipfrac 0.01-0.04, max_ratio 2.0-4.1. Looking healthy.
- **Rollout ~810 (steps 6550-6559)**: ESS drops to 0.965, max_ratio hits 7.05. Entropy still 0.13-0.23.
- **Rollout ~820 (steps 6660-6670)**: Entropy surges to 0.72, then 1.22, then **1.89** (step 6734). ESS hits 0.927. max_ratio hits 8.5. clip_frac_02 reaches 0.21.
- **Rollout ~837 (steps 6800-6807)**: Entropy cycling 0.15-0.58 but ESS recovers to 0.999 at rollout start.
- **Rollouts 848-850**: **Total collapse.** All rewards = 0. Model generates degenerate repetitive outputs ("n n n n n..."). 88% of responses truncated at max length. Response log-probs around -0.27.

**Key finding**: lr=5e-5 with the prescriptive fix causes **entropy explosion → mode collapse**. Entropy rises 10x faster than at lr=1e-5, overshoots, and the model degenerates into repetitive nonsense. Clipping at ε=0.05 is insufficient to prevent this at higher LR.

**Implications**:
1. The prescriptive fix has a **narrow effective LR band** — works at 1e-5 but breaks at 5e-5
2. The collapse mechanism: high LR → large policy updates → entropy explosion → model exits useful distribution → repetitive degeneration → 0 reward → no gradient signal → stuck
3. The fix needs **LR-adaptive clipping** or a **tighter ε** at higher LR to remain stable

### Self-Debate Before Next Experiment

**Option A**: Try lr=2e-5 or 3e-5 to find the exact boundary.
- Pro: Precisely maps the LR range where the fix works
- Con: Just narrows a known boundary, doesn't test new mechanisms

**Option B**: Try lr=5e-5 with even tighter clipping (ε=0.01 or 0.02).
- Pro: Tests whether tighter clipping can rescue the fix at higher LR
- Con: Very tight clipping might prevent any learning

**Option C**: Try lr=5e-5 with fewer rollout steps (N=4 or N=2).
- Pro: Less data reuse per rollout = less off-policy = might prevent collapse
- Con: Changes two variables at once

**Decision**: Option B is most informative. We know the collapse comes from unbounded entropy increase. If tighter clipping (ε=0.02) can prevent the entropy explosion while keeping the beneficial properties, it would show that the fix is generalizable with LR-dependent ε tuning. This would be a stronger result than just finding the lr boundary.

---

## Experiment 002 — Tighter Clipping (ε=0.02) at lr=5e-5

**Config**: Same as 001 but `--eps-clip 0.02 --eps-clip-high 0.02`
**Hypothesis**: Tighter clipping can prevent entropy explosion at lr=5e-5 while preserving the beneficial entropy increase. If ε=0.02 is sufficient, it demonstrates the fix is generalizable with LR-proportional ε.
**Success criteria**: Entropy increases moderately (0.3-0.8 range, not >1.5), ESS stays >0.95, reward improves, no degeneration.

### NEXT PLANNED EXPERIMENT:
Experiment 002: `--lr 5e-5 --eps-clip 0.02 --eps-clip-high 0.02 --use-rollout-logprobs --num-steps-per-rollout 8 --num-rollout 1000 --no-load-optim`
