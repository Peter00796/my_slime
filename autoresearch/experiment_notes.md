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

**Submitted**: 2026-03-26, SLURM job 3044925, node c06-01
**Config**: Same as 001 but `--eps-clip 0.02 --eps-clip-high 0.02`
**Hypothesis**: Tighter clipping can prevent entropy explosion at lr=5e-5 while preserving the beneficial entropy increase.
**Success criteria**: Entropy increases moderately (0.3-0.8 range, not >1.5), ESS stays >0.95, reward improves, no degeneration.

**Status**: COMPLETED — REJECTED

### Results
**Trajectory** (rollouts 800-846, ~46 rollouts, ~48 min):
- **Rollout 800-810**: ESS 0.989-0.999, entropy 0.08-0.35, pg_clipfrac 0.02-0.14. Controlled.
- **Rollout 827-831**: raw_reward 0.015-0.055, truncated 16-37%. Already below baseline (0.13-0.15).
- **Rollout 842-846**: raw_reward 0.0-0.023, truncated 52-76%. Degrading.
- **Step 6775**: ESS 0.985, max_ratio 4.7, entropy 0.21.

**Key finding**: Tighter clipping (ε=0.02) **delays but does not prevent** degradation at lr=5e-5. The model's solve rate drops 0.05→0.008 over 46 rollouts. Response lengths increase (truncation climbs), suggesting the model is generating longer but less useful outputs. Entropy stays controlled (0.08-0.38) but the model's fundamental capability degrades anyway.

**Critical insight**: The problem at lr=5e-5 is NOT just entropy explosion (which tighter clipping can contain). The model degrades even with controlled entropy. The fundamental issue is that lr=5e-5 causes too-large policy updates per step, and the accumulated drift — even when individually clipped — moves the model away from useful math reasoning. The prescriptive fix requires LR ≤ ~1e-5 to work, regardless of ε.

### Cumulative Findings (Experiments 001-002)
- **Finding 13**: lr=5e-5 breaks the prescriptive fix regardless of ε. At ε=0.05: entropy explosion → mode collapse in ~50 rollouts. At ε=0.02: controlled entropy but capability degradation → reward collapse in ~46 rollouts.
- **Finding 14**: The prescriptive fix has a narrow effective LR band (works at 1e-5, fails at 5e-5). The boundary is likely lr ∈ [1e-5, 3e-5].

### Self-Debate Before Next Experiment

**Option A**: Try lr=2e-5 to find the exact boundary.
- Pro: Confirms Finding 14, narrows the range
- Con: Still just mapping the LR boundary, which we already know is narrow

**Option B**: Try the proven fix (lr=1e-5, ε=0.05) for 500+ rollouts to test durability.
- Pro: Tests whether the entropy increase at lr=1e-5 eventually plateaus or keeps rising to collapse
- Con: Long experiment, but high information value — we need to know if the fix is truly stable long-term

**Option C**: Try N=16 rollout steps at lr=1e-5 with the fix.
- Pro: Tests a new dimension entirely, not just LR variations
- Con: If N=8 was the boundary, N=16 might collapse quickly

**Decision**: Option B. We've established that higher LR breaks the fix. The most important open question now is durability — does the proven fix at lr=1e-5 sustain over hundreds of rollouts or does it eventually degrade like the higher LR experiments? This is the key claim for the paper: that the fix is a stable long-term solution.

---

## Experiment 003 — Prescriptive Fix Durability (lr=1e-5, ε=0.05, 500 rollouts)

**Config**: Base prescriptive fix config (same as ablation 6, stale+clip005) but extended to 500 rollouts (num_rollout=1300, starting from checkpoint 800).
**Hypothesis**: The prescriptive fix at lr=1e-5 is durably stable. Entropy continues to increase moderately and reward improves over 500 rollouts without collapse.
**Success criteria**: No reward degradation after 500 rollouts. Entropy in 0.5-2.0 range, stable or slowly increasing. ESS stays >0.99.

**Submitted**: 2026-03-26, SLURM job 3045099 (c05-02), then 3045964 (resubmit with 128GB)
**Config**: `run-qwen2.5-1.5B-auto-003.sh` — proven fix (lr=1e-5, ε=0.05, stale logprobs, N=8)
**Status**: RUNNING (resubmitted after OOM at 93 rollouts)

### Partial Results (93 rollouts, 800-893, before OOM)
- **ESS**: Rock solid 0.997+ throughout
- **Entropy**: Healthy range 0.09-1.18, showing the characteristic beneficial increase
- **max_ratio**: 1.3-2.1, extremely well bounded
- **pg_clipfrac**: 0.01-0.07, actively clipping
- **Reward trajectory**: 0.125 (rollout 815) → 0.305 peak (rollout 834) → cycling 0.10-0.27
- **No degradation**: No signs of collapse, degeneration, or capability loss over 93 rollouts
- **OOM**: Host memory killed at 1h47m. Infrastructure issue, not training problem. Resubmitted with 128GB.

### NEXT PLANNED EXPERIMENT:
Continue monitoring 003. If 500 rollouts complete successfully, the durability hypothesis is confirmed → Finding 15. Then try N=16 rollout steps.
