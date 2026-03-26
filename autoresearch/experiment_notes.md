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

### NEXT PLANNED EXPERIMENT:
Experiment 1 — Prescriptive fix at lr=5e-5. This is the highest-value experiment because it tests robustness of the core finding at a more aggressive learning rate. If the fix holds at 5e-5, it dramatically strengthens the thesis. If it breaks, we learn the LR boundary of the fix.

Config: `--use-rollout-logprobs --eps-clip 0.05 --lr 5e-5 --num-rollout 200 --no-load-optim`
