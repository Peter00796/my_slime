# Novelty Assessment — Final (2026-03-26)

## Verdict: NOVEL — with careful positioning needed

### The finding IS novel:
1. **Root cause diagnosis** (recomputed log-probs) — nobody has identified this
2. **Two-cause model** (architectural + hyperparameter) — nobody has both
3. **Prescriptive fix** (stale logprobs + tight clip) — nobody has proposed or tested
4. **Synergistic 2x2 interaction** — unique experimental design and result
5. **ESS as GRPO offlineness metric** — first use
6. **Entropy increase regime** — unique; all others see entropy decrease

### Papers we MUST cite and differentiate from:
1. **Chen et al. ICLR 2026 (2512.16912)**: clip < 1%, attributes to small LR. We add: root cause is recomputed log-probs, not just LR.
2. **"Prosperity before Collapse" (2510.01161)**: clip 0.05% at s=0, proposes M2PO for high staleness. We explain WHY 0.05%, and our fix is for making standard-setting clipping work.
3. **TIC-GRPO (2508.02833)**: removing IS works fine, attributes to "frequent refresh." We explain the mechanism.
4. **"GRPO secretly off-policy" (2509.24203)**: IS non-essential, clipping critical. Different conclusion from ours.
5. **verl Rollout Correction docs + three-policy blog**: architectural distinction exists but framed as precision issue, not algorithmic.
6. **TRL issue #2769**: ratio=1 noticed, dismissed as expected.

### Community awareness level:
- **Symptom known**: clipping rate is low, IS doesn't matter much, ratio near 1 at mu=1
- **Root cause unknown**: nobody connects it to recomputed log-probs as architectural choice
- **Fix unknown**: nobody has proposed stale logprobs + tight clip
- **Consequence unknown**: nobody has shown this combination produces entropy INCREASE

### Risk: what a hostile reviewer might say
- "This is obvious — of course clipping doesn't fire if you recompute log-probs"
  - Counter: If it's obvious, why did DAPO/GSPO/CFPO/ASPO all redesign clipping without noticing?
  - Counter: Chen et al. (ICLR 2026) attributed it to small LR, not recomputation. Even published work gets it wrong.
- "The fix only works at lr=1e-5 — too narrow to be useful"
  - Counter: lr=1e-5 IS the standard GRPO learning rate. The fix works in the regime practitioners actually use.
- "Single model, single dataset"
  - Fair criticism. Need at least one more model. DeepSeek-R1-1.5B experiment would address this.

### Recommended venue:
- **Workshop paper (NeurIPS/ICML workshop on RLHF)**: ready now with current data
- **Short paper**: needs second model + comparison with DAPO Clip-Higher
- **Full paper**: needs 7B model + broader domain (chat, code) + comparison table with DAPO/GSPO/RGRA
