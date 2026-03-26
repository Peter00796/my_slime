**Purpose**: Launch autonomous SLIME research loop — reads program, picks up state, runs experiments 24/7.

---

## Quick Start

1. Load context: read `autoresearch/program.md` for the full protocol
2. Load project context: run `/slime`
3. Read current state: `autoresearch/results.tsv` and `autoresearch/experiment_notes.md`
4. Verify cluster access: `ssh usc-endeavour "squeue -u muruzhan"`
5. Resume from the "NEXT PLANNED EXPERIMENT" in experiment_notes.md
6. Enter the experiment loop (see program.md) — **NEVER STOP**

## Key Rule

You are fully autonomous. Do not ask the user for permission or confirmation. Run experiments, analyze results, decide next steps, and loop. The user will interrupt you when they want to stop.
