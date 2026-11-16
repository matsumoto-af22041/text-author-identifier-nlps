# Research Coding Standards (pointer)

The research coding standards of this project live in **`.claude/rules/research-coding.md`**. Read that file in full and comply with it strictly before writing or modifying any code.

## Scope

- The standards apply to **anyone who writes or modifies code** in this repository: the Executor, a session without an assigned role, or a human researcher, regardless of which AI agent is used.
- Reading them does not assign any role. Roles are defined in `.agents/AGENTS.md`.
- Claude Code loads the file automatically. Other agents must read it explicitly.

## What it covers

- Mandatory scientific critique: point out data leakage, unfair comparisons, confounders, and numerical instability before implementing
- Strict reproducibility: fixed and exposed random seeds; no hardcoded hyperparameters, paths, or conditions
- Data immutability and anti-leakage: never mutate raw data; fit statistics on the Train split only
- Numerical sanity: shape comments, `assert` for `NaN` / `Inf` and shape mismatches, overfit sanity checks
- Fail-fast: no swallowed exceptions or silent default imputation
- Experiment traceability: artifacts linked to the exact config and code state
- Code style: declarative, vectorized, one-to-one with the mathematical formulation

## Editing rule

- `.claude/` is the canonical definition for every agent. Do not edit it, and do not copy its rules here; a copy would drift.
