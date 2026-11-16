# Project Instructions

This repository is an academic research project driven by Claude Code with two agent roles: the **Orchestrator** (planning) and the **Executor** (implementation). This file is role-neutral and is loaded into every session and every sub-agent.

## Research Context

The research specification (overview, technical context, directory rules, notation) lives in `plans/research-context.md` so that the Orchestrator can update it. It is imported here:

@plans/research-context.md

## Agent Roles

- **No role by default.** A session acts as the Orchestrator only after the user runs `/orchestrator`, and as the Executor only when running as the `executor` sub-agent. Never assume either role on your own.
- **Orchestrator** (`/orchestrator`): research specification, experimental design, and task breakdown. Writes only under `plans/`. Defined in `.claude/skills/orchestrator/SKILL.md`.
- **Executor** (`executor` sub-agent, or `/execute <ticket path>`): implements and verifies exactly one ticket in `plans/`. Defined in `.claude/agents/executor.md`.
- Research coding standards in `.claude/rules/research-coding.md` apply to every session that writes or modifies code, with or without a role.
- In a session without a role, do not rewrite `plans/` (including `plans/research-context.md`) unless the user explicitly asks.

## Execution Model

- Tickets are executed **one at a time**, in the main working tree. Parallel execution and git worktrees are not used.
- Long-running experiments (anything beyond the 10-minute limit of a single command) are **not run by agents**. The Executor prepares and sanity-checks the command; the researcher runs it and the results are reviewed afterwards.
