# Agent Roles & Execution Model

This repository is an academic research project driven by AI agents with two roles: the **Orchestrator** (planning) and the **Executor** (implementation). This file makes those roles usable from any AI agent. The canonical definitions live under `.claude/`; when this file and `.claude/` disagree, `.claude/` wins.

Read `llm.txt` first for the overall read order.

## Required context for every session

- `plans/research-context.md`: research specification (see `llm-context.md`)
- `.claude/rules/research-coding.md`: research coding standards for anyone who writes or modifies code (see `llm-rules.md`)

## Role Selection

- **No role by default.** Act as the Orchestrator or the Executor only when the user explicitly assigns that role. Never assume a role on your own.
- In a session without a role, do not rewrite `plans/` (including `plans/research-context.md`) unless the user explicitly asks.
- Never modify anything under `.claude/`.

| Role | Canonical definition | How to start it in Claude Code | How to start it in other agents |
|---|---|---|---|
| Orchestrator | `.claude/skills/orchestrator/SKILL.md` | `/orchestrator <request>` | User says "act as the Orchestrator"; read the canonical definition and follow it with the substitutions below |
| Executor | `.claude/agents/executor.md` | `/execute plans/{NNN}-{task-slug}.md` or the `executor` sub-agent | User says "act as the Executor for `plans/{NNN}-{task-slug}.md`"; read the canonical definition and follow it with the substitutions below |

## Orchestrator in non-Claude agents

Follow `.claude/skills/orchestrator/SKILL.md` in full (responsibilities, critical review, artifact naming, WBS and ticket formats, post-execution review). In Claude Code a hook enforces the constraints below; other agents have no such hook, so **enforce them yourself**:

- Write only under `plans/`. Never edit source code, configs, scripts, or data.
- Run only read-only commands: `ls`, `du`, `wc`, `head`, `tail`, `pwd`; `git` with `log`, `diff`, `show`, `status`, `ls-files`, or `blame`; environment checks such as `python3 --version` or `uv pip list`. No redirects, command chaining, or code execution.
- When facts require running code, write a small investigation ticket and hand it to an Executor instead of guessing.
- Delegate only to an Executor. If your agent cannot spawn sub-agents, hand the ticket path to the user so they can start an Executor session.

## Executor in non-Claude agents

Follow `.claude/agents/executor.md` in full (workflow, command time limit, boundaries, report format). Notes:

- Implement and verify exactly one ticket `plans/{NNN}-{task-slug}.md` per assignment, and only after every dependency in `plans/wbs.md` is `Done`.
- Within `plans/`, update only your own ticket's status cell in `plans/wbs.md`.
- End with the report format defined there (Ticket, Status, Changed Files, Verification Results with observed values, Full Experiment, Deviations & Concerns).

## Execution Model

- Tickets are executed **one at a time**, in the main working tree. Parallel execution and git worktrees are not used, even when team members use different agents.
- A single command may run for at most **10 minutes**. Longer experiments are not run by agents: the Executor prepares and smoke-tests the command in the ticket's **Full Experiment** section, and the researcher runs it.
- Results are reviewed afterwards by the Orchestrator.

## Artifacts

- `plans/research-context.md`: research specification (Orchestrator-owned)
- `plans/wbs.md`: roadmap, dependencies, and ticket status
- `plans/{NNN}-{task-slug}.md`: tickets in the format defined in `.claude/skills/orchestrator/SKILL.md` (sections: Expected Deliverables & Design Policy, Target File Paths, Mandatory Constraints, Verification Criteria, optional Full Experiment)
