"""Orchestrator 用 PreToolUse hook．/orchestrator 実行後のセッションで有効．

plans/ 配下以外への書き込み，読み取り専用の許可リスト外の Bash，executor 以外の subagent 起動を拒否．
executor subagent 内の呼び出しは制限対象外．
"""

from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path

# Explore / Plan は Bash を持つため，ファイル書き込みの抜け道になりうるので除外
ALLOWED_SUBAGENTS = frozenset({"executor"})

# 引数を自由に取れる読み取り専用コマンド
FREE_ARG_COMMANDS = frozenset({"ls", "du", "wc", "head", "tail", "pwd"})

# 引数まで完全一致で許可する環境確認コマンド
EXACT_COMMANDS = frozenset(
    {
        ("nvidia-smi",),
        ("nvidia-smi", "-L"),
        ("nproc",),
        ("uname", "-a"),
        ("python", "--version"),
        ("python3", "--version"),
        ("pip", "list"),
        ("pip", "freeze"),
        ("uv", "pip", "list"),
    }
)

GIT_READ_SUBCOMMANDS = frozenset({"log", "diff", "show", "status", "ls-files", "blame"})

# git の読み取り系 subcommand でもファイル書き込みや外部プログラム実行につながる option
GIT_FORBIDDEN_OPTION_PREFIXES = ("--output", "--ext-diff", "--textconv")

# シェルの連結・置換・リダイレクトに使われる文字．パイプは別途分割して検査
SHELL_METACHARACTERS = frozenset(";&<>`$\n\r")


def deny(reason: str) -> None:
    print(f"orchestrator-guard: {reason}", file=sys.stderr)
    sys.exit(2)


def is_read_only_segment(tokens: list[str]) -> bool:
    if not tokens:
        return False
    command, *args = tokens
    if command in FREE_ARG_COMMANDS:
        return True
    if tuple(tokens) in EXACT_COMMANDS:
        return True
    if command == "git":
        return (
            bool(args)
            and args[0] in GIT_READ_SUBCOMMANDS
            and not any(arg.startswith(GIT_FORBIDDEN_OPTION_PREFIXES) for arg in args)
        )
    return False


def check_bash(command: str) -> None:
    rejection = (
        "Orchestrator の Bash は読み取り専用の許可リストのみ使用可能．"
        "コードの実行が必要なら調査用 ticket を作成して executor に委譲すること．"
        f"拒否したコマンド: {command}"
    )
    if any(char in SHELL_METACHARACTERS for char in command):
        deny(rejection)
    try:
        segments = [shlex.split(segment) for segment in command.split("|")]
    except ValueError:
        deny(rejection)
    if not all(is_read_only_segment(tokens) for tokens in segments):
        deny(rejection)


def check_write(target: str | None, payload: dict) -> None:
    if not target:
        deny("書き込み先 path を特定できないため拒否．")
    project_dir = Path(os.environ.get("CLAUDE_PROJECT_DIR") or payload.get("cwd") or ".").resolve()
    plans_dir = project_dir / "plans"
    resolved = (project_dir / target).resolve()
    if not resolved.is_relative_to(plans_dir):
        deny(f"Orchestrator は plans/ 配下のみ編集可能．拒否した path: {target}")


payload = json.load(sys.stdin)
tool_name = payload.get("tool_name", "")
tool_input = payload.get("tool_input", {})

# Orchestrator から委譲された executor の操作は制限しない
if payload.get("agent_type") == "executor":
    sys.exit(0)

if tool_name == "Bash":
    check_bash(tool_input.get("command", ""))
elif tool_name == "Agent":
    subagent = tool_input.get("subagent_type")
    if subagent not in ALLOWED_SUBAGENTS:
        deny(f"Orchestrator が起動できる subagent は executor のみ．拒否: {subagent}")
else:
    check_write(tool_input.get("file_path") or tool_input.get("notebook_path"), payload)
