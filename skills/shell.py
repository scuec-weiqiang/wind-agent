from __future__ import annotations

import os
import shlex
import subprocess

from .base import Skill


def _truncate(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


class ShellSkill(Skill):
    name = "shell"
    description = "Executes a shell command (disabled by default; opt-in required)."

    def run(self, args: str = "", session=None) -> str:
        command = args.strip()
        if not command:
            return "Usage: :skill shell <command>"

        if os.environ.get("ENABLE_SHELL_SKILL", "0") != "1":
            return "Shell skill is disabled. Set ENABLE_SHELL_SKILL=1 to enable it."

        try:
            tokens = shlex.split(command)
        except ValueError as exc:
            return f"Invalid command: {exc}"

        if not tokens:
            return "No command provided."

        if any(token in {">", "<", "|", "&&", "||", ";"} for token in tokens):
            return (
                "Command contains shell operators (>, <, |, &&, ||, ;) which are not supported. "
                "Use a direct executable command."
            )

        allowed_prefixes = [
            part.strip()
            for part in os.environ.get(
                "SHELL_ALLOWED_PREFIXES",
                "ls,pwd,cat,echo,whoami,date,rg,touch,mkdir,tee,printf,gcc,cc,clang,make,python3,head,tail,wc,sed",
            ).split(",")
            if part.strip()
        ]

        executable = tokens[0]
        if allowed_prefixes and executable not in allowed_prefixes:
            return "Command blocked. Allowed prefixes: " + ", ".join(allowed_prefixes)

        timeout = int(os.environ.get("SHELL_TIMEOUT", "20"))
        cwd = os.environ.get("SHELL_CWD") or os.getcwd()

        try:
            result = subprocess.run(
                tokens,
                shell=False,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout}s."
        except Exception as exc:
            return f"Command execution failed: {exc}"

        stdout = _truncate((result.stdout or "").strip())
        stderr = _truncate((result.stderr or "").strip())

        lines = [f"exit_code: {result.returncode}"]
        if stdout:
            lines.append("stdout:")
            lines.append(stdout)
        if stderr:
            lines.append("stderr:")
            lines.append(stderr)
        if not stdout and not stderr:
            lines.append("(no output)")
        return "\n".join(lines)


skill = ShellSkill()
