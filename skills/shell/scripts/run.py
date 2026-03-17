from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def _truncate(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="")
    parser.add_argument("--session-id", default="")
    args = parser.parse_args()

    command = (args.input or "").strip()
    if not command:
        print("Usage: shell <command>")
        return

    cwd = os.environ.get("SHELL_CWD") or str(Path.cwd())
    timeout = int(os.environ.get("SHELL_TIMEOUT", "0"))
    timeout_value = timeout if timeout > 0 else None
    try:
        result = subprocess.run(
            ["bash", "-lc", command],
            shell=False,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_value,
        )
    except subprocess.TimeoutExpired:
        print(f"Command timed out after {timeout}s.")
        return
    except Exception as exc:
        print(f"Command execution failed: {exc}")
        return

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

    print("\n".join(lines))


if __name__ == "__main__":
    main()
