#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/w/miniconda3/envs/wind_agent/bin/python}"
MODE="${1:-web}"

if [[ "$MODE" == "-h" || "$MODE" == "--help" ]]; then
  cat <<'HELP'
Usage:
  ./start.sh [web|cli]

Env:
  PYTHON_BIN   Python interpreter path
  ENABLE_SHELL_SKILL         Enable shell skill (default: 1)
  OLLAMA_URL   Ollama chat endpoint
  OLLAMA_MODEL Model name for CLI
HELP
  exit 0
fi

export ENABLE_SHELL_SKILL="${ENABLE_SHELL_SKILL:-1}"

case "$MODE" in
  web)
    exec "$PYTHON_BIN" "$ROOT_DIR/main.py" web
    ;;
  cli)
    exec "$PYTHON_BIN" "$ROOT_DIR/main.py" cli
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    echo "Use './start.sh --help' for usage." >&2
    exit 1
    ;;
esac
