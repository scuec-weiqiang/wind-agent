#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/ceww1999/miniconda3/envs/wind_agent/bin/python}"
MODE="${1:-web}"

if [[ "$MODE" == "-h" || "$MODE" == "--help" ]]; then
  cat <<'HELP'
Usage:
  ./start.sh [web|cli]

Env:
  PYTHON_BIN   Python interpreter path
  CONFIG_FILE               Runtime config JSON path (default: config/runtime.json)
  MODEL_PROVIDER            Model provider: ollama | openai_compatible (default: ollama)
  MODEL_NAME                Model name (fallback: OLLAMA_MODEL)
  MODEL_BASE_URL            API URL (fallback: OLLAMA_URL)
  MODEL_API_KEY             API key for compatible providers (fallback: OPENAI_API_KEY)
  SHELL_TIMEOUT              Shell timeout seconds (0 = unlimited, default: 0)
  OLLAMA_URL   Ollama chat endpoint
  OLLAMA_MODEL Model name for CLI
HELP
  exit 0
fi

export SHELL_TIMEOUT="${SHELL_TIMEOUT:-0}"

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
