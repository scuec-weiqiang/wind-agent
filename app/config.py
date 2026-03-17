from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict


DEFAULT_CONFIG_PATH = Path("config/runtime.json")


KEY_ENV_MAP = {
    "model_provider": "MODEL_PROVIDER",
    "model_name": "MODEL_NAME",
    "model_base_url": "MODEL_BASE_URL",
    "model_api_key": "MODEL_API_KEY",
    "system_prompt": "SYSTEM_PROMPT",
    "enable_shell_skill": "ENABLE_SHELL_SKILL",
    "shell_allow_operators": "SHELL_ALLOW_OPERATORS",
    "shell_allowed_prefixes": "SHELL_ALLOWED_PREFIXES",
    "shell_timeout": "SHELL_TIMEOUT",
    "shell_cwd": "SHELL_CWD",
    "enable_thinking": "ENABLE_THINKING",
}


def _stringify(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def load_runtime_config(config_file: str | None = None) -> Path | None:
    """Loads runtime config from JSON into environment variables.

    Priority:
    1) Existing environment variable values (highest)
    2) config file values
    """
    if config_file:
        path = Path(config_file).expanduser()
    else:
        path = Path(os.environ.get("CONFIG_FILE", "")).expanduser() if os.environ.get("CONFIG_FILE") else DEFAULT_CONFIG_PATH

    if not path.exists():
        return None

    with open(path, encoding="utf-8") as fh:
        payload: Dict[str, Any] = json.load(fh)

    for key, env_name in KEY_ENV_MAP.items():
        if env_name in os.environ:
            continue
        if key not in payload:
            continue
        value = payload[key]
        if value is None:
            continue
        os.environ[env_name] = _stringify(value)

    # Allow extra explicit env mapping block:
    # { "env": { "FOO": "bar" } }
    env_block = payload.get("env")
    if isinstance(env_block, dict):
        for env_name, value in env_block.items():
            if not isinstance(env_name, str) or not env_name:
                continue
            if env_name in os.environ:
                continue
            if value is None:
                continue
            os.environ[env_name] = _stringify(value)

    return path
