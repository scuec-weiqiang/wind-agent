from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from app.chat import (
    DEFAULT_MODEL,
    DEFAULT_OPENAI_COMPAT_URL,
    DEFAULT_PROVIDER,
    DEFAULT_THINKING_MODE,
    DEFAULT_URL,
    normalize_thinking_mode,
)


@dataclass(frozen=True)
class RuntimeSettings:
    model_provider: str
    model_name: str
    model_base_url: str
    model_api_key: str
    thinking_mode: str
    system_prompt: str
    enable_thinking: bool
    agent_trace: bool
    max_web_sessions: int
    project_root: Path
    sessions_dir: Path
    uploads_dir: Path
    max_upload_size_bytes: int | None
    max_attach_chars_per_file: int
    max_attach_total_chars: int
    max_attach_files: int | None
    shell_timeout_seconds: int | None
    shell_cwd: Path
    enable_skill_autorun: bool


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


def _parse_optional_positive_int(raw: str) -> int | None:
    text = str(raw or "").strip().lower()
    if not text or text in {"0", "none", "unlimited", "inf", "infinite", "-1"}:
        return None
    try:
        value = int(text)
    except ValueError:
        return None
    return value if value > 0 else None


def _resolve_model_base_url(provider: str) -> str:
    explicit = os.environ.get("MODEL_BASE_URL", os.environ.get("OLLAMA_URL", "")).strip()
    if explicit:
        return explicit
    if provider in {"openai", "openai_compatible", "openai-compatible"}:
        return DEFAULT_OPENAI_COMPAT_URL
    return DEFAULT_URL


def load_runtime_settings() -> RuntimeSettings:
    provider = os.environ.get("MODEL_PROVIDER", DEFAULT_PROVIDER).strip().lower()
    model_name = os.environ.get("MODEL_NAME", os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL))
    project_root = Path(__file__).resolve().parent.parent
    sessions_dir = Path(
        os.environ.get("WEB_SESSIONS_DIR", str(project_root / "data" / "sessions"))
    )
    uploads_dir = Path(os.environ.get("WEB_UPLOADS_DIR", str(sessions_dir / "uploads")))
    shell_timeout = _parse_optional_positive_int(os.environ.get("SHELL_TIMEOUT", "0"))
    shell_cwd = Path(os.environ.get("SHELL_CWD", str(project_root)))

    return RuntimeSettings(
        model_provider=provider,
        model_name=model_name,
        model_base_url=_resolve_model_base_url(provider),
        model_api_key=os.environ.get(
            "MODEL_API_KEY", os.environ.get("OPENAI_API_KEY", "")
        ).strip(),
        thinking_mode=normalize_thinking_mode(
            os.environ.get("THINKING_MODE", DEFAULT_THINKING_MODE)
        ),
        system_prompt=os.environ.get(
            "SYSTEM_PROMPT",
            "You are a helpful agent coordinating between the user and available skills.",
        ),
        enable_thinking=_parse_bool_env("ENABLE_THINKING", True),
        agent_trace=_parse_bool_env("AGENT_TRACE", False),
        max_web_sessions=int(os.environ.get("MAX_WEB_SESSIONS", "100")),
        project_root=project_root,
        sessions_dir=sessions_dir,
        uploads_dir=uploads_dir,
        max_upload_size_bytes=_parse_optional_positive_int(
            os.environ.get("MAX_UPLOAD_SIZE_BYTES", "")
        ),
        max_attach_chars_per_file=int(
            os.environ.get("MAX_ATTACH_CHARS_PER_FILE", "12000")
        ),
        max_attach_total_chars=int(
            os.environ.get("MAX_ATTACH_TOTAL_CHARS", "36000")
        ),
        max_attach_files=_parse_optional_positive_int(
            os.environ.get("MAX_ATTACH_FILES", "")
        ),
        shell_timeout_seconds=shell_timeout,
        shell_cwd=shell_cwd,
        enable_skill_autorun=_parse_bool_env("ENABLE_SKILL_AUTORUN", False),
    )
