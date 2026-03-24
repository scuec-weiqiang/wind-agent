from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import requests
from flask import Flask, Response, request, send_from_directory

from app.chat import (
    ChatSession,
    DEFAULT_MODEL,
    DEFAULT_OPENAI_COMPAT_URL,
    DEFAULT_PROVIDER,
    DEFAULT_THINKING_MODE,
    DEFAULT_URL,
    _extract_reasoning_text,
    normalize_thinking_mode,
)
from app.skill_manager import (
    SkillAmbiguousError,
    SkillDisabled,
    SkillManager,
    SkillNotFoundError,
    SkillRegistryError,
    SkillUnavailable,
)

MODEL_PROVIDER = os.environ.get("MODEL_PROVIDER", DEFAULT_PROVIDER).strip().lower()
MODEL_NAME = os.environ.get("MODEL_NAME", os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL))
MODEL_BASE_URL = os.environ.get("MODEL_BASE_URL", os.environ.get("OLLAMA_URL", "")).strip()
if not MODEL_BASE_URL:
    MODEL_BASE_URL = DEFAULT_OPENAI_COMPAT_URL if MODEL_PROVIDER in {"openai", "openai_compatible", "openai-compatible"} else DEFAULT_URL
MODEL_API_KEY = os.environ.get("MODEL_API_KEY", os.environ.get("OPENAI_API_KEY", "")).strip()
THINKING_MODE = normalize_thinking_mode(os.environ.get("THINKING_MODE", DEFAULT_THINKING_MODE))
SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "You are a helpful agent coordinating between the user and available skills.",
)
SKILL_ROUTER_PROMPT_TEMPLATE = (
    "You are a skill router. Decide whether to call skill packs.\n"
    "Below is the current skill catalog.\n"
    "Treat it as an available_skills index, not as full skill instructions.\n"
    "Below is the available tool catalog.\n"
    "Treat it as the low-level action reference.\n"
    "Use the skill_id exactly as listed if you choose one.\n"
    "Return JSON only, one of:\n"
    '{"action":"none"}\n'
    '{"action":"read_skill","skill":"<skill_id>"}\n'
    '{"action":"list_dir","path":"<relative directory path>"}\n'
    '{"action":"read_file","path":"<relative file path>"}\n'
    '{"action":"search_text","pattern":"<text or regex>","path":"<relative path optional>"}\n'
    '{"action":"list_uploaded_files"}\n'
    '{"action":"read_uploaded_file","file_id":"<uploaded file id>"}\n'
    '{"action":"exec_command","command":"<shell command>"}\n'
    '{"action":"tool_calls","calls":[{"tool":"read_file","path":"README.md"},{"tool":"exec_command","command":"pwd"}]}\n'
    '{"action":"use_skill","skill":"<skill_id>","input":"<input for skill>"}\n'
    '{"action":"use_skill","skill":"<skill_id>","input":{"key":"value"}}\n'
    '{"action":"use_skills","calls":[{"skill":"<skill_id>","input":"..."},{"skill":"<skill_id>","input":"..."}]}\n'
    '{"action":"use_skills","calls":[{"skill":"<skill_id>","input":{"key":"value"}}]}\n'
    "Rules:\n"
    "- no markdown, no prose, JSON only.\n"
    "- if no skill fits, return action=none.\n"
    "- if one packaged_runtime skill clearly matches the user goal, prefer action=use_skill directly.\n"
    "- choose read_skill when the user is asking about what a skill does or how to use it.\n"
    "- if a skill is marked invocation=manual_only, do not return use_skill for it; return read_skill first.\n"
    "- do not use list_dir/read_file/search_text for routine tasks when a suitable packaged_runtime skill already exists.\n"
    "- use list_dir/read_file/search_text only when required to disambiguate, inspect unknown files, or follow manual_only skills.\n"
    "- use exec_command when a shell command is the most direct way to answer.\n"
    "- if user asks to use uploaded files, prefer list_uploaded_files/read_uploaded_file before other actions.\n"
    "- use tool_calls when the task needs multiple low-level steps.\n"
    "- input may be a string or a JSON object.\n"
    "- prefer using description and when_to_use to decide if a skill fits.\n"
    "- if user asks to analyze vibration/frequency data and provides a directory path, prefer fft-frequency via use_skill.\n"
    "- for shell skill input, output executable command text only.\n"
    "- at most 3 calls for use_skills.\n\n"
    "Skill catalog:\n"
    "<<SKILL_CATALOG>>\n"
    "\nTool catalog:\n"
    "<<TOOL_CATALOG>>\n"
)
MAX_SESSIONS = int(os.environ.get("MAX_WEB_SESSIONS", "100"))
DEFAULT_SESSION_ID = "default"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENABLE_THINKING = os.environ.get("ENABLE_THINKING", "1").lower() not in {"0", "false", "no"}
AGENT_TRACE = os.environ.get("AGENT_TRACE", "0").lower() not in {"0", "false", "no"}
SESSIONS_DIR = Path(
    os.environ.get(
        "WEB_SESSIONS_DIR",
        str(PROJECT_ROOT / "data" / "sessions"),
    )
)
UPLOADS_DIR = Path(
    os.environ.get("WEB_UPLOADS_DIR", str(SESSIONS_DIR / "uploads"))
)
TEXT_FILE_EXTENSIONS = {
    ".txt",
    ".md",
    ".csv",
    ".json",
    ".log",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".go",
    ".rs",
    ".sh",
    ".sql",
    ".xml",
    ".html",
    ".css",
}
MAX_UPLOAD_SIZE_BYTES = int(os.environ.get("MAX_UPLOAD_SIZE_BYTES", str(2 * 1024 * 1024)))
MAX_ATTACH_FILES = int(os.environ.get("MAX_ATTACH_FILES", "8"))
MAX_ATTACH_CHARS_PER_FILE = int(os.environ.get("MAX_ATTACH_CHARS_PER_FILE", "12000"))
MAX_ATTACH_TOTAL_CHARS = int(os.environ.get("MAX_ATTACH_TOTAL_CHARS", "36000"))

app = Flask(__name__, static_folder=".", static_url_path="")
skill_manager: SkillManager | None = None

try:
    skill_manager = SkillManager()
except SkillRegistryError:
    skill_manager = None


@dataclass
class ConversationState:
    assistant: ChatSession
    created_at: float
    updated_at: float
    debug_trace: list[dict[str, object]] = field(default_factory=list)
    attached_file_ids: list[str] = field(default_factory=list)


_sessions: dict[str, ConversationState] = {}


def _migrate_legacy_uploads() -> None:
    # Keep backward compatibility for existing deployments that used data/uploads.
    if "WEB_UPLOADS_DIR" in os.environ:
        return
    legacy_dir = PROJECT_ROOT / "data" / "uploads"
    if not legacy_dir.exists() or not legacy_dir.is_dir():
        return
    if legacy_dir.resolve() == UPLOADS_DIR.resolve():
        return

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    for item in legacy_dir.iterdir():
        target = UPLOADS_DIR / item.name
        if target.exists():
            continue
        try:
            shutil.move(str(item), str(target))
        except Exception:
            continue

    try:
        legacy_dir.rmdir()
    except OSError:
        pass


def _safe_session_filename(session_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", session_id.strip() or DEFAULT_SESSION_ID)
    return f"{safe}.json"


def _session_file_path(session_id: str) -> Path:
    return SESSIONS_DIR / _safe_session_filename(session_id)


def _new_conversation_state() -> ConversationState:
    now = time.time()
    return ConversationState(
        assistant=ChatSession(
            model=MODEL_NAME,
            base_url=MODEL_BASE_URL,
            provider=MODEL_PROVIDER,
            api_key=MODEL_API_KEY,
            system_prompt=SYSTEM_PROMPT,
            think=ENABLE_THINKING,
            thinking_mode=THINKING_MODE,
        ),
        created_at=now,
        updated_at=now,
    )


def _save_session(session_id: str, state: ConversationState) -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = _session_file_path(session_id)
    payload = {
        "session_id": session_id,
        "created_at": state.created_at,
        "updated_at": state.updated_at,
        "assistant_messages": state.assistant.messages,
        "debug_trace": state.debug_trace[-200:],
        "attached_file_ids": state.attached_file_ids[-MAX_ATTACH_FILES:],
    }
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    tmp_path.replace(path)


def _load_sessions_from_disk() -> None:
    if not SESSIONS_DIR.exists():
        return

    loaded: list[tuple[str, ConversationState]] = []
    for path in SESSIONS_DIR.glob("*.json"):
        try:
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception:
            continue

        session_id = str(payload.get("session_id", "")).strip()
        if not session_id:
            continue

        state = _new_conversation_state()
        state.created_at = float(payload.get("created_at", time.time()))
        state.updated_at = float(payload.get("updated_at", state.created_at))

        assistant_messages = payload.get("assistant_messages")
        if isinstance(assistant_messages, list):
            state.assistant.messages = assistant_messages
        debug_trace = payload.get("debug_trace")
        if isinstance(debug_trace, list):
            state.debug_trace = [item for item in debug_trace if isinstance(item, dict)][-200:]
        attached_file_ids = payload.get("attached_file_ids")
        if isinstance(attached_file_ids, list):
            state.attached_file_ids = [
                str(item).strip()
                for item in attached_file_ids
                if str(item).strip()
            ][:MAX_ATTACH_FILES]

        loaded.append((session_id, state))

    loaded.sort(key=lambda item: item[1].updated_at, reverse=True)
    for session_id, state in loaded[:MAX_SESSIONS]:
        _sessions[session_id] = state


def _ensure_session(session_id: str) -> ConversationState:
    state = _sessions.get(session_id)
    if state is not None:
        return state

    if len(_sessions) >= MAX_SESSIONS:
        oldest = min(_sessions.items(), key=lambda item: item[1].updated_at)[0]
        _sessions.pop(oldest, None)

    state = _new_conversation_state()
    _sessions[session_id] = state
    _save_session(session_id, state)
    return state


def _extract_session_id() -> str:
    payload = request.get_json(silent=True) or {}
    raw = payload.get("session_id", "") or request.args.get("session_id", "")
    session_id = str(raw).strip()
    return session_id or DEFAULT_SESSION_ID


def _extract_session_id_from_form() -> str:
    raw = request.form.get("session_id", "") or request.args.get("session_id", "")
    session_id = str(raw).strip()
    return session_id or DEFAULT_SESSION_ID


def _safe_upload_name(name: str) -> str:
    candidate = Path(name or "").name
    sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", candidate).strip("._")
    return sanitized or f"upload_{uuid.uuid4().hex[:8]}.csv"


def _session_upload_dir(session_id: str) -> Path:
    return UPLOADS_DIR / _safe_session_filename(session_id).replace(".json", "")


def _safe_uploaded_path(session_id: str, file_id: str) -> Path | None:
    base = _session_upload_dir(session_id).resolve()
    candidate = (base / str(file_id or "").strip()).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        return None
    return candidate


def _is_text_upload(name: str) -> bool:
    suffix = Path(name or "").suffix.lower()
    return suffix in TEXT_FILE_EXTENSIONS


def _read_text_file_with_fallback(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Unsupported text encoding: {path}")


def _list_uploaded_files_content(session_id: str, file_ids: list[str]) -> str:
    if not file_ids:
        return "(none)"
    lines: list[str] = []
    for file_id in file_ids[:MAX_ATTACH_FILES]:
        path = _safe_uploaded_path(session_id, file_id)
        if path is None or not path.exists() or not path.is_file():
            continue
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        lines.append(f"{file_id} ({size} bytes)")
    return "\n".join(lines) if lines else "(none)"


def _read_uploaded_file_content(session_id: str, file_id: str) -> str:
    path = _safe_uploaded_path(session_id, file_id)
    if path is None:
        return "读取失败：附件路径无效。"
    if not path.exists() or not path.is_file():
        return f"读取失败：附件不存在：{file_id}"
    try:
        content = _read_text_file_with_fallback(path)
    except Exception as exc:
        return f"读取失败：{exc}"
    if len(content) > MAX_ATTACH_CHARS_PER_FILE:
        remain = len(content) - MAX_ATTACH_CHARS_PER_FILE
        return content[:MAX_ATTACH_CHARS_PER_FILE] + f"\n\n... ({remain} more chars)"
    return content


def _build_attachment_manifest(session_id: str, attachment_ids: list[str]) -> str:
    listing = _list_uploaded_files_content(session_id, attachment_ids)
    if listing == "(none)":
        return ""
    return (
        "User uploaded files are available in local storage.\n"
        "Use list_uploaded_files/read_uploaded_file tools to inspect on demand.\n"
        "<uploaded_files>\n"
        f"{listing}\n"
        "</uploaded_files>"
    )


def _extract_thinking_mode() -> str:
    payload = request.get_json(silent=True) or {}
    return normalize_thinking_mode(payload.get("thinking_mode", ""))


def _touch_session(session_id: str, state: ConversationState) -> None:
    state.updated_at = time.time()
    _save_session(session_id, state)


def _clip_trace_value(value: object, limit: int = 800) -> object:
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + f"... ({len(value) - limit} more chars)"
    if isinstance(value, list):
        return [_clip_trace_value(item, limit=limit) for item in value[:10]]
    if isinstance(value, dict):
        return {str(key): _clip_trace_value(item, limit=limit) for key, item in list(value.items())[:20]}
    return value


def _trace_event(session_id: str, state: ConversationState, event: str, **payload: object) -> None:
    entry = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "event": event,
        "data": {key: _clip_trace_value(value) for key, value in payload.items()},
    }
    state.debug_trace.append(entry)
    if len(state.debug_trace) > 200:
        state.debug_trace = state.debug_trace[-200:]
    if AGENT_TRACE:
        print(
            f"[agent-trace][session={session_id}][{event}] "
            f"{json.dumps(entry['data'], ensure_ascii=False)}",
            flush=True,
        )


def _derive_session_title(state: ConversationState) -> str:
    for message in state.assistant.messages:
        if message.get("role") == "user":
            text = message.get("content", "").strip()
            if text:
                return text[:40]
    return "新会话"


def _delete_session(session_id: str) -> bool:
    removed = _sessions.pop(session_id, None) is not None
    path = _session_file_path(session_id)
    if path.exists():
        try:
            path.unlink()
        except OSError:
            pass
        removed = True
    upload_dir = _session_upload_dir(session_id)
    if upload_dir.exists() and upload_dir.is_dir():
        for child in sorted(upload_dir.rglob("*"), reverse=True):
            try:
                if child.is_file() or child.is_symlink():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            except OSError:
                continue
        try:
            upload_dir.rmdir()
        except OSError:
            pass
        removed = True
    return removed


def _extract_explicit_command(user_input: str) -> str | None:
    text = user_input.strip()
    if not text:
        return None

    known_prefixes = (
        "ls ", "pwd", "cat ", "rg ", "touch ", "mkdir ", "echo ", "date", "whoami",
        "head ", "tail ", "wc ", "sed ", "python3 ", "gcc ", "cc ", "clang ", "make ",
    )
    lowered = text.lower()
    if any(lowered.startswith(prefix) for prefix in known_prefixes):
        return text

    if lowered.startswith("!"):
        return text[1:].strip() or None

    run_markers = ("执行命令", "运行命令", "run command", "execute command")
    for marker in run_markers:
        idx = lowered.find(marker)
        if idx >= 0:
            cmd = text[idx + len(marker):].strip(" ：:;,")
            return cmd or None

    return None


def _should_autorun_fft_for_uploads(user_input: str, state: ConversationState) -> bool:
    text = (user_input or "").strip().lower()
    if not text:
        return False
    if not state.attached_file_ids:
        return False
    has_analyze_intent = any(keyword in text for keyword in ("分析", "analy", "fft", "频率", "振动"))
    if not has_analyze_intent:
        return False
    data_file_count = sum(
        1
        for file_id in state.attached_file_ids
        if str(file_id).lower().endswith((".csv", ".txt"))
    )
    return data_file_count > 0


def _format_shell_result(result: str) -> str:
    if not result.startswith("exit_code:"):
        return result

    exit_code = None
    stdout = ""
    stderr = ""
    current = None
    for line in result.splitlines():
        if line.startswith("exit_code:"):
            try:
                exit_code = int(line.split(":", 1)[1].strip())
            except ValueError:
                exit_code = None
            current = None
            continue
        if line == "stdout:":
            current = "stdout"
            continue
        if line == "stderr:":
            current = "stderr"
            continue
        if current == "stdout":
            stdout = f"{stdout}\n{line}".strip()
        elif current == "stderr":
            stderr = f"{stderr}\n{line}".strip()

    if exit_code is not None and exit_code != 0:
        return f"命令执行失败：{stderr or '请检查命令参数。'}"
    if stdout:
        return stdout
    if stderr:
        return stderr
    return "已执行完成。"


def _run_system_command(command: str) -> str:
    command = command.strip()
    if not command:
        return "Usage: execute command <cmd>"
    timeout = int(os.environ.get("SHELL_TIMEOUT", "0"))
    timeout_value = timeout if timeout > 0 else None
    cwd = os.environ.get("SHELL_CWD") or str(PROJECT_ROOT)
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
        return f"Command timed out after {timeout}s."
    except Exception as exc:
        return f"Command execution failed: {exc}"

    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    parts = [f"exit_code: {result.returncode}"]
    if stdout:
        parts.append("stdout:")
        parts.append(stdout)
    if stderr:
        parts.append("stderr:")
        parts.append(stderr)
    if not stdout and not stderr:
        parts.append("(no output)")
    return "\n".join(parts)


def _parse_router_json(raw: str) -> dict | None:
    text = raw.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        return None


def _safe_project_path(path_value: str) -> Path | None:
    raw = str(path_value or "").strip()
    if not raw:
        return None
    candidate = (PROJECT_ROOT / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
    try:
        candidate.relative_to(PROJECT_ROOT)
    except ValueError:
        return None
    return candidate


def _read_project_file(path_value: str) -> str:
    path = _safe_project_path(path_value)
    if path is None:
        return "读取失败：路径无效或超出项目目录。"
    if not path.exists():
        return f"读取失败：文件不存在：{path}"
    if not path.is_file():
        return f"读取失败：不是文件：{path}"
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"读取失败：文件不是 UTF-8 文本：{path}"
    except Exception as exc:
        return f"读取失败：{exc}"


def _list_project_dir(path_value: str) -> str:
    raw = str(path_value or "").strip() or "."
    path = _safe_project_path(raw)
    if path is None:
        return "列目录失败：路径无效或超出项目目录。"
    if not path.exists():
        return f"列目录失败：目录不存在：{path}"
    if not path.is_dir():
        return f"列目录失败：不是目录：{path}"
    try:
        items = sorted(path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
    except Exception as exc:
        return f"列目录失败：{exc}"
    if not items:
        return "(empty)"
    lines = []
    for item in items[:200]:
        suffix = "/" if item.is_dir() else ""
        rel = item.relative_to(PROJECT_ROOT)
        lines.append(f"{rel}{suffix}")
    if len(items) > 200:
        lines.append(f"... ({len(items) - 200} more)")
    return "\n".join(lines)


def _search_project_text(pattern: str, path_value: str = "") -> str:
    pattern = str(pattern or "").strip()
    if not pattern:
        return "搜索失败：缺少 pattern。"
    raw_path = str(path_value or "").strip() or "."
    path = _safe_project_path(raw_path)
    if path is None:
        return "搜索失败：路径无效或超出项目目录。"
    if not path.exists():
        return f"搜索失败：路径不存在：{path}"

    cwd = os.environ.get("SHELL_CWD") or str(PROJECT_ROOT)
    target = str(path.relative_to(PROJECT_ROOT))
    try:
        result = subprocess.run(
            ["rg", "-n", "--no-heading", "--color", "never", pattern, target],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except FileNotFoundError:
        return "搜索失败：未找到 rg。"
    except subprocess.TimeoutExpired:
        return "搜索失败：搜索超时。"
    except Exception as exc:
        return f"搜索失败：{exc}"

    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    if result.returncode not in {0, 1}:
        return stderr or stdout or f"搜索失败：exit_code={result.returncode}"
    if not stdout:
        return "(no matches)"
    lines = stdout.splitlines()
    if len(lines) > 200:
        return "\n".join(lines[:200] + [f"... ({len(lines) - 200} more matches)"])
    return stdout


def _plan_system_commands(session_id: str, state: ConversationState, user_input: str) -> dict | None:
    if not skill_manager or not list(skill_manager.names()):
        return None

    router_prompt = _build_router_prompt()
    attachment_manifest = _build_attachment_manifest(session_id, state.attached_file_ids)
    extra_context = f"\n\n{attachment_manifest}\n" if attachment_manifest else ""
    return _request_router_decision(
        [
            {"role": "system", "content": router_prompt},
            {
                "role": "user",
                "content": (
                    f"User request:\n{user_input}{extra_context}\n\n"
                    f"Project root: {PROJECT_ROOT}\n"
                    "Only return JSON."
                ),
            },
        ],
        session_id=session_id,
        state=state,
        phase="phase1_router",
    )


def _request_router_decision(
    messages: list[dict[str, str]],
    *,
    session_id: str,
    state: ConversationState,
    phase: str,
) -> dict | None:
    messages = [
        *messages,
    ]
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
    }
    if MODEL_PROVIDER == "ollama":
        payload["think"] = False
    try:
        headers = {}
        if MODEL_PROVIDER in {"openai", "openai_compatible", "openai-compatible"}:
            headers["Content-Type"] = "application/json"
            if MODEL_API_KEY:
                headers["Authorization"] = f"Bearer {MODEL_API_KEY}"
        resp = requests.post(MODEL_BASE_URL, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if MODEL_PROVIDER in {"openai", "openai_compatible", "openai-compatible"}:
            choices = data.get("choices") or []
            raw = (choices[0].get("message") or {}).get("content", "") if choices else ""
        else:
            raw = data.get("message", {}).get("content", "")
    except Exception as exc:
        _trace_event(session_id, state, phase, status="error", error=str(exc))
        return None
    parsed = _parse_router_json(raw)
    _trace_event(session_id, state, phase, status="ok", raw=raw, parsed=parsed)
    return parsed


def _build_router_prompt() -> str:
    catalog = skill_manager.available_skills_catalog() if skill_manager else "<available_skills>\n(none)\n</available_skills>"
    tool_catalog = _available_tool_catalog()
    return (
        SKILL_ROUTER_PROMPT_TEMPLATE
        .replace("<<SKILL_CATALOG>>", catalog)
        .replace("<<TOOL_CATALOG>>", tool_catalog)
    )


def _available_tool_catalog() -> str:
    return "\n".join(
        [
            "<available_tools>",
            "- name: read_skill",
            "  description: Read the rendered SKILL.md body for a named skill.",
            "  parameters: { skill: string }",
            "- name: list_dir",
            "  description: List files and directories under a project-relative directory.",
            "  parameters: { path: string }",
            "- name: read_file",
            "  description: Read a UTF-8 text file inside the project.",
            "  parameters: { path: string }",
            "- name: search_text",
            "  description: Search project files using ripgrep-style text matching.",
            "  parameters: { pattern: string, path?: string }",
            "- name: list_uploaded_files",
            "  description: List uploaded files currently attached to this session.",
            "  parameters: {}",
            "- name: read_uploaded_file",
            "  description: Read one uploaded text file by file_id.",
            "  parameters: { file_id: string }",
            "- name: exec_command",
            "  description: Run a shell command in the project workspace.",
            "  parameters: { command: string }",
            "- name: tool_calls",
            "  description: Execute multiple low-level tools in order.",
            "  parameters: { calls: [{ tool: string, ...tool specific params }] }",
            "- name: use_skill",
            "  description: Call a skill's packaged runtime entrypoint. Only valid for skills with invocation=packaged_runtime.",
            "  parameters: { skill: string, input: string | object }",
            "</available_tools>",
        ]
    )


def _plan_with_skill_doc(session_id: str, state: ConversationState, user_input: str, skill_name: str) -> dict | None:
    if not skill_manager:
        return None
    try:
        skill = skill_manager.get(skill_name)
        skill_doc = skill.render_markdown()
    except SkillRegistryError:
        return None
    _trace_event(
        session_id,
        state,
        "read_skill_doc",
        skill=skill_name,
        skill_doc_preview=skill_doc[:1200],
    )

    router_prompt = _build_router_prompt()
    return _request_router_decision(
        [
            {"role": "system", "content": router_prompt},
            {
                "role": "system",
                "content": (
                    f"You have selected the skill '{skill_name}'.\n"
                    "Below is the full SKILL.md content. Follow it as the skill manual "
                    "when deciding the final action.\n"
                    f"This skill invocation mode is: {'packaged_runtime' if skill.has_runtime else 'manual_only'}.\n"
                    "If invocation mode is packaged_runtime and user asks to execute/analyze with this skill, "
                    "prefer action=use_skill with the user request as input.\n"
                    "If the skill is manual_only, do not return use_skill.\n"
                    "For manual_only skills, prefer low-level tool actions such as read_file, "
                    "list_dir, search_text, and exec_command.\n"
                    "If a manual_only task involves more than one step, prefer action=tool_calls "
                    "with an ordered sequence of low-level actions.\n"
                    "If the manual suggests exploring files or reading config before execution, "
                    "model that explicitly with tool_calls.\n\n"
                    f"<skill_doc name=\"{skill_name}\">\n{skill_doc}\n</skill_doc>"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User request:\n{user_input}\n\n"
                    f"Project root: {PROJECT_ROOT}\n"
                    f"Selected skill directory: {skill.pack_dir}\n"
                    "Resolve example commands against the selected skill directory.\n"
                    "If the manual contains stale repo paths like skills/email/... but the selected "
                    "skill directory is different, rewrite the command to the actual selected skill path.\n"
                    "Return the final JSON action now."
                ),
            },
        ],
        session_id=session_id,
        state=state,
        phase="phase2_router",
    )


def _run_planned_commands(session_id: str, state: ConversationState, decision: dict) -> str | None:
    action = str(decision.get("action", "")).strip().lower()
    _trace_event(session_id, state, "planned_action", decision=decision)
    if action == "none":
        return None

    if action == "tool_calls":
        calls = decision.get("calls")
        if not isinstance(calls, list):
            return None
        context = decision.get("_skill_context")
        outputs: list[str] = []
        for i, call in enumerate(calls[:5], start=1):
            if not isinstance(call, dict):
                continue
            if context and "_skill_context" not in call:
                call = {**call, "_skill_context": context}
            result = _run_single_tool_action(session_id, state, call)
            if not result:
                continue
            outputs.append(f"步骤{i}:\n{result}")
        return "\n\n".join(outputs) if outputs else None

    return _run_single_tool_action(session_id, state, decision)


def _normalize_skill_command(command: str, skill_name: str) -> str:
    if not skill_manager:
        return command
    try:
        skill = skill_manager.get(skill_name)
    except SkillRegistryError:
        return command

    normalized = command
    rel_pack = str(skill.pack_dir.relative_to(PROJECT_ROOT))
    skill_dir_markers = {
        f"cd {skill.pack_dir}",
        f"cd '{skill.pack_dir}'",
        f'cd "{skill.pack_dir}"',
        f"cd {rel_pack}",
        f"cd '{rel_pack}'",
        f'cd "{rel_pack}"',
    }
    uses_skill_cwd = any(marker in normalized for marker in skill_dir_markers)

    # Rewrite stale repo-prefixed paths found in imported OpenClaw skills.
    stale_prefixes = (
        "skills/email/scripts/",
        "skills/baidu-search/scripts/",
    )
    for prefix in stale_prefixes:
        if prefix in normalized:
            suffix = normalized.split(prefix, 1)[1]
            file_name = suffix.split()[0]
            candidate = skill.pack_dir / "scripts" / Path(file_name).name
            if candidate.exists():
                target = (
                    f"scripts/{Path(file_name).name}"
                    if uses_skill_cwd
                    else f"{rel_pack}/scripts/{Path(file_name).name}"
                )
                normalized = normalized.replace(prefix + file_name, target)

    # If the command already targets scripts/<file>, anchor it to the selected skill dir.
    if not uses_skill_cwd:
        normalized = re.sub(
            r"(?<!\S)scripts/([A-Za-z0-9_.-]+)",
            lambda match: f"{rel_pack}/scripts/{match.group(1)}",
            normalized,
        )
    return normalized


def _run_single_tool_action(session_id: str, state: ConversationState, decision: dict) -> str | None:
    action = str(decision.get("action", decision.get("tool", ""))).strip().lower()
    _trace_event(session_id, state, "tool_action_start", action=action, payload=decision)
    if action == "read_skill":
        if not skill_manager:
            return None
        skill_name = str(decision.get("skill", "")).strip()
        if not skill_name:
            return None
        try:
            doc = skill_manager.read_skill_doc(skill_name)
        except Exception:
            return None
        result = f"[{skill_name}] SKILL.md\n\n{doc}"
        _trace_event(session_id, state, "tool_action_end", action=action, result=result[:1200])
        return result

    if action == "list_dir":
        path_value = str(decision.get("path", "")).strip() or "."
        content = _list_project_dir(path_value)
        result = f"[list_dir] {path_value}\n\n{content}"
        _trace_event(session_id, state, "tool_action_end", action=action, result=result[:1200])
        return result

    if action == "read_file":
        path_value = str(decision.get("path", "")).strip()
        if not path_value:
            return None
        content = _read_project_file(path_value)
        result = f"[read_file] {path_value}\n\n{content}"
        _trace_event(session_id, state, "tool_action_end", action=action, result=result[:1200])
        return result

    if action == "search_text":
        pattern = str(decision.get("pattern", "")).strip()
        path_value = str(decision.get("path", "")).strip()
        if not pattern:
            return None
        content = _search_project_text(pattern, path_value)
        label = path_value or "."
        result = f"[search_text] {pattern} @ {label}\n\n{content}"
        _trace_event(session_id, state, "tool_action_end", action=action, result=result[:1200])
        return result

    if action == "list_uploaded_files":
        content = _list_uploaded_files_content(session_id, state.attached_file_ids)
        result = f"[list_uploaded_files]\n\n{content}"
        _trace_event(session_id, state, "tool_action_end", action=action, result=result[:1200])
        return result

    if action == "read_uploaded_file":
        file_id = str(decision.get("file_id", "")).strip()
        if not file_id:
            return "读取失败：缺少 file_id。"
        content = _read_uploaded_file_content(session_id, file_id)
        result = f"[read_uploaded_file] {file_id}\n\n{content}"
        _trace_event(session_id, state, "tool_action_end", action=action, result=result[:1200])
        return result

    if action == "exec_command":
        command = str(decision.get("command", "")).strip()
        if not command:
            return None
        skill_context = decision.get("_skill_context")
        skill_name = ""
        if isinstance(skill_context, dict):
            skill_name = str(skill_context.get("skill", "")).strip()
        if skill_name:
            normalized = _normalize_skill_command(command, skill_name)
            if normalized != command:
                _trace_event(
                    session_id,
                    state,
                    "tool_action_rewrite",
                    action=action,
                    skill=skill_name,
                    original_command=command,
                    normalized_command=normalized,
                )
                command = normalized
        result = _run_system_command(command)
        formatted = f"[exec_command] {command}\n\n{_format_shell_result(result)}"
        _trace_event(session_id, state, "tool_action_end", action=action, command=command, result=formatted[:1200])
        return formatted

    if action == "use_skill":
        if not skill_manager:
            return None
        skill_name = str(decision.get("skill", "")).strip()
        skill_input = decision.get("input", "")
        if not skill_name:
            payload = {
                "code": "SKILL_NOT_FOUND",
                "message": "No skill provided.",
                "requested_name": skill_name,
                "candidates": [],
            }
            return json.dumps(payload, ensure_ascii=False)
        try:
            result = skill_manager.execute(
                skill_name,
                skill_input,
                session=state.assistant,
            )
        except SkillNotFoundError as exc:
            payload = {
                "code": "SKILL_NOT_FOUND",
                "message": str(exc),
                "requested_name": exc.requested_name,
                "candidates": exc.candidates,
            }
            _trace_event(session_id, state, "tool_action_error", action=action, error=payload)
            return json.dumps(payload, ensure_ascii=False)
        except SkillAmbiguousError as exc:
            payload = {
                "code": "SKILL_AMBIGUOUS",
                "message": str(exc),
                "requested_name": exc.requested_name,
                "candidates": exc.candidates,
            }
            _trace_event(session_id, state, "tool_action_error", action=action, error=payload)
            return json.dumps(payload, ensure_ascii=False)
        except SkillRegistryError as exc:
            payload = {
                "code": "SKILL_EXECUTION_FAILED",
                "message": str(exc),
                "requested_name": skill_name,
                "candidates": [],
            }
            _trace_event(session_id, state, "tool_action_error", action=action, error=payload)
            return json.dumps(payload, ensure_ascii=False)
        except Exception as exc:
            payload = {
                "code": "SKILL_EXECUTION_FAILED",
                "message": str(exc),
                "requested_name": skill_name,
                "candidates": [],
            }
            _trace_event(session_id, state, "tool_action_error", action=action, error=payload)
            return json.dumps(payload, ensure_ascii=False)
        _trace_event(session_id, state, "tool_action_end", action=action, skill=skill_name, input=skill_input, result=result[:1200])
        return result

    if action == "use_skills":
        if not skill_manager:
            return None
        calls = decision.get("calls")
        if not isinstance(calls, list):
            return None
        calls = calls[:3]
        outputs: list[str] = []
        for i, call in enumerate(calls, start=1):
            if not isinstance(call, dict):
                continue
            skill_name = str(call.get("skill", "")).strip()
            skill_input = call.get("input", "")
            if not skill_name:
                continue
            try:
                result = skill_manager.execute(
                    skill_name,
                    skill_input,
                    session=state.assistant,
                )
            except SkillNotFoundError as exc:
                payload = {
                    "code": "SKILL_NOT_FOUND",
                    "message": str(exc),
                    "requested_name": exc.requested_name,
                    "candidates": exc.candidates,
                }
                result = json.dumps(payload, ensure_ascii=False)
            except SkillAmbiguousError as exc:
                payload = {
                    "code": "SKILL_AMBIGUOUS",
                    "message": str(exc),
                    "requested_name": exc.requested_name,
                    "candidates": exc.candidates,
                }
                result = json.dumps(payload, ensure_ascii=False)
            except Exception as exc:
                payload = {
                    "code": "SKILL_EXECUTION_FAILED",
                    "message": str(exc),
                    "requested_name": skill_name,
                    "candidates": [],
                }
                result = json.dumps(payload, ensure_ascii=False)
            outputs.append(f"步骤{i}: [{skill_name}] {result}")
        final = "\n\n".join(outputs) if outputs else None
        if final:
            _trace_event(session_id, state, "tool_action_end", action=action, result=final[:1200])
        return final

    # Backward compatibility for older router outputs.
    if action == "run_time":
        if not skill_manager or "time" not in set(skill_manager.names()):
            return None
        result = f"当前时间：{skill_manager.execute('time', '%Y-%m-%d %H:%M:%S')}"
        _trace_event(session_id, state, "tool_action_end", action=action, result=result)
        return result
    if action == "run_command":
        command = str(decision.get("command", "")).strip()
        if not command:
            return None
        result = _run_system_command(command)
        formatted = _format_shell_result(result)
        _trace_event(session_id, state, "tool_action_end", action=action, command=command, result=formatted[:1200])
        return formatted

    if action == "run_commands":
        commands = decision.get("commands")
        if not isinstance(commands, list):
            return None
        commands = [str(cmd).strip() for cmd in commands if str(cmd).strip()]
        if not commands:
            return None
        commands = commands[:3]

        outputs = []
        for i, cmd in enumerate(commands, start=1):
            result = _run_system_command(cmd)
            formatted = _format_shell_result(result)
            outputs.append(f"步骤{i}: `{cmd}`\n{formatted}")
            if formatted.startswith("命令执行失败："):
                break
        final = "\n\n".join(outputs)
        _trace_event(session_id, state, "tool_action_end", action=action, commands=commands, result=final[:1200])
        return final

    return None


def _try_system_response(
    session_id: str,
    state: ConversationState,
    user_input: str,
    display_user_input: str | None = None,
) -> str | None:
    shown_input = display_user_input if display_user_input is not None else user_input
    _trace_event(session_id, state, "user_request", input=shown_input)
    explicit_command = _extract_explicit_command(user_input)
    if explicit_command:
        result = _run_system_command(explicit_command)
        message = _format_shell_result(result)
        _trace_event(session_id, state, "explicit_command", command=explicit_command, result=message[:1200])
        state.assistant.add_user_message(shown_input)
        state.assistant.add_assistant_message(message)
        _touch_session(session_id, state)
        return message

    if _should_autorun_fft_for_uploads(shown_input, state) and skill_manager is not None:
        upload_dir = _session_upload_dir(session_id)
        try:
            fft_result = skill_manager.execute(
                "fft-frequency",
                {
                    "path": str(upload_dir),
                    "files": state.attached_file_ids,
                },
                session=state.assistant,
            )
            _trace_event(
                session_id,
                state,
                "autorun_fft",
                upload_dir=str(upload_dir),
                attached_files=state.attached_file_ids,
            )
            state.assistant.add_user_message(shown_input)
            state.assistant.add_assistant_message(fft_result)
            _touch_session(session_id, state)
            return fft_result
        except Exception as exc:
            _trace_event(session_id, state, "autorun_fft_error", error=str(exc))

    decision = _plan_system_commands(session_id, state, user_input)
    if not decision:
        return None
    action = str(decision.get("action", "")).strip().lower()
    skill_name = str(decision.get("skill", "")).strip()
    if skill_manager and skill_name and action in {"read_skill", "use_skill"}:
        try:
            skill = skill_manager.get(skill_name)
        except SkillRegistryError:
            skill = None
        if action == "read_skill" or (skill is not None and not skill.has_runtime):
            if action == "use_skill" and skill is not None and not skill.has_runtime:
                _trace_event(
                    session_id,
                    state,
                    "manual_only_reroute",
                    skill=skill_name,
                    reason="router selected use_skill for a manual-only skill",
                )
            followup = _plan_with_skill_doc(session_id, state, user_input, skill_name)
            if followup:
                followup["_skill_context"] = {"skill": skill_name}
                decision = followup
    planned = _run_planned_commands(session_id, state, decision)
    if not planned:
        return None
    state.assistant.add_user_message(shown_input)
    state.assistant.add_assistant_message(planned)
    _touch_session(session_id, state)
    return planned


def _stream_chat(
    session_id: str,
    state: ConversationState,
    user_input: str,
    model_input: str | None = None,
) -> Response:
    state.assistant.add_user_message(user_input)
    effective_input = model_input if model_input is not None else user_input
    if state.assistant.messages:
        state.assistant.messages[-1]["content"] = effective_input
    _touch_session(session_id, state)

    def generate():
        reply = ""
        thinking = ""
        try:
            response = requests.post(
                state.assistant.base_url,
                json=state.assistant._build_payload(stream=True),
                headers=state.assistant._build_headers(),
                stream=True,
                timeout=state.assistant.timeout,
            )
            response.raise_for_status()
            provider = (state.assistant.provider or DEFAULT_PROVIDER).strip().lower()

            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if provider == "ollama":
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if "message" in payload:
                        message = payload.get("message", {}) or {}
                        if isinstance(message, dict):
                            think_chunk = (
                                message.get("thinking")
                                or message.get("reasoning_content")
                                or payload.get("thinking")
                                or payload.get("reasoning_content")
                                or ""
                            )
                            if think_chunk:
                                thinking += think_chunk
                                yield f"event: thinking\ndata: {json.dumps({'text': think_chunk}, ensure_ascii=False)}\n\n"

                            chunk = message.get("content", "") or ""
                            if chunk:
                                reply += chunk
                                yield f"event: answer\ndata: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"

                    if payload.get("done"):
                        break
                    continue

                raw = line.strip()
                if raw.startswith("data:"):
                    raw = raw[5:].strip()
                if raw == "[DONE]":
                    break
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                choices = payload.get("choices") or []
                if not choices:
                    continue
                think_chunk = _extract_reasoning_text(payload, provider)
                if think_chunk:
                    thinking += think_chunk
                    yield f"event: thinking\ndata: {json.dumps({'text': think_chunk}, ensure_ascii=False)}\n\n"
                delta = choices[0].get("delta") or {}
                chunk = delta.get("content", "") or ""
                if chunk:
                    reply += chunk
                    yield f"event: answer\ndata: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"
        except requests.RequestException as exc:
            reply = f"[network error] {exc}"
            yield f"event: error\ndata: {json.dumps({'text': reply}, ensure_ascii=False)}\n\n"
        finally:
            if state.assistant.messages and state.assistant.messages[-1].get("role") == "user":
                state.assistant.messages[-1]["content"] = user_input
            if thinking:
                state.assistant.messages.append(
                    {"role": "assistant", "content": reply, "thinking": thinking}
                )
            else:
                state.assistant.add_assistant_message(reply)
            _touch_session(session_id, state)
            yield "event: done\ndata: {}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


_migrate_legacy_uploads()
_load_sessions_from_disk()


@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    user_input = str(payload.get("message", "")).strip()
    if not user_input:
        return Response("No message provided.", status=400)
    raw_attachment_ids = payload.get("attachment_ids", [])
    attachment_ids = [
        str(item).strip()
        for item in (raw_attachment_ids if isinstance(raw_attachment_ids, list) else [])
        if str(item).strip()
    ]

    session_id = _extract_session_id()
    state = _ensure_session(session_id)
    state.assistant.thinking_mode = _extract_thinking_mode()
    if attachment_ids:
        unique_ids: list[str] = []
        seen: set[str] = set()
        for file_id in [*state.attached_file_ids, *attachment_ids]:
            if file_id in seen:
                continue
            seen.add(file_id)
            unique_ids.append(file_id)
        state.attached_file_ids = unique_ids[:MAX_ATTACH_FILES]
        _trace_event(
            session_id,
            state,
            "attachments_bound",
            attachment_count=len(state.attached_file_ids),
            files=state.attached_file_ids,
        )
        _touch_session(session_id, state)
    model_input = user_input

    system_message = _try_system_response(
        session_id,
        state,
        model_input,
        display_user_input=user_input,
    )
    if system_message is not None:
        def system_generate():
            yield f"event: answer\ndata: {json.dumps({'text': system_message}, ensure_ascii=False)}\n\n"
            yield "event: done\ndata: {}\n\n"

        return Response(
            system_generate(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return _stream_chat(session_id, state, user_input, model_input=model_input)


@app.route("/session/new", methods=["POST"])
def new_session():
    session_id = str(uuid.uuid4())
    state = _ensure_session(session_id)
    return {
        "session_id": session_id,
        "title": _derive_session_title(state),
        "updated_at": state.updated_at,
    }


@app.route("/session/delete", methods=["POST"])
def delete_session():
    session_id = _extract_session_id()
    deleted = _delete_session(session_id)
    return {"session_id": session_id, "deleted": deleted}


@app.route("/upload", methods=["POST"])
def upload_files():
    session_id = _extract_session_id_from_form()
    _ensure_session(session_id)

    files = request.files.getlist("files")
    if not files:
        return {"ok": False, "error": "No files uploaded."}, 400

    upload_dir = _session_upload_dir(session_id)
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_files: list[dict[str, object]] = []
    rejected_files: list[str] = []
    for file_item in files:
        original_name = str(getattr(file_item, "filename", "") or "").strip()
        if not original_name:
            continue
        file_item.stream.seek(0, os.SEEK_END)
        size = int(file_item.stream.tell() or 0)
        file_item.stream.seek(0)
        if size > MAX_UPLOAD_SIZE_BYTES:
            rejected_files.append(original_name)
            continue
        safe_name = _safe_upload_name(original_name)
        if not _is_text_upload(safe_name):
            rejected_files.append(original_name)
            continue
        target = upload_dir / safe_name
        file_item.save(target)
        saved_files.append(
            {
                "file_id": safe_name,
                "name": original_name,
                "saved_path": str(target),
                "size": size,
            }
        )

    if not saved_files:
        return {
            "ok": False,
            "error": "No valid text files uploaded.",
            "rejected_files": rejected_files,
        }, 400

    state = _ensure_session(session_id)
    latest_ids = [
        str(item.get("file_id", "")).strip()
        for item in saved_files
        if str(item.get("file_id", "")).strip()
    ]
    state.attached_file_ids = latest_ids[:MAX_ATTACH_FILES]
    _touch_session(session_id, state)

    return {
        "ok": True,
        "session_id": session_id,
        "upload_dir": str(upload_dir),
        "saved_files": saved_files,
        "rejected_files": rejected_files,
        "attachment_ids": state.attached_file_ids,
    }


@app.route("/sessions", methods=["GET"])
def list_sessions():
    data = []
    for session_id, state in _sessions.items():
        data.append(
            {
                "session_id": session_id,
                "title": _derive_session_title(state),
                "updated_at": state.updated_at,
                "message_count": len(state.assistant.messages),
            }
        )
    data.sort(key=lambda item: item["updated_at"], reverse=True)
    return {"sessions": data}


@app.route("/skills", methods=["GET"])
def list_skills():
    if not skill_manager:
        return {"skills": [], "available": False}
    return {
        "available": True,
        "skills": skill_manager.as_dicts(),
    }


@app.route("/skills/doc", methods=["GET"])
def read_skill_doc():
    skill_name = str(request.args.get("skill", "")).strip()
    if not skill_name:
        return {"ok": False, "error": "No skill provided."}, 400
    if not skill_manager:
        return {"ok": False, "error": "Skills are unavailable."}, 400
    try:
        content = skill_manager.read_skill_doc(skill_name)
    except SkillRegistryError as exc:
        return {"ok": False, "error": str(exc)}, 400
    return {"ok": True, "skill": skill_name, "content": content}


@app.route("/tools/list_dir", methods=["GET"])
def list_dir_tool():
    path_value = str(request.args.get("path", "")).strip() or "."
    return {"ok": True, "path": path_value, "content": _list_project_dir(path_value)}


@app.route("/tools/read_file", methods=["GET"])
def read_file_tool():
    path_value = str(request.args.get("path", "")).strip()
    if not path_value:
        return {"ok": False, "error": "No path provided."}, 400
    return {"ok": True, "path": path_value, "content": _read_project_file(path_value)}


@app.route("/tools/search_text", methods=["GET"])
def search_text_tool():
    pattern = str(request.args.get("pattern", "")).strip()
    path_value = str(request.args.get("path", "")).strip()
    if not pattern:
        return {"ok": False, "error": "No pattern provided."}, 400
    return {
        "ok": True,
        "pattern": pattern,
        "path": path_value or ".",
        "content": _search_project_text(pattern, path_value),
    }


@app.route("/skills/reload", methods=["POST"])
def reload_skills():
    global skill_manager
    try:
        if skill_manager is None:
            skill_manager = SkillManager()
        else:
            skill_manager.reload()
    except SkillRegistryError as exc:
        return {"ok": False, "error": str(exc)}, 400

    return {
        "ok": True,
        "skills": skill_manager.as_dicts(),
    }


@app.route("/skills/execute", methods=["POST"])
def execute_skill():
    payload = request.get_json(silent=True) or {}
    skill_name = str(payload.get("skill", "")).strip()
    skill_input = str(payload.get("input", "")).strip()
    if not skill_name:
        return {"ok": False, "error": "No skill provided."}, 400
    if not skill_manager:
        return {"ok": False, "error": "Skills are unavailable."}, 400

    try:
        skill = skill_manager.get(skill_name)
    except SkillRegistryError as exc:
        return {"ok": False, "error": str(exc)}, 400
    if skill.user_invocable is False:
        return {"ok": False, "error": f"Skill '{skill_name}' is not user-invocable."}, 400

    session_id = _extract_session_id()
    state = _ensure_session(session_id)
    try:
        result = skill_manager.execute(skill_name, skill_input, session=state.assistant)
    except SkillDisabled as exc:
        return {"ok": False, "error": str(exc)}, 400
    except SkillUnavailable as exc:
        return {"ok": False, "error": str(exc)}, 400
    except SkillRegistryError as exc:
        return {"ok": False, "error": str(exc)}, 400
    except Exception as exc:
        return {"ok": False, "error": str(exc)}, 500

    return {"ok": True, "result": result, "skill": skill_name}


@app.route("/skills/toggle", methods=["POST"])
def toggle_skill():
    payload = request.get_json(silent=True) or {}
    skill_name = str(payload.get("skill", "")).strip()
    enabled = bool(payload.get("enabled", True))
    if not skill_name:
        return {"ok": False, "error": "No skill provided."}, 400
    if not skill_manager:
        return {"ok": False, "error": "Skills are unavailable."}, 400
    try:
        skill = skill_manager.set_enabled(skill_name, enabled)
    except SkillRegistryError as exc:
        return {"ok": False, "error": str(exc)}, 400

    return {"ok": True, "skill": skill.to_dict()}


@app.route("/session/history", methods=["GET"])
def session_history():
    session_id = _extract_session_id()
    state = _ensure_session(session_id)
    messages = []
    for message in state.assistant.messages:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        item = {"role": role, "content": message.get("content", "")}
        thinking = message.get("thinking")
        if isinstance(thinking, str) and thinking:
            item["thinking"] = thinking
        messages.append(item)
    return {"session_id": session_id, "messages": messages}


@app.route("/session/trace", methods=["GET"])
def session_trace():
    session_id = _extract_session_id()
    state = _ensure_session(session_id)
    return {"session_id": session_id, "trace": state.debug_trace[-200:]}


@app.route("/session/attachments", methods=["GET"])
def session_attachments():
    session_id = _extract_session_id()
    state = _ensure_session(session_id)
    items: list[dict[str, object]] = []
    for file_id in state.attached_file_ids:
        path = _safe_uploaded_path(session_id, file_id)
        if path is None or not path.exists() or not path.is_file():
            continue
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        items.append({"file_id": file_id, "size": size, "path": str(path)})
    return {"session_id": session_id, "files": items}


@app.route("/")
def homepage():
    return send_from_directory(str(PROJECT_ROOT / "web"), "index.html")


if __name__ == "__main__":
    app.run(port=5000)
