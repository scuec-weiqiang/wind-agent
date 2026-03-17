from __future__ import annotations

import json
import os
import re
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
from app.skill_manager import SkillDisabled, SkillManager, SkillRegistryError, SkillUnavailable

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
    "Use the skill name exactly as listed if you choose one.\n"
    "Return JSON only, one of:\n"
    '{"action":"none"}\n'
    '{"action":"read_skill","skill":"<name>"}\n'
    '{"action":"list_dir","path":"<relative directory path>"}\n'
    '{"action":"read_file","path":"<relative file path>"}\n'
    '{"action":"search_text","pattern":"<text or regex>","path":"<relative path optional>"}\n'
    '{"action":"exec_command","command":"<shell command>"}\n'
    '{"action":"tool_calls","calls":[{"tool":"read_file","path":"README.md"},{"tool":"exec_command","command":"pwd"}]}\n'
    '{"action":"use_skill","skill":"<name>","input":"<input for skill>"}\n'
    '{"action":"use_skill","skill":"<name>","input":{"key":"value"}}\n'
    '{"action":"use_skills","calls":[{"skill":"<name>","input":"..."},{"skill":"<name>","input":"..."}]}\n'
    '{"action":"use_skills","calls":[{"skill":"<name>","input":{"key":"value"}}]}\n'
    "Rules:\n"
    "- no markdown, no prose, JSON only.\n"
    "- if no skill fits, return action=none.\n"
    "- choose read_skill when the user is asking about what a skill does or how to use it.\n"
    "- if a skill is marked invocation=manual_only, do not return use_skill for it; return read_skill first.\n"
    "- use list_dir to inspect folders before picking files.\n"
    "- use read_file when you need file contents before deciding.\n"
    "- use search_text when you need to find matching text in project files.\n"
    "- use exec_command when a shell command is the most direct way to answer.\n"
    "- use tool_calls when the task needs multiple low-level steps.\n"
    "- input may be a string or a JSON object.\n"
    "- prefer using description and when_to_use to decide if a skill fits.\n"
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


_sessions: dict[str, ConversationState] = {}


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
    return _request_router_decision(
        [
            {"role": "system", "content": router_prompt},
            {
                "role": "user",
                "content": (
                    f"User request:\n{user_input}\n\n"
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
                    "At this stage, prefer low-level tool actions such as read_file, "
                    "list_dir, search_text, and exec_command.\n"
                    "If the task involves more than one step, prefer returning action=tool_calls "
                    "with an ordered sequence of low-level tool actions.\n"
                    "Use a single low-level action only when one step is enough.\n"
                    "Only use use_skill if the skill truly needs its own dedicated runtime "
                    "wrapper or if the manual clearly implies a single packaged entrypoint.\n"
                    f"This skill invocation mode is: {'packaged_runtime' if skill.has_runtime else 'manual_only'}.\n"
                    "If the skill is manual_only, do not return use_skill.\n"
                    "If the SKILL.md shows concrete commands, prefer converting them into "
                    "exec_command instead of falling back to use_skill.\n"
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
            return None
        try:
            result = skill_manager.execute(skill_name, skill_input)
        except Exception:
            return None
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
                result = skill_manager.execute(skill_name, skill_input)
            except Exception:
                result = f"Skill '{skill_name}' execution failed."
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


def _try_system_response(session_id: str, state: ConversationState, user_input: str) -> str | None:
    _trace_event(session_id, state, "user_request", input=user_input)
    explicit_command = _extract_explicit_command(user_input)
    if explicit_command:
        result = _run_system_command(explicit_command)
        message = _format_shell_result(result)
        _trace_event(session_id, state, "explicit_command", command=explicit_command, result=message[:1200])
        state.assistant.add_user_message(user_input)
        state.assistant.add_assistant_message(message)
        _touch_session(session_id, state)
        return message

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
    state.assistant.add_user_message(user_input)
    state.assistant.add_assistant_message(planned)
    _touch_session(session_id, state)
    return planned


def _stream_chat(session_id: str, state: ConversationState, user_input: str) -> Response:
    state.assistant.add_user_message(user_input)
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


_load_sessions_from_disk()


@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    user_input = str(payload.get("message", "")).strip()
    if not user_input:
        return Response("No message provided.", status=400)

    session_id = _extract_session_id()
    state = _ensure_session(session_id)
    state.assistant.thinking_mode = _extract_thinking_mode()

    system_message = _try_system_response(session_id, state, user_input)
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

    return _stream_chat(session_id, state, user_input)


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


@app.route("/")
def homepage():
    return send_from_directory(str(PROJECT_ROOT / "web"), "index.html")


if __name__ == "__main__":
    app.run(port=5000)
