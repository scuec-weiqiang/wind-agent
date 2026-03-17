from __future__ import annotations

import json
import os
import re
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import requests
from flask import Flask, Response, request, send_from_directory

from app.chat import (
    ChatSession,
    DEFAULT_MODEL,
    DEFAULT_OPENAI_COMPAT_URL,
    DEFAULT_PROVIDER,
    DEFAULT_URL,
)
from skills.manager import SkillManager, SkillRegistryError

MODEL_PROVIDER = os.environ.get("MODEL_PROVIDER", DEFAULT_PROVIDER).strip().lower()
MODEL_NAME = os.environ.get("MODEL_NAME", os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL))
MODEL_BASE_URL = os.environ.get("MODEL_BASE_URL", os.environ.get("OLLAMA_URL", "")).strip()
if not MODEL_BASE_URL:
    MODEL_BASE_URL = DEFAULT_OPENAI_COMPAT_URL if MODEL_PROVIDER in {"openai", "openai_compatible", "openai-compatible"} else DEFAULT_URL
MODEL_API_KEY = os.environ.get("MODEL_API_KEY", os.environ.get("OPENAI_API_KEY", "")).strip()
SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "You are a helpful agent coordinating between the user and available skills.",
)
SKILL_ROUTER_PROMPT_TEMPLATE = (
    "You are a skill router. Decide whether to call skill packs.\n"
    "Return JSON only, one of:\n"
    '{"action":"none"}\n'
    '{"action":"use_skill","skill":"<name>","input":"<input for skill>"}\n'
    '{"action":"use_skills","calls":[{"skill":"<name>","input":"..."},{"skill":"<name>","input":"..."}]}\n'
    "Rules:\n"
    "- no markdown, no prose, JSON only.\n"
    "- if no skill fits, return action=none.\n"
    "- for shell skill input, output executable command text only.\n"
    "- at most 3 calls for use_skills.\n\n"
    "Available skills:\n"
    "<<SKILL_CATALOG>>\n"
)
MAX_SESSIONS = int(os.environ.get("MAX_WEB_SESSIONS", "100"))
DEFAULT_SESSION_ID = "default"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENABLE_THINKING = os.environ.get("ENABLE_THINKING", "1").lower() not in {"0", "false", "no"}
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


def _touch_session(session_id: str, state: ConversationState) -> None:
    state.updated_at = time.time()
    _save_session(session_id, state)


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


def _plan_system_commands(user_input: str) -> dict | None:
    if not skill_manager or not list(skill_manager.names()):
        return None

    router_prompt = SKILL_ROUTER_PROMPT_TEMPLATE.replace(
        "<<SKILL_CATALOG>>", skill_manager.router_catalog()
    )
    messages = [
        {"role": "system", "content": router_prompt},
        {
            "role": "user",
            "content": (
                f"User request:\n{user_input}\n\n"
                f"Project root: {PROJECT_ROOT}\n"
                "Only return JSON."
            ),
        },
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
    except Exception:
        return None
    return _parse_router_json(raw)


def _run_planned_commands(decision: dict) -> str | None:
    action = str(decision.get("action", "")).strip().lower()
    if action == "none":
        return None

    if action == "use_skill":
        if not skill_manager:
            return None
        skill_name = str(decision.get("skill", "")).strip()
        skill_input = str(decision.get("input", "")).strip()
        if not skill_name:
            return None
        try:
            return skill_manager.execute(skill_name, skill_input)
        except Exception:
            return None

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
            skill_input = str(call.get("input", "")).strip()
            if not skill_name:
                continue
            try:
                result = skill_manager.execute(skill_name, skill_input)
            except Exception:
                result = f"Skill '{skill_name}' execution failed."
            outputs.append(f"步骤{i}: [{skill_name}] {result}")
        return "\n\n".join(outputs) if outputs else None

    # Backward compatibility for older router outputs.
    if action == "run_time":
        if not skill_manager or "time" not in set(skill_manager.names()):
            return None
        return f"当前时间：{skill_manager.execute('time', '%Y-%m-%d %H:%M:%S')}"
    if action == "run_command":
        command = str(decision.get("command", "")).strip()
        if not command:
            return None
        result = _run_system_command(command)
        return _format_shell_result(result)

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
        return "\n\n".join(outputs)

    return None


def _try_system_response(session_id: str, state: ConversationState, user_input: str) -> str | None:
    explicit_command = _extract_explicit_command(user_input)
    if explicit_command:
        result = _run_system_command(explicit_command)
        message = _format_shell_result(result)
        state.assistant.add_user_message(user_input)
        state.assistant.add_assistant_message(message)
        _touch_session(session_id, state)
        return message

    decision = _plan_system_commands(user_input)
    if not decision:
        return None
    planned = _run_planned_commands(decision)
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


@app.route("/")
def homepage():
    return send_from_directory(str(PROJECT_ROOT / "web"), "index.html")


if __name__ == "__main__":
    app.run(port=5000)
