from __future__ import annotations

import json
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from app.chat import ChatSession, normalize_thinking_mode
from app.runtime_settings import RuntimeSettings

DEFAULT_SESSION_ID = "default"

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


@dataclass
class ConversationState:
    assistant: ChatSession
    created_at: float
    updated_at: float
    debug_trace: list[dict[str, object]] = field(default_factory=list)
    attached_file_ids: list[str] = field(default_factory=list)
    runtime_state: dict[str, object] = field(default_factory=dict)


class SessionStore:
    def __init__(self, settings: RuntimeSettings) -> None:
        self.settings = settings
        self._sessions: dict[str, ConversationState] = {}
        self._migrate_legacy_uploads()
        self._load_sessions_from_disk()

    def create_session(self) -> tuple[str, ConversationState]:
        session_id = str(uuid.uuid4())
        return session_id, self.ensure_session(session_id)

    def ensure_session(self, session_id: str) -> ConversationState:
        state = self._sessions.get(session_id)
        if state is not None:
            state.assistant.id = session_id
            return state

        if len(self._sessions) >= self.settings.max_web_sessions:
            oldest = min(self._sessions.items(), key=lambda item: item[1].updated_at)[0]
            self._sessions.pop(oldest, None)

        state = self._new_conversation_state(session_id)
        self._sessions[session_id] = state
        self.save_session(session_id, state)
        return state

    def delete_session(self, session_id: str) -> bool:
        removed = self._sessions.pop(session_id, None) is not None
        path = self._session_file_path(session_id)
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass
            removed = True

        upload_dir = self.session_upload_dir(session_id)
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

    def list_sessions(self) -> list[dict[str, object]]:
        data = []
        for session_id, state in self._sessions.items():
            data.append(
                {
                    "session_id": session_id,
                    "title": self.derive_session_title(state),
                    "updated_at": state.updated_at,
                    "message_count": len(state.assistant.messages),
                }
            )
        data.sort(key=lambda item: float(item["updated_at"]), reverse=True)
        return data

    def save_session(self, session_id: str, state: ConversationState) -> None:
        self.settings.sessions_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "session_id": session_id,
            "created_at": state.created_at,
            "updated_at": state.updated_at,
            "assistant_messages": state.assistant.messages,
            "debug_trace": state.debug_trace[-200:],
            "attached_file_ids": self.limit_attachment_ids(state.attached_file_ids),
            "runtime_state": self._sanitize_runtime_state(state.runtime_state),
        }
        path = self._session_file_path(session_id)
        tmp_path = path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        tmp_path.replace(path)

    def touch_session(self, session_id: str, state: ConversationState) -> None:
        state.updated_at = time.time()
        self.save_session(session_id, state)

    def set_thinking_mode(self, state: ConversationState, raw_value: str) -> None:
        state.assistant.thinking_mode = normalize_thinking_mode(raw_value)

    def bind_attachments(
        self, session_id: str, state: ConversationState, attachment_ids: list[str]
    ) -> None:
        if not attachment_ids:
            return
        seen: set[str] = set()
        unique_ids: list[str] = []
        for file_id in [*state.attached_file_ids, *attachment_ids]:
            if file_id in seen:
                continue
            seen.add(file_id)
            unique_ids.append(file_id)
        state.attached_file_ids = self.limit_attachment_ids(unique_ids)
        self.touch_session(session_id, state)

    def history_payload(self, session_id: str) -> dict[str, object]:
        state = self.ensure_session(session_id)
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

    def trace_payload(self, session_id: str) -> dict[str, object]:
        state = self.ensure_session(session_id)
        return {"session_id": session_id, "trace": state.debug_trace[-200:]}

    def attachments_payload(self, session_id: str) -> dict[str, object]:
        state = self.ensure_session(session_id)
        files: list[dict[str, object]] = []
        for file_id in state.attached_file_ids:
            path = self.safe_uploaded_path(session_id, file_id)
            if path is None or not path.exists() or not path.is_file():
                continue
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            files.append({"file_id": file_id, "size": size, "path": str(path)})
        return {"session_id": session_id, "files": files}

    def runtime_payload(self, session_id: str) -> dict[str, object]:
        state = self.ensure_session(session_id)
        return {
            "session_id": session_id,
            "runtime_state": self._sanitize_runtime_state(state.runtime_state),
        }

    def derive_session_title(self, state: ConversationState) -> str:
        for message in state.assistant.messages:
            if message.get("role") == "user":
                text = str(message.get("content", "")).strip()
                if text:
                    return text[:40]
        return "新会话"

    def limit_attachment_ids(self, file_ids: list[str]) -> list[str]:
        max_files = self.settings.max_attach_files
        if max_files is None:
            return list(file_ids)
        return list(file_ids[:max_files])

    def session_upload_dir(self, session_id: str) -> Path:
        return self.settings.uploads_dir / self._safe_session_filename(session_id).replace(
            ".json", ""
        )

    def safe_uploaded_path(self, session_id: str, file_id: str) -> Path | None:
        base = self.session_upload_dir(session_id).resolve()
        candidate = (base / str(file_id or "").strip()).resolve()
        try:
            candidate.relative_to(base)
        except ValueError:
            return None
        return candidate

    def safe_upload_name(self, name: str) -> str:
        candidate = Path(name or "").name
        sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", candidate).strip("._")
        return sanitized or f"upload_{uuid.uuid4().hex[:8]}.csv"

    def is_text_upload(self, name: str) -> bool:
        return Path(name or "").suffix.lower() in TEXT_FILE_EXTENSIONS

    def read_uploaded_file_content(self, session_id: str, file_id: str) -> str:
        path = self.safe_uploaded_path(session_id, file_id)
        if path is None:
            return "读取失败：附件路径无效。"
        if not path.exists() or not path.is_file():
            return f"读取失败：附件不存在：{file_id}"
        try:
            content = self._read_text_file_with_fallback(path)
        except Exception as exc:
            return f"读取失败：{exc}"
        limit = self.settings.max_attach_chars_per_file
        if len(content) > limit:
            remain = len(content) - limit
            return content[:limit] + f"\n\n... ({remain} more chars)"
        return content

    def list_uploaded_files_content(self, session_id: str, file_ids: list[str]) -> str:
        if not file_ids:
            return "(none)"
        lines: list[str] = []
        for file_id in self.limit_attachment_ids(file_ids):
            path = self.safe_uploaded_path(session_id, file_id)
            if path is None or not path.exists() or not path.is_file():
                continue
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            lines.append(f"{file_id} ({size} bytes)")
        return "\n".join(lines) if lines else "(none)"

    def attachment_manifest(self, session_id: str, attachment_ids: list[str]) -> str:
        listing = self.list_uploaded_files_content(session_id, attachment_ids)
        if listing == "(none)":
            return ""
        return (
            "User uploaded files are available in local storage.\n"
            "Use list_uploaded_files/read_uploaded_file tools to inspect on demand.\n"
            "<uploaded_files>\n"
            f"{listing}\n"
            "</uploaded_files>"
        )

    def extract_session_id(self, payload: dict[str, object] | None, fallback: str = "") -> str:
        raw = ""
        if isinstance(payload, dict):
            raw = str(payload.get("session_id", "") or "")
        session_id = raw.strip() or fallback.strip()
        return session_id or DEFAULT_SESSION_ID

    def _new_conversation_state(self, session_id: str) -> ConversationState:
        now = time.time()
        return ConversationState(
            assistant=ChatSession(
                id=session_id,
                model=self.settings.model_name,
                base_url=self.settings.model_base_url,
                provider=self.settings.model_provider,
                api_key=self.settings.model_api_key,
                system_prompt=self.settings.system_prompt,
                think=self.settings.enable_thinking,
                thinking_mode=self.settings.thinking_mode,
            ),
            created_at=now,
            updated_at=now,
        )

    def _safe_session_filename(self, session_id: str) -> str:
        safe = re.sub(
            r"[^A-Za-z0-9_-]", "_", (session_id or "").strip() or DEFAULT_SESSION_ID
        )
        return f"{safe}.json"

    def _session_file_path(self, session_id: str) -> Path:
        return self.settings.sessions_dir / self._safe_session_filename(session_id)

    def _migrate_legacy_uploads(self) -> None:
        if "WEB_UPLOADS_DIR" in os.environ:
            return
        legacy_dir = self.settings.project_root / "data" / "uploads"
        if not legacy_dir.exists() or not legacy_dir.is_dir():
            return
        if legacy_dir.resolve() == self.settings.uploads_dir.resolve():
            return

        self.settings.uploads_dir.mkdir(parents=True, exist_ok=True)
        for item in legacy_dir.iterdir():
            target = self.settings.uploads_dir / item.name
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

    def _load_sessions_from_disk(self) -> None:
        if not self.settings.sessions_dir.exists():
            return
        loaded: list[tuple[str, ConversationState]] = []
        for path in self.settings.sessions_dir.glob("*.json"):
            try:
                with open(path, encoding="utf-8") as fh:
                    payload = json.load(fh)
            except Exception:
                continue

            session_id = str(payload.get("session_id", "")).strip()
            if not session_id:
                continue

            state = self._new_conversation_state(session_id)
            state.created_at = float(payload.get("created_at", time.time()))
            state.updated_at = float(payload.get("updated_at", state.created_at))
            assistant_messages = payload.get("assistant_messages")
            if isinstance(assistant_messages, list):
                state.assistant.messages = assistant_messages
            debug_trace = payload.get("debug_trace")
            if isinstance(debug_trace, list):
                state.debug_trace = [item for item in debug_trace if isinstance(item, dict)][
                    -200:
                ]
            attached_file_ids = payload.get("attached_file_ids")
            if isinstance(attached_file_ids, list):
                state.attached_file_ids = self.limit_attachment_ids(
                    [
                        str(item).strip()
                        for item in attached_file_ids
                        if str(item).strip()
                    ]
                )
            runtime_state = payload.get("runtime_state")
            if isinstance(runtime_state, dict):
                state.runtime_state = self._sanitize_runtime_state(runtime_state)
            loaded.append((session_id, state))

        loaded.sort(key=lambda item: item[1].updated_at, reverse=True)
        for session_id, state in loaded[: self.settings.max_web_sessions]:
            self._sessions[session_id] = state

    @staticmethod
    def _read_text_file_with_fallback(path: Path) -> str:
        for encoding in ("utf-8", "utf-8-sig", "gbk"):
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f"Unsupported text encoding: {path}")

    @staticmethod
    def _sanitize_runtime_state(value: dict[str, object]) -> dict[str, object]:
        sanitized: dict[str, object] = {}
        for key, item in list(value.items())[:24]:
            name = str(key).strip()
            if not name:
                continue
            if isinstance(item, str):
                sanitized[name] = item[:2000]
            elif isinstance(item, (int, float, bool)) or item is None:
                sanitized[name] = item
            elif isinstance(item, list):
                sanitized[name] = item[:12]
            elif isinstance(item, dict):
                sanitized[name] = {
                    str(child_key): child_value
                    for child_key, child_value in list(item.items())[:20]
                }
            else:
                sanitized[name] = str(item)[:500]
        return sanitized
