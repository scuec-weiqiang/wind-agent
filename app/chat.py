import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

import requests

StreamCallback = Callable[[str], None]

DEFAULT_MODEL = "qwen3.5:9b"
DEFAULT_URL = "http://localhost:11434/api/chat"
DEFAULT_PROVIDER = "ollama"
DEFAULT_OPENAI_COMPAT_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_THINKING_MODE = "auto"


def _extract_reasoning_text(payload: Dict[str, Any], provider: str) -> str:
    if provider == "ollama":
        message = payload.get("message", {}) or {}
        if isinstance(message, dict):
            return (
                message.get("thinking")
                or message.get("reasoning_content")
                or payload.get("thinking")
                or payload.get("reasoning_content")
                or ""
            )
        return ""

    if provider in {"openai", "openai_compatible", "openai-compatible"}:
        choices = payload.get("choices") or []
        if choices:
            first = choices[0] or {}
            delta = first.get("delta") or {}
            message = first.get("message") or {}
            return (
                delta.get("reasoning_content")
                or delta.get("reasoning")
                or delta.get("thinking")
                or message.get("reasoning_content")
                or message.get("reasoning")
                or message.get("thinking")
                or payload.get("reasoning_content")
                or payload.get("reasoning")
                or payload.get("thinking")
                or ""
            )
    return ""


@dataclass
class ChatSession:
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_URL
    provider: str = DEFAULT_PROVIDER
    api_key: str = ""
    think: bool = False
    thinking_mode: str = DEFAULT_THINKING_MODE
    system_prompt: Optional[str] = None
    timeout: int = 120
    messages: List[Dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.reset(self.system_prompt)

    def reset(self, system_prompt: Optional[str] = None) -> None:
        """Resets the conversation while preserving an optional system prompt."""
        if system_prompt is not None:
            self.system_prompt = system_prompt
        self.messages = []
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})

    def _build_payload(self, stream: bool) -> Dict[str, Any]:
        provider = (self.provider or DEFAULT_PROVIDER).strip().lower()
        thinking_mode = (self.thinking_mode or DEFAULT_THINKING_MODE).strip().lower()
        if provider == "ollama":
            think_value = self.think
            if thinking_mode == "on":
                think_value = True
            elif thinking_mode == "off":
                think_value = False
            return {
                "model": self.model,
                "messages": self.messages,
                "stream": stream,
                "think": think_value,
            }
        if provider in {"openai", "openai_compatible", "openai-compatible"}:
            payload = {
                "model": _resolve_model_name_for_thinking(self.model, thinking_mode),
                "messages": self.messages,
                "stream": stream,
            }
            thinking_payload = _build_openai_compatible_thinking_payload(
                payload["model"], thinking_mode
            )
            if thinking_payload is not None:
                payload["thinking"] = thinking_payload
            return payload
        raise ValueError(f"Unsupported model provider: {self.provider}")

    def _build_headers(self) -> Dict[str, str]:
        provider = (self.provider or DEFAULT_PROVIDER).strip().lower()
        headers: Dict[str, str] = {}
        if provider in {"openai", "openai_compatible", "openai-compatible"}:
            headers["Content-Type"] = "application/json"
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _extract_content_nonstream(self, data: Dict[str, Any]) -> str:
        provider = (self.provider or DEFAULT_PROVIDER).strip().lower()
        if provider == "ollama":
            return data.get("message", {}).get("content", "")
        if provider in {"openai", "openai_compatible", "openai-compatible"}:
            choices = data.get("choices") or []
            if choices:
                message = choices[0].get("message") or {}
                return message.get("content", "") or ""
            return data.get("output_text", "") or ""
        return ""

    def _extract_reasoning_nonstream(self, data: Dict[str, Any]) -> str:
        provider = (self.provider or DEFAULT_PROVIDER).strip().lower()
        return _extract_reasoning_text(data, provider)

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def add_assistant_message_with_reasoning(self, content: str, reasoning: str) -> None:
        payload: Dict[str, str] = {"role": "assistant", "content": content}
        if reasoning:
            payload["thinking"] = reasoning
        self.messages.append(payload)

    def ask(self, user_content: str) -> str:
        """Sends a user message and waits for the full completion."""
        self.add_user_message(user_content)
        response = requests.post(
            self.base_url,
            json=self._build_payload(stream=False),
            headers=self._build_headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        reply = self._extract_content_nonstream(data)
        reasoning = self._extract_reasoning_nonstream(data)
        self.add_assistant_message_with_reasoning(reply, reasoning)
        return reply

    def stream_chat(
        self, user_content: str, on_chunk: Optional[StreamCallback] = None
    ) -> str:
        """Streams the assistant reply while optionally emitting chunks via callback."""
        self.add_user_message(user_content)
        response = requests.post(
            self.base_url,
            json=self._build_payload(stream=True),
            headers=self._build_headers(),
            stream=True,
            timeout=self.timeout,
        )
        response.raise_for_status()
        reply = ""
        reasoning = ""
        provider = (self.provider or DEFAULT_PROVIDER).strip().lower()

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if provider == "ollama":
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if "message" in payload:
                    think_chunk = _extract_reasoning_text(payload, provider)
                    if think_chunk:
                        reasoning += think_chunk
                    chunk = payload["message"].get("content", "")
                    if chunk:
                        if on_chunk:
                            on_chunk(chunk)
                        reply += chunk

                if payload.get("done"):
                    break
                continue

            if provider in {"openai", "openai_compatible", "openai-compatible"}:
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
                    reasoning += think_chunk
                delta = choices[0].get("delta") or {}
                chunk = delta.get("content", "") or ""
                if chunk:
                    if on_chunk:
                        on_chunk(chunk)
                    reply += chunk
                continue

        self.add_assistant_message_with_reasoning(reply, reasoning)
        return reply

    def history(self) -> Iterable[Dict[str, str]]:
        yield from self.messages


_default_session = ChatSession()


def ask(user_msg: str) -> str:
    return _default_session.ask(user_msg)


def ask_stream(user_msg: str, on_chunk: Optional[StreamCallback] = None) -> str:
    return _default_session.stream_chat(user_msg, on_chunk=on_chunk)


def normalize_thinking_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    if mode in {"on", "off", "auto"}:
        return mode
    return DEFAULT_THINKING_MODE


def _resolve_model_name_for_thinking(model: str, thinking_mode: str) -> str:
    normalized_model = (model or "").strip()
    mode = normalize_thinking_mode(thinking_mode)
    if mode == "auto":
        return normalized_model
    if normalized_model == "deepseek-chat" and mode == "on":
        return "deepseek-reasoner"
    if normalized_model == "deepseek-reasoner" and mode == "off":
        return "deepseek-chat"
    return normalized_model


def _build_openai_compatible_thinking_payload(
    model: str, thinking_mode: str
) -> Dict[str, str] | None:
    mode = normalize_thinking_mode(thinking_mode)
    if mode == "auto":
        return None

    normalized_model = (model or "").strip().lower()
    if normalized_model.startswith("deepseek-"):
        return {"type": "enabled" if mode == "on" else "disabled"}

    return None
