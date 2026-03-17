import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

import requests

StreamCallback = Callable[[str], None]

DEFAULT_MODEL = "qwen3.5:9b"
DEFAULT_URL = "http://localhost:11434/api/chat"
DEFAULT_PROVIDER = "ollama"
DEFAULT_OPENAI_COMPAT_URL = "https://api.openai.com/v1/chat/completions"


@dataclass
class ChatSession:
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_URL
    provider: str = DEFAULT_PROVIDER
    api_key: str = ""
    think: bool = False
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
        if provider == "ollama":
            return {
                "model": self.model,
                "messages": self.messages,
                "stream": stream,
                "think": self.think,
            }
        if provider in {"openai", "openai_compatible", "openai-compatible"}:
            return {
                "model": self.model,
                "messages": self.messages,
                "stream": stream,
            }
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

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

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
        self.add_assistant_message(reply)
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
                delta = choices[0].get("delta") or {}
                chunk = delta.get("content", "") or ""
                if chunk:
                    if on_chunk:
                        on_chunk(chunk)
                    reply += chunk
                continue

        self.add_assistant_message(reply)
        return reply

    def history(self) -> Iterable[Dict[str, str]]:
        yield from self.messages


_default_session = ChatSession()


def ask(user_msg: str) -> str:
    return _default_session.ask(user_msg)


def ask_stream(user_msg: str, on_chunk: Optional[StreamCallback] = None) -> str:
    return _default_session.stream_chat(user_msg, on_chunk=on_chunk)
