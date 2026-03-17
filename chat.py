import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

import requests

StreamCallback = Callable[[str], None]

DEFAULT_MODEL = "qwen3.5:9b"
DEFAULT_URL = "http://localhost:11434/api/chat"


@dataclass
class ChatSession:
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_URL
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
        return {
            "model": self.model,
            "messages": self.messages,
            "stream": stream,
            "think": self.think,
        }

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
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        reply = data["message"]["content"]
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
            stream=True,
            timeout=self.timeout,
        )
        response.raise_for_status()
        reply = ""

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
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

        self.add_assistant_message(reply)
        return reply

    def history(self) -> Iterable[Dict[str, str]]:
        yield from self.messages


_default_session = ChatSession()


def ask(user_msg: str) -> str:
    return _default_session.ask(user_msg)


def ask_stream(user_msg: str, on_chunk: Optional[StreamCallback] = None) -> str:
    return _default_session.stream_chat(user_msg, on_chunk=on_chunk)
