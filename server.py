from __future__ import annotations

import json
import os

import requests
from flask import Flask, Response, request, send_from_directory

DEFAULT_MODEL = "qwen3.5:9b"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "You are a helpful agent coordinating between the user and available skills.",
)

app = Flask(__name__, static_folder=".", static_url_path="")


def _build_messages(user_input: str) -> list[dict[str, str]]:
    messages = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})

    messages.append({"role": "user", "content": user_input})
    return messages


def _stream_ollama(messages: list[dict[str, str]]) -> Response:
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": DEFAULT_MODEL,
            "messages": messages,
            "stream": True,
            "think": False,
        },
        stream=True,
        timeout=120,
    )
    resp.raise_for_status()

    def generate():
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "message" in payload:
                chunk = payload["message"].get("content", "")
                if chunk:
                    yield chunk

            if payload.get("done"):
                break

    return Response(generate(), mimetype="text/plain; charset=utf-8")


@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip()
    if not user_input:
        return Response("No message provided.", status=400)

    messages = _build_messages(user_input)
    return _stream_ollama(messages)


@app.route("/")
def homepage():
    return send_from_directory(".", "index.html")


if __name__ == "__main__":
    app.run(port=5000)
