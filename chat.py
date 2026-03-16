import requests
import json

OLLAMA_URL = "http://localhost:11434/api/chat"

messages = []

def ask(user_msg):
    while True:
        messages.append({
            "role":"user",
            "content": user_msg
        })

        data = {
            "model": "qwen3.5:9b",
            "messages": messages,
            "stream": False,
            "think": False
        }

        r = requests.post(OLLAMA_URL, json=data, stream=True)

        reply = r.json()["message"]["content"]

        messages.append({
            "role": "assistant",
            "content": reply
        })

        return reply

def ask_stream(user_msg):
    messages.append({
        "role": "user",
        "content": user_msg
    })

    r = requests.post(
        OLLAMA_URL,
        json={
            "model": "qwen3.5:9b",
            "messages": messages,
            "stream": True,
            "think": False
        },
        stream=True
    )

    reply = ""

    for line in r.iter_lines():

        if not line:
            continue

        data = json.loads(line)

        if "message" in data:
            content = data["message"].get("content", "")
            print(content, end="", flush=True)   # 实时输出
            reply += content

        if data.get("done"):
            break

    print()

    messages.append({
        "role": "assistant",
        "content": reply
    })

    return reply  