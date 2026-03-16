import requests
import json

url = "http://localhost:11434/api/chat"

messages = []

while True:
    user_input = input("You:")
    messages.append({
        "role":"user",
        "content": user_input
    })

    data = {
        "model": "qwen3.5:9b",
        "messages": messages,
        "stream": True,
        "think":False
    }

    r = requests.post(url, json=data, stream=True)

    print("AI: ", end="", flush=True)

    reply = ""

    for line in r.iter_lines():
        if line:
            chunk = json.loads(line)
            if "message" in chunk:
                content = chunk["message"].get("content", "")
                print(content, end="", flush=True)
                reply += content

    print()

    messages.append({
        "role": "assistant",
        "content": reply
    })
