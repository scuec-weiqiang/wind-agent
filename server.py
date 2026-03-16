from flask import Flask, request, Response
import requests, json

app = Flask(__name__)

OLLAMA = "http://localhost:11434/api/chat"

@app.route("/v1/chat/completions", methods=["POST"])
def chat():
    data = request.json
    messages = data["messages"]

    r = requests.post(
        OLLAMA,
        json={
            "model": "qwen3.5:9b",
            "messages": messages,
            "stream": True
        },
        stream=True
    )

    def generate():
        for line in r.iter_lines():
            if line:
                obj = json.loads(line)
                token = obj["message"]["content"]

                yield f"data: {json.dumps({'choices':[{'delta':{'content':token}}]})}\n\n"

    return Response(generate(), mimetype="text/event-stream")

app.run(port=5000)