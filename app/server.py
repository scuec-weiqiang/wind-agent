from __future__ import annotations

from flask import Flask, Response, request, send_from_directory

from app.agent_runtime import AgentRuntime
from app.config import load_runtime_config
from app.gateway_core import GatewayCore
from app.runtime_settings import load_runtime_settings
from app.session_store import SessionStore
from app.skill_manager import SkillManager, SkillRegistryError

load_runtime_config()
settings = load_runtime_settings()
session_store = SessionStore(settings)
runtime = AgentRuntime()
app = Flask(__name__, static_folder=".", static_url_path="")

try:
    skill_manager: SkillManager | None = SkillManager()
except SkillRegistryError:
    skill_manager = None

gateway = GatewayCore(
    settings=settings,
    session_store=session_store,
    skill_manager=skill_manager,
    runtime=runtime,
)


def _json_payload() -> dict[str, object]:
    payload = request.get_json(silent=True)
    return payload if isinstance(payload, dict) else {}


@app.route("/chat", methods=["POST"])
def chat():
    payload = _json_payload()
    user_input = str(payload.get("message", "")).strip()
    if not user_input:
        return Response("No message provided.", status=400)
    attachment_ids = [
        str(item).strip()
        for item in (payload.get("attachment_ids", []) if isinstance(payload.get("attachment_ids"), list) else [])
        if str(item).strip()
    ]
    session_id = session_store.extract_session_id(payload)
    thinking_mode = str(payload.get("thinking_mode", "")).strip()

    return Response(
        gateway.handle_chat(session_id, user_input, attachment_ids, thinking_mode),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/session/new", methods=["POST"])
def new_session():
    return gateway.create_session()


@app.route("/session/delete", methods=["POST"])
def delete_session():
    return gateway.delete_session(session_store.extract_session_id(_json_payload()))


@app.route("/upload", methods=["POST"])
def upload_files():
    session_id = session_store.extract_session_id(
        {"session_id": request.form.get("session_id", "")},
        fallback=request.args.get("session_id", ""),
    )
    session_store.ensure_session(session_id)

    files = request.files.getlist("files")
    if not files:
        return {"ok": False, "error": "No files uploaded."}, 400

    upload_dir = session_store.session_upload_dir(session_id)
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_files: list[dict[str, object]] = []
    rejected_files: list[str] = []
    for file_item in files:
        original_name = str(getattr(file_item, "filename", "") or "").strip()
        if not original_name:
            continue
        file_item.stream.seek(0, 2)
        size = int(file_item.stream.tell() or 0)
        file_item.stream.seek(0)
        if settings.max_upload_size_bytes is not None and size > settings.max_upload_size_bytes:
            rejected_files.append(original_name)
            continue
        safe_name = session_store.safe_upload_name(original_name)
        if not session_store.is_text_upload(safe_name):
            rejected_files.append(original_name)
            continue
        target = upload_dir / safe_name
        file_item.save(target)
        saved_files.append(
            {
                "file_id": safe_name,
                "name": original_name,
                "saved_path": str(target),
                "size": size,
            }
        )

    if not saved_files:
        return {
            "ok": False,
            "error": "No valid text files uploaded.",
            "rejected_files": rejected_files,
        }, 400

    state = session_store.ensure_session(session_id)
    latest_ids = [
        str(item.get("file_id", "")).strip()
        for item in saved_files
        if str(item.get("file_id", "")).strip()
    ]
    state.attached_file_ids = session_store.limit_attachment_ids(latest_ids)
    session_store.touch_session(session_id, state)

    return {
        "ok": True,
        "session_id": session_id,
        "upload_dir": str(upload_dir),
        "saved_files": saved_files,
        "rejected_files": rejected_files,
        "attachment_ids": state.attached_file_ids,
    }


@app.route("/sessions", methods=["GET"])
def list_sessions():
    return gateway.list_sessions()


@app.route("/skills", methods=["GET"])
def list_skills():
    return gateway.list_skills()


@app.route("/skills/doc", methods=["GET"])
def read_skill_doc():
    skill_name = str(request.args.get("skill", "")).strip()
    if not skill_name:
        return {"ok": False, "error": "No skill provided."}, 400
    try:
        return gateway.read_skill_doc(skill_name)
    except SkillRegistryError as exc:
        return {"ok": False, "error": str(exc)}, 400


@app.route("/tools/list_dir", methods=["GET"])
def list_dir_tool():
    path_value = str(request.args.get("path", "")).strip()
    return gateway.list_dir_tool(path_value)


@app.route("/tools/read_file", methods=["GET"])
def read_file_tool():
    path_value = str(request.args.get("path", "")).strip()
    if not path_value:
        return {"ok": False, "error": "No path provided."}, 400
    return gateway.read_file_tool(path_value)


@app.route("/tools/search_text", methods=["GET"])
def search_text_tool():
    pattern = str(request.args.get("pattern", "")).strip()
    path_value = str(request.args.get("path", "")).strip()
    if not pattern:
        return {"ok": False, "error": "No pattern provided."}, 400
    return gateway.search_text_tool(pattern, path_value)


@app.route("/skills/reload", methods=["POST"])
def reload_skills():
    try:
        result = gateway.reload_skills()
    except SkillRegistryError as exc:
        return {"ok": False, "error": str(exc)}, 400
    return result


@app.route("/skills/execute", methods=["POST"])
def execute_skill():
    payload = _json_payload()
    skill_name = str(payload.get("skill", "")).strip()
    skill_input = str(payload.get("input", "")).strip()
    if not skill_name:
        return {"ok": False, "error": "No skill provided."}, 400
    try:
        return gateway.execute_skill(
            session_store.extract_session_id(payload), skill_name, skill_input
        )
    except SkillRegistryError as exc:
        return {"ok": False, "error": str(exc)}, 400
    except Exception as exc:
        return {"ok": False, "error": str(exc)}, 500


@app.route("/skills/toggle", methods=["POST"])
def toggle_skill():
    payload = _json_payload()
    skill_name = str(payload.get("skill", "")).strip()
    enabled = bool(payload.get("enabled", True))
    if not skill_name:
        return {"ok": False, "error": "No skill provided."}, 400
    try:
        return gateway.toggle_skill(skill_name, enabled)
    except SkillRegistryError as exc:
        return {"ok": False, "error": str(exc)}, 400


@app.route("/session/history", methods=["GET"])
def session_history():
    session_id = session_store.extract_session_id(None, fallback=request.args.get("session_id", ""))
    return gateway.session_history(session_id)


@app.route("/session/runtime", methods=["GET"])
def session_runtime():
    session_id = session_store.extract_session_id(None, fallback=request.args.get("session_id", ""))
    return gateway.session_runtime(session_id)


@app.route("/agent/wait", methods=["GET"])
def agent_wait():
    run_id = str(request.args.get("run_id", "")).strip()
    if not run_id:
        return {"ok": False, "error": "Missing run_id."}, 400
    timeout_ms_raw = str(request.args.get("timeout_ms", "30000")).strip()
    try:
        timeout_ms = max(1, int(timeout_ms_raw))
    except ValueError:
        return {"ok": False, "error": "Invalid timeout_ms."}, 400
    return gateway.wait_for_run(run_id, timeout_ms=timeout_ms)


@app.route("/session/trace", methods=["GET"])
def session_trace():
    session_id = session_store.extract_session_id(None, fallback=request.args.get("session_id", ""))
    return gateway.session_trace(session_id)


@app.route("/session/attachments", methods=["GET"])
def session_attachments():
    session_id = session_store.extract_session_id(None, fallback=request.args.get("session_id", ""))
    return gateway.session_attachments(session_id)


@app.route("/session/file", methods=["GET"])
def session_file():
    session_id = session_store.extract_session_id(None, fallback=request.args.get("session_id", ""))
    file_id = str(request.args.get("file_id", "")).strip()
    download = str(request.args.get("download", "")).strip().lower() in {"1", "true", "yes"}
    if not file_id:
        return {"ok": False, "error": "Missing file_id."}, 400
    path = session_store.safe_uploaded_path(session_id, file_id)
    if path is None or not path.exists() or not path.is_file():
        return {"ok": False, "error": "File not found."}, 404
    response = send_from_directory(str(path.parent), path.name, as_attachment=download, max_age=0)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/")
def homepage():
    response = send_from_directory(str(settings.project_root / "web"), "index.html", max_age=0)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


if __name__ == "__main__":
    app.run(port=5000)
