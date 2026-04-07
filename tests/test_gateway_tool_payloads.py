from __future__ import annotations

from app.gateway_core import GatewayCore


def test_extract_list_dir_structured_payload() -> None:
    payload = GatewayCore._structured_tool_payload(
        GatewayCore.__new__(GatewayCore),  # type: ignore[misc]
        "list_dir",
        {"path": "skills"},
        "[list_dir] skills\n\nskills/a/\nskills/b/\n",
        session_id="demo",
        state=None,  # type: ignore[arg-type]
    )

    assert payload == {"path": "skills", "entries": ["skills/a/", "skills/b/"]}


def test_extract_read_file_structured_payload() -> None:
    payload = GatewayCore._structured_tool_payload(
        GatewayCore.__new__(GatewayCore),  # type: ignore[misc]
        "read_file",
        {"path": "README.md"},
        "[read_file] README.md\n\nhello\nworld",
        session_id="demo",
        state=None,  # type: ignore[arg-type]
    )

    assert payload == {"path": "README.md", "content": "hello\nworld"}
