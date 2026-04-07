from __future__ import annotations

from app.agent_runtime import AgentRuntime
from app.gateway_core import GatewayCore
from app.runtime_settings import load_runtime_settings
from app.session_store import SessionStore
from app.skill_manager import SkillManager


def test_workflow_hint_prefers_report_writer_for_report_requests() -> None:
    settings = load_runtime_settings()
    core = GatewayCore(settings, SessionStore(settings), SkillManager(), AgentRuntime())

    hints = core._build_workflow_hints(
        "请基于以上分析生成报告",
        [
            {
                "tool": "use_skill",
                "summary": "Analysis complete",
                "structured_data": {"score": 0.92, "status": "ok"},
            }
        ],
    )

    assert "report-writer" in hints


def test_build_report_writer_input_uses_structured_history() -> None:
    settings = load_runtime_settings()
    core = GatewayCore(settings, SessionStore(settings), SkillManager(), AgentRuntime())

    payload = core._build_report_writer_input(
        user_input="请根据以上分析生成报告",
        tool_history=[
            {
                "tool": "use_skill",
                "summary": "Anomaly analysis complete",
                "result": "Analysis complete",
                "structured_data": {
                    "status": "warning",
                    "findings": ["Gearbox vibration exceeded threshold"],
                    "recommendations": ["Inspect gearbox bearings within 24 hours"],
                    "report_path": "/tmp/analysis.json",
                },
            },
            {
                "tool": "read_file",
                "summary": "Loaded source file",
                "result": "Loaded maintenance note",
                "structured_data": {
                    "files": [{"path": "/data/maintenance-note.txt"}],
                },
            },
        ],
    )

    assert payload is not None
    assert payload["title"] == "Diagnostic Report"
    assert "Gearbox vibration exceeded threshold" in payload["findings"]
    assert "Inspect gearbox bearings within 24 hours" in payload["recommendations"]
    assert "/tmp/analysis.json" in payload["artifacts"]
    assert "/data/maintenance-note.txt" in payload["artifacts"]
    assert "use_skill" in payload["source_tools"]
    assert "read_file" in payload["source_tools"]
