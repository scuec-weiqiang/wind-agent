from __future__ import annotations

import json

from app.agent_runtime import AgentRuntime
from app.gateway_core import GatewayCore
from app.runtime_settings import load_runtime_settings
from app.session_store import SessionStore
from app.skill_manager import SkillManager


def test_runtime_context_prompt_contains_task_state_and_recent_tools() -> None:
    settings = load_runtime_settings()
    store = SessionStore(settings)
    core = GatewayCore(settings, store, SkillManager(), AgentRuntime())
    session_id, state = store.create_session()

    core._initialize_runtime_state(
        session_id=session_id,
        state=state,
        user_input="Analyze uploaded data and generate a report",
        run_id="run-demo",
    )
    core._set_runtime_stage(
        session_id,
        state,
        stage="executing_tool",
        status="running",
        current_step=2,
        max_steps=4,
        current_tool="use_skill",
        tool_count=1,
    )
    core._record_tool_evidence(
        session_id,
        state,
        tool_name="use_skill",
        summary="Analysis complete",
        structured_data={"status": "warning"},
    )

    prompt = core._runtime_context_prompt(
        state,
        [
            {
                "tool": "use_skill",
                "summary": "Analysis complete",
                "structured_data": {"status": "warning"},
            }
        ],
    )
    payload = json.loads(prompt)

    assert payload["task_state"]["run_id"] == "run-demo"
    assert payload["task_state"]["stage"] == "executing_tool"
    assert payload["task_state"]["current_tool"] == "use_skill"
    assert payload["task_state"]["tool_count"] >= 1
    assert payload["recent_tools"][0]["has_structured_data"] is True


def test_initialize_runtime_state_preserves_existing_plan_and_evidence() -> None:
    settings = load_runtime_settings()
    store = SessionStore(settings)
    core = GatewayCore(settings, store, SkillManager(), AgentRuntime())
    session_id, state = store.create_session()

    state.runtime_state = {
        "plan": {
            "goal": "Analyze uploaded data",
            "steps": [{"id": 1, "phase": "analysis", "expected_action": "use_skill"}],
        },
        "completed_steps": [1],
        "evidence": [{"tool": "use_skill", "summary": "Analysis complete", "structured": True}],
    }

    core._initialize_runtime_state(
        session_id=session_id,
        state=state,
        user_input="继续生成报告",
        run_id="run-next",
    )

    assert state.runtime_state["run_id"] == "run-next"
    assert "plan" in state.runtime_state
    assert state.runtime_state["completed_steps"] == [1]
    assert len(state.runtime_state["evidence"]) == 1


def test_update_plan_progress_requires_matching_action_and_skill() -> None:
    settings = load_runtime_settings()
    store = SessionStore(settings)
    core = GatewayCore(settings, store, SkillManager(), AgentRuntime())
    session_id, state = store.create_session()

    state.runtime_state = {
        "plan": {
            "goal": "Generate report",
            "steps": [
                {
                    "id": 1,
                    "phase": "report",
                    "description": "Write report",
                    "expected_action": "use_skill",
                    "skill": "report-writer",
                    "completion_criteria": "Report generated",
                }
            ],
        },
        "completed_steps": [],
    }

    core._update_plan_progress(
        session_id,
        state,
        "use_skill",
        "Used a different skill",
        decision={"action": "use_skill", "skill": "fft-frequency"},
    )
    assert state.runtime_state["completed_steps"] == []

    core._update_plan_progress(
        session_id,
        state,
        "use_skill",
        "Report generated",
        decision={"action": "use_skill", "skill": "report-writer"},
    )
    assert state.runtime_state["completed_steps"] == [1]
