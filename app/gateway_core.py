from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Iterator

import requests

from app.agent_runtime import AgentRuntime
from app.chat import DEFAULT_PROVIDER, _extract_reasoning_text
from app.planner import TaskPlanner
from app.runtime_settings import RuntimeSettings
from app.session_store import ConversationState, SessionStore
from app.skill_manager import (
    SkillAmbiguousError,
    SkillDisabled,
    SkillExecutionResult,
    SkillManager,
    SkillNotFoundError,
    SkillRegistryError,
    SkillUnavailable,
)

SKILL_ROUTER_PROMPT_TEMPLATE = (
    "You are a skill router. Decide whether to call skill packs.\n"
    "Below is the current skill catalog.\n"
    "Treat it as an available_skills index, not as full skill instructions.\n"
    "Below is the available tool catalog.\n"
    "Treat it as the low-level action reference.\n"
    "Use the skill_id exactly as listed if you choose one.\n"
    "Return JSON only, one of:\n"
    '{"action":"none"}\n'
    '{"action":"read_skill","skill":"<skill_id>"}\n'
    '{"action":"list_dir","path":"<relative directory path>"}\n'
    '{"action":"read_file","path":"<relative file path>"}\n'
    '{"action":"search_text","pattern":"<text or regex>","path":"<relative path optional>"}\n'
    '{"action":"list_uploaded_files"}\n'
    '{"action":"read_uploaded_file","file_id":"<uploaded file id>"}\n'
    '{"action":"exec_command","command":"<shell command>"}\n'
    '{"action":"tool_calls","calls":[{"tool":"read_file","path":"README.md"},{"tool":"exec_command","command":"pwd"}]}\n'
    '{"action":"use_skill","skill":"<skill_id>","input":"<input for skill>"}\n'
    '{"action":"use_skill","skill":"<skill_id>","input":{"key":"value"}}\n'
    '{"action":"use_skills","calls":[{"skill":"<skill_id>","input":"..."},{"skill":"<skill_id>","input":"..."}]}\n'
    '{"action":"use_skills","calls":[{"skill":"<skill_id>","input":{"key":"value"}}]}\n'
    "Rules:\n"
    "- no markdown, no prose, JSON only.\n"
    "- if no skill fits, return action=none.\n"
    "- if one packaged_runtime skill clearly matches the user goal, prefer action=use_skill directly.\n"
    "- choose read_skill when the user is asking about what a skill does or how to use it.\n"
    "- if a skill is marked invocation=manual_only, do not return use_skill for it; return read_skill first.\n"
    "- do not use list_dir/read_file/search_text for routine tasks when a suitable packaged_runtime skill already exists.\n"
    "- use list_dir/read_file/search_text only when required to disambiguate, inspect unknown files, or follow manual_only skills.\n"
    "- use exec_command when a shell command is the most direct way to answer.\n"
    "- if user asks to use uploaded files, prefer list_uploaded_files/read_uploaded_file before other actions.\n"
    "- use tool_calls when the task needs multiple low-level steps.\n"
    "- input may be a string or a JSON object.\n"
    "- prefer using description and when_to_use to decide if a skill fits.\n"
    "- if user asks to analyze vibration/frequency data and provides a directory path, prefer fft-frequency via use_skill.\n"
    "- if user asks to analyze power curve / 风速功率 / 发电健康 / 功率曲线 and provides csv data, prefer power-curve-assessment via use_skill.\n"
    "- for shell skill input, output executable command text only.\n"
    "- at most 3 calls for use_skills.\n\n"
    "Skill catalog:\n"
    "<<SKILL_CATALOG>>\n"
    "\nTool catalog:\n"
    "<<TOOL_CATALOG>>\n"
)

AGENT_ORCHESTRATOR_PROMPT_TEMPLATE = (
    "You are the runtime brain for a Linux-deployed local assistant.\n"
    "Your job is to complete the user's task by deciding the next best action.\n"
    "You may inspect files, read uploaded data, search text, execute shell commands, or use packaged skills.\n"
    "Prefer evidence and concrete tool use over guessing.\n"
    "When you have enough evidence, return a final answer.\n"
    "Return JSON only. No markdown fences.\n"
    "Schema:\n"
    '{"action":"final","message":"<final answer to user>"}\n'
    '{"action":"read_skill","skill":"<skill_id>"}\n'
    '{"action":"list_dir","path":"<relative directory path>"}\n'
    '{"action":"read_file","path":"<relative file path>"}\n'
    '{"action":"search_text","pattern":"<pattern>","path":"<relative path optional>"}\n'
    '{"action":"list_uploaded_files"}\n'
    '{"action":"read_uploaded_file","file_id":"<uploaded file id>"}\n'
    '{"action":"exec_command","command":"<shell command>"}\n'
    '{"action":"use_skill","skill":"<skill_id>","input":"<string or object>"}\n'
    '{"action":"tool_calls","calls":[{"tool":"read_file","path":"README.md"},{"tool":"exec_command","command":"pwd"}]}\n'
    "Rules:\n"
    "- Return one action at a time unless you use tool_calls.\n"
    "- Use skills when they clearly fit the task.\n"
    "- Use read_skill first if the right skill is unclear or manual_only.\n"
    "- For analysis, automation, or reports, gather evidence before final.\n"
    "- Prefer structured_result skills when you need machine-readable data for downstream steps.\n"
    "- Use workflow_hint when present to decide whether a skill fits a multi-step task.\n"
    "- Use exploration tools first when you still need paths, file contents, or search results.\n"
    "- Use packaged skills first when a domain skill clearly matches the task better than generic tools.\n"
    "- If the user asks for a report and you already have structured_data from prior steps, prefer a report-oriented skill over writing the report manually.\n"
    "- Final answers should be concise but specific, and should mention evidence when tools were used.\n"
    "- When a task plan is available (in task_state.plan_summary), consider following the planned steps while adapting to new evidence.\n"
    "- The plan provides suggested phases and actions; you may deviate if the evidence suggests a better approach.\n"
)


class GatewayCore:
    def __init__(
        self,
        settings: RuntimeSettings,
        session_store: SessionStore,
        skill_manager: SkillManager | None,
        runtime: AgentRuntime,
    ) -> None:
        self.settings = settings
        self.session_store = session_store
        self.skill_manager = skill_manager
        self.runtime = runtime
        self.planner = TaskPlanner(settings)

    def trace_event(
        self, session_id: str, state: ConversationState, event: str, **payload: object
    ) -> None:
        entry = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "event": event,
            "data": {key: self._clip_trace_value(value) for key, value in payload.items()},
        }
        state.debug_trace.append(entry)
        if len(state.debug_trace) > 200:
            state.debug_trace = state.debug_trace[-200:]
        if self.settings.agent_trace:
            print(
                f"[agent-trace][session={session_id}][{event}] "
                f"{json.dumps(entry['data'], ensure_ascii=False)}",
                flush=True,
            )

    def _initialize_runtime_state(
        self,
        *,
        session_id: str,
        state: ConversationState,
        user_input: str,
        run_id: str,
    ) -> None:
        previous_state = state.runtime_state if isinstance(state.runtime_state, dict) else {}
        state.runtime_state = {
            "run_id": run_id,
            "status": "accepted",
            "stage": "accepted",
            "goal": str(user_input or "").strip()[:800],
            "current_step": 0,
            "max_steps": 0,
            "current_tool": "",
            "tool_count": 0,
            "evidence": list(previous_state.get("evidence", []))[:8]
            if isinstance(previous_state.get("evidence"), list)
            else [],
            "artifacts": list(previous_state.get("artifacts", []))[:8]
            if isinstance(previous_state.get("artifacts"), list)
            else [],
            "attachments": list(state.attached_file_ids[:8]),
            "final_answer": "",
            "error": "",
        }
        plan = previous_state.get("plan")
        if isinstance(plan, dict):
            state.runtime_state["plan"] = plan
        completed_steps = previous_state.get("completed_steps")
        if isinstance(completed_steps, list):
            state.runtime_state["completed_steps"] = [
                int(item) for item in completed_steps if isinstance(item, int)
            ]
        self.session_store.touch_session(session_id, state)

    def _set_runtime_stage(
        self,
        session_id: str,
        state: ConversationState,
        *,
        stage: str,
        status: str,
        current_step: int | None = None,
        max_steps: int | None = None,
        current_tool: str | None = None,
        tool_count: int | None = None,
        final_answer: str | None = None,
        error: str | None = None,
    ) -> None:
        runtime_state = state.runtime_state or {}
        runtime_state["stage"] = stage
        runtime_state["status"] = status
        if current_step is not None:
            runtime_state["current_step"] = current_step
        if max_steps is not None:
            runtime_state["max_steps"] = max_steps
        if current_tool is not None:
            runtime_state["current_tool"] = current_tool
        if tool_count is not None:
            runtime_state["tool_count"] = tool_count
        if final_answer is not None:
            runtime_state["final_answer"] = str(final_answer)[:1200]
        if error is not None:
            runtime_state["error"] = str(error)[:1200]
        runtime_state["attachments"] = list(state.attached_file_ids[:8])
        state.runtime_state = runtime_state
        self.session_store.touch_session(session_id, state)

    def _record_tool_evidence(
        self,
        session_id: str,
        state: ConversationState,
        *,
        tool_name: str,
        summary: str,
        structured_data: object,
        decision: dict[str, object] | None = None,
    ) -> None:
        runtime_state = state.runtime_state or {}
        evidence = runtime_state.get("evidence")
        evidence_items = evidence if isinstance(evidence, list) else []
        skill_name = ""
        if isinstance(decision, dict):
            skill_name = str(decision.get("skill", "")).strip()
        evidence_items.append(
            {
                "tool": str(tool_name or "").strip()[:120],
                "skill": skill_name[:120],
                "summary": str(summary or "").strip()[:400],
                "structured": structured_data is not None,
            }
        )
        runtime_state["evidence"] = evidence_items[-8:]
        artifacts = runtime_state.get("artifacts")
        artifact_items = artifacts if isinstance(artifacts, list) else []
        artifact_items.extend(self._extract_artifacts_from_structured_data(structured_data))
        runtime_state["artifacts"] = self._dedupe_text_items(
            [str(item).strip() for item in artifact_items if str(item).strip()]
        )[:8]
        runtime_state["tool_count"] = int(runtime_state.get("tool_count", 0) or 0) + 1
        runtime_state["current_tool"] = str(tool_name or "").strip()[:120]
        state.runtime_state = runtime_state
        self.session_store.touch_session(session_id, state)

        # Update plan progress if a plan exists
        self._update_plan_progress(session_id, state, tool_name, summary, decision=decision)

    def _update_plan_progress(
        self,
        session_id: str,
        state: ConversationState,
        tool_name: str,
        summary: str,
        *,
        decision: dict[str, object] | None = None,
    ) -> None:
        """Update plan progress based on completed tool execution."""
        runtime_state = state.runtime_state or {}
        plan = runtime_state.get("plan")
        if not isinstance(plan, dict):
            return

        completed_steps = runtime_state.get("completed_steps")
        if not isinstance(completed_steps, list):
            completed_steps = []
            runtime_state["completed_steps"] = completed_steps

        step = self._current_plan_step(state)
        if not isinstance(step, dict):
            return

        if not self._decision_matches_plan_step(tool_name, step, summary, decision=decision):
            return

        step_id = step.get("id")
        if not isinstance(step_id, int) or step_id in completed_steps:
            return

        completed_steps.append(step_id)
        runtime_state["completed_steps"] = completed_steps
        state.runtime_state = runtime_state
        self.session_store.touch_session(session_id, state)

        self.trace_event(
            session_id,
            state,
            "plan_step_completed",
            step_id=step_id,
            step_description=step.get("description", "")[:80],
            tool_name=tool_name,
        )

    def _current_plan_step(self, state: ConversationState) -> dict[str, object] | None:
        runtime_state = state.runtime_state or {}
        plan = runtime_state.get("plan")
        if not isinstance(plan, dict):
            return None
        completed_steps = runtime_state.get("completed_steps")
        executed_steps = completed_steps if isinstance(completed_steps, list) else []
        step = self.planner.get_current_step(plan, executed_steps)
        return step if isinstance(step, dict) else None

    @staticmethod
    def _decision_matches_plan_step(
        tool_name: str,
        step: dict[str, object],
        summary: str,
        *,
        decision: dict[str, object] | None = None,
    ) -> bool:
        expected_action = str(step.get("expected_action", "")).strip().lower()
        planned_skill = str(step.get("skill", "")).strip().lower()
        actual_action = str(tool_name or "").strip().lower()
        if not expected_action:
            return False
        if actual_action != expected_action:
            return False
        if expected_action == "use_skill" and planned_skill:
            actual_skill = ""
            if isinstance(decision, dict):
                actual_skill = str(decision.get("skill", "")).strip().lower()
            if actual_skill and actual_skill != planned_skill:
                return False
        return bool(str(summary or "").strip()) or expected_action in {
            "list_dir",
            "read_file",
            "search_text",
            "list_uploaded_files",
            "read_uploaded_file",
        }

    def _runtime_state_view(self, state: ConversationState) -> dict[str, object]:
        runtime_state = state.runtime_state or {}
        evidence = runtime_state.get("evidence")
        if not isinstance(evidence, list):
            evidence = []

        # Extract plan summary if present
        plan = runtime_state.get("plan")
        plan_summary = None
        if isinstance(plan, dict):
            steps = plan.get("steps", [])
            if isinstance(steps, list):
                plan_summary = {
                    "goal": plan.get("goal", "")[:100],
                    "total_steps": len(steps),
                    "completed_steps": len(runtime_state.get("completed_steps", [])),
                    "current_step_index": min(len(steps) - 1, len(runtime_state.get("completed_steps", []))),
                }
        current_plan_step = None
        if isinstance(plan, dict):
            completed_steps = runtime_state.get("completed_steps")
            executed_steps = completed_steps if isinstance(completed_steps, list) else []
            step = self.planner.get_current_step(plan, executed_steps)
            if isinstance(step, dict):
                current_plan_step = {
                    "id": step.get("id"),
                    "phase": str(step.get("phase", "")).strip(),
                    "description": str(step.get("description", "")).strip()[:180],
                    "expected_action": str(step.get("expected_action", "")).strip(),
                    "skill": str(step.get("skill", "")).strip(),
                    "completion_criteria": str(step.get("completion_criteria", "")).strip()[:180],
                }

        result = {
            "run_id": str(runtime_state.get("run_id", "")).strip(),
            "status": str(runtime_state.get("status", "")).strip(),
            "stage": str(runtime_state.get("stage", "")).strip(),
            "goal": str(runtime_state.get("goal", "")).strip(),
            "current_step": int(runtime_state.get("current_step", 0) or 0),
            "max_steps": int(runtime_state.get("max_steps", 0) or 0),
            "current_tool": str(runtime_state.get("current_tool", "")).strip(),
            "tool_count": int(runtime_state.get("tool_count", 0) or 0),
            "attachments": list((runtime_state.get("attachments") or [])[:8]),
            "evidence": evidence[-5:],
            "artifacts": list((runtime_state.get("artifacts") or [])[:8]),
            "final_answer": str(runtime_state.get("final_answer", "")).strip()[:800],
            "error": str(runtime_state.get("error", "")).strip()[:400],
        }

        if plan_summary:
            result["plan_summary"] = plan_summary
        if current_plan_step:
            result["current_plan_step"] = current_plan_step

        return result

    def _runtime_context_prompt(
        self, state: ConversationState, tool_history: list[dict[str, object]]
    ) -> str:
        runtime_state = self._runtime_state_view(state)
        context = {
            "task_state": runtime_state,
            "recent_tools": [
                {
                    "tool": str(item.get("tool", "")).strip(),
                    "summary": str(item.get("summary", "")).strip(),
                    "has_structured_data": item.get("structured_data") is not None,
                }
                for item in tool_history[-5:]
            ],
        }
        return json.dumps(context, ensure_ascii=False, indent=2)

    def create_session(self) -> dict[str, object]:
        session_id, state = self.session_store.create_session()
        return {
            "session_id": session_id,
            "title": self.session_store.derive_session_title(state),
            "updated_at": state.updated_at,
        }

    def delete_session(self, session_id: str) -> dict[str, object]:
        deleted = self.session_store.delete_session(session_id)
        return {"session_id": session_id, "deleted": deleted}

    def list_sessions(self) -> dict[str, object]:
        return {"sessions": self.session_store.list_sessions()}

    def session_history(self, session_id: str) -> dict[str, object]:
        return self.session_store.history_payload(session_id)

    def session_trace(self, session_id: str) -> dict[str, object]:
        return self.session_store.trace_payload(session_id)

    def session_attachments(self, session_id: str) -> dict[str, object]:
        return self.session_store.attachments_payload(session_id)

    def session_runtime(self, session_id: str) -> dict[str, object]:
        return self.session_store.runtime_payload(session_id)

    def list_skills(self) -> dict[str, object]:
        if not self.skill_manager:
            return {"skills": [], "available": False}
        return {"available": True, "skills": self.skill_manager.as_dicts()}

    def read_skill_doc(self, skill_name: str) -> dict[str, object]:
        if not self.skill_manager:
            raise SkillRegistryError("Skills are unavailable.")
        return {
            "ok": True,
            "skill": skill_name,
            "content": self.skill_manager.read_skill_doc(skill_name),
        }

    def reload_skills(self) -> dict[str, object]:
        if self.skill_manager is None:
            self.skill_manager = SkillManager()
        else:
            self.skill_manager.reload()
        return {"ok": True, "skills": self.skill_manager.as_dicts()}

    def toggle_skill(self, skill_name: str, enabled: bool) -> dict[str, object]:
        if not self.skill_manager:
            raise SkillRegistryError("Skills are unavailable.")
        skill = self.skill_manager.set_enabled(skill_name, enabled)
        return {"ok": True, "skill": skill.to_dict()}

    def execute_skill(
        self, session_id: str, skill_name: str, skill_input: str
    ) -> dict[str, object]:
        if not self.skill_manager:
            raise SkillRegistryError("Skills are unavailable.")
        skill = self.skill_manager.get(skill_name)
        if skill.user_invocable is False:
            raise SkillRegistryError(f"Skill '{skill_name}' is not user-invocable.")
        state = self.session_store.ensure_session(session_id)
        result = self.skill_manager.execute_result(
            skill_name, skill_input, session=state.assistant
        )
        return {
            "ok": result.ok,
            "result": result.output_text,
            "summary": result.summary,
            "structured_data": result.structured_data,
            "skill": skill_name,
        }

    def list_dir_tool(self, path_value: str) -> dict[str, object]:
        path_value = path_value.strip() or "."
        return {"ok": True, "path": path_value, "content": self._list_project_dir(path_value)}

    def read_file_tool(self, path_value: str) -> dict[str, object]:
        return {"ok": True, "path": path_value, "content": self._read_project_file(path_value)}

    def search_text_tool(self, pattern: str, path_value: str) -> dict[str, object]:
        return {
            "ok": True,
            "pattern": pattern,
            "path": path_value or ".",
            "content": self._search_project_text(pattern, path_value),
        }

    def handle_chat(
        self,
        session_id: str,
        user_input: str,
        attachment_ids: list[str],
        thinking_mode: str,
    ) -> Iterator[str]:
        run = self.runtime.create_run(session_id)
        state = self.session_store.ensure_session(session_id)
        self.session_store.set_thinking_mode(state, thinking_mode)
        self._initialize_runtime_state(
            session_id=session_id,
            state=state,
            user_input=user_input,
            run_id=run.run_id,
        )
        yield self._sse_event(
            "lifecycle",
            {
                "phase": "accepted",
                "runId": run.run_id,
                "sessionId": session_id,
                "acceptedAt": run.accepted_at,
                "lane": self.runtime.lane_snapshot(session_id),
                "taskState": self._runtime_state_view(state),
            },
        )
        try:
            with self.runtime.session_lane(session_id, run.run_id):
                self._set_runtime_stage(
                    session_id,
                    state,
                    stage="running",
                    status="running",
                )
                yield self._sse_event(
                    "lifecycle",
                    {
                        "phase": "start",
                        "runId": run.run_id,
                        "sessionId": session_id,
                        "lane": self.runtime.lane_snapshot(session_id),
                        "taskState": self._runtime_state_view(state),
                    },
                )
                if attachment_ids:
                    self.session_store.bind_attachments(session_id, state, attachment_ids)
                    self.trace_event(
                        session_id,
                        state,
                        "attachments_bound",
                        attachment_count=len(state.attached_file_ids),
                        files=state.attached_file_ids,
                    )

                model_input = user_input
                system_message = self._try_system_response(
                    session_id,
                    state,
                    model_input,
                    display_user_input=user_input,
                    run_id=run.run_id,
                )
                if system_message is not None:
                    yield self._sse_event("answer", {"text": system_message, "runId": run.run_id})
                    self._set_runtime_stage(
                        session_id,
                        state,
                        stage="completed",
                        status="ok",
                        final_answer=system_message,
                    )
                    self.runtime.finish_run(run.run_id, status="ok")
                    yield self._sse_event(
                        "lifecycle",
                        {
                            "phase": "end",
                            "runId": run.run_id,
                            "sessionId": session_id,
                            "taskState": self._runtime_state_view(state),
                        },
                    )
                    yield self._sse_event("done", {"runId": run.run_id})
                    return

                yield from self._run_agent_loop(
                    session_id,
                    state,
                    user_input,
                    run_id=run.run_id,
                )
                self._set_runtime_stage(
                    session_id,
                    state,
                    stage="completed",
                    status="ok",
                )
                self.runtime.finish_run(run.run_id, status="ok")
        except Exception as exc:
            self._set_runtime_stage(
                session_id,
                state,
                stage="failed",
                status="error",
                error=str(exc),
            )
            self.runtime.finish_run(run.run_id, status="error", error=str(exc))
            yield self._sse_event("error", {"text": str(exc), "runId": run.run_id})
            yield self._sse_event(
                "lifecycle",
                {
                    "phase": "error",
                    "runId": run.run_id,
                    "sessionId": session_id,
                    "error": str(exc),
                    "taskState": self._runtime_state_view(state),
                },
            )
            yield self._sse_event("done", {"runId": run.run_id})
            return
        yield self._sse_event(
            "lifecycle",
            {
                "phase": "end",
                "runId": run.run_id,
                "sessionId": session_id,
                "taskState": self._runtime_state_view(state),
            },
        )

    def _stream_chat(
        self,
        session_id: str,
        state: ConversationState,
        user_input: str,
        model_input: str | None = None,
        run_id: str | None = None,
    ) -> Iterator[str]:
        state.assistant.add_user_message(user_input)
        effective_input = model_input if model_input is not None else user_input
        if state.assistant.messages:
            state.assistant.messages[-1]["content"] = effective_input
        self.session_store.touch_session(session_id, state)

        reply = ""
        thinking = ""
        try:
            response = requests.post(
                state.assistant.base_url,
                json=state.assistant._build_payload(stream=True),
                headers=state.assistant._build_headers(),
                stream=True,
                timeout=state.assistant.timeout,
            )
            response.raise_for_status()
            provider = (state.assistant.provider or DEFAULT_PROVIDER).strip().lower()

            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if provider == "ollama":
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "message" in payload:
                        message = payload.get("message", {}) or {}
                        if isinstance(message, dict):
                            think_chunk = (
                                message.get("thinking")
                                or message.get("reasoning_content")
                                or payload.get("thinking")
                                or payload.get("reasoning_content")
                                or ""
                            )
                            if think_chunk:
                                thinking += think_chunk
                                yield self._sse_event(
                                    "thinking", {"text": think_chunk, "runId": run_id}
                                )
                            chunk = message.get("content", "") or ""
                            if chunk:
                                reply += chunk
                                yield self._sse_event(
                                    "assistant", {"text": chunk, "runId": run_id}
                                )
                                yield self._sse_event("answer", {"text": chunk, "runId": run_id})
                    if payload.get("done"):
                        break
                    continue

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
                think_chunk = _extract_reasoning_text(payload, provider)
                if think_chunk:
                    thinking += think_chunk
                    yield self._sse_event("thinking", {"text": think_chunk, "runId": run_id})
                delta = choices[0].get("delta") or {}
                chunk = delta.get("content", "") or ""
                if chunk:
                    reply += chunk
                    yield self._sse_event("assistant", {"text": chunk, "runId": run_id})
                    yield self._sse_event("answer", {"text": chunk, "runId": run_id})
        except requests.RequestException as exc:
            reply = f"[network error] {exc}"
            yield self._sse_event("error", {"text": reply, "runId": run_id})
        finally:
            if state.assistant.messages and state.assistant.messages[-1].get("role") == "user":
                state.assistant.messages[-1]["content"] = user_input
            if thinking:
                state.assistant.messages.append(
                    {"role": "assistant", "content": reply, "thinking": thinking}
                )
            else:
                state.assistant.add_assistant_message(reply)
            self.session_store.touch_session(session_id, state)
            yield self._sse_event("done", {"runId": run_id})

    def _try_system_response(
        self,
        session_id: str,
        state: ConversationState,
        user_input: str,
        display_user_input: str | None = None,
        run_id: str | None = None,
    ) -> str | None:
        shown_input = display_user_input if display_user_input is not None else user_input
        self.trace_event(session_id, state, "user_request", input=shown_input)

        explicit_command = self._extract_explicit_command(user_input)
        if explicit_command:
            result = self._run_system_command(explicit_command)
            message = self._format_shell_result(result)
            self.trace_event(
                session_id,
                state,
                "explicit_command",
                command=explicit_command,
                result=message[:1200],
            )
            state.assistant.add_user_message(shown_input)
            state.assistant.add_assistant_message(message)
            self.session_store.touch_session(session_id, state)
            return message

        if self.settings.enable_skill_autorun:
            auto_result = self._maybe_autorun_special_skill(
                session_id, state, shown_input, run_id=run_id
            )
            if auto_result is not None:
                state.assistant.add_user_message(shown_input)
                state.assistant.add_assistant_message(auto_result)
                self.session_store.touch_session(session_id, state)
                return auto_result

        return None

    def _ensure_plan_exists(
        self,
        *,
        session_id: str,
        state: ConversationState,
        user_input: str,
        tool_history: list[dict[str, object]],
    ) -> dict[str, object]:
        """Ensure a task plan exists in runtime_state, generating one if needed."""
        runtime_state = state.runtime_state or {}

        # Check if plan already exists
        if "plan" in runtime_state and isinstance(runtime_state["plan"], dict):
            plan = runtime_state["plan"]
            # Validate basic structure
            if plan.get("steps") and isinstance(plan["steps"], list) and self._should_reuse_existing_plan(
                user_input=user_input,
                runtime_state=runtime_state,
            ):
                # Ensure completed_steps exists
                if "completed_steps" not in runtime_state or not isinstance(runtime_state["completed_steps"], list):
                    runtime_state["completed_steps"] = []
                    state.runtime_state = runtime_state
                    self.session_store.touch_session(session_id, state)
                return plan

        # Generate new plan
        conversation_context = self._conversation_context_for_agent(state)
        available_skills = self._get_structured_skill_list()
        available_tools = self._get_structured_tool_list()

        try:
            plan = self.planner.generate_plan(
                user_input=user_input,
                available_skills=available_skills,
                available_tools=available_tools,
                conversation_context=conversation_context,
            )
        except Exception as e:
            # Fallback plan
            plan = {
                "goal": user_input[:200],
                "steps": [
                    {
                        "id": 1,
                        "phase": "exploration",
                        "description": "Explore available files and data",
                        "expected_action": "list_dir",
                        "completion_criteria": "Understand the directory structure"
                    },
                    {
                        "id": 2,
                        "phase": "analysis",
                        "description": "Analyze relevant data",
                        "expected_action": "use_skill",
                        "completion_criteria": "Extract insights from the data"
                    },
                    {
                        "id": 3,
                        "phase": "report",
                        "description": "Generate final answer or report",
                        "expected_action": "final",
                        "completion_criteria": "Provide comprehensive answer to user"
                    }
                ],
                "max_steps": 3,
                "constraints": ["Fallback plan due to error"],
                "completed_steps": [],
            }

        # Store plan in runtime_state
        runtime_state["plan"] = plan
        runtime_state["completed_steps"] = []
        runtime_state["goal"] = str(plan.get("goal", user_input[:200])).strip()[:800]
        state.runtime_state = runtime_state
        self.session_store.touch_session(session_id, state)

        # Emit plan event
        self.trace_event(
            session_id,
            state,
            "plan_generated",
            plan_summary=f"{len(plan.get('steps', []))} steps",
            goal=plan.get("goal", "")[:100],
        )

        return plan

    @staticmethod
    def _should_reuse_existing_plan(
        *,
        user_input: str,
        runtime_state: dict[str, object],
    ) -> bool:
        normalized_input = str(user_input or "").strip().casefold()
        if not normalized_input:
            return True
        continuation_markers = (
            "continue",
            "go on",
            "继续",
            "接着",
            "然后",
            "基于上面",
            "根据上面",
            "基于以上",
            "根据以上",
        )
        if any(marker in normalized_input for marker in continuation_markers):
            return True
        goal = str(runtime_state.get("goal", "")).strip().casefold()
        if not goal:
            return False
        if normalized_input == goal:
            return True
        goal_tokens = {token for token in re.split(r"\W+", goal) if token}
        input_tokens = {token for token in re.split(r"\W+", normalized_input) if token}
        if not goal_tokens or not input_tokens:
            return False
        overlap_ratio = len(goal_tokens & input_tokens) / max(1, len(input_tokens))
        return overlap_ratio >= 0.5

    def _run_agent_loop(
        self,
        session_id: str,
        state: ConversationState,
        user_input: str,
        run_id: str | None = None,
    ) -> Iterator[str]:
        max_steps = max(1, int(os.environ.get("AGENT_MAX_STEPS", "4")))
        tool_history: list[dict[str, object]] = []
        parse_failures = 0

        # Ensure a task plan exists
        plan = self._ensure_plan_exists(
            session_id=session_id,
            state=state,
            user_input=user_input,
            tool_history=tool_history,
        )
        completed_steps = state.runtime_state.get("completed_steps", []) if state.runtime_state else []

        for step in range(max_steps):
            current_plan_step = self._current_plan_step(state)
            self._set_runtime_stage(
                session_id,
                state,
                stage=str((current_plan_step or {}).get("phase", "planning")) or "planning",
                status="running",
                current_step=step + 1,
                max_steps=max_steps,
                tool_count=len(tool_history),
            )
            decision = self._request_agent_decision(
                session_id=session_id,
                state=state,
                user_input=user_input,
                tool_history=tool_history,
                step=step,
                current_plan_step=current_plan_step,
            )
            if not decision:
                parse_failures += 1
                if parse_failures >= 2:
                    yield from self._stream_chat(
                        session_id,
                        state,
                        user_input,
                        model_input=user_input,
                        run_id=run_id,
                    )
                    return
                continue

            decision = self._align_decision_to_plan_step(
                decision=decision,
                current_plan_step=current_plan_step,
                user_input=user_input,
            )

            action = str(decision.get("action", "")).strip().lower()
            self.trace_event(
                session_id,
                state,
                "agent_loop_decision",
                runId=run_id,
                step=step + 1,
                decision=decision,
            )

            if action == "final":
                message = str(decision.get("message", "")).strip()
                if not message:
                    break
                self._set_runtime_stage(
                    session_id,
                    state,
                    stage="finalizing",
                    status="running",
                    final_answer=message,
                    tool_count=len(tool_history),
                )
                state.assistant.add_user_message(user_input)
                state.assistant.add_assistant_message(message)
                self.session_store.touch_session(session_id, state)
                yield from self._emit_answer_chunks(message, run_id=run_id)
                return

            yield from self._execute_tool_decision_stream(
                session_id=session_id,
                state=state,
                decision=decision,
                tool_history=tool_history,
                user_input=user_input,
                run_id=run_id,
            )

        fallback = self._generate_final_from_tool_history(
            session_id=session_id,
            state=state,
            user_input=user_input,
            tool_history=tool_history,
        )
        self._set_runtime_stage(
            session_id,
            state,
            stage="finalizing",
            status="running",
            final_answer=fallback,
            tool_count=len(tool_history),
        )
        state.assistant.add_user_message(user_input)
        state.assistant.add_assistant_message(fallback)
        self.session_store.touch_session(session_id, state)
        yield from self._emit_answer_chunks(fallback, run_id=run_id)

    def _request_agent_decision(
        self,
        *,
        session_id: str,
        state: ConversationState,
        user_input: str,
        tool_history: list[dict[str, object]],
        step: int,
        current_plan_step: dict[str, object] | None = None,
    ) -> dict[str, object] | None:
        messages = [
            {"role": "system", "content": self._build_agent_orchestrator_prompt()},
            {
                "role": "system",
                "content": (
                    f"Session id: {session_id}\n"
                    f"Step: {step + 1}\n"
                    f"Project root: {self.settings.project_root}\n"
                    "Recent conversation:\n"
                    f"{self._conversation_context_for_agent(state)}"
                ),
            },
            {
                "role": "system",
                "content": (
                    "Runtime task context:\n"
                    f"{self._runtime_context_prompt(state, tool_history)}"
                ),
            },
        ]

        if current_plan_step:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Current plan step:\n"
                        + json.dumps(current_plan_step, ensure_ascii=False, indent=2)
                        + "\nPrefer satisfying this step before moving on unless evidence clearly suggests otherwise."
                    ),
                }
            )

        attachment_manifest = self.session_store.attachment_manifest(
            session_id, state.attached_file_ids
        )
        if attachment_manifest:
            messages.append({"role": "system", "content": attachment_manifest})

        if tool_history:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Tool results so far:\n"
                        f"{self._render_tool_history(tool_history)}"
                    ),
                }
            )
            workflow_hints = self._build_workflow_hints(user_input, tool_history)
            if workflow_hints:
                messages.append(
                    {
                        "role": "system",
                        "content": "Workflow hints:\n" + workflow_hints,
                    }
                )

        messages.append(
            {
                "role": "user",
                "content": (
                    f"User task:\n{user_input}\n\n"
                    "Decide the next best action. If enough evidence is available, return final.\n"
                    "When tool_history contains structured_data, prefer using that structured data over re-parsing text."
                ),
            }
        )

        return self._request_router_decision(
            messages,
            session_id=session_id,
            state=state,
            phase=f"agent_loop_step_{step + 1}",
        )

    def _align_decision_to_plan_step(
        self,
        *,
        decision: dict[str, object],
        current_plan_step: dict[str, object] | None,
        user_input: str,
    ) -> dict[str, object]:
        if not current_plan_step:
            return decision
        expected_action = str(current_plan_step.get("expected_action", "")).strip().lower()
        if not expected_action:
            return decision

        action = str(decision.get("action", "")).strip().lower()
        if action == expected_action:
            if action != "use_skill":
                return decision
            planned_skill = str(current_plan_step.get("skill", "")).strip()
            if not planned_skill:
                return decision
            actual_skill = str(decision.get("skill", "")).strip()
            if actual_skill.lower() == planned_skill.lower():
                return decision

        if expected_action == "use_skill":
            planned_skill = str(current_plan_step.get("skill", "")).strip()
            if planned_skill:
                return {
                    "action": "use_skill",
                    "skill": planned_skill,
                    "input": decision.get("input") or user_input,
                }

        return decision

    def _execute_tool_decision_stream(
        self,
        *,
        session_id: str,
        state: ConversationState,
        decision: dict[str, object],
        tool_history: list[dict[str, object]],
        user_input: str,
        run_id: str | None,
    ) -> Iterator[str]:
        action = str(decision.get("action", "")).strip().lower()
        decision = self._prepare_workflow_decision(
            user_input=user_input, decision=decision, tool_history=tool_history
        )
        if action == "tool_calls":
            calls = decision.get("calls")
            if not isinstance(calls, list):
                return
            for call in calls[:5]:
                if not isinstance(call, dict):
                    continue
                call = self._prepare_workflow_decision(
                    user_input=user_input, decision=dict(call), tool_history=tool_history
                )
                tool_name = str(call.get("tool", call.get("action", ""))).strip().lower()
                yield self._sse_event(
                    "tool",
                    {
                        "phase": "start",
                        "name": tool_name,
                        "runId": run_id,
                        "args": call,
                        "summary": self._tool_summary(tool_name, dict(call)),
                        "taskState": self._runtime_state_view(state),
                    },
                )
                self._set_runtime_stage(
                    session_id,
                    state,
                    stage="executing_tool",
                    status="running",
                    current_tool=tool_name,
                )
                result = self._run_single_tool_action_result(
                    session_id, state, dict(call), run_id=run_id
                )
                preview_text = str(result.get("text") or "")[:500]
                structured_data = result.get("structured_data")
                tool_history.append(
                    {
                        "tool": tool_name,
                        "args": dict(call),
                        "result": result.get("text") or "(no output)",
                        "structured_data": result.get("structured_data"),
                        "summary": result.get("summary", ""),
                    }
                )
                self._record_tool_evidence(
                    session_id,
                    state,
                    tool_name=tool_name,
                    summary=str(result.get("summary") or ""),
                    structured_data=result.get("structured_data"),
                    decision=dict(call),
                )
                yield self._sse_event(
                    "tool",
                    {
                        "phase": "end",
                        "name": tool_name,
                        "runId": run_id,
                        "resultPreview": preview_text,
                        "summary": str(result.get("summary") or ""),
                        "structuredPreview": self._preview_structured_data(structured_data),
                        "taskState": self._runtime_state_view(state),
                    },
                )
            return

        yield self._sse_event(
            "tool",
            {
                "phase": "start",
                "name": action,
                "runId": run_id,
                "args": decision,
                "summary": self._tool_summary(action, decision),
                "taskState": self._runtime_state_view(state),
            },
        )
        self._set_runtime_stage(
            session_id,
            state,
            stage="executing_tool",
            status="running",
            current_tool=action,
        )
        result = self._run_single_tool_action_result(
            session_id, state, decision, run_id=run_id
        )
        preview_text = str(result.get("text") or "")[:500]
        structured_data = result.get("structured_data")
        tool_history.append(
            {
                "tool": action,
                "args": dict(decision),
                "result": result.get("text") or "(no output)",
                "structured_data": result.get("structured_data"),
                "summary": result.get("summary", ""),
            }
        )
        self._record_tool_evidence(
            session_id,
            state,
            tool_name=action,
            summary=str(result.get("summary") or ""),
            structured_data=result.get("structured_data"),
            decision=dict(decision),
        )
        yield self._sse_event(
            "tool",
            {
                "phase": "end",
                "name": action,
                "runId": run_id,
                "resultPreview": preview_text,
                "summary": str(result.get("summary") or ""),
                "structuredPreview": self._preview_structured_data(structured_data),
                "taskState": self._runtime_state_view(state),
            },
        )

    def _generate_final_from_tool_history(
        self,
        *,
        session_id: str,
        state: ConversationState,
        user_input: str,
        tool_history: list[dict[str, object]],
    ) -> str:
        if not tool_history:
            return "我暂时没有拿到足够的信息来可靠完成这个任务。请再提供更具体的目标、文件或设备数据。"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are producing the final user-facing answer for an industrial assistant.\n"
                    "Use the gathered evidence. Be concrete and avoid guessing.\n"
                    "If relevant, provide a short conclusion and next steps.\n"
                    "Prefer structured_data fields when they exist in tool results."
                ),
            },
            {
                "role": "system",
                "content": (
                    f"Session id: {session_id}\n"
                    "Recent conversation:\n"
                    f"{self._conversation_context_for_agent(state)}\n\n"
                    "Runtime task context:\n"
                    f"{self._runtime_context_prompt(state, tool_history)}\n\n"
                    "Tool results:\n"
                    f"{self._render_tool_history(tool_history)}"
                ),
            },
            {"role": "user", "content": user_input},
        ]
        text = self._request_text_completion(
            messages,
            session_id=session_id,
            state=state,
            phase="agent_loop_final",
        )
        return text or "任务已执行，但最终总结生成失败。请查看工具结果后重试。"

    def _request_text_completion(
        self,
        messages: list[dict[str, str]],
        *,
        session_id: str,
        state: ConversationState,
        phase: str,
    ) -> str | None:
        payload: dict[str, Any] = {
            "model": self.settings.model_name,
            "messages": messages,
            "stream": False,
        }
        if self.settings.model_provider == "ollama":
            payload["think"] = self.settings.enable_thinking
        try:
            headers: dict[str, str] = {}
            if self.settings.model_provider in {
                "openai",
                "openai_compatible",
                "openai-compatible",
            }:
                headers["Content-Type"] = "application/json"
                if self.settings.model_api_key:
                    headers["Authorization"] = f"Bearer {self.settings.model_api_key}"
            response = requests.post(
                self.settings.model_base_url,
                json=payload,
                headers=headers,
                timeout=90,
            )
            response.raise_for_status()
            data = response.json()
            if self.settings.model_provider in {
                "openai",
                "openai_compatible",
                "openai-compatible",
            }:
                choices = data.get("choices") or []
                text = (choices[0].get("message") or {}).get("content", "") if choices else ""
            else:
                text = data.get("message", {}).get("content", "")
        except Exception as exc:
            self.trace_event(session_id, state, phase, status="error", error=str(exc))
            return None
        self.trace_event(session_id, state, phase, status="ok", text=text[:1000])
        return str(text or "").strip()

    def _build_agent_orchestrator_prompt(self) -> str:
        skill_catalog = (
            self.skill_manager.available_skills_catalog()
            if self.skill_manager
            else "<available_skills>\n(none)\n</available_skills>"
        )
        return (
            AGENT_ORCHESTRATOR_PROMPT_TEMPLATE
            + "\nAvailable skills:\n"
            + skill_catalog
            + "\n\nAvailable tools:\n"
            + self._available_tool_catalog()
            + "\n\nTreat skills as the place for domain-specific behavior. The runtime itself should stay generic."
        )

    def _build_workflow_hints(
        self, user_input: str, tool_history: list[dict[str, object]]
    ) -> str:
        hints: list[str] = []
        lowered = str(user_input or "").lower()
        has_structured = any(item.get("structured_data") is not None for item in tool_history)
        report_related = any(
            token in lowered
            for token in ("report", "summary", "总结", "报告", "汇总", "markdown")
        )
        if report_related and has_structured and self._has_skill("report-writer"):
            hints.append(
                "- Structured tool results are already available. A report-oriented next step is likely a `use_skill` call for `report-writer`."
            )
            hints.append(
                "- When calling `report-writer`, you may pass a compact JSON object with title, summary, findings, recommendations, and artifacts derived from structured_data."
            )

        if not has_structured and self._has_analysis_skill():
            hints.append(
                "- No structured_data has been gathered yet. Prefer exploration or analysis skills before finalizing."
            )

        if any(item.get("tool") == "read_skill" for item in tool_history):
            hints.append(
                "- A skill manual was already read. If that skill fits, prefer executing it instead of repeating documentation reads."
            )

        return "\n".join(hints)

    def _has_skill(self, skill_name: str) -> bool:
        if not self.skill_manager:
            return False
        try:
            self.skill_manager.get(skill_name)
        except SkillRegistryError:
            return False
        return True

    def _has_analysis_skill(self) -> bool:
        if not self.skill_manager:
            return False
        for skill in self.skill_manager.list_skills(include_disabled=False):
            if skill.workflow_hint == "analysis":
                return True
        return False

    def _prepare_workflow_decision(
        self,
        *,
        user_input: str,
        decision: dict[str, object],
        tool_history: list[dict[str, object]],
    ) -> dict[str, object]:
        action = str(decision.get("action", decision.get("tool", ""))).strip().lower()
        if action != "use_skill":
            return decision
        skill_name = str(decision.get("skill", "")).strip().lower()
        if skill_name != "report-writer":
            return decision

        raw_input = decision.get("input")
        if isinstance(raw_input, dict) and raw_input:
            return decision
        if isinstance(raw_input, str) and raw_input.strip().startswith("{"):
            return decision

        synthesized = self._build_report_writer_input(user_input=user_input, tool_history=tool_history)
        if synthesized is None:
            return decision
        next_decision = dict(decision)
        next_decision["input"] = synthesized
        return next_decision

    def _build_report_writer_input(
        self,
        *,
        user_input: str,
        tool_history: list[dict[str, object]],
    ) -> dict[str, object] | None:
        structured_items = [
            item for item in tool_history if item.get("structured_data") is not None
        ]
        if not structured_items:
            return None

        findings: list[str] = []
        artifacts: list[str] = []
        summary_parts: list[str] = []
        source_tools: list[str] = []

        for item in structured_items[-6:]:
            tool_name = str(item.get("tool", "")).strip()
            summary = str(item.get("summary", "")).strip()
            result_text = str(item.get("result", "")).strip()
            structured = item.get("structured_data")
            if tool_name:
                source_tools.append(tool_name)
            if summary:
                summary_parts.append(f"{tool_name}: {summary}")
            elif result_text:
                summary_parts.append(f"{tool_name}: {result_text[:120]}")

            findings.extend(self._extract_findings_from_structured_data(structured))
            artifacts.extend(self._extract_artifacts_from_structured_data(structured))

        title = "Analysis Report"
        lowered = str(user_input or "").lower()
        if any(token in lowered for token in ("报告", "report")):
            title = "Diagnostic Report"
        elif any(token in lowered for token in ("总结", "summary")):
            title = "Analysis Summary"

        payload: dict[str, object] = {
            "title": title,
            "summary": " | ".join(summary_parts[:3]) or "Analysis completed",
            "findings": self._dedupe_text_items(findings)[:8],
            "recommendations": self._build_report_recommendations(tool_history),
            "artifacts": self._dedupe_text_items(artifacts)[:8],
            "source_tools": self._dedupe_text_items(source_tools)[:8],
        }
        return payload

    @staticmethod
    def _dedupe_text_items(values: list[str]) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = str(value).strip()
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            unique.append(text)
        return unique

    def _build_report_recommendations(
        self, tool_history: list[dict[str, object]]
    ) -> list[str]:
        recommendations: list[str] = []
        for item in tool_history[-6:]:
            structured = item.get("structured_data")
            if not isinstance(structured, dict):
                continue
            for key in ("recommendations", "actions", "next_steps"):
                value = structured.get(key)
                if isinstance(value, list):
                    recommendations.extend(
                        str(entry).strip() for entry in value if str(entry).strip()
                    )
                elif isinstance(value, str) and value.strip():
                    recommendations.append(value.strip())

        deduped = self._dedupe_text_items(recommendations)
        if deduped:
            return deduped[:8]
        return ["Review the findings and validate the recommended next actions."]

    @staticmethod
    def _extract_findings_from_structured_data(structured: object) -> list[str]:
        findings: list[str] = []
        if isinstance(structured, dict):
            for key in ("findings", "issues", "warnings", "messages"):
                value = structured.get(key)
                if isinstance(value, list):
                    findings.extend(str(item).strip() for item in value if str(item).strip())
            if not findings:
                for key in ("summary", "status", "title", "report_path"):
                    value = structured.get(key)
                    if isinstance(value, str) and value.strip():
                        findings.append(f"{key}: {value.strip()}")
        elif isinstance(structured, list):
            findings.extend(str(item).strip() for item in structured if str(item).strip())
        return findings

    @staticmethod
    def _extract_artifacts_from_structured_data(structured: object) -> list[str]:
        artifacts: list[str] = []
        if not isinstance(structured, dict):
            return artifacts
        for key, value in structured.items():
            if key.endswith("_path") and isinstance(value, str) and value.strip():
                artifacts.append(value.strip())
            elif key in {"artifacts", "files"} and isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item.strip():
                        artifacts.append(item.strip())
                    elif isinstance(item, dict):
                        path = item.get("path")
                        if isinstance(path, str) and path.strip():
                            artifacts.append(path.strip())
        return artifacts

    @staticmethod
    def _render_tool_history(tool_history: list[dict[str, object]]) -> str:
        lines: list[str] = []
        for index, item in enumerate(tool_history[-8:], start=1):
            tool = str(item.get("tool", "unknown"))
            args = item.get("args", {})
            result = str(item.get("result", ""))[:1200]
            summary = str(item.get("summary", "")).strip()
            structured_data = item.get("structured_data")
            lines.append(f"[{index}] tool={tool}")
            lines.append(f"args={json.dumps(args, ensure_ascii=False)}")
            if summary:
                lines.append(f"summary={summary}")
            lines.append(f"result={result}")
            if structured_data is not None:
                lines.append(
                    "structured_data="
                    + json.dumps(structured_data, ensure_ascii=False)[:2000]
                )
        return "\n".join(lines) if lines else "(none)"

    @staticmethod
    def _conversation_context_for_agent(state: ConversationState, limit: int = 12) -> str:
        messages = [
            message
            for message in state.assistant.messages
            if message.get("role") in {"user", "assistant"}
        ]
        if not messages:
            return "(none)"
        lines: list[str] = []
        for message in messages[-limit:]:
            role = str(message.get("role", "assistant")).upper()
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            lines.append(f"{role}: {content[:1200]}")
        return "\n".join(lines) if lines else "(none)"

    def _emit_answer_chunks(self, message: str, *, run_id: str | None) -> Iterator[str]:
        text = str(message or "")
        if not text:
            yield self._sse_event("answer", {"text": "", "runId": run_id})
            return
        chunk_size = 220
        for index in range(0, len(text), chunk_size):
            chunk = text[index : index + chunk_size]
            yield self._sse_event("assistant", {"text": chunk, "runId": run_id})
            yield self._sse_event("answer", {"text": chunk, "runId": run_id})

    def _maybe_autorun_special_skill(
        self, session_id: str, state: ConversationState, user_input: str, run_id: str | None = None
    ) -> str | None:
        if not self.skill_manager:
            return None
        upload_dir = self.session_store.session_upload_dir(session_id)
        try:
            if self._should_autorun_power_curve_for_uploads(user_input, state):
                self.trace_event(
                    session_id,
                    state,
                    "autorun_power_curve",
                    upload_dir=str(upload_dir),
                    attached_files=state.attached_file_ids,
                )
                self.trace_event(session_id, state, "tool_stream", runId=run_id, phase="start", name="power-curve-assessment")
                skill_result = self.skill_manager.execute_result(
                    "power-curve-assessment",
                    {"path": str(upload_dir), "files": state.attached_file_ids},
                    session=state.assistant,
                )
                return self._render_skill_result(skill_result)
            if self._should_autorun_fft_for_uploads(user_input, state):
                self.trace_event(
                    session_id,
                    state,
                    "autorun_fft",
                    upload_dir=str(upload_dir),
                    attached_files=state.attached_file_ids,
                )
                self.trace_event(session_id, state, "tool_stream", runId=run_id, phase="start", name="fft-frequency")
                skill_result = self.skill_manager.execute_result(
                    "fft-frequency",
                    {"path": str(upload_dir), "files": state.attached_file_ids},
                    session=state.assistant,
                )
                return self._render_skill_result(skill_result)
        except Exception as exc:
            self.trace_event(session_id, state, "autorun_error", error=str(exc))
        return None

    def _plan_system_commands(
        self, session_id: str, state: ConversationState, user_input: str
    ) -> dict[str, object] | None:
        if not self.skill_manager or not list(self.skill_manager.names()):
            return None
        attachment_manifest = self.session_store.attachment_manifest(
            session_id, state.attached_file_ids
        )
        extra_context = f"\n\n{attachment_manifest}\n" if attachment_manifest else ""
        return self._request_router_decision(
            [
                {"role": "system", "content": self._build_router_prompt()},
                {
                    "role": "user",
                    "content": (
                        f"User request:\n{user_input}{extra_context}\n\n"
                        f"Project root: {self.settings.project_root}\n"
                        "Only return JSON."
                    ),
                },
            ],
            session_id=session_id,
            state=state,
            phase="phase1_router",
        )

    def _plan_with_skill_doc(
        self,
        session_id: str,
        state: ConversationState,
        user_input: str,
        skill_name: str,
    ) -> dict[str, object] | None:
        if not self.skill_manager:
            return None
        try:
            skill = self.skill_manager.get(skill_name)
            skill_doc = skill.render_markdown()
        except SkillRegistryError:
            return None

        self.trace_event(
            session_id,
            state,
            "read_skill_doc",
            skill=skill_name,
            skill_doc_preview=skill_doc[:1200],
        )

        return self._request_router_decision(
            [
                {"role": "system", "content": self._build_router_prompt()},
                {
                    "role": "system",
                    "content": (
                        f"You have selected the skill '{skill_name}'.\n"
                        "Below is the full SKILL.md content. Follow it as the skill manual "
                        "when deciding the final action.\n"
                        f"This skill invocation mode is: {'packaged_runtime' if skill.has_runtime else 'manual_only'}.\n"
                        "If invocation mode is packaged_runtime and user asks to execute/analyze with this skill, "
                        "prefer action=use_skill with the user request as input.\n"
                        "If the skill is manual_only, do not return use_skill.\n"
                        "For manual_only skills, prefer low-level tool actions such as read_file, "
                        "list_dir, search_text, and exec_command.\n\n"
                        f"<skill_doc name=\"{skill_name}\">\n{skill_doc}\n</skill_doc>"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"User request:\n{user_input}\n\n"
                        f"Project root: {self.settings.project_root}\n"
                        f"Selected skill directory: {skill.pack_dir}\n"
                        "Resolve example commands against the selected skill directory.\n"
                        "Return the final JSON action now."
                    ),
                },
            ],
            session_id=session_id,
            state=state,
            phase="phase2_router",
        )

    def _request_router_decision(
        self,
        messages: list[dict[str, str]],
        *,
        session_id: str,
        state: ConversationState,
        phase: str,
    ) -> dict[str, object] | None:
        payload: dict[str, Any] = {
            "model": self.settings.model_name,
            "messages": messages,
            "stream": False,
        }
        if self.settings.model_provider == "ollama":
            payload["think"] = False
        try:
            headers: dict[str, str] = {}
            if self.settings.model_provider in {
                "openai",
                "openai_compatible",
                "openai-compatible",
            }:
                headers["Content-Type"] = "application/json"
                if self.settings.model_api_key:
                    headers["Authorization"] = f"Bearer {self.settings.model_api_key}"
            response = requests.post(
                self.settings.model_base_url,
                json=payload,
                headers=headers,
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            if self.settings.model_provider in {
                "openai",
                "openai_compatible",
                "openai-compatible",
            }:
                choices = data.get("choices") or []
                raw = (choices[0].get("message") or {}).get("content", "") if choices else ""
            else:
                raw = data.get("message", {}).get("content", "")
        except Exception as exc:
            self.trace_event(session_id, state, phase, status="error", error=str(exc))
            return None
        parsed = self._parse_router_json(raw)
        self.trace_event(session_id, state, phase, status="ok", raw=raw, parsed=parsed)
        return parsed

    def _run_planned_commands(
        self,
        session_id: str,
        state: ConversationState,
        decision: dict[str, object],
        run_id: str | None = None,
    ) -> str | None:
        action = str(decision.get("action", "")).strip().lower()
        self.trace_event(session_id, state, "planned_action", decision=decision)
        if action == "none":
            return None
        if action == "tool_calls":
            calls = decision.get("calls")
            if not isinstance(calls, list):
                return None
            context = decision.get("_skill_context")
            outputs: list[str] = []
            for index, call in enumerate(calls[:5], start=1):
                if not isinstance(call, dict):
                    continue
                payload = dict(call)
                if context and "_skill_context" not in payload:
                    payload["_skill_context"] = context
                result = self._run_single_tool_action(session_id, state, payload, run_id=run_id)
                if result:
                    outputs.append(f"步骤{index}:\n{result}")
            return "\n\n".join(outputs) if outputs else None
        return self._run_single_tool_action(session_id, state, decision, run_id=run_id)

    def _run_single_tool_action(
        self,
        session_id: str,
        state: ConversationState,
        decision: dict[str, object],
        run_id: str | None = None,
    ) -> str | None:
        action = str(decision.get("action", decision.get("tool", ""))).strip().lower()
        self.trace_event(session_id, state, "tool_action_start", action=action, payload=decision)
        self.trace_event(
            session_id,
            state,
            "tool_stream",
            runId=run_id,
            phase="start",
            name=action,
            args={k: v for k, v in decision.items() if k not in {"_skill_context"}},
        )

        if action == "read_skill":
            if not self.skill_manager:
                return None
            skill_name = str(decision.get("skill", "")).strip()
            if not skill_name:
                return None
            doc = self.skill_manager.read_skill_doc(skill_name)
            result = f"[{skill_name}] SKILL.md\n\n{doc}"
            self.trace_event(session_id, state, "tool_action_end", action=action, result=result[:1200])
            self.trace_event(session_id, state, "tool_stream", runId=run_id, phase="end", name=action, resultPreview=result[:200])
            return result

        if action == "list_dir":
            path_value = str(decision.get("path", "")).strip() or "."
            result = f"[list_dir] {path_value}\n\n{self._list_project_dir(path_value)}"
            self.trace_event(session_id, state, "tool_action_end", action=action, result=result[:1200])
            self.trace_event(session_id, state, "tool_stream", runId=run_id, phase="end", name=action, resultPreview=result[:200])
            return result

        if action == "read_file":
            path_value = str(decision.get("path", "")).strip()
            if not path_value:
                return None
            result = f"[read_file] {path_value}\n\n{self._read_project_file(path_value)}"
            self.trace_event(session_id, state, "tool_action_end", action=action, result=result[:1200])
            self.trace_event(session_id, state, "tool_stream", runId=run_id, phase="end", name=action, resultPreview=result[:200])
            return result

        if action == "search_text":
            pattern = str(decision.get("pattern", "")).strip()
            path_value = str(decision.get("path", "")).strip()
            if not pattern:
                return None
            label = path_value or "."
            result = f"[search_text] {pattern} @ {label}\n\n{self._search_project_text(pattern, path_value)}"
            self.trace_event(session_id, state, "tool_action_end", action=action, result=result[:1200])
            self.trace_event(session_id, state, "tool_stream", runId=run_id, phase="end", name=action, resultPreview=result[:200])
            return result

        if action == "list_uploaded_files":
            content = self.session_store.list_uploaded_files_content(
                session_id, state.attached_file_ids
            )
            result = f"[list_uploaded_files]\n\n{content}"
            self.trace_event(session_id, state, "tool_action_end", action=action, result=result[:1200])
            self.trace_event(session_id, state, "tool_stream", runId=run_id, phase="end", name=action, resultPreview=result[:200])
            return result

        if action == "read_uploaded_file":
            file_id = str(decision.get("file_id", "")).strip()
            if not file_id:
                return "读取失败：缺少 file_id。"
            result = (
                f"[read_uploaded_file] {file_id}\n\n"
                f"{self.session_store.read_uploaded_file_content(session_id, file_id)}"
            )
            self.trace_event(session_id, state, "tool_action_end", action=action, result=result[:1200])
            self.trace_event(session_id, state, "tool_stream", runId=run_id, phase="end", name=action, resultPreview=result[:200])
            return result

        if action == "exec_command":
            command = str(decision.get("command", "")).strip()
            if not command:
                return None
            skill_context = decision.get("_skill_context")
            skill_name = ""
            if isinstance(skill_context, dict):
                skill_name = str(skill_context.get("skill", "")).strip()
            if skill_name:
                command = self._normalize_skill_command(command, skill_name)
            result = f"[exec_command] {command}\n\n{self._format_shell_result(self._run_system_command(command))}"
            self.trace_event(session_id, state, "tool_action_end", action=action, result=result[:1200])
            self.trace_event(session_id, state, "tool_stream", runId=run_id, phase="end", name=action, resultPreview=result[:200])
            return result

        if action == "use_skill":
            return self._invoke_skill_action_result(
                session_id, state, decision, run_id=run_id
            )["text"]

        if action == "use_skills":
            calls = decision.get("calls")
            if not isinstance(calls, list) or not self.skill_manager:
                return None
            outputs: list[str] = []
            for index, call in enumerate(calls[:3], start=1):
                if not isinstance(call, dict):
                    continue
                payload = {
                    "action": "use_skill",
                    "skill": call.get("skill", ""),
                    "input": call.get("input", ""),
                }
                result = self._invoke_skill_action_result(
                    session_id, state, payload, run_id=run_id
                )
                if result["text"]:
                    outputs.append(f"步骤{index}: {result['text']}")
            final = "\n\n".join(outputs) if outputs else None
            if final:
                self.trace_event(session_id, state, "tool_action_end", action=action, result=final[:1200])
            return final

        return None

    def _run_single_tool_action_result(
        self,
        session_id: str,
        state: ConversationState,
        decision: dict[str, object],
        run_id: str | None = None,
    ) -> dict[str, object]:
        action = str(decision.get("action", decision.get("tool", ""))).strip().lower()
        if action == "use_skill":
            return self._invoke_skill_action_result(
                session_id, state, decision, run_id=run_id
            )
        text = self._run_single_tool_action(session_id, state, decision, run_id=run_id)
        return {
            "text": text or "",
            "structured_data": self._structured_tool_payload(
                action, decision, text or "", session_id=session_id, state=state
            ),
            "summary": self._tool_summary(action, decision),
        }

    def _invoke_skill_action(
        self,
        session_id: str,
        state: ConversationState,
        decision: dict[str, object],
        run_id: str | None = None,
    ) -> str | None:
        return self._invoke_skill_action_result(
            session_id, state, decision, run_id=run_id
        )["text"]

    def _invoke_skill_action_result(
        self,
        session_id: str,
        state: ConversationState,
        decision: dict[str, object],
        run_id: str | None = None,
    ) -> dict[str, object]:
        if not self.skill_manager:
            return {"text": "", "structured_data": None, "summary": ""}
        skill_name = str(decision.get("skill", "")).strip()
        skill_input = decision.get("input", "")
        if not skill_name:
            result = json.dumps(
                {
                    "code": "SKILL_NOT_FOUND",
                    "message": "No skill provided.",
                    "requested_name": "",
                    "candidates": [],
                },
                ensure_ascii=False,
            )
            return {"text": result, "structured_data": None, "summary": "Missing skill name"}
        try:
            skill_result = self.skill_manager.execute_result(
                skill_name, skill_input, session=state.assistant
            )
            result = self._render_skill_result(skill_result)
            structured_data = skill_result.structured_data
            summary = skill_result.summary
        except SkillNotFoundError as exc:
            result = json.dumps(
                {
                    "code": "SKILL_NOT_FOUND",
                    "message": str(exc),
                    "requested_name": exc.requested_name,
                    "candidates": exc.candidates,
                },
                ensure_ascii=False,
            )
            structured_data = None
            summary = str(exc)
        except SkillAmbiguousError as exc:
            result = json.dumps(
                {
                    "code": "SKILL_AMBIGUOUS",
                    "message": str(exc),
                    "requested_name": exc.requested_name,
                    "candidates": exc.candidates,
                },
                ensure_ascii=False,
            )
            structured_data = None
            summary = str(exc)
        except (SkillDisabled, SkillUnavailable, SkillRegistryError) as exc:
            result = json.dumps(
                {
                    "code": "SKILL_EXECUTION_FAILED",
                    "message": str(exc),
                    "requested_name": skill_name,
                    "candidates": [],
                },
                ensure_ascii=False,
            )
            structured_data = None
            summary = str(exc)
        self.trace_event(
            session_id,
            state,
            "tool_action_end",
            action="use_skill",
            skill=skill_name,
            result=result[:1200],
        )
        self.trace_event(
            session_id,
            state,
            "tool_stream",
            runId=run_id,
            phase="end",
            name="use_skill",
            skill=skill_name,
            resultPreview=result[:200],
        )
        return {"text": result, "structured_data": structured_data, "summary": summary}

    @staticmethod
    def _render_skill_result(result: SkillExecutionResult) -> str:
        if result.structured_data is None:
            return result.output_text
        if result.summary and result.output_text and result.output_text != result.summary:
            return f"{result.summary}\n\n{result.output_text}"
        return result.output_text or result.summary or "(no output)"

    def _structured_tool_payload(
        self,
        action: str,
        decision: dict[str, object],
        text: str,
        *,
        session_id: str,
        state: ConversationState,
    ) -> object | None:
        if not text:
            return None

        if action == "read_skill":
            return {
                "skill": str(decision.get("skill", "")).strip(),
                "content": self._extract_tool_body(text),
            }

        if action == "list_dir":
            body = self._extract_tool_body(text)
            entries = [line.strip() for line in body.splitlines() if line.strip()]
            return {
                "path": str(decision.get("path", "")).strip() or ".",
                "entries": entries,
            }

        if action == "read_file":
            return {
                "path": str(decision.get("path", "")).strip(),
                "content": self._extract_tool_body(text),
            }

        if action == "search_text":
            body = self._extract_tool_body(text)
            matches = [line.strip() for line in body.splitlines() if line.strip()]
            return {
                "pattern": str(decision.get("pattern", "")).strip(),
                "path": str(decision.get("path", "")).strip() or ".",
                "matches": matches,
            }

        if action == "list_uploaded_files":
            files = []
            for file_id in state.attached_file_ids:
                path = self.session_store.safe_uploaded_path(session_id, file_id)
                if path is None or not path.exists():
                    continue
                files.append({"file_id": file_id, "path": str(path)})
            return {"files": files}

        if action == "read_uploaded_file":
            return {
                "file_id": str(decision.get("file_id", "")).strip(),
                "content": self._extract_tool_body(text),
            }

        if action == "exec_command":
            return {
                "command": str(decision.get("command", "")).strip(),
                "output": self._extract_tool_body(text),
            }

        if action == "use_skills":
            return None

        return None

    @staticmethod
    def _tool_summary(action: str, decision: dict[str, object]) -> str:
        if action == "read_skill":
            return f"Read skill doc for {str(decision.get('skill', '')).strip()}"
        if action == "list_dir":
            return f"Listed directory {str(decision.get('path', '')).strip() or '.'}"
        if action == "read_file":
            return f"Read file {str(decision.get('path', '')).strip()}"
        if action == "search_text":
            return f"Searched for {str(decision.get('pattern', '')).strip()}"
        if action == "list_uploaded_files":
            return "Listed uploaded files"
        if action == "read_uploaded_file":
            return f"Read uploaded file {str(decision.get('file_id', '')).strip()}"
        if action == "exec_command":
            return f"Executed command {str(decision.get('command', '')).strip()}"
        return action or "tool"

    @staticmethod
    def _extract_tool_body(text: str) -> str:
        parts = str(text or "").split("\n\n", 1)
        if len(parts) == 2:
            return parts[1]
        return str(text or "")

    @staticmethod
    def _preview_structured_data(structured_data: object) -> str:
        if structured_data is None:
            return ""
        try:
            text = json.dumps(structured_data, ensure_ascii=False)
        except TypeError:
            text = str(structured_data)
        return text[:600]

    def wait_for_run(self, run_id: str, timeout_ms: int = 30_000) -> dict[str, object]:
        return self.runtime.wait_for_run(run_id, timeout_ms=timeout_ms)

    def _build_router_prompt(self) -> str:
        skill_catalog = (
            self.skill_manager.available_skills_catalog()
            if self.skill_manager
            else "<available_skills>\n(none)\n</available_skills>"
        )
        return (
            SKILL_ROUTER_PROMPT_TEMPLATE.replace("<<SKILL_CATALOG>>", skill_catalog).replace(
                "<<TOOL_CATALOG>>", self._available_tool_catalog()
            )
        )

    def _get_structured_tool_list(self) -> list[dict[str, object]]:
        """Get structured list of available tools for planner."""
        return [
            {"name": "read_skill", "description": "Read the rendered SKILL.md body for a named skill."},
            {"name": "list_dir", "description": "List files and directories under a project-relative directory."},
            {"name": "read_file", "description": "Read a UTF-8 text file inside the project."},
            {"name": "search_text", "description": "Search project files using ripgrep-style text matching."},
            {"name": "list_uploaded_files", "description": "List uploaded files currently attached to this session."},
            {"name": "read_uploaded_file", "description": "Read one uploaded text file by file_id."},
            {"name": "exec_command", "description": "Run a shell command in the project workspace."},
            {"name": "use_skill", "description": "Call a skill packaged runtime entrypoint."},
        ]

    def _get_structured_skill_list(self) -> list[dict[str, object]]:
        """Get structured list of available skills for planner."""
        if not self.skill_manager:
            return []

        skills = []
        for skill in self.skill_manager.list_skills(include_disabled=False):
            skills.append({
                "id": skill.skill_id,
                "description": skill.description or "No description",
                "workflow_hint": skill.workflow_hint or "",
                "structured_result": skill.structured_result or False,
            })
        return skills

    @staticmethod
    def _available_tool_catalog() -> str:
        return "\n".join(
            [
                "<available_tools>",
                "- name: read_skill",
                "  description: Read the rendered SKILL.md body for a named skill.",
                "  parameters: { skill: string }",
                "- name: list_dir",
                "  description: List files and directories under a project-relative directory.",
                "  parameters: { path: string }",
                "- name: read_file",
                "  description: Read a UTF-8 text file inside the project.",
                "  parameters: { path: string }",
                "- name: search_text",
                "  description: Search project files using ripgrep-style text matching.",
                "  parameters: { pattern: string, path?: string }",
                "- name: list_uploaded_files",
                "  description: List uploaded files currently attached to this session.",
                "  parameters: {}",
                "- name: read_uploaded_file",
                "  description: Read one uploaded text file by file_id.",
                "  parameters: { file_id: string }",
                "- name: exec_command",
                "  description: Run a shell command in the project workspace.",
                "  parameters: { command: string }",
                "- name: use_skill",
                "  description: Call a skill packaged runtime entrypoint.",
                "  parameters: { skill: string, input: string | object }",
                "</available_tools>",
            ]
        )

    def _safe_project_path(self, path_value: str) -> Path | None:
        raw = str(path_value or "").strip()
        if not raw:
            return None
        path = Path(raw)
        candidate = (
            (self.settings.project_root / raw).resolve() if not path.is_absolute() else path.resolve()
        )
        try:
            candidate.relative_to(self.settings.project_root)
        except ValueError:
            return None
        return candidate

    def _read_project_file(self, path_value: str) -> str:
        path = self._safe_project_path(path_value)
        if path is None:
            return "读取失败：路径无效或超出项目目录。"
        if not path.exists():
            return f"读取失败：文件不存在：{path}"
        if not path.is_file():
            return f"读取失败：不是文件：{path}"
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"读取失败：文件不是 UTF-8 文本：{path}"
        except Exception as exc:
            return f"读取失败：{exc}"

    def _list_project_dir(self, path_value: str) -> str:
        path = self._safe_project_path(path_value or ".")
        if path is None:
            return "列目录失败：路径无效或超出项目目录。"
        if not path.exists():
            return f"列目录失败：目录不存在：{path}"
        if not path.is_dir():
            return f"列目录失败：不是目录：{path}"
        try:
            items = sorted(path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
        except Exception as exc:
            return f"列目录失败：{exc}"
        if not items:
            return "(empty)"
        lines: list[str] = []
        for item in items[:200]:
            suffix = "/" if item.is_dir() else ""
            rel = item.relative_to(self.settings.project_root)
            lines.append(f"{rel}{suffix}")
        if len(items) > 200:
            lines.append(f"... ({len(items) - 200} more)")
        return "\n".join(lines)

    def _search_project_text(self, pattern: str, path_value: str = "") -> str:
        pattern = str(pattern or "").strip()
        if not pattern:
            return "搜索失败：缺少 pattern。"
        path = self._safe_project_path(path_value or ".")
        if path is None:
            return "搜索失败：路径无效或超出项目目录。"
        if not path.exists():
            return f"搜索失败：路径不存在：{path}"
        target = str(path.relative_to(self.settings.project_root))
        try:
            result = subprocess.run(
                ["rg", "-n", "--no-heading", "--color", "never", pattern, target],
                cwd=str(self.settings.shell_cwd),
                capture_output=True,
                text=True,
                timeout=20,
            )
        except FileNotFoundError:
            return "搜索失败：未找到 rg。"
        except subprocess.TimeoutExpired:
            return "搜索失败：搜索超时。"
        except Exception as exc:
            return f"搜索失败：{exc}"
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        if result.returncode not in {0, 1}:
            return stderr or stdout or f"搜索失败：exit_code={result.returncode}"
        if not stdout:
            return "(no matches)"
        lines = stdout.splitlines()
        if len(lines) > 200:
            return "\n".join(lines[:200] + [f"... ({len(lines) - 200} more matches)"])
        return stdout

    def _run_system_command(self, command: str) -> str:
        command = command.strip()
        if not command:
            return "Usage: execute command <cmd>"
        try:
            result = subprocess.run(
                ["bash", "-lc", command],
                shell=False,
                cwd=str(self.settings.shell_cwd),
                capture_output=True,
                text=True,
                timeout=self.settings.shell_timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            timeout = self.settings.shell_timeout_seconds or 0
            return f"Command timed out after {timeout}s."
        except Exception as exc:
            return f"Command execution failed: {exc}"

        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        parts = [f"exit_code: {result.returncode}"]
        if stdout:
            parts.extend(["stdout:", stdout])
        if stderr:
            parts.extend(["stderr:", stderr])
        if not stdout and not stderr:
            parts.append("(no output)")
        return "\n".join(parts)

    @staticmethod
    def _format_shell_result(result: str) -> str:
        if not result.startswith("exit_code:"):
            return result
        exit_code = None
        stdout = ""
        stderr = ""
        current = None
        for line in result.splitlines():
            if line.startswith("exit_code:"):
                try:
                    exit_code = int(line.split(":", 1)[1].strip())
                except ValueError:
                    exit_code = None
                current = None
                continue
            if line == "stdout:":
                current = "stdout"
                continue
            if line == "stderr:":
                current = "stderr"
                continue
            if current == "stdout":
                stdout = f"{stdout}\n{line}".strip()
            elif current == "stderr":
                stderr = f"{stderr}\n{line}".strip()
        if exit_code is not None and exit_code != 0:
            return f"命令执行失败：{stderr or '请检查命令参数。'}"
        if stdout:
            return stdout
        if stderr:
            return stderr
        return "已执行完成。"

    @staticmethod
    def _extract_explicit_command(user_input: str) -> str | None:
        text = user_input.strip()
        if not text:
            return None
        known_prefixes = (
            "ls ",
            "pwd",
            "cat ",
            "rg ",
            "touch ",
            "mkdir ",
            "echo ",
            "date",
            "whoami",
            "head ",
            "tail ",
            "wc ",
            "sed ",
            "python3 ",
            "gcc ",
            "cc ",
            "clang ",
            "make ",
        )
        lowered = text.lower()
        if any(lowered.startswith(prefix) for prefix in known_prefixes):
            return text
        if lowered.startswith("!"):
            return text[1:].strip() or None
        for marker in ("执行命令", "运行命令", "run command", "execute command"):
            idx = lowered.find(marker)
            if idx >= 0:
                cmd = text[idx + len(marker) :].strip(" ：:;,")
                return cmd or None
        return None

    @staticmethod
    def _parse_router_json(raw: str) -> dict[str, object] | None:
        text = raw.strip()
        if not text:
            return None
        try:
            payload = json.loads(text)
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def _normalize_skill_command(self, command: str, skill_name: str) -> str:
        if not self.skill_manager:
            return command
        try:
            skill = self.skill_manager.get(skill_name)
        except SkillRegistryError:
            return command
        normalized = command
        rel_pack = str(skill.pack_dir.relative_to(self.settings.project_root))
        skill_dir_markers = {
            f"cd {skill.pack_dir}",
            f"cd '{skill.pack_dir}'",
            f'cd "{skill.pack_dir}"',
            f"cd {rel_pack}",
            f"cd '{rel_pack}'",
            f'cd "{rel_pack}"',
        }
        uses_skill_cwd = any(marker in normalized for marker in skill_dir_markers)

        stale_prefixes = (
            "skills/email/scripts/",
            "skills/email-mail-master-1.0.0/scripts/",
            f"{rel_pack}/scripts/",
        )
        for prefix in stale_prefixes:
            if prefix not in normalized:
                continue
            normalized = re.sub(
                re.escape(prefix) + r"([A-Za-z0-9_.-]+)",
                (
                    lambda match: (
                        f"scripts/{match.group(1)}"
                        if uses_skill_cwd
                        else f"{rel_pack}/scripts/{match.group(1)}"
                    )
                ),
                normalized,
            )

        if not uses_skill_cwd:
            normalized = re.sub(
                r"(?<!\S)scripts/([A-Za-z0-9_.-]+)",
                lambda match: f"{rel_pack}/scripts/{match.group(1)}",
                normalized,
            )
        return normalized

    @staticmethod
    def _should_autorun_power_curve_for_uploads(
        user_input: str, state: ConversationState
    ) -> bool:
        text = (user_input or "").strip().lower()
        if not text or not state.attached_file_ids:
            return False
        has_analyze_intent = any(
            keyword in text
            for keyword in (
                "分析",
                "评估",
                "功率曲线",
                "power curve",
                "powercurve",
                "风速功率",
                "出力",
                "发电健康",
                "健康评分",
            )
        )
        return has_analyze_intent and any(
            str(file_id).lower().endswith(".csv") for file_id in state.attached_file_ids
        )

    @staticmethod
    def _should_autorun_fft_for_uploads(user_input: str, state: ConversationState) -> bool:
        text = (user_input or "").strip().lower()
        if not text or not state.attached_file_ids:
            return False
        fft_keywords = ("fft", "频率", "振动", "加速度", "塔架", "塔筒", "主频", "频谱")
        power_curve_keywords = (
            "功率曲线",
            "power curve",
            "powercurve",
            "风速功率",
            "出力",
            "发电健康",
            "健康评分",
        )
        has_fft_intent = any(keyword in text for keyword in fft_keywords)
        has_power_curve_intent = any(keyword in text for keyword in power_curve_keywords)
        return has_fft_intent and not has_power_curve_intent and any(
            str(file_id).lower().endswith((".csv", ".txt"))
            for file_id in state.attached_file_ids
        )

    @staticmethod
    def _clip_trace_value(value: object, limit: int = 800) -> object:
        if isinstance(value, str):
            return value if len(value) <= limit else value[:limit] + f"... ({len(value) - limit} more chars)"
        if isinstance(value, list):
            return [GatewayCore._clip_trace_value(item, limit=limit) for item in value[:10]]
        if isinstance(value, dict):
            return {
                str(key): GatewayCore._clip_trace_value(item, limit=limit)
                for key, item in list(value.items())[:20]
            }
        return value

    @staticmethod
    def _sse_event(event: str, payload: dict[str, object]) -> str:
        return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
