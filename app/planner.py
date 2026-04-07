from __future__ import annotations

import json
import os
import time
from typing import Any

from app.runtime_settings import RuntimeSettings, load_runtime_settings


class TaskPlanner:
    """Generates and manages task execution plans."""

    def __init__(self, settings: RuntimeSettings):
        self.settings = settings

    def generate_plan(
        self,
        *,
        user_input: str,
        available_skills: list[dict[str, Any]],
        available_tools: list[dict[str, Any]],
        conversation_context: str = "",
    ) -> dict[str, Any]:
        """Generate a multi-step plan for accomplishing the user's task."""

        # Build prompt for planning
        prompt = self._build_planning_prompt(
            user_input=user_input,
            available_skills=available_skills,
            available_tools=available_tools,
            conversation_context=conversation_context,
        )

        # Call LLM for planning decision
        messages = [
            {"role": "system", "content": self._planner_system_prompt()},
            {"role": "user", "content": prompt},
        ]

        response = self._request_planning_decision(messages)

        # Parse and validate plan
        plan = self._parse_plan_response(response, user_input)

        return plan

    def _planner_system_prompt(self) -> str:
        return """You are a task planner for an AI assistant runtime.
Your job is to analyze the user's request and create a step-by-step execution plan.

Output must be valid JSON with this schema:
{
  "goal": "concise description of the main objective",
  "steps": [
    {
      "id": 1,
      "phase": "exploration|analysis|execution|report|validation",
      "description": "what this step should accomplish",
      "expected_action": "list_dir|read_file|search_text|exec_command|use_skill|tool_calls",
      "skill": "skill_id (only if expected_action is use_skill)",
      "completion_criteria": "how to know when this step is complete"
    }
  ],
  "max_steps": 10,
  "constraints": ["list", "of", "constraints", "or", "considerations"]
}

Rules:
- Keep plans concise but actionable.
- Focus on evidence gathering before analysis.
- Prefer packaged skills over generic tools when they fit the task.
- Include exploration steps when file paths or data sources are unknown.
- Plan for report generation when the user asks for summaries or documentation.
- Limit to 3-6 steps for most tasks.
- Phases should progress naturally: exploration -> analysis -> execution -> report -> validation.
- Each step should have clear completion criteria.
"""

    def _build_planning_prompt(
        self,
        *,
        user_input: str,
        available_skills: list[dict[str, Any]],
        available_tools: list[dict[str, Any]],
        conversation_context: str,
    ) -> str:
        skills_text = "\n".join(
            f"- {skill.get('id', 'unknown')}: {skill.get('description', 'No description')}"
            for skill in available_skills[:20]
        )

        tools_text = "\n".join(
            f"- {tool.get('name', 'unknown')}: {tool.get('description', 'No description')}"
            for tool in available_tools[:20]
        )

        return f"""User request: {user_input}

Conversation context:
{conversation_context}

Available skills (prefer these for domain-specific tasks):
{skills_text if skills_text else "(no skills available)"}

Available tools (for exploration and general operations):
{tools_text if tools_text else "(no tools available)"}

Analyze the request and create a step-by-step execution plan.
Consider what evidence needs to be gathered, what analysis should be performed,
and what final output the user expects."""

    def _request_planning_decision(self, messages: list[dict[str, str]]) -> str:
        """Call LLM for planning decision."""
        import requests

        try:
            # Build payload similar to gateway_core
            payload = {
                "model": self.settings.model_name,
                "messages": messages,
                "stream": False,
            }

            # Provider-specific adjustments
            if self.settings.model_provider == "ollama":
                payload["think"] = False

            headers = {}
            if self.settings.model_provider in {"openai", "openai_compatible", "openai-compatible"}:
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

            # Extract response based on provider
            if self.settings.model_provider in {"openai", "openai_compatible", "openai-compatible"}:
                choices = data.get("choices") or []
                raw = (choices[0].get("message") or {}).get("content", "") if choices else ""
            else:
                raw = data.get("message", {}).get("content", "")

            return raw

        except Exception as e:
            # Fallback to simple plan
            return json.dumps({
                "goal": "Complete user request",
                "steps": [
                    {
                        "id": 1,
                        "phase": "exploration",
                        "description": "Explore available files and data",
                        "expected_action": "list_dir",
                        "completion_criteria": "Understand the directory structure and locate relevant files"
                    },
                    {
                        "id": 2,
                        "phase": "analysis",
                        "description": "Analyze relevant data based on exploration results",
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
                "constraints": ["Plan generated as fallback due to LLM error"]
            })

    def _parse_plan_response(self, response: str, user_input: str) -> dict[str, Any]:
        """Parse and validate the plan response."""
        try:
            # Try to extract JSON from response
            response = response.strip()

            # Find JSON object
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                plan = json.loads(json_str)
            else:
                # Fallback if no JSON found
                plan = {}
        except (json.JSONDecodeError, ValueError):
            plan = {}

        # Ensure required fields
        if not plan.get("goal"):
            plan["goal"] = user_input[:200]

        if not plan.get("steps") or not isinstance(plan["steps"], list):
            plan["steps"] = [
                {
                    "id": 1,
                    "phase": "exploration",
                    "description": "Explore available files and data",
                    "expected_action": "list_dir",
                    "completion_criteria": "Understand the directory structure"
                }
            ]

        # Validate and normalize steps
        for i, step in enumerate(plan["steps"]):
            if not isinstance(step, dict):
                step = {}
                plan["steps"][i] = step

            step["id"] = step.get("id", i + 1)
            step["phase"] = step.get("phase", "execution")
            step["description"] = step.get("description", f"Step {i + 1}")
            step["expected_action"] = step.get("expected_action", "tool_calls")
            step["completion_criteria"] = step.get("completion_criteria", "Step completed")

            # Normalize phase
            valid_phases = {"exploration", "analysis", "execution", "report", "validation"}
            if step["phase"] not in valid_phases:
                step["phase"] = "execution"

        plan["max_steps"] = plan.get("max_steps", len(plan["steps"]))
        plan["constraints"] = plan.get("constraints", [])

        # Add metadata
        plan["_plan_version"] = "1.0"
        plan["_generated_at"] = int(time.time() if "time" in dir(__builtins__) else 0)

        return plan

    def get_current_step(self, plan: dict[str, Any], executed_steps: list[int]) -> dict[str, Any] | None:
        """Get the next step to execute based on completed steps."""
        if not plan.get("steps"):
            return None

        for step in plan["steps"]:
            step_id = step.get("id")
            if step_id is not None and step_id not in executed_steps:
                return step

        return None

    def mark_step_completed(self, plan: dict[str, Any], step_id: int) -> dict[str, Any]:
        """Mark a step as completed and return updated plan."""
        if "completed_steps" not in plan:
            plan["completed_steps"] = []

        if step_id not in plan["completed_steps"]:
            plan["completed_steps"].append(step_id)

        return plan


# For testing
if __name__ == "__main__":
    settings = load_runtime_settings()
    planner = TaskPlanner(settings)

    # Example usage
    plan = planner.generate_plan(
        user_input="Analyze vibration data in the data/ directory",
        available_skills=[
            {"id": "fft-frequency", "description": "FFT frequency analysis"},
            {"id": "report-writer", "description": "Generate markdown reports"}
        ],
        available_tools=[
            {"name": "list_dir", "description": "List directory contents"},
            {"name": "read_file", "description": "Read file contents"}
        ],
        conversation_context=""
    )

    print(json.dumps(plan, indent=2))
