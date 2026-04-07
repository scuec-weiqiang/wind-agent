from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _coerce_payload(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {"summary": text}
    return payload if isinstance(payload, dict) else {"summary": text}


def _lines(values: Any) -> list[str]:
    if isinstance(values, list):
        return [str(item).strip() for item in values if str(item).strip()]
    if isinstance(values, str) and values.strip():
        return [values.strip()]
    return []


def build_report(payload: dict[str, Any], session_id: str) -> tuple[str, Path]:
    title = str(payload.get("title", "Analysis Report")).strip() or "Analysis Report"
    summary = str(payload.get("summary", "")).strip()
    findings = _lines(payload.get("findings"))
    recommendations = _lines(payload.get("recommendations"))
    artifacts = _lines(payload.get("artifacts"))
    source_tools = _lines(payload.get("source_tools"))

    reports_dir = Path(__file__).resolve().parents[1] / "outputs"
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{session_id or 'session'}-{timestamp}.md"
    report_path = reports_dir / filename

    lines = [f"# {title}", ""]
    if summary:
        lines.extend(["## Summary", summary, ""])
    if findings:
        lines.append("## Findings")
        lines.extend([f"- {item}" for item in findings])
        lines.append("")
    if recommendations:
        lines.append("## Recommendations")
        lines.extend([f"- {item}" for item in recommendations])
        lines.append("")
    if artifacts:
        lines.append("## Artifacts")
        lines.extend([f"- {item}" for item in artifacts])
        lines.append("")
    if source_tools:
        lines.append("## Source Tools")
        lines.extend([f"- {item}" for item in source_tools])
        lines.append("")

    if len(lines) <= 2:
        lines.extend(["## Summary", "No report content was provided.", ""])

    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return "\n".join(lines).rstrip(), report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="")
    parser.add_argument("--session-id", default="")
    args = parser.parse_args()

    payload = _coerce_payload(args.input)
    report_text, report_path = build_report(payload, args.session_id)
    summary = str(payload.get("summary", "")).strip() or "Report generated"

    result = {
        "kind": "openclaw_skill_result",
        "ok": True,
        "summary": summary,
        "output_text": report_text,
        "data": {
            "report_path": str(report_path),
            "title": str(payload.get("title", "Analysis Report")).strip() or "Analysis Report",
        },
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
