from __future__ import annotations

from app.skill_manager import _parse_skill_execution_result


def test_parse_plain_json_as_structured_result() -> None:
    result = _parse_skill_execution_result(
        "demo-skill",
        '{"count": 2, "items": ["a", "b"]}',
        "",
    )

    assert result.ok is True
    assert result.structured_data == {"count": 2, "items": ["a", "b"]}
    assert "count" in result.output_text


def test_parse_openclaw_skill_result_envelope() -> None:
    result = _parse_skill_execution_result(
        "report-skill",
        (
            '{"kind":"openclaw_skill_result","ok":true,'
            '"summary":"Report generated","output_text":"A report is ready.",'
            '"data":{"report_path":"reports/daily.md"}}'
        ),
        "",
    )

    assert result.ok is True
    assert result.summary == "Report generated"
    assert result.output_text == "A report is ready."
    assert result.structured_data == {"report_path": "reports/daily.md"}
