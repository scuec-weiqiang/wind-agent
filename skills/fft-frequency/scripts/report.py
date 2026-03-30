from __future__ import annotations

import csv
import html
from datetime import datetime
from pathlib import Path
from typing import Sequence


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _status_class(state: str) -> str:
    if state == "异常":
        return "status-alert"
    if state == "关注":
        return "status-watch"
    return "status-normal"


def _assessment_comment(row: dict[str, str]) -> str:
    warning = row.get("预警", "")
    deviation = row.get("偏离程度", "")
    if warning == "二级预警":
        return f"偏离较大（{deviation}），建议重点复核塔架、基础与施工质量。"
    if warning == "一级预警":
        return f"存在偏离（{deviation}），建议持续跟踪并安排现场复核。"
    return f"当前偏离较小（{deviation}），可继续常规监测。"


def _build_assessment_rows_html(rows: Sequence[dict[str, str]]) -> str:
    chunks: list[str] = []
    for row in rows:
        chunks.append(
            "<tr>"
            f"<td>{html.escape(row.get('时刻', ''))}</td>"
            f"<td class=\"{_status_class(row.get('当前状态', ''))}\">{html.escape(row.get('当前状态', ''))}</td>"
            f"<td>{html.escape(row.get('识别频率Hz', ''))} Hz</td>"
            f"<td>{html.escape(row.get('偏离程度', ''))}</td>"
            f"<td>{html.escape(row.get('预警', ''))}</td>"
            f"<td>{html.escape(_assessment_comment(row))}</td>"
            "</tr>"
        )
    return "\n".join(chunks)


def _build_attention_rows_html(rows: Sequence[dict[str, str]]) -> str:
    alert_rows = [row for row in rows if row.get("预警") in {"一级预警", "二级预警"}]
    if not alert_rows:
        return (
            "<tr>"
            "<td colspan=\"5\">本批数据未识别出需要重点关注的异常机组。</td>"
            "</tr>"
        )
    chunks: list[str] = []
    for row in alert_rows:
        warning = row.get("预警", "")
        action = "建议尽快现场复核塔架、基础、连接螺栓及施工质量。" if warning == "二级预警" else "建议纳入跟踪清单，持续观察频率变化趋势。"
        chunks.append(
            "<tr>"
            f"<td>{html.escape(row.get('时刻', ''))}</td>"
            f"<td>{html.escape(row.get('时刻', ''))}</td>"
            f"<td>{html.escape(row.get('识别频率Hz', ''))} Hz</td>"
            f"<td>{html.escape(warning)}，偏离 {html.escape(row.get('偏离程度', ''))}</td>"
            f"<td>{html.escape(action)}</td>"
            "</tr>"
        )
    return "\n".join(chunks)


def _build_window_rows_html(rows: Sequence[dict[str, str]], limit: int = 30) -> str:
    chunks: list[str] = []
    for row in rows[:limit]:
        chunks.append(
            "<tr>"
            f"<td>{html.escape(row.get('文件', ''))}</td>"
            f"<td>{html.escape(row.get('时刻', ''))}</td>"
            f"<td>{html.escape(row.get('窗口中心秒', ''))}</td>"
            f"<td>{html.escape(row.get('识别频率Hz', ''))} Hz</td>"
            f"<td>{html.escape(row.get('置信度', ''))}</td>"
            f"<td>{html.escape(row.get('使用方向', ''))}</td>"
            "</tr>"
        )
    if len(rows) > limit:
        chunks.append(
            f"<tr><td colspan=\"6\">窗口明细较多，报告仅展示前 {limit} 行，完整结果请查看 tower_frequency_windows.csv。</td></tr>"
        )
    return "\n".join(chunks)


def _render_template(template: str, mapping: dict[str, str]) -> str:
    result = template
    for key, value in mapping.items():
        result = result.replace("{{ " + key + " }}", value)
    return result


def generate_report(
    output_dir: Path,
    reference_frequency: float,
    template_path: Path,
) -> Path:
    assessment_path = output_dir / "tower_frequency_assessment.csv"
    windows_path = output_dir / "tower_frequency_windows.csv"
    trend_svg_path = output_dir / "frequency_trend.svg"

    assessment_rows = _read_csv_rows(assessment_path)
    window_rows = _read_csv_rows(windows_path)
    trend_svg = trend_svg_path.read_text(encoding="utf-8") if trend_svg_path.exists() else "<p>未生成趋势图。</p>"

    times = [row.get("时刻", "") for row in assessment_rows if row.get("时刻")]
    data_range = f"{times[0]} 至 {times[-1]}" if times else "-"
    alert_count = sum(1 for row in assessment_rows if row.get("预警") == "二级预警")
    watch_count = sum(1 for row in assessment_rows if row.get("预警") == "一级预警")
    if alert_count:
        overall_conclusion = f"识别出 {alert_count} 个二级预警时刻，建议重点复核异常机组。"
    elif watch_count:
        overall_conclusion = f"识别出 {watch_count} 个一级预警时刻，建议持续跟踪。"
    else:
        overall_conclusion = "本批数据整体处于正常范围，可继续常规监测。"

    template = template_path.read_text(encoding="utf-8")
    html_text = _render_template(
        template,
        {
            "report_title": "风电机组塔架频率识别分析报告",
            "site_name": html.escape(output_dir.parent.name or "默认风场"),
            "turbine_id": html.escape(output_dir.name),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_range": html.escape(data_range),
            "reference_frequency_hz": f"{reference_frequency:.4f}",
            "file_count": str(len(assessment_rows)),
            "overall_conclusion": html.escape(overall_conclusion),
            "frequency_trend_svg": trend_svg,
            "assessment_rows_html": _build_assessment_rows_html(assessment_rows),
            "attention_rows_html": _build_attention_rows_html(assessment_rows),
            "window_rows_html": _build_window_rows_html(window_rows),
        },
    )

    output_path = output_dir / "tower_frequency_report.html"
    output_path.write_text(html_text, encoding="utf-8")
    return output_path
