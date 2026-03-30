from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any


def _parse_input(raw: str, session_id: str = "") -> tuple[Path, list[str]]:
    text = (raw or "").strip()
    if not text:
        return _default_analysis_dir(session_id), []
    if text.startswith("{"):
        try:
            payload: dict[str, Any] = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON input: {exc}") from exc
        path_value = str(payload.get("path", "")).strip() or "."
        files_raw = payload.get("files", [])
        files = [
            str(item).strip()
            for item in (files_raw if isinstance(files_raw, list) else [])
            if str(item).strip()
        ]
        return Path(path_value).expanduser().resolve(), files
    natural_path = _extract_path_from_text(text)
    if natural_path:
        return natural_path, []
    if _looks_like_path(text):
        return Path(text).expanduser().resolve(), []
    # If natural language has no resolvable path, fallback to upload dir first, then project root.
    return _default_analysis_dir(session_id), []


def _extract_path_from_text(text: str) -> Path | None:
    # Support natural-language input like:
    # "请帮我分析 /home/w/wind-agent 的风机振动数据"
    candidates = re.findall(r"(/[-A-Za-z0-9_./]+)", text)
    for candidate in candidates:
        path = Path(candidate).expanduser().resolve()
        if path.exists() and path.is_dir():
            return path
    return None


def _looks_like_path(text: str) -> bool:
    value = (text or "").strip()
    if not value:
        return False
    return (
        value.startswith(("/", "./", "../", "~"))
        or "/" in value
        or "\\" in value
    )


def _safe_session_dir_name(session_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]", "_", (session_id or "").strip() or "default")
    return normalized


def _default_analysis_dir(session_id: str) -> Path:
    project_root = Path(__file__).resolve().parents[3]
    safe_session = _safe_session_dir_name(session_id)
    upload_dirs = [
        project_root / "data" / "sessions" / "uploads" / safe_session,
        project_root / "data" / "uploads" / safe_session,
    ]
    for upload_dir in upload_dirs:
        if upload_dir.exists() and upload_dir.is_dir():
            has_candidates = any(upload_dir.glob("*.csv")) or any(upload_dir.glob("*.txt"))
            if has_candidates:
                return upload_dir.resolve()
    return project_root


def _has_candidate_files(path: Path) -> bool:
    return any(path.glob("*.csv")) or any(path.glob("*.txt"))


def _load_fft_module(skill_scripts_dir: Path):
    fft_path = skill_scripts_dir / "fft.py"
    if not fft_path.exists():
        raise FileNotFoundError(f"fft.py not found: {fft_path}")
    spec = importlib.util.spec_from_file_location("wind_fft_module", fft_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load fft.py module spec.")
    module = importlib.util.module_from_spec(spec)
    # Ensure decorators relying on sys.modules (e.g. dataclass) can resolve the module.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_report_module(skill_scripts_dir: Path):
    report_path = skill_scripts_dir / "report.py"
    if not report_path.exists():
        raise FileNotFoundError(f"report.py not found: {report_path}")
    spec = importlib.util.spec_from_file_location("wind_fft_report_module", report_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load report.py module spec.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _build_natural_summary(assessments, reference_frequency: float) -> list[str]:
    total = len(assessments)
    alert_items = [item for item in assessments if item.warning == "二级预警"]
    watch_items = [item for item in assessments if item.warning == "一级预警"]
    normal_count = total - len(alert_items) - len(watch_items)

    summary = [
        f"本次共完成 {total} 个时刻样本的塔架频率识别，现场参考固有频率约为 {reference_frequency:.4f} Hz。"
    ]
    if alert_items:
        summary.append(
            f"结果中识别出 {len(alert_items)} 个二级预警时刻，建议优先复核对应机组的塔架、基础与施工质量。"
        )
    elif watch_items:
        summary.append(
            f"结果中暂无二级预警，但识别出 {len(watch_items)} 个一级预警时刻，建议持续跟踪频率变化趋势。"
        )
    else:
        summary.append("本批数据未识别出预警时刻，整体处于可继续常规监测的状态。")

    summary.append(
        f"状态分布为：正常 {normal_count} 个，一级预警 {len(watch_items)} 个，二级预警 {len(alert_items)} 个。"
    )

    focus_items = sorted(
        [item for item in assessments if item.warning in {"一级预警", "二级预警"}],
        key=lambda item: (item.warning != "二级预警", -item.deviation_pct, item.timestamp),
    )[:3]
    if focus_items:
        summary.append("")
        summary.append("建议重点关注：")
        for item in focus_items:
            ts = item.representative_time or item.timestamp
            summary.append(
                f"- {ts:%Y-%m-%d %H:%M:%S}：识别频率 {item.natural_frequency_hz:.4f} Hz，偏离 {item.deviation_pct:.2f}%，{item.warning}。"
            )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="")
    parser.add_argument("--session-id", default="")
    args = parser.parse_args()

    try:
        target_dir, include_files = _parse_input(args.input, args.session_id)
    except Exception as exc:
        print(f"FFT input parse failed: {exc}")
        sys.exit(1)

    # Backward-compatible guard:
    # if target resolves to skill dir (old behavior) and contains no candidate data files,
    # fallback to upload dir/project root so natural-language calls still work.
    scripts_dir = Path(__file__).resolve().parent
    skill_dir = scripts_dir.parent
    if target_dir.resolve() == skill_dir.resolve() and not _has_candidate_files(target_dir):
        target_dir = _default_analysis_dir(args.session_id)

    if not target_dir.exists() or not target_dir.is_dir():
        print(f"FFT analysis failed: directory not found: {target_dir}")
        sys.exit(1)

    try:
        fft_module = _load_fft_module(scripts_dir)
    except Exception as exc:
        print(f"FFT analysis failed: {exc}")
        sys.exit(1)

    try:
        report_module = _load_report_module(scripts_dir)
    except Exception as exc:
        print(f"FFT report failed: {exc}")
        sys.exit(1)

    try:
        assessments, reference_frequency = fft_module.analyse_directory(
            target_dir,
            include_files=include_files,
        )
    except Exception as exc:
        print(f"FFT analysis failed: {exc}")
        sys.exit(1)

    lines = _build_natural_summary(assessments, reference_frequency)

    template_path = skill_dir / "report_template.html"
    try:
        report_path = report_module.generate_report(
            target_dir,
            reference_frequency,
            template_path,
        )
    except Exception as exc:
        lines.append("")
        lines.append(f"HTML 报告生成失败: {exc}")

    trend_svg = target_dir / "frequency_trend.svg"
    if args.session_id and trend_svg.exists():
        lines.append("")
        lines.append("频率随时间变化曲线：")
        lines.append(f"![塔架频率随时间变化曲线](/session/file?session_id={args.session_id}&file_id=frequency_trend.svg)")

    if args.session_id and 'report_path' in locals() and report_path.exists():
        lines.append("")
        lines.append("报告文件：")
        lines.append(f"[查看 HTML 报告](/session/file?session_id={args.session_id}&file_id={report_path.name})")
        lines.append(f"[下载 HTML 报告](/session/file?session_id={args.session_id}&file_id={report_path.name}&download=1)")

    lines.append("")
    lines.append(f"结果文件：`{target_dir / 'tower_frequency_assessment.csv'}`")
    lines.append(f"窗口明细：`{target_dir / 'tower_frequency_windows.csv'}`")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
