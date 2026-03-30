from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any


def _safe_session_dir_name(session_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", (session_id or "").strip() or "default")


def _default_analysis_dir(session_id: str) -> Path:
    project_root = Path(__file__).resolve().parents[3]
    safe_session = _safe_session_dir_name(session_id)
    upload_dirs = [
        project_root / "data" / "sessions" / "uploads" / safe_session,
        project_root / "data" / "uploads" / safe_session,
    ]
    for upload_dir in upload_dirs:
        if upload_dir.exists() and upload_dir.is_dir() and any(upload_dir.glob("*.csv")):
            return upload_dir.resolve()
    return project_root / "tests"


def _extract_path_from_text(text: str) -> Path | None:
    candidates = re.findall(r"(/[-A-Za-z0-9_./]+)", text)
    for candidate in candidates:
        path = Path(candidate).expanduser().resolve()
        if path.exists():
            return path
    return None


def _looks_like_path(text: str) -> bool:
    value = (text or "").strip()
    return bool(value) and (
        value.startswith(("/", "./", "../", "~"))
        or "/" in value
        or "\\" in value
        or value.lower().endswith(".csv")
    )


def _parse_input(raw: str, session_id: str) -> Path:
    text = (raw or "").strip()
    if not text:
        return _default_analysis_dir(session_id)
    if text.startswith("{"):
        try:
            payload: dict[str, Any] = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON input: {exc}") from exc
        return Path(str(payload.get("path", "") or ".")).expanduser().resolve()
    natural_path = _extract_path_from_text(text)
    if natural_path:
        return natural_path
    if _looks_like_path(text):
        return Path(text).expanduser().resolve()
    return _default_analysis_dir(session_id)


def _load_module(scripts_dir: Path):
    module_path = scripts_dir / "power_curve.py"
    spec = importlib.util.spec_from_file_location("wind_power_curve_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load power_curve.py module spec.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _build_summary(summary) -> list[str]:
    lines = [
        f"本次共读取 {summary.total_points} 个样本点，其中 {summary.modeling_points} 个样本参与建模，风速范围 {summary.wind_min:.2f}~{summary.wind_max:.2f} m/s。",
        f"按 0.1 m/s 分仓并完成仓内异常剔除后，生成了 {summary.fit_points} 个功率曲线状态仓。",
        f"当前状态为 {summary.assessment.state}，健康状态评分 {summary.assessment.score} 分，预警 {summary.assessment.warning}。",
    ]
    lines.append(
        f"包络带外样本占比约 {summary.assessment.outside_ratio * 100:.2f}%，低于下边缘样本占比约 {summary.assessment.below_ratio * 100:.2f}%。"
    )
    lines.append(f"建议：{summary.assessment.advice}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="")
    parser.add_argument("--session-id", default="")
    args = parser.parse_args()

    try:
        target = _parse_input(args.input, args.session_id)
    except Exception as exc:
        print(f"Power curve input parse failed: {exc}")
        sys.exit(1)

    scripts_dir = Path(__file__).resolve().parent
    try:
        module = _load_module(scripts_dir)
    except Exception as exc:
        print(f"Power curve analysis failed: {exc}")
        sys.exit(1)

    try:
        csv_path = module.select_input_file(target)
        summary, points_csv, fit_csv, assessment_csv, svg_path, report_path = module.analyse_file(csv_path)
    except Exception as exc:
        print(f"Power curve analysis failed: {exc}")
        sys.exit(1)

    lines = _build_summary(summary)
    if args.session_id and svg_path.exists():
        lines.append("")
        lines.append("功率曲线图：")
        lines.append(f"![风电机组功率曲线](/session/file?session_id={args.session_id}&file_id={svg_path.name})")
    if args.session_id and report_path.exists():
        lines.append("")
        lines.append("报告文件：")
        lines.append(f"[查看 HTML 报告](/session/file?session_id={args.session_id}&file_id={report_path.name})")
        lines.append(f"[下载 HTML 报告](/session/file?session_id={args.session_id}&file_id={report_path.name}&download=1)")
    lines.append("")
    lines.append(f"健康评估：`{assessment_csv}`")
    lines.append(f"散点明细：`{points_csv}`")
    lines.append(f"拟合曲线：`{fit_csv}`")
    lines.append(f"图像文件：`{svg_path}`")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
