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
    upload_dir = project_root / "data" / "uploads" / safe_session
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
        assessments, reference_frequency = fft_module.analyse_directory(
            target_dir,
            include_files=include_files,
        )
    except Exception as exc:
        print(f"FFT analysis failed: {exc}")
        sys.exit(1)

    lines = [f"现场参考固有频率: {reference_frequency:.4f} Hz"]
    lines.append("时刻, 当前状态, 识别频率Hz, 偏离程度, 预警")
    for item in assessments:
        ts = item.representative_time or item.timestamp
        lines.append(
            f"{ts:%Y-%m-%d %H:%M:%S}, {item.state}, "
            f"{item.natural_frequency_hz:.4f}, {item.deviation_pct:.2f}%, {item.warning}"
        )
    lines.append(f"输出文件: {target_dir / 'tower_frequency_windows.csv'}")
    lines.append(f"输出文件: {target_dir / 'tower_frequency_assessment.csv'}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
