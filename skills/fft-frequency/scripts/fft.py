#!/usr/bin/env python3
"""风电塔架频率识别与离散预警分析.

无第三方依赖，直接基于目录中的历史 csv 数据完成：
1. 振动主导频率提取
2. 单文件窗口频率聚类，识别固有频率
3. 多文件固有频率聚类，形成现场参考频率
4. 输出偏离程度、趋势及预警结果
"""

from __future__ import annotations

import csv
import html
import math
import re
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Sequence


ENCODINGS = ("gbk", "utf-8-sig", "utf-8")
TIME_COLUMN = "时间"
X_COLUMN = "X振动"
Y_COLUMN = "Y振动"
DEFAULT_PATTERNS = ("*data.csv", "*data.txt", "*.csv", "*.txt")

# 频率搜索参数，可根据机型再微调
WINDOW_SECONDS = 20.0
STEP_SECONDS = 10.0
FREQ_MIN = 0.15
FREQ_MAX = 1.20
CLUSTER_TOLERANCE_HZ = 0.08
CONSENSUS_TOLERANCE_HZ = 0.08


@dataclass
class WindowFrequency:
    center_time: float
    frequency_hz: float
    confidence: float
    axis: str


@dataclass
class FileAssessment:
    file_name: str
    timestamp: datetime
    natural_frequency_hz: float
    confidence: float
    sample_count: int
    window_count: int
    representative_time: datetime | None = None
    deviation_hz: float = 0.0
    deviation_pct: float = 0.0
    trend_pct_per_day: float = 0.0
    state: str = ""
    warning: str = ""


def median(values: Sequence[float], default: float = 0.0) -> float:
    return statistics.median(values) if values else default


def mean(values: Sequence[float], default: float = 0.0) -> float:
    return statistics.fmean(values) if values else default


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    total = sum(weights)
    if total <= 0:
        return mean(values)
    return sum(v * w for v, w in zip(values, weights)) / total


def parse_timestamp(file_name: str) -> datetime:
    stem = Path(file_name).stem
    stamp = stem.replace("data", "")
    return datetime.strptime(stamp, "%Y%m%d_%H%M")


def open_csv_reader(path: Path) -> list[dict[str, str]]:
    last_error: Exception | None = None
    for encoding in ENCODINGS:
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                text = handle.read()
            rows = _parse_delimited_rows(text)
            if rows:
                return rows
        except UnicodeDecodeError as exc:
            last_error = exc
    raise RuntimeError(f"无法解码文件 {path}") from last_error


def _parse_delimited_rows(text: str) -> list[dict[str, str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    header_line = lines[0].lstrip("\ufeff")
    data_lines = lines[1:]
    if not data_lines:
        return []

    if "," in header_line:
        header = [item.strip() for item in next(csv.reader([header_line]))]
        return _rows_from_csv_reader(header, data_lines, delimiter=",")

    if "\t" in header_line:
        header = [item.strip() for item in next(csv.reader([header_line], delimiter="\t"))]
        return _rows_from_csv_reader(header, data_lines, delimiter="\t")

    header = [item.strip() for item in re.split(r"\s+", header_line) if item.strip()]
    return _rows_from_whitespace(header, data_lines)


def _rows_from_csv_reader(
    header: list[str],
    data_lines: list[str],
    delimiter: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in data_lines:
        values = [item.strip() for item in next(csv.reader([line], delimiter=delimiter))]
        if len(values) < len(header):
            continue
        rows.append({key: values[index] for index, key in enumerate(header)})
    return rows


def _rows_from_whitespace(header: list[str], data_lines: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in data_lines:
        values = [item.strip() for item in re.split(r"\s+", line.strip()) if item.strip()]
        if len(values) < len(header):
            continue
        rows.append({key: values[index] for index, key in enumerate(header)})
    return rows


def read_signal(path: Path) -> tuple[list[float], list[float], list[float]]:
    rows = open_csv_reader(path)
    if not rows:
        raise ValueError(f"{path.name} 没有有效数据")
    times, x_values, y_values = [], [], []
    for row in rows:
        try:
            times.append(float(row[TIME_COLUMN]))
            x_values.append(float(row[X_COLUMN]))
            y_values.append(float(row[Y_COLUMN]))
        except (KeyError, ValueError) as exc:
            raise ValueError(f"{path.name} 中缺少必要列或存在非法数值") from exc
    return times, x_values, y_values


def shifted_seconds(times: Sequence[float]) -> list[float]:
    if not times:
        return []
    origin = min(times)
    return [time_value - origin for time_value in times]


def detrend(values: Sequence[float]) -> list[float]:
    n = len(values)
    if n < 2:
        return list(values)
    x_mean = (n - 1) / 2.0
    y_mean = mean(values)
    sxx = 0.0
    sxy = 0.0
    for index, value in enumerate(values):
        dx = index - x_mean
        sxx += dx * dx
        sxy += dx * (value - y_mean)
    slope = sxy / sxx if sxx else 0.0
    intercept = y_mean - slope * x_mean
    return [value - (intercept + slope * index) for index, value in enumerate(values)]


def hanning(values: Sequence[float]) -> list[float]:
    n = len(values)
    if n <= 1:
        return list(values)
    windowed = []
    for index, value in enumerate(values):
        weight = 0.5 - 0.5 * math.cos(2.0 * math.pi * index / (n - 1))
        windowed.append(value * weight)
    return windowed


def spectrum_peak(
    values: Sequence[float],
    dt: float,
    f_min: float = FREQ_MIN,
    f_max: float = FREQ_MAX,
) -> tuple[float, float, float]:
    """返回主频、峰值幅值、峰值置信度."""
    n = len(values)
    prepared = hanning(detrend(values))
    span = n * dt
    resolution = 1.0 / span
    k_min = max(1, math.ceil(f_min / resolution))
    k_max = min(n // 2, math.floor(f_max / resolution))
    amplitudes: list[tuple[float, float]] = []
    for k in range(k_min, k_max + 1):
        freq = k * resolution
        real = 0.0
        imag = 0.0
        for index, value in enumerate(prepared):
            angle = 2.0 * math.pi * k * index / n
            real += value * math.cos(angle)
            imag -= value * math.sin(angle)
        amplitude = math.hypot(real, imag)
        amplitudes.append((freq, amplitude))
    if not amplitudes:
        return 0.0, 0.0, 0.0
    best_freq, best_amp = max(amplitudes, key=lambda item: item[1])
    background = median([amp for _, amp in amplitudes], default=1e-12)
    confidence = best_amp / max(background, 1e-12)
    return best_freq, best_amp, confidence


def sliding_windows(
    times: Sequence[float],
    x_values: Sequence[float],
    y_values: Sequence[float],
    window_seconds: float = WINDOW_SECONDS,
    step_seconds: float = STEP_SECONDS,
) -> list[WindowFrequency]:
    if len(times) < 3:
        return []
    dt = times[1] - times[0]
    window_size = max(8, int(round(window_seconds / dt)))
    step_size = max(1, int(round(step_seconds / dt)))
    results: list[WindowFrequency] = []
    for start in range(0, len(times) - window_size + 1, step_size):
        end = start + window_size
        center_time = times[start + window_size // 2]
        window_x = x_values[start:end]
        window_y = y_values[start:end]
        fx, _, cx = spectrum_peak(window_x, dt)
        fy, _, cy = spectrum_peak(window_y, dt)
        if not fx and not fy:
            continue
        if fx and fy and abs(fx - fy) <= CONSENSUS_TOLERANCE_HZ:
            frequency = (fx * cx + fy * cy) / max(cx + cy, 1e-12)
            confidence = max(cx, cy)
            axis = "XY"
        elif cx >= cy:
            frequency = fx
            confidence = cx
            axis = "X"
        else:
            frequency = fy
            confidence = cy
            axis = "Y"
        results.append(
            WindowFrequency(
                center_time=center_time,
                frequency_hz=frequency,
                confidence=confidence,
                axis=axis,
            )
        )
    return results


def cluster_1d(
    values: Sequence[float],
    weights: Sequence[float],
    tolerance: float = CLUSTER_TOLERANCE_HZ,
) -> list[tuple[list[float], list[float]]]:
    if not values:
        return []
    pairs = sorted(zip(values, weights), key=lambda item: item[0])
    clusters: list[tuple[list[float], list[float]]] = []
    current_values = [pairs[0][0]]
    current_weights = [pairs[0][1]]
    for value, weight in pairs[1:]:
        center = weighted_mean(current_values, current_weights)
        if abs(value - center) <= tolerance:
            current_values.append(value)
            current_weights.append(weight)
        else:
            clusters.append((current_values, current_weights))
            current_values = [value]
            current_weights = [weight]
    clusters.append((current_values, current_weights))
    return clusters


def select_reference_cluster(
    values: Sequence[float],
    weights: Sequence[float],
    tolerance: float = CLUSTER_TOLERANCE_HZ,
) -> tuple[float, float, int]:
    clusters = cluster_1d(values, weights, tolerance=tolerance)
    if not clusters:
        return 0.0, 0.0, 0
    ranked = sorted(
        clusters,
        key=lambda item: (
            sum(item[1]),
            len(item[0]),
            -statistics.pvariance(item[0]) if len(item[0]) > 1 else 0.0,
        ),
        reverse=True,
    )
    cluster_values, cluster_weights = ranked[0]
    center = weighted_mean(cluster_values, cluster_weights)
    confidence = sum(cluster_weights) / max(sum(weights), 1e-12)
    return center, confidence, len(cluster_values)


def dominant_cluster_mask(
    values: Sequence[float],
    weights: Sequence[float],
    tolerance: float = CLUSTER_TOLERANCE_HZ,
) -> list[bool]:
    if not values:
        return []
    indexed_pairs = sorted(
        enumerate(zip(values, weights)),
        key=lambda item: item[1][0],
    )
    clusters: list[list[int]] = []
    current = [indexed_pairs[0][0]]
    current_values = [indexed_pairs[0][1][0]]
    current_weights = [indexed_pairs[0][1][1]]
    for original_index, (value, weight) in indexed_pairs[1:]:
        center = weighted_mean(current_values, current_weights)
        if abs(value - center) <= tolerance:
            current.append(original_index)
            current_values.append(value)
            current_weights.append(weight)
        else:
            clusters.append(current)
            current = [original_index]
            current_values = [value]
            current_weights = [weight]
    clusters.append(current)

    def cluster_key(indices: list[int]) -> tuple[float, int, float]:
        cluster_values = [values[index] for index in indices]
        cluster_weights = [weights[index] for index in indices]
        variance = statistics.pvariance(cluster_values) if len(cluster_values) > 1 else 0.0
        return (sum(cluster_weights), len(indices), -variance)

    best_cluster = max(clusters, key=cluster_key)
    mask = [False] * len(values)
    for index in best_cluster:
        mask[index] = True
    return mask


def linear_slope(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    if len(x_values) < 2 or len(x_values) != len(y_values):
        return 0.0
    x_mean = mean(x_values)
    y_mean = mean(y_values)
    sxx = 0.0
    sxy = 0.0
    for x_value, y_value in zip(x_values, y_values):
        dx = x_value - x_mean
        sxx += dx * dx
        sxy += dx * (y_value - y_mean)
    return sxy / sxx if sxx else 0.0


def classify_state(
    deviation_pct: float,
    trend_pct_per_day: float,
    cluster_confidence: float,
) -> tuple[str, str]:
    if deviation_pct >= 10.0 or (deviation_pct >= 6.0 and trend_pct_per_day >= 0.5):
        return "异常", "二级预警"
    if deviation_pct >= 5.0 or trend_pct_per_day >= 0.3 or cluster_confidence < 0.55:
        return "关注", "一级预警"
    return "正常", "正常"


def assess_file(path: Path) -> tuple[FileAssessment, list[WindowFrequency]]:
    times, x_values, y_values = read_signal(path)
    shifted_times = shifted_seconds(times)
    windows = sliding_windows(shifted_times, x_values, y_values)
    frequencies = [item.frequency_hz for item in windows]
    confidences = [item.confidence for item in windows]
    natural_frequency, cluster_confidence, sample_count = select_reference_cluster(
        frequencies,
        confidences,
    )
    mask = dominant_cluster_mask(frequencies, confidences)
    cluster_times = [
        window.center_time
        for window, selected in zip(windows, mask)
        if selected
    ]
    cluster_weights = [
        window.confidence
        for window, selected in zip(windows, mask)
        if selected
    ]
    representative_offset = weighted_mean(cluster_times, cluster_weights) if cluster_times else 0.0
    representative_time = parse_timestamp(path.name) + timedelta(seconds=round(representative_offset))
    assessment = FileAssessment(
        file_name=path.name,
        timestamp=parse_timestamp(path.name),
        natural_frequency_hz=natural_frequency,
        confidence=cluster_confidence,
        sample_count=sample_count,
        window_count=len(windows),
        representative_time=representative_time,
    )
    return assessment, windows


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _svg_polyline(
    x_values: Sequence[float],
    y_values: Sequence[float],
    width: int,
    height: int,
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> str:
    if not x_values or not y_values:
        return ""
    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(y_values), max(y_values)
    if math.isclose(min_x, max_x):
        max_x = min_x + 1.0
    if math.isclose(min_y, max_y):
        pad = 0.1 if min_y == 0 else abs(min_y) * 0.05
        min_y -= pad
        max_y += pad
    plot_width = width - left - right
    plot_height = height - top - bottom
    points = []
    for x_value, y_value in zip(x_values, y_values):
        x_pos = left + (x_value - min_x) / (max_x - min_x) * plot_width
        y_pos = top + plot_height - (y_value - min_y) / (max_y - min_y) * plot_height
        points.append(f"{x_pos:.1f},{y_pos:.1f}")
    return " ".join(points)


def _svg_smooth_path(points: Sequence[tuple[float, float]]) -> str:
    if not points:
        return ""
    if len(points) == 1:
        x_value, y_value = points[0]
        return f"M {x_value:.1f} {y_value:.1f}"
    commands = [f"M {points[0][0]:.1f} {points[0][1]:.1f}"]
    for index in range(len(points) - 1):
        p0 = points[index - 1] if index > 0 else points[index]
        p1 = points[index]
        p2 = points[index + 1]
        p3 = points[index + 2] if index + 2 < len(points) else p2
        cp1_x = p1[0] + (p2[0] - p0[0]) / 6.0
        cp1_y = p1[1] + (p2[1] - p0[1]) / 6.0
        cp2_x = p2[0] - (p3[0] - p1[0]) / 6.0
        cp2_y = p2[1] - (p3[1] - p1[1]) / 6.0
        commands.append(
            f"C {cp1_x:.1f} {cp1_y:.1f}, {cp2_x:.1f} {cp2_y:.1f}, {p2[0]:.1f} {p2[1]:.1f}"
        )
    return " ".join(commands)


def build_frequency_trend_svg(
    assessments: Sequence[FileAssessment],
    reference_frequency: float,
) -> str:
    width, height = 960, 420
    left, top, right, bottom = 72, 58, 28, 56
    plot_width = width - left - right
    plot_height = height - top - bottom
    timestamps = [item.representative_time or item.timestamp for item in assessments]
    x_values = [item.timestamp() for item in timestamps]
    y_values = [item.natural_frequency_hz for item in assessments]
    min_y = min(y_values) if y_values else 0.0
    max_y = max(y_values) if y_values else 1.0
    if math.isclose(min_y, max_y):
        min_y -= 0.1
        max_y += 0.1
    else:
        pad = (max_y - min_y) * 0.12
        min_y -= pad
        max_y += pad

    def scale_x(value: float) -> float:
        min_x = min(x_values) if x_values else 0.0
        max_x = max(x_values) if x_values else 1.0
        if math.isclose(min_x, max_x):
            max_x = min_x + 1.0
        return left + (value - min_x) / (max_x - min_x) * plot_width

    def scale_y(value: float) -> float:
        return top + plot_height - (value - min_y) / (max_y - min_y) * plot_height

    screen_points = [
        (scale_x(x_value), scale_y(y_value))
        for x_value, y_value in zip(x_values, y_values)
    ]
    smooth_path = _svg_smooth_path(screen_points)

    y_ticks = []
    y_tick_count = 3
    for index in range(y_tick_count):
        ratio = index / (y_tick_count - 1) if y_tick_count > 1 else 0.0
        value = max_y - (max_y - min_y) * ratio
        y_pos = top + plot_height * ratio
        y_ticks.append(
            f'<line x1="{left}" y1="{y_pos:.1f}" x2="{width - right}" y2="{y_pos:.1f}" '
            f'stroke="#d9e3ea" stroke-width="1" />'
            f'<text x="{left - 10}" y="{y_pos + 4:.1f}" text-anchor="end" '
            f'font-size="12" fill="#597184">{value:.2f}</text>'
        )

    x_ticks = []
    if timestamps:
        min_dt = min(timestamps)
        max_dt = max(timestamps)
        hour_start = min_dt.replace(minute=0, second=0, microsecond=0)
        hour_end = max_dt.replace(minute=0, second=0, microsecond=0)
        total_hours = max(1, int((hour_end - hour_start).total_seconds() // 3600) + 1)
        max_hour_labels = 8
        hour_step = max(1, math.ceil(total_hours / max_hour_labels))
        tick_time = hour_start
        while tick_time <= hour_end:
            x_pos = scale_x(tick_time.timestamp())
            tick_label = tick_time.strftime("%H:00")
            if min_dt.date() != max_dt.date():
                tick_label = tick_time.strftime("%m-%d %H:00")
            x_ticks.append(
                f'<line x1="{x_pos:.1f}" y1="{top}" x2="{x_pos:.1f}" y2="{top + plot_height}" '
                f'stroke="#eef3f7" stroke-width="1" />'
                f'<text x="{x_pos:.1f}" y="{height - 24}" text-anchor="middle" '
                f'font-size="11" fill="#597184">{html.escape(tick_label)}</text>'
            )
            tick_time += timedelta(hours=hour_step)

    points = []
    labels = []
    for item, (x_pos, y_pos) in zip(assessments, screen_points):
        if item.warning != "二级预警":
            continue
        points.append(
            f'<circle cx="{x_pos:.1f}" cy="{y_pos:.1f}" r="5.5" fill="#a12622" stroke="#ffffff" stroke-width="2" />'
        )
        labels.append(
            f'<text x="{x_pos:.1f}" y="{y_pos - 12:.1f}" text-anchor="middle" font-size="11" fill="#7f1d1d">'
            f'{item.natural_frequency_hz:.4f}</text>'
        )

    ref_y = scale_y(reference_frequency)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{left}" y="24" font-size="20" font-weight="700" fill="#1c2731">塔架频率随时间变化曲线</text>
  <text x="{width - right}" y="24" text-anchor="end" font-size="12" fill="#5d6b78">参考固有频率: {reference_frequency:.4f} Hz</text>
  <rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" rx="10" fill="#f9fbfc" stroke="#d8e2e8"/>
  {''.join(y_ticks)}
  {''.join(x_ticks)}
  <line x1="{left}" y1="{ref_y:.1f}" x2="{width - right}" y2="{ref_y:.1f}" stroke="#0e6b74" stroke-width="2" stroke-dasharray="6 5" />
  <path d="{smooth_path}" fill="none" stroke="#2d6cdf" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />
  {''.join(points)}
  {''.join(labels)}
  <text x="{left + plot_width / 2:.1f}" y="{height - 8}" text-anchor="middle" font-size="12" fill="#597184">时间</text>
  <text x="20" y="{top + plot_height / 2:.1f}" text-anchor="middle" font-size="12" fill="#597184" transform="rotate(-90 20 {top + plot_height / 2:.1f})">频率 (Hz)</text>
</svg>
"""
    return svg


def analyse_directory(
    base_dir: Path,
    include_files: Sequence[str] | None = None,
) -> tuple[list[FileAssessment], float]:
    file_paths: list[Path] = []
    seen: set[Path] = set()
    include_set = {str(name).strip() for name in (include_files or []) if str(name).strip()}
    for pattern in DEFAULT_PATTERNS:
        for path in sorted(base_dir.glob(pattern)):
            if path in seen:
                continue
            if include_set and path.name not in include_set:
                continue
            seen.add(path)
            file_paths.append(path)
    if not file_paths:
        raise FileNotFoundError(
            f"目录 {base_dir} 下未找到可分析文件（支持: {', '.join(DEFAULT_PATTERNS)}）"
        )

    assessments: list[FileAssessment] = []
    window_rows: list[dict[str, object]] = []
    skipped: list[str] = []
    for path in file_paths:
        try:
            assessment, windows = assess_file(path)
        except Exception:
            skipped.append(path.name)
            continue
        assessments.append(assessment)
        for window in windows:
            window_rows.append(
                {
                    "文件": assessment.file_name,
                    "时刻": (assessment.timestamp + timedelta(seconds=round(window.center_time))).strftime("%Y-%m-%d %H:%M:%S"),
                    "窗口中心秒": f"{window.center_time:.2f}",
                    "识别频率Hz": f"{window.frequency_hz:.4f}",
                    "置信度": f"{window.confidence:.2f}",
                    "使用方向": window.axis,
                }
            )
    if not assessments:
        raise FileNotFoundError(
            f"目录 {base_dir} 下未找到可分析的原始振动数据文件。已跳过: {', '.join(skipped) if skipped else '(none)'}"
        )

    assessments.sort(key=lambda item: item.timestamp)
    frequencies = [item.natural_frequency_hz for item in assessments]
    weights = [max(item.confidence, 0.01) for item in assessments]
    reference_frequency, _, _ = select_reference_cluster(frequencies, weights)

    start_time = assessments[0].timestamp
    elapsed_days: list[float] = []
    abs_deviations: list[float] = []
    for assessment in assessments:
        assessment.deviation_hz = assessment.natural_frequency_hz - reference_frequency
        assessment.deviation_pct = (
            abs(assessment.deviation_hz) / reference_frequency * 100.0
            if reference_frequency
            else 0.0
        )
        day_offset = (assessment.timestamp - start_time).total_seconds() / 86400.0
        elapsed_days.append(day_offset)
        abs_deviations.append(assessment.deviation_pct)
        assessment.trend_pct_per_day = max(
            0.0,
            linear_slope(elapsed_days, abs_deviations),
        )
        assessment.state, assessment.warning = classify_state(
            assessment.deviation_pct,
            assessment.trend_pct_per_day,
            assessment.confidence,
        )

    result_rows = []
    for assessment in assessments:
        result_rows.append(
            {
                "时刻": assessment.representative_time.strftime("%Y-%m-%d %H:%M:%S")
                if assessment.representative_time
                else assessment.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "当前状态": assessment.state,
                "识别频率Hz": f"{assessment.natural_frequency_hz:.4f}",
                "偏离程度": f"{assessment.deviation_pct:.2f}%",
                "预警": assessment.warning,
                "偏离量Hz": f"{assessment.deviation_hz:+.4f}",
                "趋势%每天": f"{assessment.trend_pct_per_day:.3f}",
                "聚类置信度": f"{assessment.confidence:.2f}",
                "有效窗口数": assessment.window_count,
            }
        )

    write_csv(
        base_dir / "tower_frequency_windows.csv",
        ["文件", "时刻", "窗口中心秒", "识别频率Hz", "置信度", "使用方向"],
        window_rows,
    )
    write_csv(
        base_dir / "tower_frequency_assessment.csv",
        [
            "时刻",
            "当前状态",
            "识别频率Hz",
            "偏离程度",
            "预警",
            "偏离量Hz",
            "趋势%每天",
            "聚类置信度",
            "有效窗口数",
        ],
        result_rows,
    )
    write_text(
        base_dir / "frequency_trend.svg",
        build_frequency_trend_svg(assessments, reference_frequency),
    )
    return assessments, reference_frequency


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    assessments, reference_frequency = analyse_directory(base_dir)
    print(f"现场参考固有频率: {reference_frequency:.4f} Hz")
    print("时刻, 当前状态, 识别频率Hz, 偏离程度, 预警")
    for assessment in assessments:
        print(
            f"{assessment.representative_time or assessment.timestamp:%Y-%m-%d %H:%M:%S}, "
            f"{assessment.state}, "
            f"{assessment.natural_frequency_hz:.4f}, "
            f"{assessment.deviation_pct:.2f}%, "
            f"{assessment.warning}"
        )


if __name__ == "__main__":
    main()
