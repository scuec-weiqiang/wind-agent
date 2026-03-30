from __future__ import annotations

import csv
import html
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Sequence

TIME_COLUMN = "TIME"
POWER_COLUMN = "10min平均有功功率"
WIND_COLUMN = "10min平均风速"
PITCH_COLUMN = "1#叶片变桨角度"
CURTAIL_COLUMN = "限功率运行状态"

ENCODINGS = ("utf-8-sig", "utf-8", "gbk")
GENERATED_OUTPUTS = {
    "power_curve_points.csv",
    "power_curve_fit.csv",
    "power_curve_assessment.csv",
}

BIN_WIDTH = 0.1
REPORT_SVG_POINT_LIMIT = 6000
MAX_VALID_WIND_SPEED = 20.0
MIN_BIN_SAMPLES = 8
LOW_BAND_QUANTILE = 0.15
HIGH_BAND_QUANTILE = 0.85
LOW_POWER_FLOOR_KW = 80.0
HIGH_POWER_FLOOR_KW = 120.0
STOP_PITCH_DEG = 80.0
STOP_POWER_KW = 50.0
SLIDE_MIN_PREFIX = 6
SLIDE_JUMP_RATIO = 1.35
SLIDE_MEDIAN_GAIN_RATIO = 0.10


@dataclass
class PowerCurvePoint:
    timestamp: datetime | None
    wind_speed: float
    power_kw: float
    pitch_deg: float
    curtailed: bool


@dataclass
class BinStats:
    wind_speed: float
    median_power_kw: float
    lower_power_kw: float
    upper_power_kw: float
    sample_count: int
    raw_count: int


@dataclass
class CurveSample:
    wind_speed: float
    power_kw: float


@dataclass
class AssessmentRow:
    timestamp_label: str
    state: str
    score: int
    warning: str
    advice: str
    outside_ratio: float
    below_ratio: float


@dataclass
class PowerCurveSummary:
    total_points: int
    modeling_points: int
    fit_points: int
    wind_min: float
    wind_max: float
    power_min: float
    power_max: float
    outside_count: int
    zero_power_count: int
    source_name: str
    assessment: AssessmentRow


def _parse_timestamp(value: str) -> datetime | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        return datetime.strptime(text, "%Y%m%d%H%M%S")
    except ValueError:
        return None


def _parse_float(value: str) -> float:
    text = (value or "").strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _read_rows(path: Path) -> list[dict[str, str]]:
    last_error: Exception | None = None
    for encoding in ENCODINGS:
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                return list(csv.DictReader(handle))
        except UnicodeDecodeError as exc:
            last_error = exc
    raise RuntimeError(f"{path.name} 无法解码") from last_error


def _is_missing_required(row: dict[str, str]) -> bool:
    return any(not (row.get(column, "") or "").strip() for column in (TIME_COLUMN, WIND_COLUMN, POWER_COLUMN))


def load_points(path: Path) -> list[PowerCurvePoint]:
    rows = _read_rows(path)
    if not rows:
        raise ValueError(f"{path.name} 没有有效数据")

    points: list[PowerCurvePoint] = []
    for row in rows:
        if _is_missing_required(row):
            continue
        wind_speed = _parse_float(row.get(WIND_COLUMN, ""))
        power_kw = _parse_float(row.get(POWER_COLUMN, ""))
        pitch_deg = _parse_float(row.get(PITCH_COLUMN, ""))
        if wind_speed < 0.0 or wind_speed > MAX_VALID_WIND_SPEED or power_kw < 0.0:
            continue
        curtailed_value = (row.get(CURTAIL_COLUMN, "") or "").strip()
        curtailed = bool(curtailed_value and curtailed_value not in {"0", "0.0", "false", "False", "否"})
        points.append(
            PowerCurvePoint(
                timestamp=_parse_timestamp(row.get(TIME_COLUMN, "")),
                wind_speed=wind_speed,
                power_kw=power_kw,
                pitch_deg=pitch_deg,
                curtailed=curtailed,
            )
        )
    if not points:
        raise ValueError(f"{path.name} 清洗后无有效样本")
    return points


def _is_modeling_point(point: PowerCurvePoint) -> bool:
    if point.wind_speed < 0.0 or point.wind_speed > MAX_VALID_WIND_SPEED:
        return False
    if point.power_kw < 0.0:
        return False
    if point.pitch_deg >= STOP_PITCH_DEG and point.power_kw <= STOP_POWER_KW:
        return False
    if point.wind_speed < 2.0 and point.power_kw > 300.0:
        return False
    return True


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    index = max(0.0, min(len(ordered) - 1.0, q * (len(ordered) - 1)))
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _remove_bin_outliers(values: Sequence[float]) -> list[float]:
    if len(values) < 4:
        return list(values)
    descending = sorted(values, reverse=True)
    prefix_stds: list[float] = [0.0]
    prefix_sum = 0.0
    prefix_sum_sq = 0.0
    for index, value in enumerate(descending, start=1):
        prefix_sum += value
        prefix_sum_sq += value * value
        if index == 1:
            continue
        subset_mean = prefix_sum / index
        variance = max(0.0, prefix_sum_sq / index - subset_mean * subset_mean)
        prefix_stds.append(math.sqrt(variance))

    best_cut = len(descending)
    best_jump = 0.0
    for index in range(SLIDE_MIN_PREFIX, len(prefix_stds) - 1):
        current_std = prefix_stds[index]
        next_std = prefix_stds[index + 1]
        if current_std <= 0.0:
            continue
        jump_ratio = next_std / current_std
        if jump_ratio > SLIDE_JUMP_RATIO and jump_ratio > best_jump:
            best_jump = jump_ratio
            best_cut = index

    slide_filtered = descending[:best_cut]
    full_median = median(descending)
    filtered_median = median(slide_filtered) if slide_filtered else full_median
    if best_cut < len(descending) and filtered_median >= full_median * (1.0 + SLIDE_MEDIAN_GAIN_RATIO):
        candidates = slide_filtered
    else:
        candidates = descending

    q1 = _quantile(candidates, 0.25)
    q3 = _quantile(candidates, 0.75)
    iqr = q3 - q1
    if math.isclose(iqr, 0.0):
        return list(candidates)
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    filtered = [value for value in candidates if lower <= value <= upper]
    return filtered or list(candidates)


def _smooth_stats(stats: Sequence[BinStats]) -> list[BinStats]:
    if len(stats) <= 2:
        return list(stats)
    smoothed: list[BinStats] = []
    for index, item in enumerate(stats):
        start = max(0, index - 1)
        end = min(len(stats), index + 2)
        neighbors = stats[start:end]
        total_weight = sum(neighbor.sample_count for neighbor in neighbors) or 1
        smoothed.append(
            BinStats(
                wind_speed=item.wind_speed,
                median_power_kw=sum(neighbor.median_power_kw * neighbor.sample_count for neighbor in neighbors) / total_weight,
                lower_power_kw=sum(neighbor.lower_power_kw * neighbor.sample_count for neighbor in neighbors) / total_weight,
                upper_power_kw=sum(neighbor.upper_power_kw * neighbor.sample_count for neighbor in neighbors) / total_weight,
                sample_count=item.sample_count,
                raw_count=item.raw_count,
            )
        )
    return smoothed


def build_bin_statistics(points: Sequence[PowerCurvePoint]) -> list[BinStats]:
    grouped: dict[float, list[float]] = {}
    raw_counts: dict[float, int] = {}
    for point in points:
        if not _is_modeling_point(point):
            continue
        wind_bin = round(round(point.wind_speed / BIN_WIDTH) * BIN_WIDTH, 2)
        grouped.setdefault(wind_bin, []).append(point.power_kw)
        raw_counts[wind_bin] = raw_counts.get(wind_bin, 0) + 1

    stats: list[BinStats] = []
    for wind_bin, powers in sorted(grouped.items()):
        if len(powers) < MIN_BIN_SAMPLES:
            continue
        filtered = _remove_bin_outliers(powers)
        stats.append(
            BinStats(
                wind_speed=wind_bin,
                median_power_kw=median(filtered),
                lower_power_kw=_quantile(filtered, LOW_BAND_QUANTILE),
                upper_power_kw=_quantile(filtered, HIGH_BAND_QUANTILE),
                sample_count=len(filtered),
                raw_count=raw_counts.get(wind_bin, len(powers)),
            )
        )
    return _smooth_stats(stats)


def _interpolate_band(stats: Sequence[BinStats], wind_speed: float) -> tuple[float, float, float]:
    if not stats:
        return 0.0, 0.0, 0.0
    if wind_speed <= stats[0].wind_speed:
        first = stats[0]
        return first.median_power_kw, first.lower_power_kw, first.upper_power_kw
    if wind_speed >= stats[-1].wind_speed:
        last = stats[-1]
        return last.median_power_kw, last.lower_power_kw, last.upper_power_kw
    for left, right in zip(stats, stats[1:]):
        if left.wind_speed <= wind_speed <= right.wind_speed:
            span = right.wind_speed - left.wind_speed
            if math.isclose(span, 0.0):
                return right.median_power_kw, right.lower_power_kw, right.upper_power_kw
            ratio = (wind_speed - left.wind_speed) / span
            median_power = left.median_power_kw + ratio * (right.median_power_kw - left.median_power_kw)
            lower_power = left.lower_power_kw + ratio * (right.lower_power_kw - left.lower_power_kw)
            upper_power = left.upper_power_kw + ratio * (right.upper_power_kw - left.upper_power_kw)
            return median_power, lower_power, upper_power
    last = stats[-1]
    return last.median_power_kw, last.lower_power_kw, last.upper_power_kw


def _point_status(point: PowerCurvePoint, stats: Sequence[BinStats]) -> str:
    median_power, lower_power, upper_power = _interpolate_band(stats, point.wind_speed)
    if median_power <= 0.0:
        return "未评估"
    if point.pitch_deg >= STOP_PITCH_DEG and point.power_kw <= STOP_POWER_KW:
        return "停机"
    if lower_power <= point.power_kw <= upper_power:
        return "有效"
    if point.power_kw < lower_power - max(LOW_POWER_FLOOR_KW, median_power * 0.10):
        return "偏低"
    if point.power_kw > upper_power + max(HIGH_POWER_FLOOR_KW, median_power * 0.10):
        return "偏高"
    return "偏离"


def build_fit_curve(stats: Sequence[BinStats]) -> list[BinStats]:
    return list(stats)


def _sigmoid_power(wind_speed: float, lower: float, upper: float, slope: float, midpoint: float) -> float:
    exponent = max(-60.0, min(60.0, -slope * (wind_speed - midpoint)))
    return lower + (upper - lower) / (1.0 + math.exp(exponent))


def _fit_smooth_curve(stats: Sequence[BinStats]) -> list[CurveSample]:
    if not stats:
        return []

    x_values = [item.wind_speed for item in stats]
    y_values = [item.median_power_kw for item in stats]

    lower = max(0.0, min(y_values))
    upper = max(y_values)
    half_power = lower + (upper - lower) * 0.5
    midpoint = min(x_values, key=lambda value: abs(_interpolate_band(stats, value)[0] - half_power))
    slope = 0.8

    def loss(params: tuple[float, float, float, float]) -> float:
        a, d, b, c = params
        if d <= a or b <= 0.0:
            return float("inf")
        total = 0.0
        for x_value, y_value in zip(x_values, y_values):
            diff = _sigmoid_power(x_value, a, d, b, c) - y_value
            total += diff * diff
        return total

    params = (lower, upper, slope, midpoint)
    step_scales = [
        (120.0, 120.0, 0.35, 1.0),
        (60.0, 60.0, 0.18, 0.5),
        (20.0, 20.0, 0.08, 0.2),
        (8.0, 8.0, 0.03, 0.1),
    ]

    best_loss = loss(params)
    for lower_step, upper_step, slope_step, midpoint_step in step_scales:
        improved = True
        while improved:
            improved = False
            candidates = []
            for delta_lower in (-lower_step, 0.0, lower_step):
                for delta_upper in (-upper_step, 0.0, upper_step):
                    for delta_slope in (-slope_step, 0.0, slope_step):
                        for delta_mid in (-midpoint_step, 0.0, midpoint_step):
                            if delta_lower == delta_upper == delta_slope == delta_mid == 0.0:
                                continue
                            candidates.append(
                                (
                                    max(0.0, params[0] + delta_lower),
                                    max(params[1] + delta_upper, params[0] + 1.0),
                                    max(0.02, params[2] + delta_slope),
                                    params[3] + delta_mid,
                                )
                            )
            for candidate in candidates:
                candidate_loss = loss(candidate)
                if candidate_loss + 1e-9 < best_loss:
                    params = candidate
                    best_loss = candidate_loss
                    improved = True

    sample_count = max(120, len(stats) * 4)
    start = min(x_values)
    end = max(x_values)
    if math.isclose(start, end):
        return [CurveSample(wind_speed=start, power_kw=params[1])]
    samples: list[CurveSample] = []
    for index in range(sample_count + 1):
        wind_speed = start + (end - start) * index / sample_count
        samples.append(
            CurveSample(
                wind_speed=wind_speed,
                power_kw=_sigmoid_power(wind_speed, *params),
            )
        )
    return samples


def _state_from_score(score: int) -> str:
    if score >= 85:
        return "良好"
    if score >= 70:
        return "正常"
    return "异常"


def _advice_from_distribution(low_count: int, high_count: int, total_outside: int) -> str:
    if total_outside == 0:
        return "维持常规监测，建议按周期复核风速仪、风向仪。"
    if low_count >= high_count * 1.5:
        return "检查风速仪、风向仪、偏航对风及叶片表面状态，必要时复核变桨控制。"
    if high_count > low_count:
        return "检查传感器标定、功率测量链路及控制参数，排查异常高估或采集偏差。"
    return "检查叶片、变桨系统、偏航系统及传动链，结合现场数据进一步诊断。"


def assess_health(points: Sequence[PowerCurvePoint], stats: Sequence[BinStats], source_name: str) -> AssessmentRow:
    evaluated = 0
    outside = 0
    below = 0
    low_region_below = 0
    high_region_below = 0
    for point in points:
        median_power, lower_power, upper_power = _interpolate_band(stats, point.wind_speed)
        if median_power <= 0.0:
            continue
        evaluated += 1
        if point.power_kw < lower_power:
            outside += 1
            below += 1
            if point.wind_speed < 8.0:
                low_region_below += 1
            else:
                high_region_below += 1
        elif point.power_kw > upper_power:
            outside += 1

    outside_ratio = outside / evaluated if evaluated else 0.0
    below_ratio = below / evaluated if evaluated else 0.0
    score = max(0, round(100 - outside_ratio * 60 - below_ratio * 30))
    state = _state_from_score(score)
    warning = "开启" if score < 70 or outside_ratio >= 0.20 else "关闭"
    advice = _advice_from_distribution(low_region_below, high_region_below, outside)
    label = next((point.timestamp.strftime("%Y%m") for point in points if point.timestamp), source_name[:6])
    return AssessmentRow(
        timestamp_label=label,
        state=state,
        score=score,
        warning=warning,
        advice=advice,
        outside_ratio=outside_ratio,
        below_ratio=below_ratio,
    )


def summarize(points: Sequence[PowerCurvePoint], fit_points: Sequence[BinStats], source_name: str) -> PowerCurveSummary:
    wind_values = [point.wind_speed for point in points] or [0.0]
    power_values = [point.power_kw for point in points] or [0.0]
    modeling_points = sum(1 for point in points if _is_modeling_point(point))
    zero_power_count = sum(1 for point in points if point.power_kw <= 0.0)
    assessment = assess_health(points, fit_points, source_name)
    return PowerCurveSummary(
        total_points=len(points),
        modeling_points=modeling_points,
        fit_points=len(fit_points),
        wind_min=min(wind_values),
        wind_max=max(wind_values),
        power_min=min(power_values),
        power_max=max(power_values),
        outside_count=round(assessment.outside_ratio * len(points)),
        zero_power_count=zero_power_count,
        source_name=source_name,
        assessment=assessment,
    )


def _nice_ticks(min_value: float, max_value: float, count: int = 5) -> list[float]:
    if math.isclose(min_value, max_value):
        return [min_value]
    span = max_value - min_value
    raw_step = span / max(count - 1, 1)
    magnitude = 10 ** math.floor(math.log10(raw_step)) if raw_step > 0 else 1
    normalized = raw_step / magnitude
    if normalized <= 1:
        step = 1 * magnitude
    elif normalized <= 2:
        step = 2 * magnitude
    elif normalized <= 5:
        step = 5 * magnitude
    else:
        step = 10 * magnitude
    start = math.floor(min_value / step) * step
    end = math.ceil(max_value / step) * step
    ticks: list[float] = []
    value = start
    while value <= end + step * 0.5:
        ticks.append(round(value, 6))
        value += step
    return ticks


def render_power_curve_svg(
    points: Sequence[PowerCurvePoint],
    fit_points: Sequence[BinStats],
    scatter_limit: int | None = None,
) -> str:
    width = 980
    height = 560
    left = 86
    right = 40
    top = 42
    bottom = 64
    inner_width = width - left - right
    inner_height = height - top - bottom

    wind_values = [point.wind_speed for point in points] or [0.0]
    power_values = [point.power_kw for point in points] or [0.0]
    x_min = 0.0
    x_max = max(max(wind_values), max((point.wind_speed for point in fit_points), default=0.0), 1.0)
    x_max = math.ceil(x_max)
    y_min = 0.0
    y_max_raw = max(max(power_values), max((point.upper_power_kw for point in fit_points), default=0.0), 1.0)
    y_max = max(1.0, math.ceil(y_max_raw / 100.0) * 100.0)

    def x_pos(value: float) -> float:
        if math.isclose(x_max, x_min):
            return left
        return left + (value - x_min) / (x_max - x_min) * inner_width

    def y_pos(value: float) -> float:
        if math.isclose(y_max, y_min):
            return top + inner_height
        return top + inner_height - (value - y_min) / (y_max - y_min) * inner_height

    x_ticks = _nice_ticks(x_min, x_max, count=6)
    y_ticks = _nice_ticks(y_min, y_max, count=6)
    smooth_curve = _fit_smooth_curve(fit_points)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs>',
        '<linearGradient id="fitLine" x1="0" y1="0" x2="1" y2="0">',
        '<stop offset="0%" stop-color="#df4a43" />',
        '<stop offset="100%" stop-color="#a91f28" />',
        '</linearGradient>',
        '<linearGradient id="bandFill" x1="0" y1="0" x2="0" y2="1">',
        '<stop offset="0%" stop-color="#f8d8d8" stop-opacity="0.65" />',
        '<stop offset="100%" stop-color="#f8d8d8" stop-opacity="0.18" />',
        '</linearGradient>',
        '</defs>',
        '<rect width="100%" height="100%" fill="#ffffff" rx="18" />',
        f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" font-size="22" font-family="Microsoft YaHei, sans-serif" font-weight="700" fill="#1f2d3a">风电机组功率曲线自主评估</text>',
    ]

    for tick in y_ticks:
        y = y_pos(tick)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="#d9e5ea" stroke-width="1" />')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="12" fill="#5e6e79">{tick:.0f}</text>')

    for tick in x_ticks:
        x = x_pos(tick)
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + inner_height}" stroke="#eef4f6" stroke-width="1" />')
        parts.append(f'<text x="{x:.2f}" y="{height - 24}" text-anchor="middle" font-size="12" fill="#5e6e79">{tick:.1f}</text>')

    parts.append(f'<rect x="{left}" y="{top}" width="{inner_width}" height="{inner_height}" fill="none" stroke="#9db2bc" stroke-width="1.2" rx="12" />')

    if fit_points:
        upper_path: list[str] = []
        lower_path: list[str] = []
        median_path: list[str] = []
        for index, stat in enumerate(fit_points):
            prefix = "M" if index == 0 else "L"
            upper_path.append(f'{prefix}{x_pos(stat.wind_speed):.2f},{y_pos(stat.upper_power_kw):.2f}')
            lower_path.append(f'{prefix}{x_pos(stat.wind_speed):.2f},{y_pos(stat.lower_power_kw):.2f}')
            median_path.append(f'{prefix}{x_pos(stat.wind_speed):.2f},{y_pos(stat.median_power_kw):.2f}')
        band_polygon = " ".join(upper_path + [segment.replace("M", "L", 1) for segment in reversed(lower_path)])
        parts.append(f'<path d="{band_polygon}" fill="url(#bandFill)" stroke="none" />')

    scatter_points = list(points)
    if scatter_limit is not None and len(scatter_points) > scatter_limit:
        step = max(1, len(scatter_points) // scatter_limit)
        scatter_points = scatter_points[::step]
        if scatter_points[-1] != points[-1]:
            scatter_points.append(points[-1])

    for point in scatter_points:
        status = _point_status(point, fit_points)
        if status == "有效":
            fill = "#48d46b"
            opacity = "0.46"
            radius = "2.0"
        elif status == "偏低":
            fill = "#b560c7"
            opacity = "0.24"
            radius = "1.5"
        elif status == "偏高":
            fill = "#ef8f32"
            opacity = "0.20"
            radius = "1.5"
        else:
            fill = "#2aa7b3"
            opacity = "0.26"
            radius = "1.7"
        parts.append(
            f'<circle cx="{x_pos(point.wind_speed):.2f}" cy="{y_pos(point.power_kw):.2f}" r="{radius}" fill="{fill}" fill-opacity="{opacity}" />'
        )

    if smooth_curve:
        path_commands = [
            f'{"M" if index == 0 else "L"}{x_pos(stat.wind_speed):.2f},{y_pos(stat.power_kw):.2f}'
            for index, stat in enumerate(smooth_curve)
        ]
        parts.append(
            f'<path d="{" ".join(path_commands)}" fill="none" stroke="url(#fitLine)" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" />'
        )

    parts.extend(
        [
            f'<text x="{width / 2:.1f}" y="{height - 8}" text-anchor="middle" font-size="13" fill="#385160">10min平均风速 (m/s)</text>',
            f'<text x="22" y="{height / 2:.1f}" text-anchor="middle" font-size="13" fill="#385160" transform="rotate(-90 22 {height / 2:.1f})">10min平均有功功率 (kW)</text>',
            '<circle cx="620" cy="30" r="4" fill="#48d46b" fill-opacity="0.58" />',
            '<text x="632" y="34" font-size="12" fill="#5e6e79">主密集有效点</text>',
            '<circle cx="760" cy="30" r="3" fill="#b560c7" fill-opacity="0.32" />',
            '<text x="772" y="34" font-size="12" fill="#5e6e79">偏低异常点</text>',
            '<line x1="872" y1="30" x2="908" y2="30" stroke="url(#fitLine)" stroke-width="4" stroke-linecap="round" />',
            '<text x="918" y="34" font-size="12" fill="#5e6e79">中值功率曲线</text>',
            '</svg>',
        ]
    )
    return "".join(parts)


def write_points_csv(path: Path, points: Sequence[PowerCurvePoint], fit_points: Sequence[BinStats]) -> Path:
    output_path = path / "power_curve_points.csv"
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["时刻", "风速(m/s)", "功率(kW)", "桨距角(deg)", "状态判定"])
        for point in points:
            writer.writerow([
                point.timestamp.strftime("%Y-%m-%d %H:%M:%S") if point.timestamp else "",
                f"{point.wind_speed:.2f}",
                f"{point.power_kw:.2f}",
                f"{point.pitch_deg:.2f}",
                _point_status(point, fit_points),
            ])
    return output_path


def write_fit_csv(path: Path, fit_points: Sequence[BinStats]) -> Path:
    output_path = path / "power_curve_fit.csv"
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["风速中心(m/s)", "中值功率(kW)", "下边缘(kW)", "上边缘(kW)", "样本数", "原始样本数"])
        for point in fit_points:
            writer.writerow([
                f"{point.wind_speed:.2f}",
                f"{point.median_power_kw:.2f}",
                f"{point.lower_power_kw:.2f}",
                f"{point.upper_power_kw:.2f}",
                point.sample_count,
                point.raw_count,
            ])
    return output_path


def write_assessment_csv(path: Path, assessment: AssessmentRow) -> Path:
    output_path = path / "power_curve_assessment.csv"
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["时刻", "当前状态", "健康状态评分", "预警", "检修建议"])
        writer.writerow([
            assessment.timestamp_label,
            assessment.state,
            assessment.score,
            assessment.warning,
            assessment.advice,
        ])
    return output_path


def write_svg(path: Path, svg_text: str) -> Path:
    output_path = path / "power_curve.svg"
    output_path.write_text(svg_text, encoding="utf-8")
    return output_path


def _fit_rows_html(fit_points: Sequence[BinStats], limit: int = 24) -> str:
    if not fit_points:
        return '<tr><td colspan="5">未能生成有效功率曲线。</td></tr>'
    rows = []
    for point in fit_points[:limit]:
        rows.append(
            "<tr>"
            f"<td>{point.wind_speed:.2f}</td>"
            f"<td>{point.median_power_kw:.2f}</td>"
            f"<td>{point.lower_power_kw:.2f}</td>"
            f"<td>{point.upper_power_kw:.2f}</td>"
            f"<td>{point.sample_count}</td>"
            "</tr>"
        )
    if len(fit_points) > limit:
        rows.append(f'<tr><td colspan="5">分仓较多，报告仅展示前 {limit} 行，完整结果请查看 power_curve_fit.csv。</td></tr>')
    return "\n".join(rows)


def _render_template(template: str, mapping: dict[str, str]) -> str:
    output = template
    for key, value in mapping.items():
        output = output.replace("{{ " + key + " }}", value)
    return output


def write_report(path: Path, template_path: Path, summary: PowerCurveSummary, fit_points: Sequence[BinStats], svg_text: str) -> Path:
    template = template_path.read_text(encoding="utf-8")
    html_text = _render_template(
        template,
        {
            "report_title": "风电机组功率曲线自主评估报告",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_name": html.escape(summary.source_name),
            "point_count": str(summary.total_points),
            "modeling_point_count": str(summary.modeling_points),
            "wind_range": f"{summary.wind_min:.2f} ~ {summary.wind_max:.2f} m/s",
            "power_range": f"{summary.power_min:.2f} ~ {summary.power_max:.2f} kW",
            "fit_count": str(summary.fit_points),
            "state": summary.assessment.state,
            "health_score": str(summary.assessment.score),
            "warning": summary.assessment.warning,
            "advice": html.escape(summary.assessment.advice),
            "outside_ratio": f"{summary.assessment.outside_ratio * 100:.2f}%",
            "below_ratio": f"{summary.assessment.below_ratio * 100:.2f}%",
            "power_curve_svg": svg_text,
            "fit_rows_html": _fit_rows_html(fit_points),
        },
    )
    output_path = path / "power_curve_report.html"
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


def analyse_file(csv_path: Path, output_dir: Path | None = None) -> tuple[PowerCurveSummary, Path, Path, Path, Path]:
    points = load_points(csv_path)
    fit_points = build_fit_curve(build_bin_statistics(points))
    summary = summarize(points, fit_points, csv_path.name)
    target_dir = output_dir or csv_path.parent
    points_csv = write_points_csv(target_dir, points, fit_points)
    fit_csv = write_fit_csv(target_dir, fit_points)
    assessment_csv = write_assessment_csv(target_dir, summary.assessment)
    svg_text = render_power_curve_svg(points, fit_points)
    svg_path = write_svg(target_dir, svg_text)
    report_svg_text = render_power_curve_svg(points, fit_points, scatter_limit=REPORT_SVG_POINT_LIMIT)
    template_path = Path(__file__).resolve().parents[1] / "report_template.html"
    report_path = write_report(target_dir, template_path, summary, fit_points, report_svg_text)
    return summary, points_csv, fit_csv, assessment_csv, svg_path, report_path


def select_input_file(path: Path) -> Path:
    if path.is_file():
        return path
    csv_files = sorted(
        candidate
        for candidate in path.glob("*.csv")
        if candidate.is_file() and candidate.name not in GENERATED_OUTPUTS
    )
    if not csv_files:
        raise FileNotFoundError(f"未在 {path} 中找到可分析的 csv 文件")
    return csv_files[0]
