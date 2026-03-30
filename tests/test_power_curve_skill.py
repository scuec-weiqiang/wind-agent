from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "skills" / "power-curve-assessment" / "scripts" / "power_curve.py"
SPEC = importlib.util.spec_from_file_location("test_power_curve_module", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_load_points_treats_missing_values_as_zero(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "TIME,10min平均有功功率,1#叶片变桨角度,10min平均风速,限功率运行状态\n"
        "20180106071059,,89.02,0.61,\n"
        "20180106071159,120.5,,5.20,1\n",
        encoding="utf-8",
    )

    points = MODULE.load_points(csv_path)

    assert len(points) == 2
    assert points[0].power_kw == 0.0
    assert points[1].pitch_deg == 0.0
    assert points[1].curtailed is True


def test_select_input_file_ignores_generated_csv_outputs(tmp_path: Path) -> None:
    (tmp_path / "power_curve_fit.csv").write_text("x,y\n", encoding="utf-8")
    (tmp_path / "power_curve_points.csv").write_text("x,y\n", encoding="utf-8")
    source = tmp_path / "turbine_a.csv"
    source.write_text(
        "TIME,10min平均有功功率,1#叶片变桨角度,10min平均风速,限功率运行状态\n"
        "20180106071059,0,89.02,0.61,\n",
        encoding="utf-8",
    )

    selected = MODULE.select_input_file(tmp_path)

    assert selected == source
