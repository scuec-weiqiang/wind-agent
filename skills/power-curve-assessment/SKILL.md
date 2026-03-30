---
skill_id: power-curve-assessment
name: 功率曲线自主评估
description: 基于 csv 风电运行数据生成风速-功率散点图、拟合功率曲线和 HTML 报告，适合批量评估功率曲线偏差与运行状态。
command: ["{python}", "scripts/run.py", "--input", "{input}", "--session-id", "{session_id}"]
aliases: ["power-curve", "powercurve", "功率曲线", "功率曲线分析", "功率曲线自主评估"]
---

# 功率曲线自主评估

## When to use
- 用户希望基于风电机组运行 csv 数据绘制功率曲线。
- 用户需要查看风速与有功功率的散点关系，并给出拟合后的参考曲线。
- 用户希望输出图片、结果表格或 HTML 报告。

## Input
- 单个 csv 文件路径：`/path/to/file.csv`
- 目录路径：`/path/to/data_dir`
- JSON：
```json
{"path": "/path/to/file_or_dir.csv"}
```
- 自然语言中包含路径时也会自动提取。

## Output
- 文本摘要
- 生成文件：
  - `power_curve_points.csv`
  - `power_curve_fit.csv`
  - `power_curve_assessment.csv`
  - `power_curve.svg`
  - `power_curve_report.html`
