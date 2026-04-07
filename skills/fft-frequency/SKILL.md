---
skill_id: fft-frequency
name: fft 风机塔台振动分析
description: 基于 skill 内置的 scripts/fft.py 对风机塔台振动 csv 数据做主频识别、偏离趋势与预警分析。
command: ["{python}", "scripts/run.py", "--input", "{input}", "--session-id", "{session_id}"]
aliases: ["fft", "tower-fft", "vibration-fft", "风机频率分析", "振动频率分析"]
metadata: {"structured-result": true, "workflow-hint": "analysis"}
---

# FFT 风机塔台振动分析

## When to use
- 用户要求分析风机塔台振动频率、主频、固有频率偏离、趋势与预警。
- 用户希望批量处理某个目录下的 `*data.csv` 并生成评估结果。

## Input
- 字符串路径：`/path/to/data_dir`
- 或 JSON：
```json
{"path": "/path/to/data_dir"}
```
- 或自然语言（会自动提取其中的目录路径）：
  - `帮我分析 /home/w/wind-agent 的风机振动数据`
  - `请用 fft 看一下 /data/tower 这批数据`

默认目录为项目根目录。

## Output
- 文本摘要（现场参考固有频率 + 每文件状态）
- 生成文件：
  - `tower_frequency_windows.csv`
  - `tower_frequency_assessment.csv`
  - `frequency_trend.svg`
  - `tower_frequency_report.html`
