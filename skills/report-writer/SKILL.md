---
skill_id: report-writer
name: 通用报告生成器
description: 将结构化分析结果、发现项和建议整理成 Markdown 报告，适用于诊断、巡检、分析总结等场景。
command: ["{python}", "scripts/run.py", "--input", "{input}", "--session-id", "{session_id}"]
aliases: ["report", "report-writer", "报告生成", "生成报告", "markdown-report"]
metadata: {"structured-result": true, "workflow-hint": "report"}
---

# 通用报告生成器

## When to use
- 用户要求生成报告、总结、分析结论、交付文档。
- 上一步 skill 已经产出结构化发现项，希望整理成统一格式文档。

## Input
- JSON：
```json
{
  "title": "日报",
  "summary": "总体情况正常",
  "findings": ["发现 1", "发现 2"],
  "recommendations": ["建议 1", "建议 2"],
  "artifacts": ["reports/a.csv", "plots/b.svg"]
}
```

## Output
- 文本摘要
- 结构化结果
- 生成 Markdown 报告文件
