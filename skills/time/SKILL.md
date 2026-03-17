---
name: time
description: 获取当前本地时间，支持自定义时间格式。
---

# time

## When to use
- 用户询问当前时间、日期、星期。
- 需要按指定格式返回本机时间字符串。

## Usage
```bash
{python} scripts/run.py --input '{input}' --session-id '{session_id}'
```

- 输入可选格式字符串（例如 `%Y-%m-%d %H:%M:%S`）。
- 留空则使用默认格式。
