---
name: echo
description: 原样回显输入内容，适合测试技能链路是否工作。
---

# echo

## When to use
- 需要验证 skill 调用是否成功。
- 用户明确要求“原样输出”。

## Usage
```bash
{python} scripts/run.py --input '{input}' --session-id '{session_id}'
```
