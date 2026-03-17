---
name: shell
description: 执行本地终端命令，用于文件创建、编译、目录查看等本地操作。
---

# shell

## When to use
- 用户希望在本地执行终端任务（如创建文件、编译运行、查看目录）。
- 需要调用系统命令获取真实输出。

## Usage
```bash
{python} scripts/run.py --input '{input}' --session-id '{session_id}'
```

- 输入应为可执行命令。
- 支持完整 shell 语法（如管道、重定向、逻辑运算符）。
- 默认不限白名单，按系统权限直接执行。
- `SHELL_TIMEOUT` 大于 0 时启用超时；`0` 表示不限时。
