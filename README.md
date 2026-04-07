# Wind Agent

一个运行在 Linux 本地的通用 Agent 平台。

## 当前能力

- Linux 本地部署
- Web 前端聊天与运行观察
- CLI 入口
- 多步 agent loop
- session 持久化
- `runId` 和同 session 串行执行 lane
- planner / executor 基础链路
- 结构化 skill 结果协议
- workflow skill 衔接
- 报告生成 skill
- 运行时任务状态展示

## 当前定位

这个项目现在适合做：

- 本地工程助手
- 数据分析与诊断助手
- 报告生成助手
- 面向某个垂直场景的 Linux 侧 Agent Gateway

它现在还不是完整复刻 OpenClaw，但已经有了比较清晰的核心骨架：

- 通用 runtime
- skill 驱动的领域能力
- planner/executor
- 结构化 tool history
- 前端控制面雏形

## 架构概览

### 1. Runtime Core

核心运行时主要在这些文件里：

- `app/gateway_core.py`
- `app/agent_runtime.py`
- `app/session_store.py`
- `app/planner.py`

职责大致是：

- 接收用户请求
- 创建并跟踪 `runId`
- 维护 session 级串行执行
- 生成或复用 task plan
- 驱动多步 agent loop
- 调工具 / 调 skill
- 累积结构化证据
- 生成最终答复或报告
- 将运行状态同步到前端

### 2. Skill Layer

领域能力通过 `skills/` 下的技能包提供，不写死在 runtime 里。

每个 skill pack 至少包含：

- `SKILL.md`
- 可执行脚本，如 `scripts/run.py`

skill 现在支持结构化结果，runtime 可以继续利用 skill 的输出推进后续步骤，而不是只把 stdout 当成一段文本。

### 3. Web UI

前端在 `web/index.html`，当前已经支持：

- 聊天会话
- session 切换
- 文件上传
- skill 面板
- trace 面板
- 运行状态展示
- 自动折叠的调用链
- plan / current step / artifacts 展示

## 目录结构

```text
old_code/
├─ app/
│  ├─ agent_runtime.py      # runId、session lane、wait
│  ├─ chat.py               # 模型请求与流式聊天基础
│  ├─ cli.py                # CLI 入口
│  ├─ config.py             # runtime.json 到环境变量映射
│  ├─ gateway_core.py       # Agent runtime 核心
│  ├─ planner.py            # 任务规划
│  ├─ runtime_settings.py   # 运行时配置读取
│  ├─ server.py             # Flask Web 服务
│  ├─ session_store.py      # session / trace / runtime_state 持久化
│  └─ skill_manager.py      # skill 加载、解析、执行
├─ config/
│  ├─ runtime.example.json
│  └─ runtime.json
├─ skills/
│  ├─ report-writer/
│  ├─ time/
│  ├─ fft-frequency/
│  ├─ power-curve-assessment/
│  └─ ...
├─ tests/
├─ web/
│  └─ index.html
├─ main.py
└─ start.sh
```

## 启动

### Web

```bash
./start.sh web
```

### CLI

```bash
./start.sh cli
```

### 帮助

```bash
./start.sh --help
```

## 配置

默认会读取：

- `config/runtime.json`

也可以指定：

```bash
CONFIG_FILE=./config/runtime.prod.json ./start.sh web
```

建议先从模板开始：

```bash
cp config/runtime.example.json config/runtime.json
```

## DeepSeek / OpenAI Compatible 配置

如果你使用 DeepSeek 这类 OpenAI 兼容接口，推荐这样配置：

```json
{
  "model_provider": "openai_compatible",
  "model_name": "deepseek-chat",
  "model_base_url": "https://api.deepseek.com/chat/completions",
  "model_api_key": "YOUR_API_KEY",
  "thinking_mode": "auto",
  "system_prompt": "You are a helpful local agent coordinating between the user and available skills.",
  "enable_thinking": true,
  "shell_timeout": 0,
  "enable_skill_autorun": false
}
```

常用字段：

- `model_provider`
- `model_name`
- `model_base_url`
- `model_api_key`
- `thinking_mode`
- `system_prompt`
- `enable_thinking`
- `shell_timeout`
- `enable_skill_autorun`

注意：

- `model_base_url` 需要直接指向 chat completions 接口
- 请不要把真实 API key 提交到 git

## Runtime 能力

### Session

session 会持久化这些内容：

- 对话历史
- trace
- 附件绑定
- runtime state

runtime state 当前包含：

- `goal`
- `status`
- `stage`
- `current_step`
- `max_steps`
- `current_tool`
- `tool_count`
- `evidence`
- `artifacts`
- `plan_summary`
- `current_plan_step`

### Planner / Executor

当前执行链大致是：

1. 收到用户请求
2. 初始化或复用 runtime state
3. 生成或复用 task plan
4. 进入 agent loop
5. 按 plan step 和当前证据请求下一步 action
6. 执行 tool / skill
7. 将结构化结果写回 tool history 和 runtime state
8. 必要时继续下一步
9. 最终总结或调用报告 skill

### Tool History

tool history 已经开始统一化，每条记录都会尽量保留：

- `tool`
- `args`
- `result`
- `summary`
- `structured_data`

这让后续 step 可以真正基于证据继续推理，而不只是重新读一遍长文本。

## Skill 机制

### Skill Pack 结构

每个 skill 位于：

```text
skills/<skill-id>/
├─ SKILL.md
└─ scripts/
   └─ run.py
```

### SKILL.md

推荐使用类似 OpenClaw 的 frontmatter：

```md
---
name: report-writer
description: Generate a markdown report from structured analysis results.
metadata:
  structured-result: true
  workflow-hint: report
command: "{python} scripts/run.py --input '{input}' --session-id '{session_id}'"
---
```

### 结构化 skill 结果

skill 可以输出普通文本，也可以输出结构化结果。

推荐使用 envelope：

```json
{
  "kind": "openclaw_skill_result",
  "ok": true,
  "summary": "Report generated",
  "output_text": "# Report\n...",
  "data": {
    "report_path": "/tmp/report.md"
  }
}
```

这样 runtime 可以：

- 更新 evidence
- 提取 artifacts
- 让后续步骤继续利用结构化结果
- 更自然地衔接 workflow skill

## 前端界面

当前前端运行卡片里可以看到：

- 运行状态
- 阶段 / 步骤 / 当前工具 / 证据数量
- Plan
- Current Step
- Artifacts
- 调用链详情

调用链默认自动折叠，避免压过主回答；出错时会自动展开，便于排查。

## 接口概览

当前常用接口包括：

- `POST /chat`
- `GET /sessions`
- `POST /session/new`
- `POST /session/delete`
- `GET /session/history`
- `GET /session/trace`
- `GET /session/attachments`
- `GET /session/runtime`
- `GET /agent/wait`
- `GET /skills`
- `GET /skills/doc`
- `POST /skills/reload`
- `POST /skills/toggle`
- `POST /skills/execute`
- `GET /tools/list_dir`
- `GET /tools/read_file`
- `GET /tools/search_text`

## 测试

当前已经有一些针对关键 runtime 行为的回归测试：

- `tests/test_skill_resolution.py`
- `tests/test_skill_results.py`
- `tests/test_gateway_tool_payloads.py`
- `tests/test_gateway_workflow_hints.py`
- `tests/test_gateway_runtime_context.py`
- `tests/test_power_curve_skill.py`

可先做基础语法检查：

```bash
python -m py_compile app/*.py tests/test_gateway_runtime_context.py
```

如果本地环境装了 `pytest`，可以继续跑：

```bash
pytest tests
```

## 当前进展

目前这个项目已经完成了从旧式单体聊天后端到通用 Agent runtime 的第一阶段重构，重点包括：

- runtime / session / planner 分层
- planner 接入执行链
- skill 结构化结果协议
- report workflow skill
- 运行时 task state
- plan/current step/artifact 前端展示
- 自动折叠的调用链 UI

## 还缺什么

离一个更完整的 OpenClaw-like 体验，当前还缺的重点主要是：

- 更强的 step-driven executor
- 更稳定的中断 / 恢复 / 重试
- 更统一的 skill input/output schema
- 更完整的 artifact 操作和预览
- 更成熟的控制面 UI

## 适合的下一步

建议后续继续沿这个方向推进：

1. 强化 planner 对 executor 的直接约束
2. 统一 workflow skill 协议
3. 做 artifact 预览和导出
4. 增加 run 恢复 / 重试 / 中断
5. 继续收敛为“通用 runtime + skill 负责领域能力”
