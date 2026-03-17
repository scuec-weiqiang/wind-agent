# wind-agent

## 目录结构

```text
wind-agent/
├─ app/            # Python 后端与 CLI
│  ├─ chat.py
│  ├─ cli.py
│  └─ server.py
├─ web/            # Web 前端
│  └─ index.html
├─ app/
│  ├─ skill_manager.py # Skill 加载与执行
├─ skills/         # OpenClaw 风格技能包
│  ├─ time/
│  │  ├─ SKILL.md
│  │  └─ scripts/run.py
│  ├─ echo/
│  └─ shell/
├─ main.py         # 统一入口
└─ start.sh        # 启动脚本
```

## 启动

```bash
./start.sh web
./start.sh cli
```

也可直接：

```bash
/home/w/miniconda3/envs/wind_agent/bin/python main.py web
/home/w/miniconda3/envs/wind_agent/bin/python main.py cli
```

## 配置文件选择

现在支持使用配置文件管理运行参数。默认读取：

- `config/runtime.json`（存在时自动生效）

你也可以指定其他配置文件：

```bash
CONFIG_FILE=./config/runtime.prod.json ./start.sh web
```

配置模板见：

- `config/runtime.example.json`

建议复制后使用：

```bash
cp config/runtime.example.json config/runtime.json
```

## 多厂商模型 API

默认使用本地 Ollama。你也可以切换到 OpenAI 兼容协议（多数厂商支持）：

```bash
MODEL_PROVIDER=openai_compatible \
MODEL_NAME=你的模型名 \
MODEL_BASE_URL=你的兼容接口地址 \
MODEL_API_KEY=你的密钥 \
./start.sh web
```

可选的思考模式开关：

```bash
THINKING_MODE=auto   # auto | on | off
```

- `ollama`：映射到请求里的 `think`
- `deepseek-chat` / `deepseek-reasoner`：优先通过 `thinking` 参数切换，并在这两个模型名之间自动映射

例如（OpenAI 官方）：

```bash
MODEL_PROVIDER=openai_compatible \
MODEL_NAME=gpt-4o-mini \
MODEL_BASE_URL=https://api.openai.com/v1/chat/completions \
MODEL_API_KEY=sk-xxxx \
./start.sh web
```

## Skill Pack 模式

当前技能系统已切换为类似 OpenClaw 的 pack 结构：

- `skills/<your_skill>/` 就是一个 skill 目录
- 每个技能目录至少包含 `SKILL.md`
- `SKILL.md` 优先使用 OpenClaw 风格 frontmatter：`name`、`description`、可选 `metadata`
- 执行命令优先从 frontmatter 的 `command` 读取；若未提供，则从 `## Usage` / `## 命令行调用` 代码块推导
- 当前示例 skill 使用 `scripts/run.py`，但执行器不再写死为 Python
- 后端会将已加载技能目录提供给路由模型，自动决策调用

示例：

```md
---
name: time
description: 获取当前本地时间
---

# Time

## Usage
```bash
{python} scripts/run.py --input '{input}' --session-id '{session_id}'
```
```

你可以在 `skills/<your_skill>/` 下新增技能包，无需改核心路由代码。

### shell 技能开关

当前 shell 技能为无白名单、支持完整 shell 语法（含 `|`、`&&`、重定向）。

```bash
./start.sh web
```

如需限制长时间命令，可设置超时（秒）：

```bash
SHELL_TIMEOUT=30 ./start.sh web
```
