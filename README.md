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
├─ skills/         # 技能模块
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
