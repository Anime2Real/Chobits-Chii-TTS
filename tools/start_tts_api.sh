#!/bin/bash
# 启动小叽 TTS 门面 (tools/server.py: 鉴权 + 限流 + OpenAI 垫片, 反向代理容器内引擎)
# 用法: bash tools/start_tts_api.sh [端口, 默认 9880]
# 需要环境变量 CHII_TTS_API_KEY (systemd 从 /etc/chobits-chii-tts.env 读取);
# 推理引擎在 Docker 容器里 (见 docs/deployment.md), 本脚本只起门面;
# 首次运行自动在仓库根目录建 .venv 并安装 requirements.txt
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${1:-9880}"
VENV="$REPO_ROOT/.venv"

if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install -r "$REPO_ROOT/requirements.txt"
fi

exec "$VENV/bin/python" "$REPO_ROOT/tools/server.py" "$PORT"
