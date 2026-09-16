#!/bin/bash
# 启动小叽 TTS 门面 (tools/server.py: 鉴权 + 限流 + OpenAI 垫片, 反向代理容器内引擎)
# 用法: bash tools/start_tts_api.sh [端口, 默认 9880]
# 需要环境变量 CHII_TTS_API_KEY (systemd 从 /etc/chobits-chii-tts.env 读取);
# 推理引擎在 Docker 容器里 (见 docs/deployment.md), 本脚本只起门面;
# 首次运行自动在仓库根目录建 .venv 并安装共享库 + requirements.txt
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${1:-9880}"
VENV="$REPO_ROOT/.venv"
# 门面公共逻辑共享库（chii_facade_common）源在兄弟仓库 Chobits-Chii-CloudDeploy，
# 生产 /home/ubuntu/Github/ 下三仓库互为兄弟目录
COMMON_LIB="$REPO_ROOT/../Chobits-Chii-CloudDeploy/tools/chii-facade-common"

install_common_lib() {
    if [ ! -d "$COMMON_LIB" ]; then
        echo "[错误] 未找到门面共享库: $COMMON_LIB" >&2
        echo "       本门面依赖兄弟仓库的共享库，请先同级 clone Chobits-Chii-CloudDeploy 后重试，" >&2
        echo "       或手动安装: pip install -e <chii-facade-common 路径>" >&2
        exit 1
    fi
    "$VENV/bin/pip" install -e "$COMMON_LIB"
}

if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
    install_common_lib
    "$VENV/bin/pip" install -r "$REPO_ROOT/requirements.txt"
elif ! "$VENV/bin/python" -c "import chii_facade_common" >/dev/null 2>&1; then
    # 旧 venv 补装共享库（共享库接入前已存在的门面环境）
    install_common_lib
fi

exec "$VENV/bin/python" "$REPO_ROOT/tools/server.py" "$PORT"
