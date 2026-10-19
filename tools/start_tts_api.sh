#!/bin/bash
# 启动小叽 TTS 门面 (tools/server.py: 鉴权 + 限流 + OpenAI 垫片, 反向代理容器内引擎)
# 用法: bash tools/start_tts_api.sh [端口, 默认 9880]
# 需要环境变量 CHII_TTS_API_KEY (systemd 从 /etc/chobits-chii-tts.env 读取);
# 推理引擎在 Docker 容器里 (见 docs/deployment.md), 本脚本只起门面;
# 首次运行自动在仓库根目录建 .venv 并安装共享库 + requirements.txt;
# 之后每次启动先验证关键 import (fastapi/httpx/uvicorn/chii_facade_common),
# 缺失/损坏自动重装 (venv 存在 ≠ 依赖可用), 装不通拒绝启动
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${1:-9880}"
VENV="$REPO_ROOT/.venv"
# 门面公共逻辑共享库（chii_facade_common）源在兄弟仓库 Chobits-Chii-ServerDeploy，
# 生产 /home/ubuntu/Github/ 下三仓库互为兄弟目录；兼容更名前的旧目录名 CloudDeploy
COMMON_LIB=""
for _sibling in "$REPO_ROOT/../Chobits-Chii-ServerDeploy" "$REPO_ROOT/../Chobits-Chii-CloudDeploy"; do
    if [ -d "$_sibling/tools/chii-facade-common" ]; then
        COMMON_LIB="$_sibling/tools/chii-facade-common"
        break
    fi
done

# 前置检查：共享库缺失时先给出可操作的错误再退出，避免建完 venv 才失败
if [ ! -d "$COMMON_LIB" ]; then
    echo "[错误] 未找到门面共享库: $REPO_ROOT/../Chobits-Chii-ServerDeploy/tools/chii-facade-common" >&2
    if [ -d "$REPO_ROOT/../Chobits-Chii-ServerDeploy" ] || [ -d "$REPO_ROOT/../Chobits-Chii-CloudDeploy" ]; then
        echo "       兄弟仓库目录已存在但缺少 tools/chii-facade-common：ServerDeploy 的 main 分支仅作" >&2
        echo "       索引（无 tools/ 目录），共享库源在 cloud 分支，请检出后重试:" >&2
        echo "       git -C $REPO_ROOT/../Chobits-Chii-ServerDeploy switch cloud" >&2
    else
        echo "       本门面依赖兄弟仓库的共享库，请同级 clone 并检出 cloud 分支后重试:" >&2
        echo "       git clone -b cloud git@github.com:Anime2Real/Chobits-Chii-ServerDeploy.git \\" >&2
        echo "           $REPO_ROOT/../Chobits-Chii-ServerDeploy" >&2
    fi
    echo "       或手动安装: pip install -e <chii-facade-common 路径>" >&2
    exit 1
fi

# 门面实际依赖的关键 import：venv 目录存在 ≠ 依赖可用（首次 pip 失败后重启
# 会带着半成品环境永远跳过安装），故以 import 验证为准，不可用则重装
KEY_IMPORTS="fastapi, httpx, uvicorn, chii_facade_common"

venv_imports_ok() {
    "$VENV/bin/python" -c "import $KEY_IMPORTS" >/dev/null 2>&1
}

run_pip() {
    # 生产 .venv 由 uv 创建、没有 pip 模块：优先 venv 自带 pip，退化为 uv pip
    # （uv pip 子命令与 pip 一致，直接透传参数 + 指定目标解释器）
    if [ -x "$VENV/bin/pip" ]; then
        "$VENV/bin/pip" "$@"
    else
        uv pip "$@" -p "$VENV/bin/python"
    fi
}

install_deps() {
    # -e 共享库 + requirements.txt；chii_facade_common 在 requirements 之外，单独装
    run_pip install -e "$COMMON_LIB"
    run_pip install -r "$REPO_ROOT/requirements.txt"
}

if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
fi
if ! venv_imports_ok; then
    install_deps
fi
if ! venv_imports_ok; then
    # 装完仍不可用（如首次安装中途失败）→ 重试一次；再失败则明确报错退出，
    # 而不是带着坏环境启动门面
    echo "[warn] .venv 关键 import 验证失败（$KEY_IMPORTS），重试安装依赖" >&2
    install_deps
fi
if ! venv_imports_ok; then
    echo "[错误] .venv 依赖安装后关键 import 仍失败: $KEY_IMPORTS" >&2
    echo "       请手动安装: uv pip install -p $VENV/bin/python -e $COMMON_LIB \\" >&2
    echo "           && uv pip install -p $VENV/bin/python -r $REPO_ROOT/requirements.txt" >&2
    exit 1
fi

exec "$VENV/bin/python" "$REPO_ROOT/tools/server.py" "$PORT"
