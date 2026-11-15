#!/bin/bash
# 在隔离训练环境中运行固定评测集生成脚本.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$REPO_ROOT/.venv-train}"
PYTHON="$VENV/bin/python"
PROD_SITE="/opt/chobits/venvs/tts-engine/lib/python3.10/site-packages"

if [ ! -x "$PYTHON" ]; then
    echo "缺少训练/评测环境: $VENV" >&2
    exit 1
fi

export LD_LIBRARY_PATH="/opt/chobits/venvs/tts-engine/lib:$PROD_SITE/nvidia/npp/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$REPO_ROOT/GPT-SoVITS/TEMP/train_stubs${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" "$REPO_ROOT/tools/eval_tts.py" "$@"
