#!/bin/bash
# 运行 Qwen3-ASR + Whisper 双模型复核.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$REPO_ROOT/.venv-asr}"
PYTHON="$VENV/bin/python"

if [ ! -x "$PYTHON" ]; then
    echo "缺少 ASR 环境: $VENV" >&2
    echo "先运行: bash tools/setup_asr_env.sh" >&2
    exit 1
fi
for model in Qwen3-ASR-1.7B whisper-large-v3 Qwen3-4B-Instruct-2507; do
    if [ ! -d "$REPO_ROOT/models/asr/$model" ]; then
        echo "缺少模型: models/asr/$model" >&2
        echo "先运行: bash tools/setup_asr_env.sh" >&2
        exit 1
    fi
done

exec "$PYTHON" "$REPO_ROOT/tools/asr_consensus.py" "$@"
