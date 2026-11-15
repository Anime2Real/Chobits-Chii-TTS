#!/bin/bash
# 建立独立的 ASR 审计环境并下载 Qwen3-ASR / Whisper / Qwen3-4B 翻译权重.
#
# 默认复用 /opt/chobits/venvs/tts-engine 已安装的 CUDA PyTorch, 避免在
# 慢速网络上重复下载数 GB 的 torch wheel. 可通过 TORCH_SITE_PACKAGES 指定
# 其他 Python 3.10 site-packages. 若不存在, 会从 TORCH_INDEX 安装 torch.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$REPO_ROOT/.venv-asr}"
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/models/asr}"
TORCH_VERSION="${TORCH_VERSION:-2.13.0+cu130}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
DEFAULT_TORCH_SITE="/opt/chobits/venvs/tts-engine/lib/python3.10/site-packages"
TORCH_SITE_PACKAGES="${TORCH_SITE_PACKAGES:-}"
BASE_PYTHON="${BASE_PYTHON:-/opt/chobits/python/cpython-3.10.22-linux-aarch64-gnu/bin/python3.10}"

if [ -z "$TORCH_SITE_PACKAGES" ] && [ -d "$DEFAULT_TORCH_SITE" ]; then
    TORCH_SITE_PACKAGES="$DEFAULT_TORCH_SITE"
fi

echo "[1/3] 创建 Python 环境: $VENV"
rm -rf "$VENV"
if [ -n "$TORCH_SITE_PACKAGES" ] && [ -x "$BASE_PYTHON" ]; then
    uv venv "$VENV" --python "$BASE_PYTHON"
    SITE="$VENV/lib/python3.10/site-packages"
    echo "$TORCH_SITE_PACKAGES" > "$SITE/tts_engine_torch.pth"
    uv pip install --python "$VENV/bin/python" "transformers==4.57.6" sentencepiece
    uv pip install --python "$VENV/bin/python" --no-deps \
        "qwen-asr==0.0.6" "accelerate==1.12.0" "qwen-omni-utils==0.0.9" "nagisa==0.2.11"
else
    uv venv "$VENV" --python 3.12
    uv pip install --python "$VENV/bin/python" --index-url "$TORCH_INDEX" "torch==$TORCH_VERSION"
    uv pip install --python "$VENV/bin/python" "qwen-asr==0.0.6" sentencepiece
fi

echo "[2/3] 从 ModelScope 下载模型到: $MODEL_DIR"
mkdir -p "$MODEL_DIR"
uvx --from modelscope-hub modelscope download \
    --model Qwen/Qwen3-ASR-1.7B --local_dir "$MODEL_DIR/Qwen3-ASR-1.7B" \
    --include model-00001-of-00002.safetensors model-00002-of-00002.safetensors \
              config.json generation_config.json preprocessor_config.json \
              tokenizer_config.json vocab.json merges.txt chat_template.json \
              model.safetensors.index.json
uvx --from modelscope-hub modelscope download \
    --model AI-ModelScope/whisper-large-v3 --local_dir "$MODEL_DIR/whisper-large-v3" \
    --include model.safetensors config.json generation_config.json \
              preprocessor_config.json tokenizer.json tokenizer_config.json \
              vocab.json merges.txt added_tokens.json normalizer.json \
              special_tokens_map.json
uvx --from modelscope-hub modelscope download \
    --model Qwen/Qwen3-4B-Instruct-2507 --local_dir "$MODEL_DIR/Qwen3-4B-Instruct-2507" \
    --include model-00001-of-00003.safetensors model-00002-of-00003.safetensors \
              model-00003-of-00003.safetensors model.safetensors.index.json \
              config.json generation_config.json tokenizer.json tokenizer_config.json \
              vocab.json merges.txt

echo "[3/3] 完成"
echo "运行: $VENV/bin/python tools/asr_consensus.py --device cuda:0"
