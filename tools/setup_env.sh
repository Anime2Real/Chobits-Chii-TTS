#!/bin/bash
# 从零搭建小叽 TTS 训练/本地推理环境 (幂等, 可重复执行).
# 注: 部署 HTTP 服务不需要本脚本 —— 推理环境在 docker/ 镜像内 (见 docs/deployment.md).
#
# 步骤:
#   1. 安装 Miniconda (~/miniconda3, 清华镜像)
#   2. 创建 conda 环境 GPTSoVits (Python 3.10)
#   3. 克隆 GPT-SoVITS 并锁定到 README 验证过的 commit
#   4. 运行上游 install.sh --device CU128 --source ModelScope
#   5. 应用版本修复 (见下), 这些坑在本仓库 README 有记录
#
# 修复项 (2026-08 实测, Ubuntu 20.04/24.04 + 国内网络):
#   a. torch 锁定 2.11.0+cu128: 上游 install.sh 的 cu128 索引现已解析出
#      cu130 构建, 需要 580+ 驱动; 锁定 cu128 兼容 525+ 驱动 (T4 可用)
#   b. fastapi 0.141/starlette 1.6 与 gradio 4.x 不兼容, 回退
#   c. torchcodec 需要 nvidia-npp-cu12 的 libnppicc.so.12 + ffmpeg<=8
#   d. nltk 3.10 需要 punkt_tab; raw.githubusercontent.com 国内被限速,
#      改走 jsDelivr 镜像
#
# 用法: bash tools/setup_env.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GS_ROOT="$REPO_ROOT/GPT-SoVITS"
GS_COMMIT=48b1a0169a28582a8984402f82cf438d3bfa6aca
MINICONDA="${MINICONDA:-$HOME/miniconda3}"
ENV_NAME=GPTSoVits
TORCH_INDEX=https://download.pytorch.org/whl/cu128

# --- 1. Miniconda ---
if [ ! -x "$MINICONDA/bin/conda" ]; then
    echo "== 安装 Miniconda =="
    curl -fL --retry 3 -o /tmp/miniconda.sh \
        https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash /tmp/miniconda.sh -b -p "$MINICONDA"
    rm -f /tmp/miniconda.sh
fi
# shellcheck disable=SC1091
source "$MINICONDA/etc/profile.d/conda.sh"

# 清华镜像 (不存在才写, 不覆盖用户已有配置)
if [ ! -f ~/.condarc ]; then
    cat > ~/.condarc <<'EOF'
channels:
  - conda-forge
channel_alias: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
EOF
fi

# --- 2. conda 环境 ---
if ! conda env list | grep -q "^$ENV_NAME "; then
    echo "== 创建 conda 环境 $ENV_NAME (Python 3.10) =="
    conda create -n "$ENV_NAME" -c conda-forge --override-channels python=3.10 pip -y -q
fi
conda activate "$ENV_NAME"

# --- 3. GPT-SoVITS ---
if [ ! -d "$GS_ROOT/.git" ]; then
    echo "== 克隆 GPT-SoVITS ($GS_COMMIT) =="
    git init "$GS_ROOT"
    git -C "$GS_ROOT" remote add origin https://github.com/RVC-Boss/GPT-SoVITS.git
    git -C "$GS_ROOT" fetch --depth 1 origin "$GS_COMMIT"
    git -C "$GS_ROOT" checkout FETCH_HEAD
fi

# --- 4. 上游 install.sh (幂等: 已下载的模型会跳过) ---
echo "== 运行 install.sh --device CU128 --source ModelScope =="
cd "$GS_ROOT"
TERM=xterm WORKFLOW=true bash install.sh --device CU128 --source ModelScope

# --- 5. 版本修复 ---
echo "== 修复 a: 锁定 torch/torchaudio/torchcodec cu128 =="
pip install --force-reinstall \
    "torch==2.11.0+cu128" "torchaudio==2.11.0+cu128" "torchcodec==0.11.1+cu128" \
    --index-url "$TORCH_INDEX"

echo "== 修复 b: fastapi/starlette/uvicorn 回退 =="
pip install 'fastapi[standard]==0.115.6' 'starlette==0.41.3' 'uvicorn==0.32.0'

echo "== 修复 c: nvidia-npp-cu12 + ffmpeg 7 =="
pip install nvidia-npp-cu12
conda install -c conda-forge --override-channels 'ffmpeg=7' -y -q

echo "== 修复 d: punkt_tab (jsDelivr 镜像) =="
if [ ! -d "$CONDA_PREFIX/nltk_data/tokenizers/punkt_tab" ]; then
    mkdir -p "$CONDA_PREFIX/nltk_data/tokenizers"
    ok=0
    for host in cdn.jsdelivr.net fastly.jsdelivr.net gcore.jsdelivr.net; do
        if curl -fL --max-time 90 -o /tmp/punkt_tab.zip \
            "https://$host/gh/nltk/nltk_data@gh-pages/packages/tokenizers/punkt_tab.zip"; then
            ok=1; break
        fi
    done
    [ "$ok" = 1 ] || { echo "punkt_tab 下载失败"; exit 1; }
    unzip -q -o /tmp/punkt_tab.zip -d "$CONDA_PREFIX/nltk_data/tokenizers"
    rm -f /tmp/punkt_tab.zip
fi

echo "== 验证 =="
python -c "import torch, torchaudio; print('torch', torch.__version__, 'cuda_avail', torch.cuda.is_available()); print('torchaudio', torchaudio.__version__)"
ffmpeg -version | head -1

echo
echo "[完成] 环境就绪。启动服务: bash tools/start_tts_api.sh [端口]"
