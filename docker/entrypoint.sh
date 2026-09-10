#!/bin/bash
# 引擎容器入口: 确保模型就位 (缺失则下载) 后裸跑上游 api_v2
# 引擎无鉴权, 部署时只发布到宿主机 127.0.0.1, 对外由门面 tools/server.py 负责
#
# /data 卷约定 (可整体挂载, 也可按子目录分别挂载; 只读挂载且文件齐全时跳过下载):
#   /data/pretrained_models/   GPT-SoVITS 预训练模型 (推理需: chinese-roberta-wwm-ext-large,
#                              chinese-hubert-base, sv, fast_langdetect)
#   /data/G2PWModel/           中文 G2PW 模型 (仅中文文本需要, 但 auto 模式可能切出中文段)
#   /data/models/              chii 权重 (chii-e10.ckpt / chii_e10_s1210.pth / ref_audio.wav / ref_text.txt)
set -euo pipefail

GS=/opt/GPT-SoVITS
DATA=/data
MODEL_SOURCE="${MODEL_SOURCE:-ms}"           # ms=ModelScope / hf / hf-mirror
CHII_WEIGHTS_REPO="${CHII_WEIGHTS_REPO:-chenxin199305/Chobits-Chii-TTS}"
DEVICE="${DEVICE:-cuda}"
IS_HALF="${IS_HALF:-true}"
ENGINE_PORT="${ENGINE_PORT:-9880}"

case "$MODEL_SOURCE" in
    ms)        BASE="https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master" ;;
    hf)        BASE="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main" ;;
    hf-mirror) BASE="https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main" ;;
    *) echo "[engine] 未知 MODEL_SOURCE: $MODEL_SOURCE (可选 ms|hf|hf-mirror)" >&2; exit 1 ;;
esac

# $1=目录 $2=用途说明; 目录不存在则创建, 不可写则报错退出 (只读挂载缺文件的场景)
ensure_writable_dir() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1" 2>/dev/null || {
            echo "[engine] 无法创建 $1 ($2): 若是只读挂载请在宿主机补齐文件" >&2; exit 1; }
    fi
    [ -w "$1" ] || {
        echo "[engine] $1 为只读挂载且缺少$2, 请在宿主机补齐后重启容器" >&2; exit 1; }
}

# --- 预训练模型 -------------------------------------------------------------
need_pretrained=0
for d in chinese-roberta-wwm-ext-large chinese-hubert-base sv fast_langdetect; do
    [ -d "$DATA/pretrained_models/$d" ] || need_pretrained=1
done
if [ "$need_pretrained" = 1 ]; then
    ensure_writable_dir "$DATA/pretrained_models" "预训练模型"
    echo "[engine] 下载预训练模型 (pretrained_models.zip, 源: $MODEL_SOURCE, 仅解出推理所需子集) ..."
    wget --tries=5 --wait=5 --read-timeout=60 -q --show-progress \
        "$BASE/pretrained_models.zip" -O /tmp/pretrained_models.zip
    unzip -q -o /tmp/pretrained_models.zip \
        'GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/*' \
        'GPT_SoVITS/pretrained_models/chinese-hubert-base/*' \
        'GPT_SoVITS/pretrained_models/sv/*' \
        'GPT_SoVITS/pretrained_models/fast_langdetect/*' -d /tmp/pm
    mkdir -p "$DATA/pretrained_models"
    cp -r /tmp/pm/GPT_SoVITS/pretrained_models/. "$DATA/pretrained_models/"
    rm -rf /tmp/pretrained_models.zip /tmp/pm
fi

# --- G2PW 模型 ---------------------------------------------------------------
if ! ls "$DATA/G2PWModel/"* >/dev/null 2>&1; then
    ensure_writable_dir "$DATA/G2PWModel" "G2PW 模型"
    echo "[engine] 下载 G2PWModel.zip (源: $MODEL_SOURCE) ..."
    wget --tries=5 --wait=5 --read-timeout=60 -q --show-progress \
        "$BASE/G2PWModel.zip" -O /tmp/G2PWModel.zip
    unzip -q -o /tmp/G2PWModel.zip -d /tmp/g2pw
    mkdir -p "$DATA/G2PWModel"
    cp -r /tmp/g2pw/G2PWModel/. "$DATA/G2PWModel/" 2>/dev/null || cp -r /tmp/g2pw/. "$DATA/G2PWModel/"
    rm -rf /tmp/G2PWModel.zip /tmp/g2pw
fi

# --- chii 权重 ---------------------------------------------------------------
if [ ! -f "$DATA/models/chii-e10.ckpt" ] || [ ! -f "$DATA/models/chii_e10_s1210.pth" ]; then
    ensure_writable_dir "$DATA/models" "chii 权重"
    echo "[engine] 下载 chii 权重 ($CHII_WEIGHTS_REPO, HF_ENDPOINT=${HF_ENDPOINT:-默认}) ..."
    hf download "$CHII_WEIGHTS_REPO" --local-dir "$DATA/models" \
        || huggingface-cli download "$CHII_WEIGHTS_REPO" --local-dir "$DATA/models"
fi

# --- 代码内硬编码相对路径 → 数据卷软链 ----------------------------------------
rm -rf "$GS/GPT_SoVITS/pretrained_models"
ln -sfn "$DATA/pretrained_models" "$GS/GPT_SoVITS/pretrained_models"
rm -rf "$GS/GPT_SoVITS/text/G2PWModel"
ln -sfn "$DATA/G2PWModel" "$GS/GPT_SoVITS/text/G2PWModel"

# --- 推理配置 (随镜像重建, 每次启动按当前挂载重新生成) --------------------------
cat > "$GS/GPT_SoVITS/configs/tts_infer_chii.yaml" <<EOF
custom:
  bert_base_path: GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large
  cnhuhbert_base_path: GPT_SoVITS/pretrained_models/chinese-hubert-base
  device: $DEVICE
  is_half: $IS_HALF
  t2s_weights_path: $DATA/models/chii-e10.ckpt
  version: v2Pro
  vits_weights_path: $DATA/models/chii_e10_s1210.pth
EOF

# shellcheck disable=SC1091
source /opt/conda/etc/profile.d/conda.sh
conda activate GPTSoVits

cd "$GS"
export PYTHONPATH="$GS:$GS/GPT_SoVITS:$GS/GPT_SoVITS/BigVGAN"
# torchcodec 需要的 npp 动态库 (conda 环境 libstdc++ 也一并前置, 对齐宿主机启动脚本)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/npp/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export version=v2Pro

echo "[engine] 启动 api_v2: port=$ENGINE_PORT device=$DEVICE is_half=$IS_HALF"
exec python api_v2.py -c GPT_SoVITS/configs/tts_infer_chii.yaml -a 0.0.0.0 -p "$ENGINE_PORT"
