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

# --- 启动前端口占用检测 --------------------------------------------------------
# api_v2 监听端口被占用会直接崩溃退出; 配合文档推荐的 --restart unless-stopped
# 会成为无退避的崩溃循环, 因此这里前置检测: 占用时明确报错并非零退出,
# 让 docker 重试也有可读日志。容器内无 ss/netstat 且读不到 /proc 时跳过检测, 不误报。
port_in_use() {
    if command -v ss >/dev/null 2>&1; then
        ss -ltn "sport = :$ENGINE_PORT" 2>/dev/null | grep -q "^LISTEN"
    elif command -v netstat >/dev/null 2>&1; then
        netstat -ltn 2>/dev/null | awk -v p=":$ENGINE_PORT" '$4 ~ p"$" { found=1 } END { exit !found }'
    elif [ -r /proc/net/tcp ] || [ -r /proc/net/tcp6 ]; then
        # local_address 端口列为十六进制 (0A = LISTEN); 逐个读存在的文件, 缺失不报错
        local hex_port f
        hex_port=$(printf '%04X' "$ENGINE_PORT")
        for f in /proc/net/tcp /proc/net/tcp6; do
            [ -r "$f" ] || continue
            if awk -v p="$hex_port" \
                '$2 ~ ":"p"$" && $4 == "0A" { found=1 } END { exit !found }' "$f"; then
                return 0
            fi
        done
        return 1
    else
        return 1  # 无法检测, 放行
    fi
}
if port_in_use; then
    echo "[engine] 端口 $ENGINE_PORT 已被占用, api_v2 无法启动。" >&2
    echo "[engine] 排查占用: ss -lntp \"sport = :$ENGINE_PORT\" (或 netstat -lntp | grep $ENGINE_PORT)" >&2
    echo "[engine] 常见原因: 上一个引擎容器/进程未退出, 或 ENGINE_PORT 与环境冲突。" >&2
    exit 1
fi

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
    if ! wget --tries=5 --wait=5 --read-timeout=60 --show-progress \
        "$BASE/pretrained_models.zip" -O /tmp/pretrained_models.zip; then
        echo "[engine] 预训练模型下载失败: $BASE/pretrained_models.zip" >&2
        echo "[engine] 手动放置: 下载该 zip 后解出 GPT_SoVITS/pretrained_models/ 下的" >&2
        echo "[engine]   chinese-roberta-wwm-ext-large、chinese-hubert-base、sv、fast_langdetect" >&2
        echo "[engine]   四个子目录, 拷入 $DATA/pretrained_models/ 后重启容器。" >&2
        exit 1
    fi
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
    if ! wget --tries=5 --wait=5 --read-timeout=60 --show-progress \
        "$BASE/G2PWModel.zip" -O /tmp/G2PWModel.zip; then
        echo "[engine] G2PW 模型下载失败: $BASE/G2PWModel.zip" >&2
        echo "[engine] 手动放置: 下载该 zip 后解出 G2PWModel/ 内容, 拷入 $DATA/G2PWModel/ 后重启容器。" >&2
        exit 1
    fi
    unzip -q -o /tmp/G2PWModel.zip -d /tmp/g2pw
    mkdir -p "$DATA/G2PWModel"
    cp -r /tmp/g2pw/G2PWModel/. "$DATA/G2PWModel/" 2>/dev/null || cp -r /tmp/g2pw/. "$DATA/G2PWModel/"
    rm -rf /tmp/G2PWModel.zip /tmp/g2pw
fi

# --- chii 权重 ---------------------------------------------------------------
if [ ! -f "$DATA/models/chii-e10.ckpt" ] || [ ! -f "$DATA/models/chii_e10_s1210.pth" ]; then
    ensure_writable_dir "$DATA/models" "chii 权重"
    echo "[engine] 下载 chii 权重 ($CHII_WEIGHTS_REPO, HF_ENDPOINT=${HF_ENDPOINT:-默认}) ..."
    if ! hf download "$CHII_WEIGHTS_REPO" --local-dir "$DATA/models" \
        && ! huggingface-cli download "$CHII_WEIGHTS_REPO" --local-dir "$DATA/models"; then
        echo "[engine] chii 权重下载失败: $CHII_WEIGHTS_REPO (HF_ENDPOINT=${HF_ENDPOINT:-默认})" >&2
        echo "[engine] 可排查网络/镜像 (HF_ENDPOINT=https://hf-mirror.com) 后重启容器;" >&2
        echo "[engine] 或手动放置 chii-e10.ckpt、chii_e10_s1210.pth (及 ref_audio.wav、ref_text.txt)" >&2
        echo "[engine]   到 $DATA/models/ 后重启。" >&2
        exit 1
    fi
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
