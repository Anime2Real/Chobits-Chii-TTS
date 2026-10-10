#!/bin/bash
# CosyVoice3 引擎容器入口: 确保模型就位 (缺失时可自动下载预训练基座) 后启动 engine_server
# 引擎无鉴权, 部署时只发布到宿主机 127.0.0.1, 对外由门面 tools/server.py 负责
#
# /data 卷约定 (可整体挂载, 也可按子目录分别挂载):
#   /data/model/    Fun-CosyVoice3 模型目录 (llm.pt / flow.pt / hift.pt / campplus.onnx /
#                   speech_tokenizer_v3.onnx / cosyvoice3.yaml / CosyVoice-BlankEN/)
#                   ★ SFT 微调权重 (Fun-CosyVoice3-0.5B-chii-sft) 无发布渠道, 必须从宿主机
#                     挂载 (MODEL_DIR 指向挂载点); MODEL_SOURCE=ms 自动下载的只是官方
#                     zero-shot 预训练基座 Fun-CosyVoice3-0.5B-2512, 供冒烟/回退用
#   /data/models/   参考音频 ref_audio.wav 等 (门面经 ref_audio_path 逐请求指定;
#                   缺失仅在启动时 WARN, 不阻断——请求到达时引擎侧 4xx 兜底)
set -euo pipefail

DATA=/data
MODEL_DIR="${MODEL_DIR:-$DATA/model}"
MODEL_SOURCE="${MODEL_SOURCE:-ms}"
ENGINE_PORT="${ENGINE_PORT:-9880}"

# --- 启动前端口占用检测 (与 ../entrypoint.sh 同款; 容器内读不到 /proc 时跳过) -----------
port_in_use() {
    if command -v ss >/dev/null 2>&1; then
        ss -ltn "sport = :$ENGINE_PORT" 2>/dev/null | grep -q "^LISTEN"
    elif [ -r /proc/net/tcp ] || [ -r /proc/net/tcp6 ]; then
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
    echo "[engine] 端口 $ENGINE_PORT 已被占用, 引擎无法启动。" >&2
    echo "[engine] 常见原因: 上一个引擎容器/进程未退出, 或 ENGINE_PORT 与环境冲突。" >&2
    exit 1
fi

# --- 模型目录检查 (缺失时可选自动下载预训练基座) ----------------------------------------
model_ready() {
    [ -f "$MODEL_DIR/llm.pt" ] && [ -f "$MODEL_DIR/flow.pt" ] && [ -f "$MODEL_DIR/hift.pt" ] \
        && [ -f "$MODEL_DIR/campplus.onnx" ] && [ -f "$MODEL_DIR/speech_tokenizer_v3.onnx" ] \
        && [ -f "$MODEL_DIR/cosyvoice3.yaml" ] && [ -d "$MODEL_DIR/CosyVoice-BlankEN" ]
}

if ! model_ready && [ "$MODEL_SOURCE" = "ms" ]; then
    if [ ! -d "$MODEL_DIR" ]; then
        mkdir -p "$MODEL_DIR" 2>/dev/null || true
    fi
    if [ -d "$MODEL_DIR" ] && [ -w "$MODEL_DIR" ]; then
        echo "[engine] 模型缺失, 自动下载预训练基座 Fun-CosyVoice3-0.5B-2512 (ModelScope) -> $MODEL_DIR"
        echo "[engine] 注意: 自动下载的是官方 zero-shot 基座; SFT 微调权重请从宿主机挂载 (见头部注释)"
        # shellcheck disable=SC1091
        source /opt/conda/etc/profile.d/conda.sh && conda activate CosyVoice
        python -c "from modelscope import snapshot_download; \
                   snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='$MODEL_DIR')"
    fi
fi

if ! model_ready; then
    echo "[engine] 模型目录不完整: $MODEL_DIR" >&2
    echo "[engine] 需要: llm.pt flow.pt hift.pt campplus.onnx speech_tokenizer_v3.onnx" >&2
    echo "[engine]       cosyvoice3.yaml CosyVoice-BlankEN/" >&2
    echo "[engine] 修复: 宿主机挂载 SFT 模型目录后重启, 例如:" >&2
    echo "[engine]   docker run -d --gpus all -p 127.0.0.1:9882:9880 \\" >&2
    echo "[engine]     -v /srv/chii-tts/cosyvoice/model:/data/model \\" >&2
    echo "[engine]     -v /srv/chii-tts/cosyvoice/models:/data/models \\" >&2
    echo "[engine]     chobits-chii-cosyvoice-engine" >&2
    echo "[engine] (SFT 权重无发布渠道, 挂载是必须路径; 或将 MODEL_SOURCE=ms 且目录可写" >&2
    echo "[engine]  让容器自动下载官方 zero-shot 基座)" >&2
    exit 1
fi

# --- 参考音频 (门面逐请求传 ref_audio_path, 缺失仅 WARN) ---------------------------------
if [ ! -f "$DATA/models/ref_audio.wav" ]; then
    echo "[engine] WARN: $DATA/models/ref_audio.wav 不存在;" >&2
    echo "[engine]   门面会以请求中的 ref_audio_path 为准, 指向缺失文件时引擎返回 4xx。" >&2
fi

# shellcheck disable=SC1091
source /opt/conda/etc/profile.d/conda.sh
conda activate CosyVoice

# 携带参数时直接执行 (CPU 冒烟/调试: docker run --rm <image> python -c "...")
if [ $# -gt 0 ]; then
    cd /opt/CosyVoice
    exec "$@"
fi

cd /app
echo "[engine] 启动 engine_server: port=$ENGINE_PORT model_dir=$MODEL_DIR"
exec uvicorn engine_server:app --host 0.0.0.0 --port "$ENGINE_PORT" --workers 1
