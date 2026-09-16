# Chobits-Chii-TTS 推理引擎镜像 (GPT-SoVITS api_v2, v2Pro)
# 引擎无鉴权, 只发布到宿主机 127.0.0.1; 鉴权/限流/OpenAI 垫片由宿主机门面 tools/server.py 负责
#
# 构建:  docker build -t chobits-chii-tts-engine docker/
#        国内可加 --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
#                  --build-arg TORCH_INDEX_URL=https://mirror.sjtu.edu.cn/pytorch-wheels/cu128
# 运行见 docs/deployment.md; 预训练模型与 chii 权重经 /data 卷挂载, 缺失时首次启动自动下载
FROM condaforge/miniforge3:25.3.1-0

# 锁定到本项目 fork 的修复提交 (fix-nonstream-threadpool 分支 HEAD @ 2026-09-16,
# 基于上游 48b1a01, 含非流式合成线程池修复); 完整 SHA 保证可复现, 仍可用
# --build-arg GS_REPO_REF=<分支名或 SHA> 覆盖
ARG GS_REPO_REF=49587af32c427e3123dcc23c03de00a3cfe0adbb
ARG GS_REPO_URL=https://github.com/Anime2Real/GPT-SoVITS.git
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
# 构建期小体积数据 (nltk_data / open_jtalk dict) 的下载源: ms=ModelScope / hf / hf-mirror
ARG MODEL_SOURCE=ms

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget unzip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# conda 环境: Python 3.10 + ffmpeg 7 (与 tools/setup_env.sh 一致; torchcodec 需要 ffmpeg<=8)
RUN conda create -n GPTSoVits -c conda-forge --override-channels python=3.10 pip -y \
    && conda install -n GPTSoVits -c conda-forge --override-channels 'ffmpeg=7' -y \
    && conda clean -afy

ENV CONDA_ENV=/opt/conda/envs/GPTSoVits
ENV PATH="$CONDA_ENV/bin:$PATH"

# GPT-SoVITS 引擎源码 (浅抓取锁定 ref, 减小传输量; GitHub 允许按完整 SHA 直抓,
# 默认 GS_REPO_REF 即 commit SHA; 覆盖为分支名时本命令同样适用。
# GitHub 不通时可 --build-arg GS_REPO_URL=https://gh-proxy.com/https://github.com/Anime2Real/GPT-SoVITS.git)
RUN git init /opt/GPT-SoVITS \
    && cd /opt/GPT-SoVITS \
    && git remote add origin "$GS_REPO_URL" \
    && git fetch --depth 1 origin "$GS_REPO_REF" \
    && git checkout --detach FETCH_HEAD

WORKDIR /opt/GPT-SoVITS

# 依赖 (README 快速开始第 2 步同款修复, 顺序敏感):
#   a) torch 锁定 cu128 构建 (cu128 索引现已解析出需 580+ 驱动的 cu130 版)
#   b) fastapi/starlette/uvicorn 回退到与上游兼容的版本
#   c) nvidia-npp-cu12: torchcodec 运行时加载 libnppicc.so.12
# build-essential/cmake 仅用于编译 opencc (requirements.txt 强制 --no-binary), 同层装完即删
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake \
    && pip install --no-cache-dir \
        "torch==2.11.0+cu128" "torchaudio==2.11.0+cu128" "torchcodec==0.11.1+cu128" \
        --index-url "$TORCH_INDEX_URL" \
    && pip install --no-cache-dir -i "$PIP_INDEX_URL" -r requirements.txt \
    && pip install --no-cache-dir -i "$PIP_INDEX_URL" \
        'fastapi[standard]==0.115.6' 'starlette==0.41.3' 'uvicorn==0.32.0' \
        nvidia-npp-cu12 huggingface_hub \
    && apt-get purge -y build-essential cmake && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# 小体积运行数据, 直接烘焙进镜像 (与 install.sh 同源):
#   nltk_data (g2p_en 需要; 含 punkt_tab 修复) + open_jtalk dict (pyopenjtalk 不自带)
RUN case "$MODEL_SOURCE" in \
        ms)        BASE="https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master" ;; \
        hf)        BASE="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main" ;; \
        hf-mirror) BASE="https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main" ;; \
        *) echo "未知 MODEL_SOURCE: $MODEL_SOURCE" >&2; exit 1 ;; \
    esac \
    && wget --tries=5 --wait=5 -q "$BASE/nltk_data.zip" -O /tmp/nltk_data.zip \
    && unzip -q -o /tmp/nltk_data.zip -d /opt && rm /tmp/nltk_data.zip \
    && ( python -m nltk.downloader -d /opt/nltk_data punkt_tab \
         || ( wget -q -O /tmp/punkt_tab.zip https://cdn.jsdelivr.net/gh/nltk/nltk_data@gh-pages/packages/tokenizers/punkt_tab.zip \
              && unzip -q -o /tmp/punkt_tab.zip -d /opt/nltk_data/tokenizers && rm /tmp/punkt_tab.zip ) ) \
    && wget --tries=5 --wait=5 -q "$BASE/open_jtalk_dic_utf_8-1.11.tar.gz" -O /tmp/open_jtalk.tar.gz \
    && tar -xzf /tmp/open_jtalk.tar.gz -C "$(python -c 'import os, pyopenjtalk; print(os.path.dirname(pyopenjtalk.__file__))')" \
    && rm /tmp/open_jtalk.tar.gz

ENV NLTK_DATA=/opt/nltk_data

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# 运行参数 (均可 docker run -e 覆盖); /data 卷约定见 entrypoint.sh 头部注释
ENV MODEL_SOURCE=ms \
    DEVICE=cuda \
    IS_HALF=true \
    ENGINE_PORT=9880

EXPOSE 9880
ENTRYPOINT ["/app/entrypoint.sh"]
