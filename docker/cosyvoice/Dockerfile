# CosyVoice3 推理引擎镜像 (Fun-CosyVoice3-0.5B-chii-sft, FastAPI shim 见 engine_server.py)
# 与 GPT-SoVITS 引擎镜像 (../Dockerfile) 同级契约: 引擎无鉴权, 只发布到宿主机 127.0.0.1;
# 鉴权/限流/OpenAI 垫片由宿主机门面 tools/server.py 负责 (一行不改, CHII_TTS_ENGINE_URL 指过来即可)
#
# 构建:  docker build -t chobits-chii-cosyvoice-engine docker/cosyvoice/
#        国内可加 --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
#                  --build-arg TORCH_INDEX_URL=https://mirror.sjtu.edu.cn/pytorch-wheels/cu121
# 运行:  docker run -d --gpus all -p 127.0.0.1:9882:9880 \
#            -v /srv/chii-tts/cosyvoice:/data chobits-chii-cosyvoice-engine
#        /data 卷约定见 entrypoint.sh 头部注释 (SFT 权重无发布渠道, 挂载是必须路径)
FROM condaforge/miniforge3:25.3.1-0

# CosyVoice 框架锁定到本仓库 CosyVoice/ 同款提交 (2026-10 实测验证);
# 完整 SHA 保证可复现, 仍可用 --build-arg GS_REPO_REF=<分支名或 SHA> 覆盖
ARG GS_REPO_REF=074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc
ARG GS_REPO_URL=https://github.com/FunAudioLLM/CosyVoice.git
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
# 构建期小体积数据 (nltk_data / open_jtalk dict) 的下载源: ms=ModelScope / hf / hf-mirror
ARG MODEL_SOURCE=ms

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget unzip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# conda 环境: Python 3.10 + ffmpeg 7 (aac/ogg 编码用; 与宿主机 CosyVoice 环境一致)
RUN conda create -n CosyVoice -c conda-forge --override-channels python=3.10 pip -y \
    && conda install -n CosyVoice -c conda-forge --override-channels 'ffmpeg=7' -y \
    && conda clean -afy

ENV CONDA_ENV=/opt/conda/envs/CosyVoice
ENV PATH="$CONDA_ENV/bin:$PATH"

# CosyVoice 引擎源码 (浅抓取锁定 ref, 含 third_party/Matcha-TTS 子模块;
# GitHub 不通时可 --build-arg GS_REPO_URL=https://gh-proxy.com/https://github.com/FunAudioLLM/CosyVoice.git)
RUN git init /opt/CosyVoice \
    && cd /opt/CosyVoice \
    && git remote add origin "$GS_REPO_URL" \
    && git fetch --depth 1 origin "$GS_REPO_REF" \
    && git checkout --detach FETCH_HEAD \
    && git submodule update --init --recursive --depth 1

WORKDIR /opt/CosyVoice

# 依赖 (与宿主机 conda CosyVoice 环境同款组合, 顺序敏感):
#   a) torch 锁定 2.3.1+cu121 (从 TORCH_INDEX_URL 装, 故先剔除 requirements 里的 torch 行)
#   b) requirements 剔除推理不需要/装不动的: deepspeed tensorrt-cu12* gdown gradio
#      lightning tensorboard fastapi-cli (--extra-index-url 行一并剔除)
#   c) openai-whisper==20231117 是 sdist (setup.py 直读 num 版本), 须先装 setuptools<81 + wheel
#      再 --no-build-isolation 单独装 (与 docs/cosyvoice-feasibility.md §9.4 同款修复)
#   d) pyopenjtalk 日语前端必需, 词典在下方 MODEL_SOURCE 分支解包
# build-essential 仅用于编译 pyworld (PyPI 无 cp310 manylinux 轮), 同层装完即删
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && pip install --no-cache-dir \
        "torch==2.3.1" "torchaudio==2.3.1" --index-url "$TORCH_INDEX_URL" \
    && pip install --no-cache-dir -i "$PIP_INDEX_URL" 'setuptools<81' wheel \
    && grep -vE '^(deepspeed|tensorrt|gdown|gradio|lightning|tensorboard|fastapi-cli|torch|torchaudio|openai-whisper|--)' \
        requirements.txt > /tmp/req-engine.txt \
    && pip install --no-cache-dir -i "$PIP_INDEX_URL" -r /tmp/req-engine.txt \
    && pip install --no-cache-dir -i "$PIP_INDEX_URL" --no-build-isolation openai-whisper==20231117 \
    && pip install --no-cache-dir -i "$PIP_INDEX_URL" pyopenjtalk==0.4.1 \
    && apt-get purge -y build-essential && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# 小体积运行数据, 直接烘焙进镜像 (与 ../Dockerfile 同源 ModelScope 仓库):
#   nltk_data (g2p_en 英文前端需要; 含 punkt_tab 修复) + open_jtalk dict (pyopenjtalk 不自带)
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

COPY engine_server.py /app/engine_server.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# 运行参数 (均可 docker run -e 覆盖); /data 卷约定见 entrypoint.sh 头部注释
ENV MODEL_SOURCE=ms \
    MODEL_DIR=/data/model \
    COSYVOICE_DIR=/opt/CosyVoice \
    ENGINE_PORT=9880

EXPOSE 9880
ENTRYPOINT ["/app/entrypoint.sh"]
