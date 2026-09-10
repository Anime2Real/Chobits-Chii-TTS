# Chobits-Chii-TTS

[![Made with Love](https://img.shields.io/badge/Made%20with-Love-ff69b4.svg)](https://madewithlove.org.in)
[![GitHub](https://img.shields.io/badge/GitHub-Chobits--Chii--TTS-181717?logo=github)](https://github.com/Anime2Real/Chobits-Chii-TTS)
[![Dataset: Chobits-Chii-Voice](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Chobits--Chii--Voice-yellow)](https://huggingface.co/datasets/chenxin199305/Chobits-Chii-Voice)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Chobits--Chii--TTS-yellow)](https://huggingface.co/chenxin199305/Chobits-Chii-TTS)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Language: Japanese](https://img.shields.io/badge/Language-Japanese-green.svg)]()

> ✅ 首轮模型已训练完成并人工试听通过（GPT-SoVITS v2Pro，e10 checkpoint），权重已发布到 [Hugging Face](https://huggingface.co/chenxin199305/Chobits-Chii-TTS)，见[模型文件](#模型文件)。

《人形电脑天使心》(Chobits) 中 **小叽 (Chii / ちぃ)** 角色的 TTS（语音合成）模型项目。

本项目以 [Chobits-Chii-Voice](https://github.com/Anime2Real/Chobits-Chii-Voice) 数据集为训练数据，基于 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 微调出小叽声线的文本转语音模型。

> ⚠️ 注意：原始动画音频的版权归其权利方所有。本项目仅供学习与研究使用，请勿用于商业用途。

## 示例试听

https://github.com/user-attachments/assets/06c1976c-84c9-4fd2-b472-d5cdfe9be485

合成文本（`examples/target_text.txt`）：`ちぃ、秀樹のこと、大好き。ずっと一緒にいる。`
（GPT-SoVITS v2Pro · e10 checkpoint 合成）

## 数据

训练数据来自 [Chobits-Chii-Voice](https://github.com/Anime2Real/Chobits-Chii-Voice) 数据集（[Hugging Face](https://huggingface.co/datasets/chenxin199305/Chobits-Chii-Voice)）：

| 项目 | 数值 |
| --- | --- |
| 片段数量 | 487 |
| 总时长 | 约 21.4 分钟 |
| 片段时长 | 中位 2.24s，平均 2.63s（0.98s – 9.05s） |
| 采样率 | 22050 Hz，单声道，16-bit PCM |
| 语言 | 日语 |
| 标注格式 | `metadata.csv`：`文件名|文本` |

数据集的 `metadata.csv` 标注格式与 GPT-SoVITS 训练输入格式兼容，可直接用于微调。

## 训练路线

基于 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 的少样本语音克隆流程（v2Pro，2026-08 已跑通首轮）：

1. ~~数据准备~~：✅ `tools/clean_dataset.py` 清洗 Chobits-Chii-Voice 数据集（修复 Whisper 误转写 18 条、剔除 4 条），输出 `data/gpt_sovits.list`（483 条 / 21.4 分钟）
2. ~~预处理~~：✅ 文本→音素、HuBERT SSL 特征、说话人嵌入（v2Pro）、语义 token（VQ），首轮产物在 `GPT-SoVITS/logs/chi/`（历史目录保持不动；EXP_NAME 已改名 `chii`，新训练产物写入 `GPT-SoVITS/logs/chii/`）
3. ~~微调 SoVITS~~：✅ v2Pro 全量微调，batch 4 × 15 epochs → `GPT-SoVITS/SoVITS_weights_v2Pro/chi_e{5,10,15}_*.pth`
4. ~~微调 GPT~~：✅ batch 4 × 15 epochs → `GPT-SoVITS/GPT_weights_v2Pro/chi-e{5,10,15}.ckpt`
5. **推理评估**：✅ 首轮合成已产出 `outputs/inference/e{5,10,15}/output.wav`（32kHz）。人工试听选定 **e10**（`chi-e10.ckpt` + `chi_e10_s1210.pth`）为当前首选
6. **迭代优化**：根据试听结果调整 epochs / 参考音频 / 数据量，必要时换 v4 对比

复现命令见下方[快速开始](#快速开始从零复现)。

## 模型文件

当前首选权重（人工试听选定，v2Pro，e10）已发布到 Hugging Face：
**[huggingface.co/chenxin199305/Chobits-Chii-TTS](https://huggingface.co/chenxin199305/Chobits-Chii-TTS)**

```bash
# 下载模型 (约 290MB)
pip install huggingface_hub
hf download chenxin199305/Chobits-Chii-TTS --local-dir models/
```

| 文件 | 说明 |
| --- | --- |
| `chii-e10.ckpt` | GPT 语义模型（149MB） |
| `chii_e10_s1210.pth` | SoVITS 声学模型（129MB） |
| `ref_audio.wav` | 参考音频（ep07「秀樹は地位を拾ってくれた」，4.2s） |
| `ref_text.txt` | 参考音频对应文本（同 `examples/ref_text.txt`） |

首轮发布的权重已在 Hugging Face 上统一更名为 `chii-*`/`chii_*` 拼写；本地历史训练产物（`GPT-SoVITS/` 目录下）保持 `chi-*`/`chi_*` 原名不动。
下载后可直接推理（见快速开始第 5 步，将权重路径替换为 `models/` 内文件），无需训练。
本地训练产生的全部 checkpoint 在 `GPT-SoVITS/GPT_weights_v2Pro/` 与 `GPT-SoVITS/SoVITS_weights_v2Pro/`（e5/e10/e15），试听样本在 `outputs/inference/`。

## 仓库结构

```
Chobits-Chii-TTS/
├── README.md               # 本文件
├── LICENSE                 # CC BY-NC-SA 4.0
├── .gitignore
├── requirements.txt        # 门面依赖 (引擎依赖在镜像内安装)
├── docker/                 # 推理引擎镜像 (GPT-SoVITS api_v2, 无鉴权)
│   ├── Dockerfile             # conda 环境 + 锁定 commit 的引擎源码 + 运行数据
│   └── entrypoint.sh          # 模型就位检查/下载 + 生成推理配置 + 拉起 api_v2
├── docs/
│   └── deployment.md          # 服务器部署实录 (docker/systemd/TLS/验证/迁移)
├── data/                   # 清洗后的训练数据 (由 tools/clean_dataset.py 生成)
│   ├── wavs/                  # 483 个音频片段 (不入库, 由脚本从 Chobits-Chii-Voice 复制)
│   ├── metadata.csv           # 文件名|文本
│   ├── gpt_sovits.list        # GPT-SoVITS 训练格式 (含本机绝对路径, 不入库)
│   └── cleaning_report.csv    # 清洗记录 (修复/剔除明细, 供人工复核)
├── tools/                  # 工具脚本
│   ├── clean_dataset.py       # 数据集清洗 (修复 Whisper 误转写, 剔除脏条目)
│   ├── setup_env.sh           # 环境一键搭建 (Miniconda + install.sh + 版本修复, 幂等)
│   ├── server.py              # TTS HTTP 门面 (API Key 鉴权 + 限流 + OpenAI 垫片, 代理容器内引擎)
│   └── start_tts_api.sh       # 启动门面 (可直接运行或供 systemd 调用)
├── training/               # 训练流水线
│   └── train_chii.py          # 预处理 + SoVITS/GPT 微调一键驱动 (幂等, 可续跑)
├── examples/               # 示例
│   ├── ref_text.txt           # 参考音频文本 (ep07_00783.96s)
│   └── target_text.txt        # 推理示例文本
├── models/                 # 导出的模型权重与参考音频 (不入库, 见"模型文件")
├── GPT-SoVITS/             # GPT-SoVITS 框架 (git clone, 不入库)
└── outputs/                # 试听音频等产物 (不入库)
```

## 快速开始（从零复现）

以下步骤在 Ubuntu 24.04 + RTX 4060 Laptop 8GB 上实测通过（2026-08）。国内网络已按镜像优化。

### 1. 获取训练数据

```bash
# 从 Hugging Face 下载 Chobits-Chii-Voice 数据集 (含 wavs/, metadata.csv, transcripts.csv)
git clone https://huggingface.co/datasets/chenxin199305/Chobits-Chii-Voice
```

### 2. 搭建环境

```bash
# conda 环境 (官方测试配置: Python 3.10 + PyTorch CUDA 12.8)
conda create -n GPTSoVits -c conda-forge --override-channels python=3.10 pip -y
conda activate GPTSoVits

# 克隆框架 (锁定到本项目验证过的 commit, 避免 main 分支漂移)
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS && git checkout 48b1a0169a28582a8984402f82cf438d3bfa6aca

# 一键安装 (依赖 + 预训练模型, 模型走 ModelScope 源; 非 TTY 环境需 TERM=xterm)
TERM=xterm bash install.sh --device CU128 --source ModelScope
```

> 也可以直接 `bash tools/setup_env.sh` 自动完成本步全部内容（Miniconda、conda 环境、
> 克隆、install.sh 及下方全部修复），幂等可重复执行。

install.sh 完成后，**按顺序**执行以下修复（2026-08 时点的上游版本兼容问题，已全部内置于 `tools/setup_env.sh`）：

```bash
# a) cu128 索引现已解析出 cu130 构建的 torch (需 580+ 驱动), 锁定到 cu128 版
pip install --force-reinstall "torch==2.11.0+cu128" "torchaudio==2.11.0+cu128" "torchcodec==0.11.1+cu128" \
  --index-url https://download.pytorch.org/whl/cu128

# b) fastapi 0.141/starlette 1.6 与 gradio 4.x 不兼容, 回退
pip install 'fastapi[standard]==0.115.6' 'starlette==0.41.3' 'uvicorn==0.32.0'

# c) torchcodec 依赖: npp 动态库 + ffmpeg<=8
pip install nvidia-npp-cu12
conda install -c conda-forge --override-channels 'ffmpeg=7' -y

# d) nltk 3.10 需要 punkt_tab (install.sh 下载的 nltk_data 不含);
#    raw.githubusercontent.com 被限速时走 jsDelivr 镜像
python -m nltk.downloader punkt_tab || \
  (curl -fL -o /tmp/punkt_tab.zip https://cdn.jsdelivr.net/gh/nltk/nltk_data@gh-pages/packages/tokenizers/punkt_tab.zip \
   && unzip -q -o /tmp/punkt_tab.zip -d "$CONDA_PREFIX/nltk_data/tokenizers")
```

> PyPI 下载缓慢时换清华源 `-i https://pypi.tuna.tsinghua.edu.cn/simple`；
> torch cuXXX 系列可用上海交大镜像
> `pip install torch torchcodec --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu128`
> （阿里云与清华均已下线 pytorch-wheels 镜像）。

### 3. 数据清洗

```bash
# 回到本仓库根目录, 生成 data/ (修复 Whisper 误转写 18 条, 剔除 4 条)
python3 tools/clean_dataset.py --src /path/to/Chobits-Chii-Voice/dataset
```

### 4. 训练

```bash
# 预处理 + SoVITS + GPT 一键完成 (幂等, 中断后可续跑)
python training/train_chii.py
```

训练产物：`GPT-SoVITS/SoVITS_weights_v2Pro/chii_e*.pth`（声学模型）与 `GPT-SoVITS/GPT_weights_v2Pro/chii-e*.ckpt`（语义模型），每 5 个 epoch 保存一次（首轮历史产物为 `chi_*`/`chi-*` 拼写，保持不动）。

### 5. 推理

```bash
cd GPT-SoVITS
version=v2Pro LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/npp/lib \
python GPT_SoVITS/inference_cli.py \
  --gpt_model GPT_weights_v2Pro/chii-e10.ckpt \
  --sovits_model SoVITS_weights_v2Pro/chii_e10_s1210.pth \
  --ref_audio ../data/wavs/ep07_00783.96s.wav \
  --ref_text ../examples/ref_text.txt --ref_language 日文 \
  --target_text ../examples/target_text.txt --target_language 日文 \
  --output_path ../outputs/inference/run1
```

`LD_LIBRARY_PATH` 前缀的作用：torchcodec 加载 `libnppicc.so.12`（见 2-c），以及
pyopenjtalk 加载新版 libstdc++（Ubuntu 20.04 系统库缺 `GLIBCXX_3.4.29`，需用 conda 环境的）。
也可以 `python webui.py` 启动浏览器界面操作（localhost:9874）。

> 权重已在 Hugging Face 发布，不想训练可直接下载（见[模型文件](#模型文件)）；
> 参考音频可用 `data/wavs/` 中任意一条 3–8 秒的片段替代
> （`ref_text.txt` 内容需与音频一致，参考 `data/metadata.csv`）。

## 部署为 HTTP 服务

推理只需要[模型文件](#模型文件)中的权重，无需训练数据。部署采用与家族其他服务一致的引擎/门面分离架构：

- **引擎**（Docker 容器）：上游 GPT-SoVITS `api_v2`，加载 chii 权重做合成，无鉴权，只发布到 `127.0.0.1:9882`
- **门面**（宿主机 `tools/server.py`）：唯一对外入口，API Key 鉴权 + 每 IP 限流 + OpenAI TTS 垫片 + 可选 TLS，监听 `0.0.0.0:9880`

```bash
# 1. 构建引擎镜像 (国内网络建议加镜像 build-arg, 见 docs/deployment.md)
docker build -t chobits-chii-tts-engine docker/

# 2. 启动引擎 (复用本仓库已有模型, 零下载; 全新机器的空卷自动下载方式见 deployment.md)
docker run -d --name chobits-chii-tts-engine \
  --gpus all --restart unless-stopped \
  -p 127.0.0.1:9882:9880 \
  -v $PWD/GPT-SoVITS/GPT_SoVITS/pretrained_models:/data/pretrained_models:ro \
  -v $PWD/GPT-SoVITS/GPT_SoVITS/text/G2PWModel:/data/G2PWModel:ro \
  -v $PWD/models:/data/models:ro \
  chobits-chii-tts-engine

# 3. 启动门面 (0.0.0.0:9880; 首次运行自动建 .venv)
export CHII_TTS_API_KEY=<随机密钥>   # 必填, 未设置会拒绝启动
bash tools/start_tts_api.sh 9880
```

生产环境用 systemd 守护门面（密钥经 `EnvironmentFile` 注入）+ docker `--restart` 守护引擎，
HTTPS（自签名证书）在门面层启用；unit / env 示例与验证步骤见 **[docs/deployment.md](docs/deployment.md)**。

OpenAI TTS 兼容调用（推荐；客户端 `baseUrl` 填 `http(s)://<服务器IP>:9880/v1`，
`GET /v1/models` 返回固定模型 `chii-tts`，`voice` 当前仅 `chii`，`response_format` 支持
`wav`/`aac`/`opus`，默认 `wav`。`wav` 为流式输出（边合成边推流，首字延迟低）；
`aac`/`opus` 为合成完成后一次性返回）：

```bash
curl -k -X POST https://<服务器IP>:9880/v1/audio/speech \
  -H "Authorization: Bearer <API_KEY>" \
  -H 'Content-Type: application/json' \
  -d '{"model": "chii-tts", "input": "ちぃ、秀樹のこと、大好き。", "voice": "chii"}' \
  -o out.wav
# 未启用 TLS 时把 https 换成 http、去掉 -k 即可
```

原生 `/tts` 调用示例（GET query / POST JSON 原样透传到引擎；`ref_audio_path` 为**引擎容器内**路径，
默认数据卷挂载下即 `/data/models/...`；密钥用 `Authorization: Bearer` 或 `?api_key=` 传递）：

```bash
curl -k -G https://<服务器IP>:9880/tts \
  -H "Authorization: Bearer <API_KEY>" \
  --data-urlencode "text=ちぃ、秀樹のこと、大好き。" \
  --data-urlencode "text_lang=ja" \
  --data-urlencode "ref_audio_path=/data/models/ref_audio.wav" \
  --data-urlencode "prompt_lang=ja" \
  --data-urlencode "prompt_text=秀樹は地位を拾ってくれた" \
  --data-urlencode "media_type=wav" -o out.wav
# 未启用 TLS 时把 https 换成 http、去掉 -k 即可
```

其他环境变量：`CHII_TTS_RATE_LIMIT`（`/tts` 与 `/v1/audio/speech` 每 IP 每分钟限流次数，默认 60，0 关闭）；
`CHII_TTS_ENGINE_URL`（引擎地址，默认 `http://127.0.0.1:9882`）；
`CHII_TTS_SSL_CERTFILE` / `CHII_TTS_SSL_KEYFILE`（同时设置时以 HTTPS 启动）；
`CHII_TTS_REF_AUDIO` / `CHII_TTS_REF_TEXT_FILE`（覆盖 OpenAI 垫片 `chii` 音色的参考音频/参考文本路径，
前者为容器内路径，后者为宿主机路径）。

注意在云安全组放行 TCP 9880（9882 只绑回环，无需放行）；对外提供服务须遵守 [CC BY-NC-SA 4.0](#许可协议)（非商业）。
面向公众分发应用时建议由后端服务代为调用，不要把唯一密钥嵌进客户端。

## 许可协议

本项目的派生内容遵循数据集的 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh-hans)（署名-非商业性使用-相同方式共享）协议：

- **署名 (BY)**：使用时须注明来源。
- **非商业性使用 (NC)**：不得用于商业目的。
- **相同方式共享 (SA)**：衍生作品须以相同协议发布。

## 免责声明

- 本项目仅用于学术研究与个人学习，不构成对原作品版权的任何主张。
- 使用本项目训练的模型所生成的内容，不得用于侵犯原作品及相关声优（田中理惠）权益的用途。
- 若权利方提出要求，本项目将被下架。

## 相关项目

- [Chobits-Chii-Voice](https://github.com/Anime2Real/Chobits-Chii-Voice) — 小叽语音数据集（本项目的数据来源）
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) — 底层语音合成框架
