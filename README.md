<div align="center">
	<h1>Chobits</h1>
	<p><b>叽～</b> 小叽的语音合成模型</p>
	<p>《人形电脑天使心》(Chobits) 中 <b>小叽 (Chii / ちぃ)</b> 角色的 TTS（语音合成）模型项目，基于 Fun-CosyVoice3-0.5B 微调。</p>
	<p>
		<a href="https://madewithlove.org.in"><img alt="Made with Love" src="https://img.shields.io/badge/Made%20with-Love-ff69b4.svg"></a>
		<a href="https://github.com/Anime2Real/Chobits-Chii-TTS"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-Chobits--Chii--TTS-181717?logo=github"></a>
		<a href="https://huggingface.co/datasets/chenxin199305/Chobits-Chii-Voice"><img alt="Dataset: Chobits-Chii-Voice" src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Chobits--Chii--Voice-yellow"></a>
		<a href="https://huggingface.co/chenxin199305/Chobits-Chii-TTS-CosyVoice"><img alt="Hugging Face Model" src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-Chobits--Chii--TTS--CosyVoice-yellow"></a>
		<a href="https://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="License: CC BY-NC-SA 4.0" src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg"></a>
		<img alt="Language: Japanese" src="https://img.shields.io/badge/Language-Japanese-green.svg">
	</p>
</div>

> 💖 如果这个项目对你有帮助，欢迎在 [GitHub](https://github.com/Anime2Real/Chobits-Chii-TTS) 点个 Star —— 你的支持能让更多人发现小叽！

> 🌿 本分支（`cosy-voice`）是 **CosyVoice3 方案**（验证通过，待上线）。仓库 `main` 分支为[分支索引](https://github.com/Anime2Real/Chobits-Chii-TTS)；
> GPT-SoVITS 生产方案（当前线上服务）见 [`gpt-sovits`](https://github.com/Anime2Real/Chobits-Chii-TTS/tree/gpt-sovits) 分支。

> ✅ CosyVoice 方案已验证通过：zero-shot 基线与 21 分钟数据 SFT 微调均在 RTX 4090 实测跑通，
> 人工试听选定 **SFT 微调版**为正式路线；引擎容器与门面端到端联调完成。
> 权重已发布到 [Hugging Face](https://huggingface.co/chenxin199305/Chobits-Chii-TTS-CosyVoice)，见[模型文件](#-模型文件)。
> 完整调研与实测记录见 **[docs/cosyvoice-feasibility.md](docs/cosyvoice-feasibility.md)**。

《人形电脑天使心》(Chobits) 中 **小叽 (Chii / ちぃ)** 角色的 TTS（语音合成）模型项目。

本分支以 [Chobits-Chii-Voice](https://github.com/Anime2Real/Chobits-Chii-Voice) 数据集为训练数据，基于
[CosyVoice](https://github.com/FunAudioLLM/CosyVoice)（Fun-CosyVoice3-0.5B，代码与权重均 Apache-2.0）
微调出小叽声线的文本转语音模型。

> ⚠️ 注意：原始动画音频的版权归其权利方所有。本项目仅供学习与研究使用，请勿用于商业用途。

## 🎧 示例试听

下载权重后本地生成（见[快速开始](#-快速开始从零复现)第 5 步）：

```bash
python tools/cosyvoice_zeroshot_eval.py \
  --model_dir CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-chii-sft \
  --ref_audio models/ref_audio.wav --ref_text models/ref_text.txt \
  --target_text examples/target_text.txt --out_dir outputs/cosyvoice_sft
```

合成文本（`examples/target_text.txt`）：`ちぃ、秀樹のこと、大好き。ずっと一緒にいる。`
（Fun-CosyVoice3-0.5B chii SFT · zero-shot 模式合成，输出 24kHz `outputs/cosyvoice_sft/zero_shot.wav`）

## 📊 数据

训练数据与 `gpt-sovits` 分支相同：[Chobits-Chii-Voice](https://huggingface.co/datasets/chenxin199305/Chobits-Chii-Voice)
（487 条 / 约 21.4 分钟日语，22050Hz 单声道），经 `tools/clean_dataset.py` 清洗后为 483 条。
CosyVoice 侧的特殊处理：**日语文本全部假名化**（pyopenjtalk NJD → 空格分隔片假名），
原因见下方"方案要点"。

## 🔑 方案要点

- **模型**：Fun-CosyVoice3-0.5B（监督多任务 speech tokenizer + Qwen 系 LLM + chunk-aware 因果 flow matching + HiFT），
  支持原生 token 级双向流式（GPT-SoVITS 无此能力，只能门面按句伪流式）
- **微调**：llm + flow 两模块全量 SFT（hifigan 不动），lr 1e-5、10 epoch、单卡 4090 约 4 分钟训完；
  小数据过拟合快，用 val_best 权重平均自动回退到轻适配 checkpoint（细节见调研文档 §9.5）
- **日语文本前端（必需）**：CosyVoice3 的 `contains_chinese()` 会把含汉字的日语文本路由到中文前端，
  轻则发音崩坏重则推理崩溃——训练与推理两侧都先做假名化（pyopenjtalk，助词は→ワ按实际发音），
  该逻辑已内置于训练数据准备脚本与推理引擎，无需手工处理

## 📈 实测性能（RTX 4090 · fp16 · 裸 HF 推理）

| 指标 | SFT 微调版 | zero-shot 基线 |
| --- | --- | --- |
| 非流式 RTF | 0.348 | 0.320 |
| 流式首包延迟 | 1268ms | 1052ms |
| 峰值显存（torch 分配） | ~5GB | ~5GB |
| whisper-small 回译 | チー、**ヒデキ**なこと、大好き、ずっと一緒にいる。 | チー、**秘敵**のこと、大好き、ずっと一緒にいる |

SFT 版发音正确性更好（人名「秀樹」说清），人工试听音色亦优于基线。官方宣称的 150ms 首包是
TensorRT-LLM + 高端卡全优化下界；裸 HF 推理约 1s，部署机可再开 TRT-LLM 压缩。

## 📦 模型文件

SFT 微调权重已发布到 Hugging Face（自包含推理包，含基座推理必需文件，下载即可推理，无需训练）：
**[huggingface.co/chenxin199305/Chobits-Chii-TTS-CosyVoice](https://huggingface.co/chenxin199305/Chobits-Chii-TTS-CosyVoice)**

```bash
# 下载模型 (约 5.6GB)
pip install huggingface_hub
hf download chenxin199305/Chobits-Chii-TTS-CosyVoice \
  --local-dir CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-chii-sft
```

| 文件 | 说明 |
| --- | --- |
| `llm.pt` / `flow.pt` | SFT 微调后的语义/声学模型（val_best 平均，各约 2.4GB/1.3GB） |
| `hift.pt` / `campplus.onnx` / `speech_tokenizer_v3.onnx` / `CosyVoice-BlankEN/` | 基座推理必需文件，来自 `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` |
| `models/ref_audio.wav` | 参考音频（ep07「秀樹は地位を拾ってくれた」，4.2s，zero-shot 推理仍需要；在 GPT-SoVITS 版模型仓库 [Chobits-Chii-TTS](https://huggingface.co/chenxin199305/Chobits-Chii-TTS) 中） |

推理不需要的基座文件（`llm.rl.pt`、`flow.decoder.estimator.fp32.onnx`、`speech_tokenizer_v3.batch.onnx`）
未包含在内，需要时从[基座仓库](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)补齐。

## 🗂 仓库结构

本分支独有的部分（门面 `tools/server.py`、`deploy/`、`tests/` 等与 `gpt-sovits` 分支共享，零改动复用）：

```
Chobits-Chii-TTS/
├── docs/
│   └── cosyvoice-feasibility.md  # CosyVoice 调研 + 4090 实测 + SFT 记录 + 容器封装 (主文档)
├── docker/
│   └── cosyvoice/                # CosyVoice 引擎镜像 (与 GPT-SoVITS 引擎并存)
│   │   ├── Dockerfile               # conda py3.10 + torch cu121 + pin 074ca6dc 的 CosyVoice 源码
│   │   ├── engine_server.py         # 引擎 shim: 与 api_v2 相同 /tts 契约, 内部假名化 + SFT 模型推理
│   │   └── entrypoint.sh            # 模型挂载检查 + 可选 ModelScope 基座下载
├── training/
│   ├── prepare_cosyvoice_data.py   # 训练数据准备 (HF 数据集 wav + 假名化 + train/dev 拆分)
│   └── cosyvoice3_chii_sft.yaml    # SFT 训练配置 (max_epoch 10, use_spk_embedding)
├── tools/
│   └── cosyvoice_zeroshot_eval.py  # zero-shot 验证脚本 (假名化 + RTF/首包/显存测量, --raw 复现 bug)
├── CosyVoice/                      # CosyVoice 框架 clone (不入库)
│   └── pretrained_models/
│       └── Fun-CosyVoice3-0.5B-chii-sft/  # SFT 微调模型目录 (不入库, 见"模型文件")
└── (其余同 gpt-sovits 分支)
```

## 🚀 快速开始（从零复现）

以下步骤在 Ubuntu 22.04 + RTX 4090 24GB 上实测通过（2026-10）。国内网络已按镜像优化。

### 1. 获取训练数据

```bash
# 从 Hugging Face 下载 Chobits-Chii-Voice 数据集 (含 wavs/)
git clone https://hf-mirror.com/datasets/chenxin199305/Chobits-Chii-Voice /tmp/chii-voice-hf
# 生成清洗后的 data/metadata.csv (修复 Whisper 误转写 18 条, 剔除 4 条)
python3 tools/clean_dataset.py --src /tmp/chii-voice-hf
```

### 2. 搭建环境

```bash
# conda 环境 (Python 3.10 + torch 2.3.1+cu121)
conda create -n CosyVoice -c conda-forge --override-channels python=3.10 pip -y
conda activate CosyVoice
pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu121

# 克隆框架 (pin 到验证过的 commit) 并装依赖
git clone https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice && git checkout 074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc
git submodule update --init --recursive
# 推理不需要 deepspeed/tensorrt-cu12*, 剔除后安装 (whisper 为 sdist 需特殊处理);
# pyopenjtalk 需另解包 open_jtalk 词典 —— 完整命令与坑见 docs/cosyvoice-feasibility.md §9.4
```

### 3. 下载预训练基座

```bash
python -c "from modelscope import snapshot_download; \
  snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', \
                    local_dir='CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B')"
```

### 4. SFT 微调

```bash
# 数据准备 (假名化 + wav.scp/text/utt2spk/instruct + train/dev 拆分)
python training/prepare_cosyvoice_data.py /tmp/chii-voice-hf/wavs
# 之后按官方 examples/libritts/cosyvoice3/run.sh 流程: 提取 embedding/speech token →
# parquet → 单卡 torch_ddp 训 llm 与 flow → val_best 权重平均。
# 完整命令、deepspeed 条件导入补丁与过拟合注意事项见 docs/cosyvoice-feasibility.md §9.5
```

### 5. 推理验证

见上方[示例试听](#-示例试听)的命令，产物与性能指标写入 `outputs/cosyvoice_sft/`。

## 🌐 部署为 HTTP 服务

部署架构与 `gpt-sovits` 分支完全一致：**引擎 Docker 容器（绑回环）+ 宿主机门面
`tools/server.py`（唯一对外入口：API Key 鉴权 + 限流 + OpenAI 垫片）**。门面零改动，
仅引擎镜像换成 `docker/cosyvoice/`：

```bash
# 1. 构建引擎镜像 (国内可加镜像 build-arg, 见 docker/cosyvoice/Dockerfile 头部注释)
docker build -t chobits-chii-cosyvoice-engine docker/cosyvoice/

# 2. 启动引擎 (挂载 SFT 模型目录与参考音频; 需要 nvidia-container-toolkit)
#    空目录挂载时容器自动从 Hugging Face 拉 chii SFT 推理包 (MODEL_SOURCE=hf-sft 默认)
docker run -d --name chobits-chii-cosyvoice-engine \
  --gpus all --restart unless-stopped \
  -p 127.0.0.1:9882:9880 \
  -v $PWD/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-chii-sft:/data/model:ro \
  -v $PWD/models:/data/models:ro \
  chobits-chii-cosyvoice-engine

# 3. 启动门面 (与 gpt-sovits 分支完全相同, 引擎地址默认值即 9882)
export CHII_TTS_API_KEY=<随机密钥>
bash tools/start_tts_api.sh 9880
```

OpenAI 兼容调用、健康检查、环境变量、systemd/Caddy 生产部署等均与 `gpt-sovits` 分支相同，
见 [docs/deployment.md](docs/deployment.md)；引擎侧契约细节见 [docs/cosyvoice-feasibility.md](docs/cosyvoice-feasibility.md) §10。

## 📄 许可协议

本项目的派生内容遵循数据集的 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh-hans)（署名-非商业性使用-相同方式共享）协议：

- **署名 (BY)**：使用时须注明来源。
- **非商业性使用 (NC)**：不得用于商业目的。
- **相同方式共享 (SA)**：衍生作品须以相同协议发布。

底层框架 CosyVoice 的代码与模型权重均为 Apache-2.0，与本项目许可无冲突。

## ⚠️ 免责声明

- 本项目仅用于学术研究与个人学习，不构成对原作品版权的任何主张。
- 使用本项目训练的模型所生成的内容，不得用于侵犯原作品及相关声优（田中理惠）权益的用途。
- 若权利方提出要求，本项目将被下架。

## 🙏 相关项目

- [Chobits-Chii-Voice](https://github.com/Anime2Real/Chobits-Chii-Voice) — 小叽语音数据集（本项目的数据来源）
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) — 底层语音合成框架
