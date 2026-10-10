# CosyVoice 引入可行性调研

> 调研日期：2026-10-09 · 分支：`cosy-voice`
> 结论先行：**技术上可引入，许可无障碍（Apache-2.0 全栈）；zero-shot 与 SFT 均已在
> RTX 4090 上实测跑通（第 9 节），日语假名化补丁验证为必需（不打补丁直接崩溃）**。
> **选型已定（2026-10-10 人工试听）：采用 SFT 微调版**（`Fun-CosyVoice3-0.5B-chii-sft`），
> 音色与发音均优于预训练 zero-shot 基线。~~下一步：封装引擎容器接入现有门面（阶段 2）~~
> **阶段 2 已完成（2026-10-10）**：引擎容器封装于 `docker/cosyvoice/`，宿主机契约验证 +
> 门面端到端 + 镜像构建全部通过，门面 `tools/server.py` 零改动（见第 10 节）。

## 1. 框架概览

[CosyVoice](https://github.com/FunAudioLLM/CosyVoice)（2026 年起仓库迁至 QwenAudio org，旧链接自动重定向）是阿里通义开源的多语言 TTS 框架，GitHub 23.9k stars。版本沿革：

| 版本 | 发布 | 架构要点 | 开源模型 |
| --- | --- | --- | --- |
| CosyVoice 1 | 2024-07 | 监督语义 token（VQ，25Hz）+ LLM + 条件 flow matching + HiFT vocoder | CosyVoice-300M / -SFT / -Instruct（~2.6GB 权重） |
| CosyVoice 2 | 2024-12 | FSQ 替代 VQ；Qwen 系 LLM 做 backbone；chunk-aware 因果 flow matching，单模型统一流式/非流式 | CosyVoice2-0.5B（~4.9GB） |
| Fun-CosyVoice 3 | 2025-05 论文 / 2025-12-15 开源 0.5B | 多任务 speech tokenizer（ASR+情感+语种+说话人）；可微 reward model RL 后训练；训练数据扩到 100 万小时 | Fun-CosyVoice3-0.5B-2512（~11.7GB，含 RL 权重）；论文中的 1.5B 未开源 |

官方评测（README）：CV3-0.5B test-zh CER 1.21% / 说话人相似度 78.0%，RL 版 CER 0.81%，处于开源第一梯队。官方当前主推 Fun-CosyVoice3-0.5B。

论文：[CV1 arXiv:2407.05407](https://arxiv.org/abs/2407.05407)、[CV2 arXiv:2412.10117](https://arxiv.org/abs/2412.10117)、[CV3 arXiv:2505.17589](https://arxiv.org/abs/2505.17589)。

## 2. 许可

- **代码与全部模型权重均为 Apache-2.0**（HF API `cardData.license` 逐一确认：Fun-CosyVoice3-0.5B-2512、CosyVoice2-0.5B、CosyVoice-300M 全系），无 gated、无额外限制条款，可商用、可分发衍生模型。
- 本项目自身受数据集 CC BY-NC-SA 4.0 约束（非商业），引入 Apache-2.0 引擎无冲突——比 GPT-SoVITS（MIT 代码 + 各预训练模型混合许可）的许可结构更清晰。

## 3. 语言支持与日语文本前端的坑

- CV1 支持中/英/日/韩/粤；CV2 收缩为以中英为主（CV3 论文明确将语言覆盖列为 CV2 短板）；**CV3 恢复 9 语言（含日语）+ 跨语言 zero-shot 克隆**。
- **已知 bug（重要）**：CV3 的 `contains_chinese()` 会把含汉字的日语文本整体路由到中文前端预处理，导致日语合成崩坏，且模型对汉字有中文读法 bias。社区 workaround：汉字→假名转换（pyopenjtalk/pykakasi）+ 助词 は→わ 替换 + prompt 前缀，修后日语 CER 可降到 0.045（[Zenn 实测，2026-06](https://zenn.dev/mohhh_ok/articles/2026-06-cosyvoice-qwen3-tts-apache2-mac?locale)）。仓库 2026-05-25 后无新提交，该 bug 大概率仍未修。
- 社区主观评测中 CosyVoice 日语自然度（★★★☆☆）明显低于日语专精方案 Style-BERT-VITS2 / AivisSpeech（[Qiita 5 模型对比](https://qiita.com/GeneLab_999/items/bc07147b589a93bf6114)）。本项目训练侧已用 pyopenjtalk，汉字→假名能力现成，打补丁的成本可控。

## 4. 与本项目数据的适配（21.4 分钟 / 483 条日语单人数据）

两条路线：

**A. Zero-shot 克隆（推荐先试）**：3–10 秒参考音频即可，零训练。CV2 论文 Table 5 零样本对比中说话人相似度碾压 GPT-SoVITS（SS 0.745 vs 0.405，人类 0.697）。可复用现有 `models/ref_audio.wav`（4.2s）直接出基线。这是相对现有方案**免训练**的最大卖点。

**B. SFT 微调**：官方提供全栈训练脚本（`examples/libritts/cosyvoice3/run.sh`，llm/flow/hifigan 均可训，DDP/DeepSpeed），流程语言无关。但：
- 官方答复**单说话人微调最低约 1 小时数据**（[Issue #943](https://github.com/FunAudioLLM/CosyVoice/issues/943)）；本项目 21 分钟低于下限，可行但有过拟合风险（社区经验：小数据 3–5 epoch）。HF 上有小语种小数据微调先例（如阿尔巴尼亚语）。
- 显存：0.5B 全量微调社区实践需 24GB 级单卡（3090/4090 也有人报 OOM）；**8GB 卡微调基本不可行**，且官方无 LoRA 脚本。要微调须借云 GPU。
- 训练数据的日语文本同样受汉字前端 bug 影响，需先做假名化预处理。

对比：GPT-SoVITS 微调路线（当前方案）正是以"1 分钟级数据即可微调、8GB 卡可训"见长，已产出人工试听通过的 chii 权重。**数据效率上现有方案占优。**

## 5. 性能

### 5.1 官方/论文指标

- 官方宣传流式**首包延迟低至 150ms**——注意这是高端 GPU + TRT-LLM + 缓存全优化的下界，论文本身只给了延迟公式，无实测毫秒表；消费级部署实测普遍远高于此。
- CV2 流式 vs 非流式质量几乎无损（test-zh CER 1.45% 持平；仅 test-hard 6.83%→8.08% 退化）。
- 官方未发布裸 PyTorch RTF（不确定项）。

### 5.2 实测数据（第三方/官方 runtime，单卡 L20 除非注明）

| 场景 | 指标 | 来源 |
| --- | --- | --- |
| CV2 HF transformers bs1 | RTF 0.249 | 官方 triton_trtllm README |
| CV2 TensorRT-LLM bs1 | RTF 0.099（~4x 加速） | 同上 |
| CV2 TRT-LLM 流式首包（并发1，开音色缓存） | 190–220ms | 同上 |
| CV3 TRT-LLM 离线 bs1 / bs16 | RTF 0.109 / 0.050 | 官方 runtime |
| CV3 流式首包（并发4） | ~750ms，明显高于 CV2 | 同上 |
| RTX 4080 + TRT + 缓存 | 首包 ~1.3s | 火山引擎·小智AI 实测 |
| 未加速 HF 实现首包 | 0.8–2s | CSDN 星图实测 |

vLLM（官方支持 ≥0.11.0 V1）第三方报告首 token 800ms→150ms（未独立验证）。

### 5.3 显存（第三方实测，官方无数字）

- CV2-0.5B 推理约 **6–8GB**；CV-300M 4–6GB；长文本（600+ 字）7–9GB+；每路并发 +2.5–3GB；流式比非流式省 0.6–0.9GB；**开 TensorRT 显存约翻倍**。
- CV3-0.5B 有"4GB 内可跑"的报告（fp16 纯推理，未经证实）。

### 5.4 对 RTX 4060 Laptop 8GB 的判断（推断，无直接实测）

- CV2/CV3-0.5B fp16 流式推理可行但紧贴上限：不能开 TRT、控制文本长度、单并发；预期 RTF 0.3–0.8、首包 0.5–1.5s 量级。
- 对比：GPT-SoVITS v2Pro 推理显存约 4GB，8GB 卡上余量明显更宽裕。
- 微调在 8GB 卡上不可行（CosyVoice 与 GPT-SoVITS 均如此——GPT-SoVITS v2Pro 首轮也是本机 8GB 跑通的，这是现有方案的既有优势）。

### 5.5 流式能力是关键差异

- CosyVoice2/3 **原生 token 级双向流式**：LLM 按 5:15 交错文本/语音 token，每 chunk ≈0.6s 音频即解码返回；可直接对接上游 LLM 的流式输出做语音对话。
- GPT-SoVITS 无原生流式，本项目目前的"门面按句串行合成 + PCM 流式转发"是伪流式。若家族 roadmap 中有 LLM→TTS 全链路低延迟对话需求，CosyVoice 的架构优势明显；对当前的"整句文本进、wav 出"用法，收益有限。

## 6. 部署集成分析

现有架构（引擎 Docker 容器绑回环 + `tools/server.py` 门面鉴权/限流/OpenAI 垫片）可直接复用：

- CosyVoice 官方 `runtime/python/` 自带 **FastAPI / gRPC server + Dockerfile**（无预构建镜像，需自建），可替换 `docker/` 内 GPT-SoVITS 引擎，门面基本不用动（`CHII_TTS_ENGINE_URL` 指向新引擎，OpenAI 垫片的 voice→参考音频映射逻辑保留）。
- 高阶部署另有 Triton + TensorRT-LLM（docker compose 一键）与 vLLM 路径，但两者在 8GB 卡上都不适用（显存翻倍/额外开销），属服务器侧升级选项。
- 生态风险：仓库 2026-05 后停止更新（673 open issues），org 迁移到 QwenAudio 后与 Qwen3-TTS 生态靠拢，CosyVoice 主线维护优先级存疑。

## 7. 风险与不确定项

1. **日语开箱质量**：`contains_chinese` bug 未修，需自带假名化前端补丁（工作量小但必须有）。
2. **微调数据量**：21 分钟低于官方 1 小时下限，SFT 效果无背书；8GB 卡无法微调，需云 GPU。
3. **8GB 显存余量**：CV2/CV3 推理 6–8GB，比 GPT-SoVITS（~4GB）紧张；4060 Laptop 无现成实测报告，RTF/首包数字为推断。
4. **维护活跃度**：主线近半年无提交；CV3 论文 1.5B 版本未开源。
5. 若日语自然度是硬指标，社区评测中 Style-BERT-VITS2 / AivisSpeech 优于 CosyVoice，应纳入横向对比。

## 8. 结论与建议路线

引入 CosyVoice **可行且有明确收益点**（Apache-2.0 全栈、原生流式、免训练 zero-shot、官方指标第一梯队），但**不构成对现有方案的全面替代**：日语前端需打补丁、微调数据与显存条件都不如现有路线宽松。建议分阶段验证，每阶段有明确去/留判据：

1. ~~**Zero-shot 基线实测**~~：✅ 已完成（2026-10-09，RTX 4090 24GB），见 9.1–9.4 节。
2. ~~**若基线音质可接受**：封装为引擎容器（复用门面），对比流式体验~~：✅ 已完成（2026-10-10），见第 10 节；日语自然度若仍不满，横向测 Style-BERT-VITS2。
3. ~~**若 zero-shot 音色相似度不足，评估 SFT 微调**~~：✅ 已完成（2026-10-10，本机 4090），见 9.5 节——24GB 可全量 SFT，小数据过拟合快但 val_best 平均有效，发音正确性有改善。

## 9. 本机实测（2026-10-09 · RTX 4090 24GB · 驱动 580.159 / CUDA 13.0）

阶段 1 已在本机完成。环境：conda `CosyVoice`（Python 3.10 + torch 2.3.1+cu121），CosyVoice 框架 clone 在
`CosyVoice/`（不入库），权重 `Fun-CosyVoice3-0.5B-2512` 经 ModelScope 下载至
`CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B`（约 9.1GB）。验证脚本 `tools/cosyvoice_zeroshot_eval.py`：
用 `models/ref_audio.wav`（4.2s）+ 假名化参考文本做 zero-shot 克隆，合成 `examples/target_text.txt`
同款文本，产物在 `outputs/cosyvoice/`（不入库）。

### 9.1 性能实测（fp16，HF transformers 原生推理，未开 TRT/vLLM）

| 指标 | 实测值 | 备注 |
| --- | --- | --- |
| 模型加载（热缓存） | ~5s | 冷启动首次约 65s（含 wetext 前端模型下载） |
| 非流式 RTF | **0.32** | 1.34s 合成 4.20s 音频（warmup 后计时） |
| 流式首包延迟 | **1052ms** | 首个 chunk 约 1.6s 音频 |
| 流式整体 RTF | 0.39 | 1.61s 合成 4.12s 音频，共 3 个 chunk |
| 峰值显存（torch 分配） | **4.97GB** | `torch.cuda.max_memory_allocated` |
| 峰值显存（nvidia-smi 整卡） | 8.39GB | 含桌面环境常驻 ~2.7GB，本进程增量约 5.7GB |

要点：4090 上裸 HF 推理首包约 1s，验证了"官方 150ms 是 TRT-LLM + 高端卡全优化下界"的判断；
RTF 0.32–0.39 意味着短句合成亚秒级出整句，整卡峰值显存增量 5.7GB——**4060 Laptop 8GB 跑
CV3-0.5B fp16 流式预计可行但贴顶**（桌面占用约 0.5–1GB 时余量不足 1GB，需实测确认）。

### 9.2 日语前端 bug 实测确认

- 假名化（pyopenjtalk NJD → 空格分隔片假名，助词は→ワ按发音处理）后合成正常：
  whisper-small 回译 zero-shot 输出为「チー、秘敵のこと、大好き、ずっと一緒にいる」，与目标文本
  「ちぃ、秀樹のこと、大好き。ずっと一緒にいる。」一致（「秘敵」系 ASR 同音误转，真人音频
  `ref_audio.wav` 亦被转作「井的」），可懂度合格。pyopenjtalk 的 `pron` 字段可能混入
  U+2019 撇号（如 ダイス’キ），脚本已做剔除。
- **不打补丁的 raw 模式直接崩溃**：含汉字文本被路由到中文前端后产出退化 token 序列，
  hifigan F0 predictor 卷积报 `Kernel size can't be greater than actual input size`。
  即日语场景下假名化补丁不是优化而是**必需**。

### 9.3 与现有方案的初步对照（待人工试听确认）

| | GPT-SoVITS v2Pro e10（现有） | Fun-CosyVoice3-0.5B zero-shot |
| --- | --- | --- |
| 训练成本 | 已微调（21min 数据 + 本机训练） | **零训练**（4.2s 参考音频） |
| 采样率 | 32kHz | 24kHz |
| whisper-small 回译 | チン、イデキのこと大好き。ずっと一緒にいる。 | チー、秘敵のこと、大好き、ずっと一緒にいる |
| 流式 | 门面按句伪流式 | 原生 token 级流式，首包 1.05s（裸跑） |

ASR 回译仅供参考（短句 + whisper-small 误差大），**音色相似度与自然度需人工 AB 试听**：
`outputs/inference/run1/output.wav`（现有）vs `outputs/cosyvoice/zero_shot.wav` / `zero_shot_stream.wav`。

### 9.4 复现步骤（本机）

```bash
# 环境 (conda env 已建好, 名为 CosyVoice; torch 走 SJTU 镜像, 其余走清华镜像)
conda create -n CosyVoice -c conda-forge --override-channels python=3.10 pip -y
conda activate CosyVoice
pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu121
pip install -r CosyVoice/requirements.txt pyopenjtalk==0.4.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
# 注意: requirements 中的 deepspeed/tensorrt-cu12* 推理不需要, 本次安装已剔除;
# openai-whisper==20231117 为 sdist, 需先 pip install 'setuptools<81' wheel 并用 --no-build-isolation 安装;
# pyopenjtalk 需解包 open_jtalk 词典 (来源同 docker/Dockerfile: ModelScope XXXXRT/GPT-SoVITS-Pretrained)

# 权重 (ModelScope)
python -c "from modelscope import snapshot_download; \
  snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', \
                    local_dir='CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B')"

# 验证 (假名化; 加 --raw 可复现汉字 bug 崩溃)
python tools/cosyvoice_zeroshot_eval.py \
  --model_dir CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B \
  --ref_audio models/ref_audio.wav --ref_text models/ref_text.txt \
  --target_text examples/target_text.txt --out_dir outputs/cosyvoice
```

### 9.5 SFT 微调实测（2026-10-10 · 同机 4090）

阶段 3 的微调路线也已在本机验证（24GB 显存使本机 SFT 可行）。数据：HF 数据集 483 条全部匹配到 wav，
文本假名化后按 seed=1986 拆 train 473 / dev 10（准备脚本 `training/prepare_cosyvoice_data.py`），
按官方 `examples/libritts/cosyvoice3` 流程提取 embedding/speech token 并转 parquet。
配置 `training/cosyvoice3_chii_sft.yaml`（官方 yaml 副本，改动：max_epoch 200→10、log_interval→10、
`use_spk_embedding` 按 yaml 注释 SFT 置 True）；lr 1e-5、constantlr、AMP、单卡 torch_ddp。

| | LLM SFT | Flow SFT |
| --- | --- | --- |
| 训练 | 10 epoch，每 epoch ~35 step / ~12s，无 OOM | 同左 |
| train loss | 1.77 → 0.32（acc 0.23→0.80） | ~0.41 → ~0.30 |
| CV loss | **3.16 → 6.58 单调上升**（10 条超短 dev 上过拟合极快） | 0.53–0.84 震荡，无明显恶化 |
| 峰值显存 | ~18.5GB | ~14.5GB |
| val_best 平均（--num 3） | epoch 0/1/2（自动回退到轻适配权重） | epoch 3/6/9 |

**微调版推理对照**（模型目录 `CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-chii-sft/`，
同参跑 `tools/cosyvoice_zeroshot_eval.py`，产物 `outputs/cosyvoice_sft/`）：

| 指标 | SFT 版 | zero-shot 基线 |
| --- | --- | --- |
| 非流式 RTF | 0.348 | 0.320 |
| 流式首包 | 1268ms | 1052ms |
| torch 峰值显存 | 4967MiB | 4969MiB |
| whisper-small 回译 | **チー、ヒデキなこと、大好き、ずっと一緒にいる。** | チー、秘敵のこと、大好き、ずっと一緒にいる |

性能与基线持平（差异在测量噪声内）；**发音正确性改善**：人名「ヒデキ」（秀樹）说清楚了
（基线误为「秘敵」），仅「のこと」偏「なこと」一处小偏差。

**人工试听结论（2026-10-10）：SFT 版优于 zero-shot 基线，选定为正式路线。**
备用产物：`outputs/cosyvoice_sft/zero_shot.wav`（SFT）vs `outputs/cosyvoice/zero_shot.wav`（基线）。

**微调注意事项**（复现必读）：
1. `cosyvoice/bin/train.py` 与 `cosyvoice/utils/train_utils.py` 顶层 import deepspeed（torch_ddp
   用不到），需最小补丁：两处 import 包 try/except + `deepspeed.add_config_arguments` 加 None 守卫
   （本次直接改在 gitignored 的 `CosyVoice/` clone 内，未入库）。
2. 这版 train.py 用严格 `parse_args()`，**不支持** `--train_conf.max_epoch` 形式的 CLI 覆盖，
   改超参须用配置副本。
3. `onnxruntime-gpu` 可能被 `~/.local` 的 CPU 版 onnxruntime 遮蔽，speech token 提取自动回退 CPU
   （21 分钟数据约 20s，无实际影响）。
4. LLM 在小数据上过拟合极快（epoch 0 后 CV loss 即升），val_best 平均会自动回退；想要更强声线
   适配可增大 dev 集或人工挑 epoch 3–5 checkpoint 做听测。

## 10. 阶段 2：引擎容器封装（2026-10-10 完成）

SFT 模型封装为 Docker 引擎容器，**与 GPT-SoVITS 引擎同一 `/tts` 契约，门面 `tools/server.py`
一行不改**（`CHII_TTS_ENGINE_URL` 指到新引擎即完成切换）。新增文件（不入库产物除外）：

- `docker/cosyvoice/engine_server.py`：FastAPI 引擎 shim。设计要点：
  - 日语假名化在引擎内部完成（`text` 与 `prompt_text` 均过 pyopenjtalk NJD → 空格分隔片假名、
    剔除撇号），`prompt_text` 前加 `You are a helpful assistant.<|endofprompt|>`（与 SFT 训练
    instruct 序列一致），调 `AutoModel(fp16).inference_zero_shot(..., stream=..., speed=...)`；
  - `streaming_mode` 为真：首块为含 fmt 的流式 WAV 头（24kHz mono 16bit，RIFF/data 长度
    `0xFFFFFFFF` 占位，与门面 `_stream_wav_header` 同约定），后续块裸 PCM；首块在返回响应前
    预读，推理零产出/异常回 5xx，**合法输入绝不返回 200 空流**（门面预读首块判定依赖此语义）；
  - `media_type`：wav 直接封装；aac/ogg 非流式时用容器内 ffmpeg 7 子进程从裸 PCM 编码
    （aac→adts、ogg→libopus）；
  - GPU 推理由全局锁串行化（门面允许 8 路在途，CosyVoice 无批推理接口）；`batch_size/top_k/
    top_p/temperature/seed/text_lang/prompt_lang` 等白名单参数接受但忽略；坏 `ref_audio_path`
    回 400；路径白名单在门面侧，shim 不重复设防。
- `docker/cosyvoice/Dockerfile`：miniforge 基座 + conda py3.10 + ffmpeg 7 + torch 2.3.1+cu121；
  CosyVoice 源码按 pin SHA `074ca6dc` 浅抓（含 Matcha-TTS 子模块）；requirements 剔除
  deepspeed/tensorrt-cu12*/gdown/gradio/lightning/tensorboard/fastapi-cli，openai-whisper sdist
  按 §9.4 同款修复（先 `setuptools<81`+wheel 再 `--no-build-isolation`）；pyopenjtalk==0.4.1 +
  open_jtalk 词典（MODEL_SOURCE 分支，与 `docker/Dockerfile` 同源）。
- `docker/cosyvoice/entrypoint.sh`：模型目录完整性检查，缺失时打印挂载指引并非零退出；
  `MODEL_SOURCE=ms` 且目录可写时自动下载官方预训练基座（**SFT 权重无发布渠道，挂载是必须路径**）；
  带参数启动时直接 exec（CPU 冒烟/调试入口）。

### 10.1 构建与运行

```bash
docker build -t chobits-chii-cosyvoice-engine docker/cosyvoice/
# 国内可加 --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
#          --build-arg TORCH_INDEX_URL=https://mirror.sjtu.edu.cn/pytorch-wheels/cu121

docker run -d --gpus all -p 127.0.0.1:9882:9880 \
  -v /srv/chii-tts/cosyvoice/model:/data/model \
  -v /srv/chii-tts/cosyvoice/models:/data/models \
  chobits-chii-cosyvoice-engine
# 门面切换: CHII_TTS_ENGINE_URL=http://127.0.0.1:9882 (恰为门面默认值)
```

### 10.2 验证结果（宿主机 + 构建，详见下文）

- **引擎宿主机直跑**（conda CosyVoice 环境，uvicorn 127.0.0.1:9883，MODEL_DIR 指 SFT 目录）：
  非流式 wav 200（24kHz/16bit/4.36s）；streaming_mode=2 首块合法 WAV 头（fmt 24kHz，
  RIFF/data=0xFFFFFFFF），PCM 总长 216960B=4.52s 与非流式一致；ogg 200（ffprobe ogg/1.65s）；
  GET /tts 与 GET 流式均 200；坏 ref_audio_path → 400、空 text → 400。
- **门面端到端**（`CHII_TTS_ENGINE_URL=http://127.0.0.1:9883`，门面零改动）：
  `/healthz/deep` 200（ref_ready=true）；`/v1/audio/speech` 多句 wav 合并流 200 可播放
  （3 句 5.44s，whisper-small 回译「チー、イデキのこと、大好き、ずっと一緒にいる。今日も元気だよ。」，
  内容完整）；opus 200（ogg/3.13s）；`/tts` GET 透传 200。
- **门面测试套件**：`python -m pytest tests/ -q` → 175 passed / 1 failed。唯一失败
  `test_ref_audio_local_symlink_escape_rejected` 为**既有问题，与本阶段无关**（工作树对 HEAD
  零改动即可复现：测试 fixture 的 "outside" 文件实际建在前缀内，realpath 复核合法放行，
  fixture 与测试意图不符）；另本机 ROS humble 的全局 pytest 插件会致收集崩溃，需
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`。
- **镜像构建**：见 10.3。

### 10.3 本机限制：docker 无 GPU

本机 docker **未安装 nvidia-container-toolkit 且无密码 sudo，容器内 GPU 不可用**，故验证策略为：
引擎服务在宿主机 conda 环境跑通完整契约（上表）+ 镜像构建成功 + 容器内 CPU 冒烟（import 级别）。
实测结果（2026-10-10，构建全程官方源，未切镜像）：

- 构建：`docker build -t chobits-chii-cosyvoice-engine docker/cosyvoice/` 成功，镜像 **8.12GB**。
  唯一踩坑：pyworld 0.3.4 在 PyPI 无 cp310 manylinux 轮需源码编译，已在 pip 层同层加
  `build-essential`（装完即删，与 `docker/Dockerfile` 编译 opencc 同款处理）。
- CPU 冒烟：`from cosyvoice.cli.cosyvoice import AutoModel` import ok；torch 2.3.1+cu121
  （容器内 `cuda_avail=False` 符合预期）；pyopenjtalk 词典就位（`run_frontend` 正常出 pron）；
  ffmpeg 7.1.1。
- entrypoint 两路径均验证：模型缺失时打印挂载指引并非零退出；`MODEL_SOURCE=ms` 且目录可写时
  自动从 ModelScope 拉取预训练基座（下载行为实测触发，基座 ~9GB 故中断未拉完）。

在带 GPU 的部署机上，需先安装 toolkit（**需用户手动执行**）：

```bash
sudo apt-get install nvidia-container-toolkit && sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
```

## 主要来源

- [FunAudioLLM/CosyVoice GitHub](https://github.com/FunAudioLLM/CosyVoice) 及 [runtime/triton_trtllm README](https://github.com/FunAudioLLM/CosyVoice/tree/main/runtime/triton_trtllm)
- [Fun-CosyVoice3-0.5B-2512 HF 模型页](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
- 论文：arXiv:2407.05407 / 2412.10117 / 2505.17589
- [Zenn: CosyVoice3 日语实测与汉字 bug workaround](https://zenn.dev/mohhh_ok/articles/2026-06-cosyvoice-qwen3-tts-apache2-mac?locale)
- [Qiita: 日语 TTS 5 模型对比](https://qiita.com/GeneLab_999/items/bc07147b589a93bf6114)
- [Issue #943: 微调数据量官方答复](https://github.com/FunAudioLLM/CosyVoice/issues/943)
- 显存/延迟实测：OnlyVoice 深度解析、火山引擎开发者社区（小智AI）、CSDN 星图平台实测
