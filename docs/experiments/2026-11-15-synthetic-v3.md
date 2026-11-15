# 2026-11-15 合成数据 v2/v3 与 chii_ep18 实验记录

## 结论摘要

- 真实数据保持 475 条，其中 24 条为真实留出集；混合训练真实部分 451 条 / 725.978s。
- 合成数据经历两版：
  - v2：`chii_mix-e15 + 当前参考 + cut5 + seed42/43/44`，299 条 / 575.520s。
  - v3：`chii_mix-e15 + ep18 参考 + cut0 + seed42`，281 条 / 575.610s。
- 两版都把合成时长配到约 50%：
  - v2 混合集 839 条 / 1453.468s，合成 388 条 / 727.490s（50.05%）。
  - v3 混合集 821 条 / 1453.558s，合成 370 条 / 727.580s（50.06%）。
- 基于 v3 训练 `chii_ep18`（GPT/SoVITS e15）。
- 自动指标上 `chii_ep18` 没有稳定超过 `chii_mix-e15`：
  - 长文最佳 seed 46：漏句 4、低质量 1、平均相似度 0.8597；seed42 漏句 6。
  - 但 seed46 在真实留出集上变差（score 中位 0.860 vs seed42 0.896、mix 0.935）。
- 生产当前是 **CosyVoice3 SFT** 引擎，不是 GPT-SoVITS；`seed` 参数被 shim 忽略。
- 生产参考音频已从原 `秀樹は地位を拾ってくれた` 切换为 ep18 `チーは最初チーンしか言えなかったから`，并已重启门面验证。
- 不推荐把生产权重切到 `chii_ep18`；GPT-SoVITS 新版仅作为实验记录保留。

## 数据血缘

| 数据集 | 条数 | 时长 | 说明 |
|---|---:|---:|---|
| `data/trimmed/reviewed` | 475 | 768.4s | 最终真实集，24 话 |
| `data/trimmed/pilot/holdout_real.csv` | 24 | 42.4s | 真实留出集，不进训练 |
| 混合真实部分 | 451 | 725.978s | reviewed 去掉留出集 |
| v1 合成（旧） | 89 | 151.970s | e15 + 当前参考 + cut0 + seed42/43 |
| v2 新合成 | 299 | 575.520s | chii_mix-e15 + 当前参考 + cut5 + seed42/43/44 |
| v3 新合成 | 281 | 575.610s | chii_mix-e15 + ep18 参考 + cut0 + seed42 |

对应混合集：

- `data/trimmed/pilot/mixed/`：451 真实 + 89 v1 = 540 条 / 877.9s。
- `data/trimmed/synth_v2/mixed/`：451 + 89 + 299 = 839 条 / 1453.468s。
- `data/trimmed/synth_v3/mixed/`：451 + 89 + 281 = 821 条 / 1453.558s。

## 合成数据生成与筛选

### v2

- prompt：`data/trimmed/synth_v2/generation_prompts.tsv`，400 条，来自 24 话台词索引，排除真实训练/试点/评测文本。
- 生成：400 prompt × seed 42/43/44 = 1200 条候选。
- 筛选：双 ASR（Qwen3-ASR-1.7B + Whisper large-v3）+ WeSpeaker 声纹 + 时长/音量/削波。
- 结果：1069/1200 候选通过；375 个 prompt 至少一条可用；按目标时长取 299 条。
- 新合成质量：score 0.897–0.979（中位 0.950），Qwen/Whisper 均值 0.982/0.993，声纹均值 0.670。

产物：

- `data/trimmed/synth_v2/synthetic_candidates.csv`
- `data/trimmed/synth_v2/selected/selection.tsv`
- `data/trimmed/synth_v2/mixed/manifest.json`
- `data/trimmed/synth_v2/review_new.html`

### v3

- 配置：`chii_mix-e15` GPT/SoVITS e15，参考 `data/trimmed/synth_v3/refs/ep18.wav`，参考文本 `チーは最初チーンしか言えなかったから`，cut0，seed 42，默认采样参数。
- 生成：400 prompt × seed42 = 400 条候选。
- 筛选：同一套双 ASR/声纹/质量规则。
- 结果：355/400 通过；281 条进入混合集。
- 新合成质量：score 0.881–0.973（中位 0.947），Qwen/Whisper 均值 0.976/0.984，声纹均值 0.653。
- 备注：v3 的 F0 起伏中位 std/range 约 124/361，低于 v2 的 136/394；听感更自然，但不是“音高变化更大”。

产物：

- `data/trimmed/synth_v3/generated/seed42/`
- `data/trimmed/synth_v3/synthetic_candidates.csv`
- `data/trimmed/synth_v3/selected/selection.tsv`
- `data/trimmed/synth_v3/mixed/manifest.json`
- `data/trimmed/synth_v3/review_new.html`

## chii_ep18 训练

- 训练脚本：`training/train_chii.py`
- 实验名：`chii_ep18`
- 数据：`data/trimmed/synth_v3/mixed/gpt_sovits.list`（821 条）
- 预处理：SoVITS 侧有效 786 条；GPT 侧 dataset 806 条。
- 训练：SoVITS 15 epoch + GPT 15 epoch。
- 权重：
  - `GPT-SoVITS/GPT_weights_v2Pro/chii_ep18-e5.ckpt` / `-e10` / `-e15`
  - `GPT-SoVITS/SoVITS_weights_v2Pro/chii_ep18_e5_s970.pth` / `e10_s1940` / `e15_s2910`
- 训练日志：`outputs/train/chii_ep18.log`
- 训练产物目录：`GPT-SoVITS/logs/chii_ep18/`

## 评测

统一条件：ep18 参考、seed42、短句 cut0 / 长文 cut5。

### 24 条真实留出集

| 模型 | 通过 | score 均值 | score 中位 | 18 条有效句均值 | 声纹均值 |
|---|---:|---:|---:|---:|---:|
| chii_ep18 | 16/24 | 0.790 | 0.896 | 0.888 | 0.659 |
| chii_mix-e15 | 17/24 | 0.829 | 0.935 | 0.896 | 0.662 |
| e15 | 14/24 | 0.710 | 0.845 | 0.822 | 0.672 |

短句/语气词（`パンツ`、`けど`、`いい？` 等）对 ASR 分数波动很大，18 条有效句更能反映差距。

### 30 条长文（seed42）

| 模型 | 漏句 | 低质量数 | 平均相似度 | 最低相似度 |
|---|---:|---:|---:|---:|
| chii_ep18 | 6 | 1 | 0.8554 | 0.611 |
| chii_mix-e15 | 5 | 2 | 0.8614 | 0.667 |
| e15 | 7 | 2 | 0.8468 | 0.590 |

### chii_ep18 seed 42–49 扫描

| seed | 漏句 | 低质量数 | 平均相似度 | 最低相似度 |
|---:|---:|---:|---:|---:|
| 46 | 4 | 1 | 0.8597 | 0.611 |
| 44 | 5 | 1 | 0.8503 | 0.472 |
| 45 | 5 | 3 | 0.8478 | 0.444 |
| 49 | 6 | 1 | 0.8580 | 0.667 |
| 42 | 6 | 1 | 0.8579 | 0.611 |
| 47 | 6 | 1 | 0.8562 | 0.667 |
| 48 | 6 | 2 | 0.8426 | 0.444 |
| 43 | 7 | 2 | 0.8493 | 0.500 |

seed46 长文最好，但留出集复测：

| 配置 | 留出集通过 | score 中位 | 18 条有效句均值 |
|---|---:|---:|---:|
| chii_ep18 seed46 | 15/24 | 0.860 | 0.852 |
| chii_ep18 seed42 | 16/24 | 0.896 | 0.888 |
| chii_mix-e15 seed42 | 17/24 | 0.935 | 0.896 |
| e15 seed42 | 14/24 | 0.845 | 0.822 |

结论：seed46 偏长文，seed42 偏短句；没有一个 seed 全面超过 `chii_mix-e15`。

### 试听页

```bash
cd outputs/eval
python3 -m http.server 8768
```

- `holdout_ep18/compare.html`：留出集三模型基线
- `holdout_ep18/compare_s46.html`：seed46 复测
- `longform_ep18/compare.html`：长文三模型基线
- `longform_ep18/compare_s46.html`：seed46 长文对比
- `cosy_ref_ab/compare.html`：生产 CosyVoice 当前参考 vs ep18 参考

## 生产参考切换

生产实际引擎：`chobits-tts-cosy-engine.service`，CosyVoice3 旧 SFT（`model.chii-sft-20261115-170330`），端口 3301；门面 `chobits-tts-cosy-facade.service`，端口 3300。（2026-11-15 晚已按 `chobits-tts-<provider>-<role>` 规范改名。）

- CosyVoice shim 接受 `seed` 字段但**忽略**，所以生产不存在 seed46 参数。
- 2026-11-15 15:51 已将生产参考从原 `秀樹は地位を拾ってくれた` 切换为 ep18：
  - `/opt/chobits/data/tts-cosy/models/ref_audio.wav`
  - `/opt/chobits/data/tts-cosy/models/ref_text.txt`
  - 备份：同目录 `ref_audio.wav.bak-20261115-155156`、`ref_text.txt.bak-20261115-155156`
- 已重启 `chobits-tts-cosy-facade`，`/healthz` 返回 `{"status":"ok","model":"chii-tts"}`，`/v1/tts/speech` 实测返回 200 / 1.76s WAV。
- 生产 CosyVoice A/B（8 句）：当前参考 F0 std/range 121/319，ep18 参考 161/390；ep18 参考的起伏更明显。
- 2026-11-15 18:24 试听回滚：Cosy 模型回旧 SFT，参考改回 ep07（`ref_cosy_ep07.wav` / `ref_cosy_ep07.txt`）；GPT-SoVITS 继续用 ep18，两套参考分离。
- 2026-11-15 18:43：Cosy 切到 `model-chii-mixed-v3` + ep07（新 SFT 权重、旧参考音频）；旧 SFT 备份保留。
- 2026-11-15 19:22：GPT-SoVITS 与 CosyVoice3 统一切到 ep16 `ep16_01061.26s.wav`（3.147s，`ヒデキが元気じゃないと、チーも元気なくなる。`）。
- 2026-11-15 19:45：GPT-SoVITS 改用 Extra `syn2_p261_s42`（5.69s，`ハッキングとは...`）；Cosy 保持 ep16。
- 2026-11-15 20:01：Cosy 低合成占比重训完成 `chii_v1`（428 真实 + 89 v1，剔除 23 条 ep10-like），见 `cosyvoice-mixed-v1.md`。
- 2026-11-15 20:15：试听确认 `chii_v1` 更好，Cosy 生产切到 `model-chii-v1` + ep16；mixed_v3 保留为回滚点。
- 2026-11-15 20:30：参考音频 6 候选对比完成，ep16 最终保留；`ep08_00921` / `extra_p261` 备选留档。

## 复现命令

```bash
# 生成 v3 候选
tools/run_eval_tts.sh \
  --gpt GPT-SoVITS/GPT_weights_v2Pro/chii_mix-e15.ckpt \
  --sovits GPT-SoVITS/SoVITS_weights_v2Pro/chii_mix_e15_s1860.pth \
  --ref-audio data/trimmed/synth_v3/refs/ep18.wav \
  --ref-text-file data/trimmed/synth_v3/refs/ep18.txt \
  --prompts data/trimmed/synth_v2/generation_prompts.tsv \
  --output-dir data/trimmed/synth_v3/generated/seed42 \
  --seed 42 --split-method cut0

# 筛选
.venv-asr/bin/python tools/filter_synthetic.py \
  --root data/trimmed/synth_v3 \
  --prompts data/trimmed/synth_v2/generation_prompts.tsv \
  --target 1000

# 构建 50/50 混合集
.venv/bin/python tools/build_mixed_dataset.py \
  --real-dir data/trimmed/pilot/mixed \
  --fixed-synthetic data/trimmed/pilot/selected/selection.tsv \
  --candidate-synthetic data/trimmed/synth_v3/selected/selection.tsv \
  --target-synthetic-duration 725.978 \
  --output-dir data/trimmed/synth_v3/mixed \
  --review-html data/trimmed/synth_v3/review_new.html \
  --generator-new "chii_mix-e15 + ep18 ref + cut0 + seed42" \
  --force

# 训练
.venv-train/bin/python training/train_chii.py \
  --exp-name chii_ep18 \
  --list data/trimmed/synth_v3/mixed/gpt_sovits.list \
  --wav-dir data/trimmed/synth_v3/mixed/wavs
```

## 待决

- `Chobits-Chii-Voice-Extra` v2 已在本地提交 `d171e21`、打 tag `v2`（281 条 v2 + 89 条 v1 = 370 条），尚未 push 到 GitHub/HF。
- 是否在 `Chobits-Chii-TTS` 中提交本实验记录；GPT-SoVITS 新权重仅作实验，不建议打生产 release。
- 是否要再训练 25–30% 合成占比的 `chii_ep18_25`，验证 50% 是否稀释了真实数据。
- 是否要切换到 GPT-SoVITS 引擎才能使用 seed46；当前生产 CosyVoice 不支持 seed。

## CosyVoice3 mixed_v3 重训与引擎对比（2026-11-15）

### 训练

- 基座：`/opt/chobits/data/tts-cosy/base/Fun-CosyVoice3-0.5B`（ModelScope `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`，8.2GB）。
- 数据：`mixed_v3` 821 条，按 seed=1986 拆 `chii_mixed_train` 801 / `chii_mixed_dev` 20；文本用 pyopenjtalk 假名化，instruct 前缀与生产 shim 一致。
- 配置：`Chobits-Chii-TTS-cosy/training/cosyvoice3_chii_sft.yaml`（max_epoch 10，use_spk_embedding=True）。
- 训练：LLM 10 epoch、Flow 10 epoch，单卡 torch_ddp；LLM val_best 取 epoch 2/1/0，Flow 取 epoch 9/2/8 平均。
- 模型目录：`Chobits-Chii-TTS-cosy/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-chii-mixed`。
- 训练日志：`Chobits-Chii-TTS-cosy/outputs/cosy_train/train_llm_nw0.log`、`train_flow_nw0.log`、`train_flow_resume.log`。

### 兼容补丁（仅训练副本）

- `cosyvoice/bin/train.py`：deepspeed 改为 try/except；无 deepspeed 时不注册 `--deepspeed_config`。
- `cosyvoice/utils/train_utils.py`：deepspeed 导入 try/except；`num_workers=0` 时 `prefetch_factor=None`；`cosyvoice_join` 的 timeout 改为 60s（torch 2.13 无 `group.options._timeout`）。
- `cosyvoice/flow/flow.py`：训练时按 `min(feat_len, token_len*2)` 对齐 mel/token，避免合成数据偶发的 80 vs 78 维度不一致。

### 评测（统一 ep18 参考）

留出集 24 条：

| 系统 | 通过 | score 均值 | score 中位 | 18 有效句均值 | Qwen | Whisper | 声纹 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cosy 新 mixed_v3 | 18/24 | 0.831 | 0.917 | 0.899 | 0.876 | 0.943 | 0.638 |
| Cosy 当前 SFT | 16/24 | 0.775 | 0.923 | 0.803 | 0.740 | 0.802 | 0.636 |
| GPT chii_ep18 | 16/24 | 0.790 | 0.896 | 0.888 | 0.818 | 0.958 | 0.650 |

长文 30 条：

| 系统 | 漏句 | 低质量数 | 平均相似度 | 最低相似度 | 字/秒 |
|---|---:|---:|---:|---:|---:|
| Cosy 当前 SFT | 4 | 1 | 0.8701 | 0.671 | 5.46 |
| Cosy 新 mixed_v3 | 5 | 0 | 0.8596 | 0.641 | 6.02 |
| GPT chii_ep18 | 6 | 1 | 0.8554 | 0.611 | 6.10 |

结论：新 Cosy 明显改善了短句 ASR/完整度，长文漏句略逊于当前 SFT，但低质量数为 0；GPT 的 Whisper/声纹略高但长文漏句最多。最终切换需试听决定。

试听页：

```bash
cd Chobits-Chii-TTS-cosy/outputs/cosy_train
python3 -m http.server 8769
```

- `holdout_engines/compare.html`
- `longform_engines/compare.html`

生产当前是 CosyVoice3 `chii_v1` + ep16 参考（3.147s，已通过 6 候选对比定稿）；GPT-SoVITS 使用 Extra `syn2_p261`（5.69s）。mixed_v3 与旧 SFT 保留为回滚点。生产引擎在评测期间临时停止，已恢复并用 `/v1/tts/speech` 实测 200。

### 生产切换（2026-11-15 17:03）

- 停 `chobits-tts-cosy-engine`，备份当前 SFT 模型目录：
  `/opt/chobits/data/tts-cosy/model` -> `/opt/chobits/data/tts-cosy/model.chii-sft-20261115-170330`
- 新模型固定到 `/opt/chobits/data/tts-cosy/model-chii-mixed-v3`；`model` 改为指向它的 symlink。
- 参考音频保持 ep18（`/opt/chobits/data/tts-cosy/models/ref_audio.wav`）。
- 重启引擎后日志显示 `model loaded, sample_rate=24000, cuda=True`；`/healthz` 正常；`/v1/tts/speech` 实测 200，输出 1.72s WAV。
- 工作区原模型路径 `Chobits-Chii-TTS-cosy/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-chii-mixed` 保留为指向生产模型目录的 symlink。

回滚：

```bash
systemctl stop chobits-tts-cosy-engine
rm /opt/chobits/data/tts-cosy/model
mv /opt/chobits/data/tts-cosy/model.chii-sft-20261115-170330 /opt/chobits/data/tts-cosy/model
systemctl start chobits-tts-cosy-engine
```

> 门面端点说明：`cosy-voice` 的 `6fbf34e` 起统一为 `/v1/tts/speech`，`/v1/audio/speech` 兼容别名已移除。上文 200 验证是在旧门面进程仍加载 `/v1/audio/speech` 时完成的；门面重启后请使用 `/v1/tts/speech`。
>
> 2026-11-15 晚双引擎上线后：对外默认 `/v1/tts/speech` → CosyVoice3；GPT-SoVITS 走 `/v1/tts/gpt/speech`。生产单元为 `chobits-tts-cosy-{engine,facade}` 与 `chobits-tts-gpt-{engine,facade}`，端口 3301/3300 与 3303/3302。
