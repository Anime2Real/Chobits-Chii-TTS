# CosyVoice3 mixed_v3 微调与引擎对比

## 数据

- `mixed_v3`：451 真实 + 89 v1 合成 + 281 v3 合成 = 821 条。
- 真实留出集 24 条不进入训练。
- 按 seed=1986 拆 `chii_mixed_train` 801 / `chii_mixed_dev` 20。
- 文本用 pyopenjtalk 假名化，instruct 前缀与生产 shim 一致。

准备数据：

```bash
python training/prepare_cosyvoice_mixed.py
# 或显式指定：
python training/prepare_cosyvoice_mixed.py \
  --mixed-dir ../Chobits-Chii-TTS/data/trimmed/synth_v3/mixed \
  --des-base CosyVoice/examples/libritts/cosyvoice3/data
```

## 训练

- 基座：`Fun-CosyVoice3-0.5B-2512`（8.2GB，放在 `CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B`）。
- 配置：`training/cosyvoice3_chii_sft.yaml`（max_epoch 10，use_spk_embedding=True）。
- 训练 LLM 10 epoch + Flow 10 epoch，单卡 torch_ddp。
- LLM val_best 取 epoch 2/1/0；Flow 取 epoch 9/2/8。
- 模型目录：`CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-chii-mixed`。

本机 torch 2.13 训练兼容补丁（在 gitignored 的 `CosyVoice/` clone 内）：

1. `cosyvoice/bin/train.py`、`cosyvoice/utils/train_utils.py`：deepspeed 改为 try/except；无 deepspeed 时不注册 `--deepspeed_config`。
2. `train_utils.py`：`num_workers=0` 时 `prefetch_factor=None`；`cosyvoice_join` 的 timeout 改为 60s（torch 2.13 无 `group.options._timeout`）。
3. `cosyvoice/flow/flow.py`：训练时按 `min(feat_len, token_len*2)` 对齐 mel/token，避免合成数据偶发的 80 vs 78 维度不一致。

训练日志在 `outputs/cosy_train/`。

## 评测

统一使用 ep18 参考。

### 24 条真实留出集

| 系统 | 通过 | score 中位 | 18 有效句均值 | Qwen | Whisper | 声纹 |
|---|---:|---:|---:|---:|---:|---:|
| Cosy 新 mixed_v3 | 18/24 | 0.917 | 0.899 | 0.876 | 0.943 | 0.638 |
| Cosy 当前 SFT | 16/24 | 0.923 | 0.803 | 0.740 | 0.802 | 0.636 |
| GPT chii_ep18 | 16/24 | 0.896 | 0.888 | 0.818 | 0.958 | 0.650 |

### 30 条长文

| 系统 | 漏句 | 低质量数 | 平均相似度 | 最低相似度 | 字/秒 |
|---|---:|---:|---:|---:|---:|
| Cosy 当前 SFT | 4 | 1 | 0.8701 | 0.671 | 5.46 |
| Cosy 新 mixed_v3 | 5 | 0 | 0.8596 | 0.641 | 6.02 |
| GPT chii_ep18 | 6 | 1 | 0.8554 | 0.611 | 6.10 |

试听页：

```bash
cd outputs/cosy_train
python3 -m http.server 8769
```

- `holdout_engines/compare.html`
- `longform_engines/compare.html`

## 生产切换

- 2026-11-15 17:03：曾切到 `model-chii-mixed-v3` + ep18。
- 2026-11-15 18:24：试听后回滚到旧 SFT `model.chii-sft-20261115-170330` + ep07。
- 2026-11-15 18:43：生产切到 `model-chii-mixed-v3` + ep07。
- 2026-11-15 19:22：GPT-SoVITS 与 CosyVoice3 统一切到 ep16
  `ep16_01061.26s.wav`（3.147s，`ヒデキが元気じゃないと、チーも元気なくなる。`）。
- 2026-11-15 20:15：试听确认 `chii_v1` 更好，Cosy 生产切到
  `model-chii-v1` + ep16；mixed_v3 成为上一版/回滚点。
- 2026-11-15 20:30：参考音频 6 候选对比完成，**ep16 最终保留**；`ep08_00921` /
  `extra_p261` 作为备选留档。
- 当前 symlink：`/opt/chobits/data/tts-cosy/model -> model-chii-v1`。
- 旧 SFT 备份：`model.chii-sft-20261115-170330`；mixed_v3 目录保留。
- CosyVoice 参考：`ep16_01061.26s.wav` / `.txt`，由
  `~/.config/chobits/chobits-chii-tts-cosy.env` 注入；GPT-SoVITS 使用 Extra `syn2_p261`。
- 服务名：`chobits-tts-cosy-engine` / `chobits-tts-cosy-facade`。
- `/healthz` 与 `/healthz/deep` 正常；chii_v1 + ep16 实测合成 200。
- CosyVoice 不读取 `seed`。

回滚到 mixed_v3 + ep16：

```bash
ln -sfn model-chii-mixed-v3 /opt/chobits/data/tts-cosy/model
sudo systemctl restart chobits-tts-cosy-engine
```

更早的旧 SFT 回滚：

```bash
ln -sfn model.chii-sft-20261115-170330 /opt/chobits/data/tts-cosy/model
sudo systemctl restart chobits-tts-cosy-engine
```

参考音频由 `chobits-chii-tts-cosy.env` 决定；换参考时改该 env 的
`CHII_TTS_REF_AUDIO` / `CHII_TTS_REF_TEXT_FILE` 后重启 `chobits-tts-cosy-facade`。

完整实验记录（含 GPT-SoVITS、数据 v2/v3、seed 扫描）见 `gpt-sovits` 分支 `docs/experiments/2026-11-15-synthetic-v3.md`。

后续低合成占比重训（428 真实 + 89 v1，剔除 23 条 ep10-like）见
[`cosyvoice-mixed-v1.md`](./cosyvoice-mixed-v1.md)；该模型只用于评测，尚未切生产。
