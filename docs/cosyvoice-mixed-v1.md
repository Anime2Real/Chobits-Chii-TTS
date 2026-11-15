# CosyVoice3 chii_v1 重训记录（428 真实 + 89 v1 合成）

## 目标

- 把合成数据占比从 mixed_v3 的 370 条降到 89 条 v1，避免合成音质/内容过多把音色带偏。
- 剔除与 `ep10_00850` 音色相近的真实片段，避免不想要的音色进入训练。

## 数据

| 项 | 数量 |
|---|---:|
| 真实（原始） | 451 |
| 剔除：与 ep10_00850 余弦相似度 ≥ 0.65 | 23 |
| 真实（保留） | 428 |
| v1 合成 | 89 |
| 合计 | 517 |
| train / dev | 497 / 20（seed 1986） |

剔除方法：

- 用 `Chobits-Chii-Voice/annotations/human_labels.npz` 的 192 维声纹 embedding，
  计算每条真实片段与 `ep10_00850.00s` 的余弦相似度。
- 阈值取约 p95 = 0.65，剔除 23 条；该阈值高于 ep16（0.585）、ep08_00921（0.547），
  低于 ep08_01107（0.643）。
- 剔除清单（23 条）：

```text
ep06_00162.28s ep06_00179.98s ep07_00191.42s ep07_00548.70s ep07_00776.42s
ep07_00783.96s ep08_00777.98s ep08_01214.40s ep09_00111.60s ep10_00723.10s
ep10_00832.44s ep10_00850.00s ep10_01451.50s ep12_00487.42s ep12_00855.50s
ep12_01305.72s ep13_01268.68s ep16_00607.00s ep16_00812.32s ep16_01163.18s
ep23_00230.68s ep23_00237.34s ep23_01276.82s
```

数据源与清单：

- 源：`Chobits-Chii-TTS/data/trimmed/synth_v3/mixed`，只保留 `ep*` + `syn_p*`（89 条 v1），
  丢弃 `syn2_*`（281 条 v3）。
- 过滤后目录：`Chobits-Chii-TTS/data/trimmed/mixed_v1_filtered/mixed`
- `manifest.json` 记录总数、真实/合成数、阈值和剔除清单。
- 准备命令：

```bash
python training/prepare_cosyvoice_mixed.py \
  --mixed-dir ../Chobits-Chii-TTS/data/trimmed/mixed_v1_filtered/mixed \
  --name-prefix chii_v1 --dev-size 20 --seed 1986
```

`prepare_cosyvoice_mixed.py` 新增 `--name-prefix`，输出 `chii_v1_train` / `chii_v1_dev`，
不会覆盖现有 `chii_mixed_*`。

## 训练

- 配置：`training/cosyvoice3_chii_sft.yaml`（max_epoch 10，use_spk_embedding=True）
- 引擎：`torch_ddp`，单卡，`--num_workers 0`，AMP
- 基座：`/opt/chobits/data/tts-cosy/base/Fun-CosyVoice3-0.5B`
- LLM 10 epoch，val_best：epoch 2/1/3（loss 3.356 / 3.373 / 3.444）
- Flow 10 epoch，val_best：epoch 6/4/1（loss 0.585 / 0.599 / 0.601）
- 日志：
  - `outputs/cosy_train/train_chii_v1_llm.log`
  - `outputs/cosy_train/train_chii_v1_flow.log`
  - `outputs/cosy_train/avg_chii_v1_llm.log`
  - `outputs/cosy_train/avg_chii_v1_flow.log`
- 训练产物：`CosyVoice/examples/libritts/cosyvoice3/exp/chii_v1/{llm,flow}/torch_ddp`
- 打包：`/opt/chobits/data/tts-cosy/model-chii-v1`（5.6G，自包含推理包）
- 临时评测引擎：`127.0.0.1:3311`（`MODEL_DIR=/opt/chobits/data/tts-cosy/model-chii-v1`）

> 训练前在 `tts-engine` venv 补了 `tensorboard==2.16.2`、`pandas==2.3.3`、
> `pyarrow==18.1.0`；protobuf/grpcio 已固定回 4.25.0 / 1.57.0，保持 ONNX Runtime 兼容。

## 评测

同一参考 ep16，6 句测试文本：

| 系统 | ASR 字符相似度均值 |
|---|---:|
| 当前 mixed_v3 | 0.915 |
| 新 chii_v1 | 0.881 |

新模型在 2 句上略低，可能是数据量减少（517 vs 821）和剔除 23 条后的适应不足；
ASR 字符相似度只作参考，最终以试听为准。

试听页：

```text
Chobits-Chii-TTS/outputs/eval/cosy_v1_ab/index.html
http://192.168.100.213:8771/
```

页面用同一份 ep16 参考，对比当前 mixed_v3 与新 chii_v1，6 句逐句并排。

## 切换与回滚

2026-11-15 20:15：试听后确认 `chii_v1` 更好，**生产已切到 chii_v1 + ep16**。
参考音频 6 候选对比后，**ep16 听感最合适，最终保留 ep16**；候选页留档：
`Chobits-Chii-TTS/outputs/eval/cosy_v1_refs/index.html`（`http://192.168.100.213:8772/`）。
备选：真实片段 `ep08_00921`，合成片段 `extra_p261`。

切换命令：

```bash
ln -sfn model-chii-v1 /opt/chobits/data/tts-cosy/model
sudo systemctl restart chobits-tts-cosy-engine
```

回滚到 mixed_v3：

```bash
ln -sfn model-chii-mixed-v3 /opt/chobits/data/tts-cosy/model
sudo systemctl restart chobits-tts-cosy-engine
```

参考音频仍由 `~/.config/chobits/chobits-chii-tts-cosy.env` 决定（当前 ep16）。
