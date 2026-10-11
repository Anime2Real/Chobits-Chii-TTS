# Genie-TTS 引入调研与实测记录（gpt-sovits-genie 分支）

> 结论：**不可行（截至 genie-tts 2.0.2）**——Genie 的 ONNX 转换器仅支持 GPT-SoVITS v2 / v2ProPlus 架构，
> 本项目生产模型为 v2Pro（VITS 512 通道），转换产物无法加载。
> 本文件记录完整验证过程与证据，供日后复查或向上游提 issue 引用。

## 1. 背景与动机

[Genie-TTS](https://github.com/High-Logic/Genie-TTS)（High-Logic）是基于 GPT-SoVITS 的轻量推理引擎：
ONNX 模型 + 纯 CPU 推理，官方标称运行时 ~200MB、模型 ~230MB、短句首包 1.13s（i7-13620H）。
对小叽项目的吸引力：生产 T4 服务器上 TTS 容器常驻 ~3.7–4GB 显存（见
[Chobits-Chii-ASR docs/hardware-adaptation.md](https://github.com/Anime2Real/Chobits-Chii-ASR/blob/main/docs/hardware-adaptation.md)），
若 TTS 能撤到 CPU，同卡的 ASR 引擎可获得全部显存。

## 2. 验证环境

| 项 | 值 |
| --- | --- |
| 机器 | 本机 RTX 4090 24GB（驱动 580.159.03 / CUDA 13.0）；Genie 为 CPU 引擎，GPU 仅用于观察「是否真的不碰显卡」 |
| Python | 3.12（venv） |
| genie-tts | 2.0.2（2026-10-11 PyPI 最新；上游 master README 仍写「仅支持 V2」，包内已含 v2ProPlus 转换器） |
| torch / onnxruntime | 2.9.1 / 1.22.1（CPU） |
| 被测模型 | [chenxin199305/Chobits-Chii-TTS-GPTSoVITS](https://huggingface.co/chenxin199305/Chobits-Chii-TTS-GPTSoVITS)：`chii-e10.ckpt`（149MB，GPT 语义）+ `chii_e10_s1210.pth`（129MB，SoVITS 声学，**v2Pro** 架构）+ `ref_audio.wav` / `ref_text.txt` |

## 3. 验证过程

### 3.1 安装与依赖模型

- `pip install genie-tts` 正常（pyopenjtalk-plus 有 Linux wheel，无需编 C 扩展）。
- 首次 import 需交互下载 GenieData（hubert / roberta / 说话人验证 / G2P 共 17 个文件，HF），
  非交互环境用 `yes y | python -c "import genie_tts"`。

### 3.2 转换尝试一：`genie.convert_to_onnx`（默认路由）

```python
genie.convert_to_onnx(torch_pth_path='chii_e10_s1210.pth',
                      torch_ckpt_path='chii-e10.ckpt',
                      output_dir='onnx-chii')
# 🎉 Conversion successful! - Model Type: V2
```

**注意这里已被错误路由**。`genie_tts/Converter/Converter.py` 的路由逻辑是按 .pth 文件大小：

```python
def convert(torch_ckpt_path, torch_pth_path, output_dir):
    if os.path.getsize(torch_pth_path) > 150 * 1024 * 1024:  # 大于 150 MB
        convert_v2pp(...)   # v2ProPlus
    else:
        convert_v2(...)     # v2
```

v2Pro 的 `chii_e10_s1210.pth` 为 134.9MB < 150MB → 走 v2 转换器。转换「成功」并产出
`t2s_*.onnx/.bin + vits_fp32.onnx + vits_fp16.bin`，但 `load_character` 时报（仅记日志不抛异常，角色实际未注册成功）：

```
Failed to load in-memory model vits_fp32.onnx: [ONNXRuntimeError]
initializer.cc:48 ... FAIL: endian_utils.cc:56 CopyLittleEndian source and destination buffer size mismatch
```

### 3.3 转换尝试二：强制 v2ProPlus 转换器

```python
from genie_tts.Converter.v2ProPlus.Converter import convert
convert(torch_ckpt_path='chii-e10.ckpt', torch_pth_path='chii_e10_s1210.pth', output_dir='onnx-chii')
# 🎉 Conversion successful! - Model Type: V2ProPlus
```

`load_character` 报**完全相同**的 vits_fp32.onnx 初始化错误。

### 3.4 根因：VITS 通道数不匹配（架构级）

Genie 的转换器不是「从 pth 重新建图导出」，而是**把 pth 权重按键位表灌进包内预置的 ONNX 资源图**
（`Data/v2/Models/vits_fp32.onnx` 与 `Data/v2ProPlus/Models/vits_fp32.onnx`）。
灌入时若 pth 张量与资源图维度不一致，bin 中记录的长度（fp32 字节）就与图的 initializer 声明不符，
加载时 ORT 逐个校验 initializer 报 buffer size mismatch。

实测维度对比（复现脚本可打印）：

| 张量 | 小叽 v2Pro pth | Genie v2ProPlus 资源图 |
| --- | --- | --- |
| `dec.conv_pre.weight` | **512**×192×7 | **768**×192×7 |
| `dec.ups.0.weight_g` | **512**×1×1 | **768**×1×1 |
| `dec.ups.0.weight_v` | **512**×256×16 | **768**×384×20 |

v2Pro 的 VITS 是 512 通道，v2ProPlus 是 768 通道，v2 又不同——三者互不兼容，
**genie-tts 2.0.2 没有 v2Pro 资源图，无法支持本模型**。650 个外部权重张量中 290 个维度不符。
上游 master README 的 roadmap 列的是 V2Proplus / V3 / V4，v2Pro 未列入。

### 3.5 附带发现：HF 发布的权重文件缺 ZIP 魔数头

`chii-e10.ckpt` / `chii_e10_s1210.pth` 在 HF 上的原始字节以 `30 35 03 04` 开头而非标准 ZIP 的
`50 4B 03 04`（"PK\x03\x04"）——文件被削去了前 2 字节 "PK"（已核对 HF 原始字节，非下载损坏）。
直接 `torch.load` 会报 `unpickling stack underflow`；Genie 的
`Converter/load_state_dict.py` 内置了修复（丢弃前 2 字节、前置 "PK" 后按 ZIP 读取），
所以转换器能正常读权重。任何自写脚本加载这两个文件都需要同样的处理：

```python
f = open(pth_path, 'rb'); f.read(2)
pth = torch.load(io.BytesIO(b'PK' + f.read()), map_location='cpu', weights_only=False)
```

（GPT-SoVITS 的 pickle 还引用其 `utils` 模块，加载 pth 需相应 import 路径。）

## 4. 显存结论

**Genie 显存占用恒为 0，与模型无关**——`genie_tts/ModelManager.py` 只注册
`providers=["CPUExecutionProvider"]`，推理全程在 CPU/内存（ONNX 权重在内存中从 fp16 bin 重建为
fp32，单角色约 0.5GB RAM 量级）。本测试中 `nvidia-smi` 在转换（CPU 导出）与加载尝试期间
无任何新 GPU 进程。也就是说：

- 「用 Genie 部署能省多少显存」的答案是**全部**（TTS 显存归零，T4 上可腾出 ~4GB 给 ASR）；
- 但第 3 节的架构问题不解决，这一点无法兑现。

## 5. 后续路径

1. **上游提 issue** 请求 v2Pro 支持：需要 512 通道 VITS 的资源 ONNX + 键位表
   （`Converter/v2ProPlus/` 同款，只是源模型换成 v2Pro 基座）。引用本文件第 3.4 节的维度证据。
2. **改用 v2ProPlus 架构重训**（GPT-SoVITS 上游已支持 v2ProPlus 训练），
   之后 Genie 的 v2pp 转换器即可用；代价是重新训练 + 试听选型（参见 `gpt-sovits` 分支训练文档）。
3. 维持现状：生产继续用 `gpt-sovits` 分支的 PyTorch 部署（显存 ~3.7–4GB），
   Genie 路线挂起，跟踪上游版本。

## 6. 复现

见 `tools/genie_v2pro_repro.py`（环境准备 → 权重加载（含 PK 修复）→ 维度对比打印 →
两次转换 → load_character 预期失败）。

- 2026-10-11，RTX 4090 24GB，genie-tts 2.0.2。
