<div align="center">
	<h1>Chobits</h1>
	<p><b>叽～</b> 小叽的语音合成模型</p>
	<p>《人形电脑天使心》(Chobits) 中 <b>小叽 (Chii / ちぃ)</b> 角色的 TTS（语音合成）模型项目。</p>
	<p>
		<a href="https://madewithlove.org.in"><img alt="Made with Love" src="https://img.shields.io/badge/Made%20with-Love-ff69b4.svg"></a>
		<a href="https://github.com/Anime2Real/Chobits-Chii-TTS"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-Chobits--Chii--TTS-181717?logo=github"></a>
		<a href="https://huggingface.co/datasets/chenxin199305/Chobits-Chii-Voice"><img alt="Dataset: Chobits-Chii-Voice" src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Chobits--Chii--Voice-yellow"></a>
		<a href="https://huggingface.co/chenxin199305/Chobits-Chii-TTS-GPTSoVITS"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-Chobits--Chii--TTS--GPTSoVITS-yellow"></a>
		<a href="https://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="License: CC BY-NC-SA 4.0" src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg"></a>
		<img alt="Language: Japanese" src="https://img.shields.io/badge/Language-Japanese-green.svg">
	</p>
</div>

> 💖 如果这个项目对你有帮助，欢迎在 [GitHub](https://github.com/Anime2Real/Chobits-Chii-TTS) 点个 Star —— 你的支持能让更多人发现小叽！

> 🌿 本分支（`gpt-sovits-genie`）为 [Genie-TTS](https://github.com/High-Logic/Genie-TTS) 引擎的**验证档案**（不是部署分支）。
> 结论：**当前不可行**——Genie 的 ONNX 转换器仅支持 GPT-SoVITS **v2 / v2ProPlus** 架构，本项目生产模型为 **v2Pro**（VITS 512 通道），转换产物无法加载。
> 生产方案见 [`gpt-sovits`](https://github.com/Anime2Real/Chobits-Chii-TTS/tree/gpt-sovits) 分支；CosyVoice 方案见 [`cosy-voice`](https://github.com/Anime2Real/Chobits-Chii-TTS/tree/cosy-voice) 分支。

> ⚠️ 注意：原始动画音频的版权归其权利方所有。本项目仅供学习与研究使用，请勿用于商业用途。

## 🔍 结论

调研动机：Genie-TTS 是 GPT-SoVITS 的轻量 CPU 推理引擎（ONNX，运行时仅 ~200MB），若可用，可把 TTS 从 GPU 上撤下来，给同卡其他服务腾显存。2026-10-11 在 RTX 4090 上实测（genie-tts 2.0.2，当时 PyPI 最新）：

| 验证项 | 结果 |
| --- | --- |
| GenieData 依赖模型下载 | ✅ 正常（17 个文件，HF） |
| 小叽 v2Pro 权重 → ONNX 转换 | ⚠️ 转换器报「成功」，但产物损坏不可加载 |
| 根因 | ❌ 架构不匹配：v2Pro 的 VITS 为 **512 通道**（`dec.conv_pre.weight` = 512×192×7），Genie 内置的 v2ProPlus 资源图期望 **768 通道**，v2 资源图亦不同；Genie 无 v2Pro 支持（上游 roadmap 也未列入） |
| 显存占用 | **恒为 0**——Genie 只注册 `CPUExecutionProvider`，纯 CPU 推理，根本不触 GPU（by design） |

完整证据与过程：[docs/genie-feasibility.md](docs/genie-feasibility.md)；一键复现：`tools/genie_v2pro_repro.py`。

## 📊 显存与资源（若能用会怎样）

Genie 显存占用为 0 是设计使然（ONNX CPU 推理），这本是对生产拓扑的最大价值：
T4 服务器上 TTS 容器常驻 ~3.7–4GB 显存，若换 Genie 可全部让给 ASR 引擎。
代价是 CPU 推理延迟（官方标称短句首包 1.13s，i7-13620H）与仅支持日语（2.0.2 起含中/英）。
但上述架构问题不解决，这些无从谈起。

## 🧭 后续路径（若仍想走 Genie 路线）

1. **上游提 issue** 请求 v2Pro 支持——需为 512 通道 VITS 重新导出资源图与键位表（Genie 的转换器是把 pth 权重按键位表灌进预置 ONNX 图，架构必须逐层一致）。
2. **改用 v2ProPlus 架构重训**小叽模型（GPT-SoVITS 上游已支持 v2ProPlus），之后 Genie 转换即可用；成本是重新训练与试听选型。
3. 放弃 Genie，维持 `gpt-sovits` 分支的 PyTorch 部署（当前生产方案，显存 ~3.7–4GB）。

## 📄 许可协议

本项目的派生内容遵循数据集的 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh-hans)（署名-非商业性使用-相同方式共享）协议。

Genie-TTS 与 GPT-SoVITS 遵循其各自上游许可。

## ⚠️ 免责声明

- 本项目仅用于学术研究与个人学习，不构成对原作品版权的任何主张。
- 使用本项目训练的模型所生成的内容，不得用于侵犯原作品及相关声优（田中理惠）权益的用途。
- 若权利方提出要求，本项目将被下架。

## 🙏 相关项目

- [Chobits-Chii-Voice](https://github.com/Anime2Real/Chobits-Chii-Voice) — 小叽语音数据集（本项目的数据来源）
- [Genie-TTS](https://github.com/High-Logic/Genie-TTS) — 本分支验证的 GPT-SoVITS 轻量 CPU 推理引擎
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) — `gpt-sovits` 分支的底层语音合成框架
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) — `cosy-voice` 分支的底层语音合成框架
