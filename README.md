<div align="center">
	<h1>Chobits-Chii-TTS</h1>
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

> ⚠️ 注意：原始动画音频的版权归其权利方所有。本项目仅供学习与研究使用，请勿用于商业用途。

本仓库以 [Chobits-Chii-Voice](https://github.com/Anime2Real/Chobits-Chii-Voice) 数据集
（487 条 / 约 21.4 分钟日语语音，[Hugging Face](https://huggingface.co/datasets/chenxin199305/Chobits-Chii-Voice)）
为训练数据，探索了两种 TTS 框架的小叽声线克隆方案。**`main` 分支仅作分支索引**，
实现代码在各方案分支上。

## 🌿 分支总览

| 分支 | 方案 | 状态 | 说明 |
| --- | --- | --- | --- |
| [`gpt-sovits`](https://github.com/Anime2Real/Chobits-Chii-TTS/tree/gpt-sovits) | GPT-SoVITS v2Pro 微调 | **生产运行中** | 首个跑通的方案：数据清洗 → 全量微调（e10 checkpoint，已发布 [Hugging Face](https://huggingface.co/chenxin199305/Chobits-Chii-TTS-GPTSoVITS)）→ Docker 引擎 + HTTP 门面（鉴权/限流/OpenAI 兼容垫片）部署。训练/复现/部署文档齐全 |
| [`cosy-voice`](https://github.com/Anime2Real/Chobits-Chii-TTS/tree/cosy-voice) | Fun-CosyVoice3-0.5B SFT | 验证通过，待上线 | 引入调研 + RTX 4090 实测（zero-shot RTF 0.32 / 流式首包 ~1s）+ 21 分钟数据 SFT 微调（人工试听优于 zero-shot 基线）+ 与 gpt-sovits 门面零改动兼容的引擎容器。完整调研与实测记录见该分支 `docs/cosyvoice-feasibility.md` |
| [`gpt-sovits-genie`](https://github.com/Anime2Real/Chobits-Chii-TTS/tree/gpt-sovits-genie) | Genie-TTS（GPT-SoVITS CPU 引擎） | 验证未通过 | Genie 纯 CPU 推理（显存恒为 0，本可给同卡服务腾显存），但 2.0.2 的 ONNX 转换器仅支持 v2/v2ProPlus 架构，本项目 v2Pro 模型（VITS 512 通道）转换产物无法加载。完整记录见该分支 `docs/genie-feasibility.md` |

gpt-sovits 与 cosy-voice 两分支共享同一部署架构（引擎 Docker 容器绑回环 + 宿主机门面唯一对外入口），门面代码一致、可互换引擎。

## 🧭 规则（2026-10-11 起）

1. `main` 只维护本 README、`LICENSE` 与 `.gitignore`，不承载实现内容提交。
2. 两方案分支独立演进、互不合并；门面与部署架构的通用改动：先提交一个分支，
   再 `git cherry-pick <commit>` 到另一分支（同一改动多处提交）。
3. 方案专属内容（引擎镜像、训练配置、方案调研与实测记录）只留在对应方案分支。
4. 本 README 引用其他仓库的内容文件时，必须指向该仓的内容分支（如 ASR 的模型分支、
   LLM 的路线分支），不指 `blob/main`——对方 `main` 收缩成分支索引后这类链接必 404。

## 📄 许可协议

本项目的派生内容遵循数据集的 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh-hans)（署名-非商业性使用-相同方式共享）协议。

## ⚠️ 免责声明

- 本项目仅用于学术研究与个人学习，不构成对原作品版权的任何主张。
- 使用本项目训练的模型所生成的内容，不得用于侵犯原作品及相关声优（田中理惠）权益的用途。
- 若权利方提出要求，本项目将被下架。

## 🙏 相关项目

- [Chobits-Chii-Voice](https://github.com/Anime2Real/Chobits-Chii-Voice) — 小叽语音数据集（本项目的数据来源）
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) — `gpt-sovits` 分支的底层语音合成框架
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) — `cosy-voice` 分支的底层语音合成框架
