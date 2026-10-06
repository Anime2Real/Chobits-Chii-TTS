# 国内镜像源汇总

Chobits-Chii 各仓库散落使用的国内镜像/加速源统一登记在此（2026-10-06 盘点）。
本文件放在 TTS 仓库的原因：ServerDeploy 是全体系部署真相源，但其 `main` 仅作分支索引
（仓库规则不承载内容）、内容在 `cloud` 分支；TTS 是镜像引用最密集的仓库且 `docs/` 已有
deployment.md 实录，故暂归集于此，ServerDeploy README 索引处有指针。

## GitHub 加速

| 镜像 | 用途 | 引用位置 |
| --- | --- | --- |
| `gh-proxy.com` | GitHub 不通时 clone 仓库的代理前缀 | TTS `docker/Dockerfile`（`--build-arg GS_REPO_URL=https://gh-proxy.com/https://github.com/...`，构建引擎镜像拉 fork 源码的备选） |
| `ghfast.top` | 同类代理前缀（历史上用过） | 曾作为 TTS 内嵌 `GPT-SoVITS/` clone 的 origin remote（clone 上游 RVC-Boss），2026-10 已移除：remote 统一改指 fork 走 SSH，当前各仓库无残留引用 |
| jsDelivr（`cdn.jsdelivr.net` / `fastly.jsdelivr.net` / `gcore.jsdelivr.net`） | `raw.githubusercontent.com` 被限速时的文件下载替代 | TTS README 快速开始与 `tools/setup_env.sh`（nltk `punkt_tab` 兜底下载，按 cdn→fastly→gcore 顺序探测）、TTS `docker/Dockerfile`（构建期同款兜底） |

## PyPI / conda / PyTorch

| 镜像 | 用途 | 引用位置 |
| --- | --- | --- |
| 清华 TUNA `pypi.tuna.tsinghua.edu.cn/simple` | pip 索引 | TTS README 快速开始；TTS `docs/deployment.md`（docker `--build-arg PIP_INDEX_URL`） |
| 清华 TUNA `mirrors.tuna.tsinghua.edu.cn/anaconda` | Miniconda 安装包 + conda channels | TTS `tools/setup_env.sh`（一键环境搭建） |
| 上海交大 `mirror.sjtu.edu.cn/pytorch-wheels/cu128` | torch / torchaudio / torchcodec cuXXX wheels（阿里云与清华均已下线 pytorch-wheels） | TTS README 快速开始；TTS `docker/Dockerfile`（`TORCH_INDEX_URL`）；TTS `docs/deployment.md` |
| 腾讯云内网 `mirrors.tencentyun.com` | PyPI / apt（仅腾讯云宿主机内网可达，公网无效） | ASR `docs/deployment.md`（生产构建实测：`PIP_INDEX_URL` / `PIP_TRUSTED_HOST` / `APT_MIRROR`） |

## 模型权重

| 镜像 | 用途 | 引用位置 |
| --- | --- | --- |
| ModelScope `modelscope.cn` | 国内模型权重源：GPT-SoVITS 预训练模型（`install.sh --source ModelScope`）、Fun-ASR-Nano-2512（ASR 引擎首启下载，~2.2GB）、Qwen 系列 LLM 权重 | TTS README / `tools/setup_env.sh` / `docker/entrypoint.sh`（`MODEL_SOURCE=ms` 默认）；ASR README / `docs/deployment.md`；LLM `docs/deployment.md` / `sft/README.md`（HF LFS 停滞时改走 ModelScope） |
| hf-mirror.com | Hugging Face 镜像（`HF_ENDPOINT=https://hf-mirror.com`） | TTS `docker/entrypoint.sh`（chii 权重下载失败时的排查提示）、TTS `docs/deployment.md`（空卷首次启动下载 chii 权重）、上游 `install.sh --source HF-Mirror` |

## Docker / Electron

| 镜像 | 用途 | 引用位置 |
| --- | --- | --- |
| `mirror.ccs.tencentyun.com` | dockerd `registry-mirrors`（腾讯云内网，拉镜像加速） | LLM `docs/deployment.md` |
| `npmmirror.com/mirrors/electron/` | Electron 二进制下载镜像 | Mascot `.npmrc`（注释态；pnpm 10+ 已不从 `.npmrc` 读该设置，当前未启用，需要时改配 pnpm-workspace.yaml） |

## 使用场景速查

- **本机开发（国内宽带）**：GitHub 走 gh-proxy.com 前缀或 SSH；pip 走清华；torch 走交大；模型权重走 ModelScope。
- **腾讯云生产宿主机**：PyPI/apt/Docker 优先走腾讯内网镜像（免费且快）；模型权重仍走 ModelScope。
- **Docker 构建**：TTS/ASR 镜像均有 `PIP_INDEX_URL` 等 build-arg，默认值即可，国内构建按各仓库 `docs/deployment.md` 实录传参。
