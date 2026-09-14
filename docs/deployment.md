# Chobits-Chii-TTS 部署实录

> 与家族其他服务一致的约定：推理引擎跑在 Docker 里（裸 api_v2，无鉴权，只绑 127.0.0.1），
> 宿主机 Python 门面负责鉴权（`Authorization: Bearer`）、每 IP 限流与 OpenAI 垫片，
> 密钥经 `/etc/chobits-chii-tts.env` 注入，systemd 守护。

## 实测环境

- 服务器：Ubuntu 24.04 + Tesla T4 16GB（v2Pro fp16，8GB 显存即可）
- Docker 24.0.7 + NVIDIA Container Toolkit 1.14.3（驱动 525.105.17，满足 cu128 要求的 ≥525.60.13）
- 日期：2026-09-10

## 1. 构建引擎镜像

```bash
# 国内网络建议加镜像 build-arg (torch 走 SJTU, PyPI 走清华)
docker build -t chobits-chii-tts-engine \
  --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  --build-arg TORCH_INDEX_URL=https://mirror.sjtu.edu.cn/pytorch-wheels/cu128 \
  docker/
```

引擎源码经 `GS_REPO_REF` 锁到 README 验证过的 commit（默认即锁定值，无需改动）；
`MODEL_SOURCE`（ms/hf/hf-mirror）控制构建期 nltk_data 与 open_jtalk 词典的下载源，默认 ModelScope。

## 2. 启动引擎容器

数据卷 `/data` 约定三个子目录：`pretrained_models/`（GPT-SoVITS 预训练模型）、
`G2PWModel/`（中文 G2PW）、`models/`（chii 权重与参考音频）。两种准备方式：

### 方式 A：复用宿主机已有模型（训练机 / 从旧 systemd 部署迁移，零下载）

```bash
REPO=$HOME/Github/Chobits-Chii-TTS
docker run -d --name chobits-chii-tts-engine \
  --gpus all --restart unless-stopped \
  -p 127.0.0.1:9882:9880 \
  -v $REPO/GPT-SoVITS/GPT_SoVITS/pretrained_models:/data/pretrained_models:ro \
  -v $REPO/GPT-SoVITS/GPT_SoVITS/text/G2PWModel:/data/G2PWModel:ro \
  -v $REPO/models:/data/models:ro \
  chobits-chii-tts-engine
```

### 方式 B：空卷自动下载（全新机器）

```bash
docker run -d --name chobits-chii-tts-engine \
  --gpus all --restart unless-stopped \
  -p 127.0.0.1:9882:9880 \
  -v chii-tts-data:/data \
  -e MODEL_SOURCE=ms \
  chobits-chii-tts-engine
```

首次启动自动下载：预训练模型推理子集（roberta/hubert/sv/fast_langdetect，约 1GB，
`MODEL_SOURCE` 可选 ms(默认)/hf/hf-mirror）+ G2PWModel（约 600MB）+ chii 权重
（约 290MB，走 Hugging Face，国内加 `-e HF_ENDPOINT=https://hf-mirror.com`）。
只读挂载缺文件时入口脚本会报错并提示补齐，不会在只读卷上尝试写入。

```bash
docker logs -f chobits-chii-tts-engine   # 等 "Uvicorn running on" 字样
```

- 引擎只发布到 `127.0.0.1:9882`（容器内固定 9880），对外统一由门面负责（鉴权/TLS 都在门面层）。
- 无 GPU 的机器加 `-e DEVICE=cpu -e IS_HALF=false`（慢，仅供测试）。

## 3. 启动门面（宿主机）

```bash
export CHII_TTS_API_KEY=<随机密钥>   # 必填, 未设置拒绝启动
bash tools/start_tts_api.sh 9880     # 首次运行自动在仓库根目录建 .venv 装依赖
```

门面环境变量（均有默认值，env 文件里只写需要覆盖的）：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `CHII_TTS_API_KEY` | （必填） | API 密钥 |
| `CHII_TTS_ENGINE_URL` | `http://127.0.0.1:9882` | 引擎地址 |
| `CHII_TTS_RATE_LIMIT` | `60` | `/tts` 与 `/v1/audio/speech` 每 IP 每分钟限流，0 关闭 |
| `CHII_TTS_MAX_TEXT_CHARS` | `2000` | 合成文本长度硬上限（两路径均生效），防长文本独占 GPU |
| `CHII_TTS_MAX_INFLIGHT` | `8` | 全局在途并发上限，超出即 429 |
| `CHII_TTS_REF_AUDIO` | `/data/models/ref_audio.wav` | OpenAI 垫片 `chii` 音色的参考音频（**引擎容器内**路径） |
| `CHII_TTS_REF_AUDIO_PREFIX` | `/data/` | `/tts` 透传的 ref_audio 路径前缀约束（防容器内任意路径探测） |
| `CHII_TTS_REF_TEXT_FILE` | `models/ref_text.txt` | 参考文本（门面**宿主机**路径，读出后内联传给引擎） |
| `CHII_TTS_SSL_CERTFILE` / `CHII_TTS_SSL_KEYFILE` | （空） | 同时设置时以 HTTPS 启动 |

> 2026-09-14 安全加固：`/tts` 透传不再原样暴露引擎全部参数面——参数白名单 +
> 数值钳制（batch_size ≤ 16、sample_steps ≤ 64 等）+ ref_audio 前缀约束；
> 非 JSON 的 POST body 不再接受（此前按原始字节透传）。

## 4. systemd 守护（生产）

```ini
# /etc/systemd/system/chobits-chii-tts.service
[Unit]
Description=Chobits Chii TTS (facade -> docker engine)
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
ExecStart=/bin/bash /home/ubuntu/Github/Chobits-Chii-TTS/tools/start_tts_api.sh 9880
Restart=on-failure
RestartSec=5
LimitNOFILE=65536
EnvironmentFile=/etc/chobits-chii-tts.env

[Install]
WantedBy=multi-user.target
```

```bash
# /etc/chobits-chii-tts.env (chmod 600), 内容:
#   CHII_TTS_API_KEY=<随机密钥>
#   CHII_TTS_SSL_CERTFILE=/etc/chobits-chii-tts.crt   (可选, 见下方 TLS)
#   CHII_TTS_SSL_KEYFILE=/etc/chobits-chii-tts.key    (可选, 与上一条同时设置)
sudo systemctl daemon-reload && sudo systemctl enable --now chobits-chii-tts
journalctl -u chobits-chii-tts -f   # 查看日志
```

启用 HTTPS（自签名证书，客户端需信任该证书或用 `-k` 跳过校验）：

```bash
sudo openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout /etc/chobits-chii-tts.key -out /etc/chobits-chii-tts.crt -days 3650 \
  -subj "/CN=chii-tts" -addext "subjectAltName=IP:<服务器IP>,IP:127.0.0.1"
sudo chmod 600 /etc/chobits-chii-tts.key
# 在 /etc/chobits-chii-tts.env 中设置 CHII_TTS_SSL_CERTFILE / CHII_TTS_SSL_KEYFILE 后重启服务
```

## 5. 验证

```bash
# 引擎直连冒烟 (ref_audio_path 为容器内路径)
curl -G http://127.0.0.1:9882/tts \
  --data-urlencode "text=ちぃ、秀樹のこと、大好き。" \
  --data-urlencode "text_lang=ja" \
  --data-urlencode "ref_audio_path=/data/models/ref_audio.wav" \
  --data-urlencode "prompt_lang=ja" \
  --data-urlencode "prompt_text=秀樹は地位を拾ってくれた" \
  --data-urlencode "media_type=wav" -o engine.wav

# 门面 OpenAI 兼容 (baseUrl 填 http(s)://<服务器IP>:9880/v1)
curl -k -X POST https://<服务器IP>:9880/v1/audio/speech \
  -H "Authorization: Bearer <API_KEY>" \
  -H 'Content-Type: application/json' \
  -d '{"model": "chii-tts", "input": "ちぃ、秀樹のこと、大好き。", "voice": "chii"}' \
  -o out.wav
# 未启用 TLS 时把 https 换成 http、去掉 -k 即可
```

`GET /v1/models` 返回固定模型 `chii-tts`；`voice` 当前仅 `chii`；`response_format` 支持
`wav`/`aac`/`opus`，默认 `wav`。`wav` 为流式输出（边合成边推流，首字延迟低）；
`aac`/`opus` 为合成完成后一次性返回。

注意在云安全组放行 TCP 9880（9882 只绑回环，无需放行）；对外提供服务须遵守
CC BY-NC-SA 4.0（非商业）。面向公众分发应用时建议由后端服务代为调用，
不要把唯一密钥嵌进客户端。

## 实测记录（2026-09-10 本机迁移）

- 镜像体积 10.9GB；构建约 30 分钟（torch 走 SJTU、PyPI 走清华镜像）。
- 引擎冷启动到就绪约 40s（权重加载 + CUDA 初始化），常驻显存约 2.4GB。
- 合成延迟（短句）：首次 21.8s（含 CUDA warmup），热请求约 1.2s。
- 切换当日发现并修复：旧 unit 的 `ExecStart` 仍指向改名前的 `Chobits-Chi-TTS` 目录
  （仓库改名后未同步，运行中的旧进程不受影响，但任何 restart/reboot 都会失败）——
  已改为现路径并补 `After=docker.service`。**迁移旧部署时务必先核对 unit 内路径。**
- 现网 env 启用了 TLS，门面日志应为 `Uvicorn running on https://...`；
  验证时用 `curl -k https://...`（http 探测会得到空响应，属预期）。

## 从旧 systemd 部署迁移（2026-09 之前的宿主内 Conda 部署）

旧部署中 `tools/start_tts_api.sh` 直接用 conda 环境拉起 `api_v2`；现该脚本只起门面，
ExecStart 不变，因此：

1. 按第 1–2 节构建镜像、启动引擎容器（方式 A 复用现有模型，零下载）；
2. `sudo systemctl restart chobits-chii-tts`（env 文件无需改动，密钥沿用）；
3. 按第 5 节验证。回滚：取迁移前旧提交的版本还原后再次 restart
   （`git log` 找到迁移前提交，`git checkout <旧提交> -- tools/ requirements.txt`；
   conda 环境与 GPT-SoVITS 目录保持不动）。
