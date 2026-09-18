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

引擎源码经 `GS_REPO_REF` 锁到 README 验证过的 commit（默认 `49587af3`，
即 fix-nonstream-threadpool 分支 2026-09-16 的 HEAD，构建时可用 `--build-arg` 覆盖）；
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
| `CHII_TTS_MAX_BODY_BYTES` | `26214400` (25MB) | `/tts` POST 与 `/v1/audio/speech` 请求体大小上限，超限 413（Content-Length 预检 + 无长度时按流累计兜底）；uvicorn/Caddy 无默认 cap |
| `CHII_TTS_MAX_TEXT_CHARS` | `2000` | 合成文本长度硬上限（两路径均生效），防长文本独占 GPU |
| `CHII_TTS_MAX_INFLIGHT` | `8` | 全局在途并发上限（保护 GPU）：超上限排队等待空位，`CHII_TTS_QUEUE_TIMEOUT` 秒内仍拿不到才 429；流式请求（wav 流式与 `/tts` 透传）的信号量持有到推流结束/客户端断开 |
| `CHII_TTS_QUEUE_TIMEOUT` | `30` | 在途满员后排队等待空位的超时秒数，超时返回 429 |
| `CHII_TTS_BATCH_SIZE` | `5` | 引擎批推理 batch_size 默认值（客户端未显式传时注入，仅非流式路径生效；流式一律钉 1） |
| `CHII_TTS_BIND` | `127.0.0.1` | 门面监听地址（生产由 Caddy 反代；绑非回环地址须配 TLS，否则拒绝启动） |
| `CHII_TTS_REF_AUDIO` | `/data/models/ref_audio.wav` | OpenAI 垫片 `chii` 音色的参考音频（**引擎容器内**路径） |
| `CHII_TTS_REF_AUDIO_PREFIX` | `/data/` | `/tts` 透传的 ref_audio 路径前缀约束（防容器内任意路径探测） |
| `CHII_TTS_REF_TEXT_FILE` | `models/ref_text.txt` | 参考文本（门面**宿主机**路径，读出后内联传给引擎） |
| `CHII_TTS_SSL_CERTFILE` / `CHII_TTS_SSL_KEYFILE` | （空） | 同时设置时以 HTTPS 启动 |
| `CHII_TTS_DEEP_PROBE_TTL` | `30` | `/healthz/deep` 探测结果缓存秒数（防高频探测烧 GPU） |
| `CHII_TTS_DEEP_PROBE_TIMEOUT` | `20` | `/healthz/deep` 单次真实合成探测的超时秒数 |

> 2026-09-15 门面加固（引擎流式模式 bug 的门面侧规避）：`wav` 流式路径下，多句文本
> 由门面按句切分（日/中标点与换行）后逐句串行调引擎并合并 PCM 流，且流式请求的
> `batch_size` 一律钉 1——引擎流式模式对多片段并行批推理会抛 "Sizes of tensors must
> match"（返回 200 但音频截断，反复触发还会拖垮引擎致所有请求 200 空流），而引擎内部
> 还会把单句按逗号/顿号等再切成片段（门面切句管不到，真实流量已观测到单句触发）；
> 对单片段文本钉 1 无影响（本就只有 1 个片段进批，推理结果一致）。同时合成开始前
> 预读上游首块，连接失败/非 200/空流返回 502 JSON 而非 200 空流。
> aac/opus 非流式路径不受影响。
> 新增 `GET /healthz/deep` 深度健康检查（真实合成探测，带缓存，须 API key）；
> 轻量存活仍用 `GET /v1/models`。

> 2026-09-14 安全加固：`/tts` 透传不再原样暴露引擎全部参数面——参数白名单 +
> 数值钳制（batch_size ≤ 20、sample_steps ≤ 64 等）+ ref_audio 前缀约束；
> 非 JSON 的 POST body 不再接受（此前按原始字节透传）。
> 2026-09-16 补充：流式透传（`streaming_mode` 为真）的 `batch_size` 一律钉 1
> （覆盖显式传值并记日志），否则 `GET /tts?streaming_mode=2` 即可触发上述引擎
> 批推理 bug 致全服务 200 空流；整型钳制参数（batch_size/top_k/sample_steps）
> 钳制后还原 int（此前 GET 透传 "5.0" 被引擎拒成 422）。

> 2026-09-18 内测前加固：① 请求体大小上限 `CHII_TTS_MAX_BODY_BYTES`（默认 25MB，
> 对齐 ASR 批量上限量级）——Content-Length 超限直接 413（错误体含稳定 code
> `payload_too_large`），chunked 无长度时按 `request.stream()` 累计兜底；
> ② `/tts` 透传的 ref_audio 路径改为 normpath 归一化后判前缀：此前仅 startswith，
> `/data/../../etc/passwd` 可穿越，现归一化后须严格落在 `CHII_TTS_REF_AUDIO_PREFIX`
> 内（`..`/冗余分隔符折叠后再判，宿主机本地可见时再解析符号链接复核且须为常规文件）；
> ③ 参考文本（`CHII_TTS_REF_TEXT_FILE`）缺失/为空不再静默置零样本：启动打 WARN 日志，
> `/healthz/deep` 响应体新增 `ref_ready` 字段（状态码语义不变，引擎健康仍 200，
> `ref_ready=false` 表示合成质量降质，运维据此告警）。

## 4. systemd 守护（生产）

unit 以仓库 [deploy/chobits-chii-tts.service](../deploy/chobits-chii-tts.service) 为唯一事实源——
本文不再内嵌全文（避免双份漂移，改动只改 deploy/ 那一处）：

```bash
sudo cp deploy/chobits-chii-tts.service /etc/systemd/system/chobits-chii-tts.service
```

unit 要点：`Restart=always` + `RestartSec=3`（崩溃/启动失败快速拉起）、
`NoNewPrivileges=true` / `PrivateTmp=true` 加固、`EnvironmentFile` 注入密钥、
`LimitNOFILE=65536`、`After=docker.service` 保证引擎容器先就绪。

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

# 门面 OpenAI 兼容（Caddy 架构下客户端 baseUrl 填 https://<服务器IP>/chobits/v1，
# 由 LLM 垫片转发本机门面；直调门面验证用本机回环 http://127.0.0.1:9880/v1）
curl -X POST http://127.0.0.1:9880/v1/audio/speech \
  -H "Authorization: Bearer <API_KEY>" \
  -H 'Content-Type: application/json' \
  -d '{"model": "chii-tts", "input": "ちぃ、秀樹のこと、大好き。", "voice": "chii"}' \
  -o out.wav
```

`GET /v1/models` 返回固定模型 `chii-tts`；`voice` 当前仅 `chii`；`response_format` 支持
`wav`/`aac`/`opus`，默认 `wav`。`wav` 为流式输出（边合成边推流，首字延迟低；多句文本
由门面按句串行合成，见第 3 节加固说明）；`aac`/`opus` 为合成完成后一次性返回。

```bash
# 深度健康检查 (真实合成探测, 结果缓存 30s; 异常时 503)
# 响应体含 ref_ready: 参考文本缺失/为空时为 false —— 引擎健康仍 200,
# 但 OpenAI 垫片在跑零样本提示, 合成质量降质 (启动日志有 WARN)
curl -H "Authorization: Bearer <API_KEY>" http://127.0.0.1:9880/healthz/deep
```

注意：Caddy 架构（2026-09-12 起）下门面绑回环、公网只放行 TCP 443，安全组
**不需要**放行 9880/9882；对外提供服务须遵守 CC BY-NC-SA 4.0（非商业）。
面向公众分发应用时应由后端服务代为调用（现网即如此：客户端 → 443 → LLM
垫片 → 本机门面），不要把唯一密钥嵌进客户端。

## 实测记录（2026-09-10 本机迁移）

- 镜像体积 10.9GB；构建约 30 分钟（torch 走 SJTU、PyPI 走清华镜像）。
- 引擎冷启动到就绪约 40s（权重加载 + CUDA 初始化），常驻显存约 2.4GB。
- 合成延迟（短句）：首次 21.8s（含 CUDA warmup），热请求约 1.2s。
- 切换当日发现并修复：旧 unit 的 `ExecStart` 仍指向改名前的 `Chobits-Chi-TTS` 目录
  （仓库改名后未同步，运行中的旧进程不受影响，但任何 restart/reboot 都会失败）——
  已改为现路径并补 `After=docker.service`。**迁移旧部署时务必先核对 unit 内路径。**
- ~~现网 env 启用了 TLS，门面日志应为 `Uvicorn running on https://...`~~
  （2026-09-12 起门面已关 TLS、绑回环，TLS 由 Caddy 终结；日志为 http://127.0.0.1）。

## 从旧 systemd 部署迁移（2026-09 之前的宿主内 Conda 部署）

旧部署中 `tools/start_tts_api.sh` 直接用 conda 环境拉起 `api_v2`；现该脚本只起门面，
ExecStart 不变，因此：

1. 按第 1–2 节构建镜像、启动引擎容器（方式 A 复用现有模型，零下载）；
2. `sudo systemctl restart chobits-chii-tts`（env 文件无需改动，密钥沿用）；
3. 按第 5 节验证。回滚：取迁移前旧提交的版本还原后再次 restart
   （`git log` 找到迁移前提交，`git checkout <旧提交> -- tools/ requirements.txt`；
   conda 环境与 GPT-SoVITS 目录保持不动）。
