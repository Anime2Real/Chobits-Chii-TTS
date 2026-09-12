"""小叽 TTS HTTP 门面: 鉴权 + 限流 + OpenAI 垫片, 反向代理 Docker 容器内的 GPT-SoVITS 引擎.

架构 (与家族 ASR 服务一致): 推理引擎 (上游 api_v2, 无鉴权) 跑在 Docker 容器里,
只发布到宿主机 127.0.0.1 (见 docs/deployment.md); 本门面是唯一对外入口:
  - API Key: 环境变量 CHII_TTS_API_KEY (必填, 未设置则拒绝启动).
    客户端通过 `Authorization: Bearer <key>` 或 `?api_key=<key>` 提供.
  - 限流: /tts 与 /v1/audio/speech 每 IP 每分钟最多 CHII_TTS_RATE_LIMIT 次 (默认 60, 设 0 关闭).
  - TLS: 同时设置 CHII_TTS_SSL_CERTFILE / CHII_TTS_SSL_KEYFILE 时以 HTTPS 启动.
  - 引擎地址: CHII_TTS_ENGINE_URL (默认 http://127.0.0.1:9882).

OpenAI TTS 兼容垫片 (客户端 baseUrl 填 http(s)://<IP>:9880/v1):
  - GET  /v1/models        → 固定返回 chii-tts
  - POST /v1/audio/speech  → OpenAI TTS 协议: {"model", "input", "voice", "response_format"?, "speed"?}
    voice 映射引擎侧参考音频 (当前仅 "chii");
    response_format 支持 wav/aac/opus (默认 wav; mp3/flac/pcm 暂不支持, 返回 400);
    speed (0.25~4.0) 映射 speed_factor; text_lang 由服务端钉死 auto,
    其余采样参数取引擎默认值. wav 为流式输出 (边合成边推流, 首字延迟低);
    aac/opus 为合成完成后一次性返回.

原生 /tts (GET/POST) 原样透传到引擎 (GET query / POST JSON); 注意其中的
ref_audio_path 是**引擎容器内**路径 (默认数据卷挂载为 /data/models/..., 见
docs/deployment.md). 引擎的 /control 与 /set_*_weights 不对外暴露.

启动: python tools/server.py [端口, 默认 9880]  (绑 0.0.0.0)
"""

from __future__ import annotations  # 宿主机 Python 3.8 (Ubuntu 20.04) 兼容

import json
import os
import sys
import time
from collections import defaultdict, deque

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


def _getenv(suffix: str, default: str = "") -> str:
    """读取 CHII_TTS_<suffix> 环境变量."""
    return os.environ.get(f"CHII_TTS_{suffix}", default)


API_KEY = _getenv("API_KEY")
if not API_KEY:
    sys.exit("[错误] 未设置 CHII_TTS_API_KEY 环境变量, 拒绝以无鉴权方式启动")

RATE_LIMIT = int(_getenv("RATE_LIMIT", "60"))

# TLS: 两个变量都设置时以 HTTPS 启动 (自签名证书见 README「部署为 HTTP 服务」)
SSL_CERTFILE = _getenv("SSL_CERTFILE")
SSL_KEYFILE = _getenv("SSL_KEYFILE")
if bool(SSL_CERTFILE) != bool(SSL_KEYFILE):
    sys.exit("[错误] CHII_TTS_SSL_CERTFILE 与 CHII_TTS_SSL_KEYFILE 必须同时设置")

ENGINE_URL = _getenv("ENGINE_URL", "http://127.0.0.1:9882").rstrip("/")

APP = FastAPI(title="chobits-chii-tts facade")
# 长文本合成耗时久, read 超时放宽; 流式响应按每次读操作计时, 正常推流不会触发
_client = httpx.AsyncClient(
    base_url=ENGINE_URL,
    timeout=httpx.Timeout(connect=10.0, read=600.0, write=120.0, pool=60.0),
)

_hits: dict[str, deque] = defaultdict(deque)

# --- OpenAI TTS 兼容垫片 -------------------------------------------------
# voice → 引擎侧参考音频映射. ref_audio_path 是引擎容器内路径 (客户端不可见),
# 用 CHII_TTS_REF_AUDIO 覆盖; prompt_text 由门面从宿主机文件读出后内联传给引擎,
# 用 CHII_TTS_REF_TEXT_FILE 覆盖 (默认取本仓库 models/ref_text.txt)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ref_text_file = _getenv("REF_TEXT_FILE", os.path.join(REPO_ROOT, "models", "ref_text.txt"))
try:
    with open(_ref_text_file, encoding="utf-8") as _f:
        _ref_text = _f.read().strip()
except OSError:
    _ref_text = ""

VOICES = {
    "chii": {
        "ref_audio_path": _getenv("REF_AUDIO", "/data/models/ref_audio.wav"),
        "prompt_text": _ref_text,
        "prompt_lang": "ja",
    },
}
TTS_MODEL = "chii-tts"
# OpenAI response_format → 引擎 media_type (ogg 即 opus 的 ogg 封装); mp3/flac/pcm 暂不支持
FORMAT_MAP = {"wav": "wav", "aac": "aac", "opus": "ogg"}
# OpenAI TTS 协议不传语言: 钉死 auto (引擎为 v2Pro, 语言列表含 auto)
TEXT_LANG = "auto"


def _openai_error(code: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=code, content={
        "error": {"message": message, "type": "invalid_request_error", "param": None, "code": None}})


async def _proxy_to_engine(method: str, **kwargs) -> StreamingResponse | JSONResponse:
    """流式转发到引擎 /tts: 状态码与 Content-Type 原样回传, 客户端断开时关闭上游连接."""
    try:
        req = _client.build_request(method, "/tts", **kwargs)
        upstream = await _client.send(req, stream=True)
    except httpx.HTTPError as exc:
        return JSONResponse(status_code=503, content={
            "message": f"tts engine unavailable ({ENGINE_URL}): {exc.__class__.__name__}"})

    headers = {}
    if ct := upstream.headers.get("content-type"):
        headers["content-type"] = ct

    async def body_iter():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(body_iter(), status_code=upstream.status_code, headers=headers)


@APP.get("/v1/models")
async def openai_models():
    return {"object": "list", "data": [
        {"id": TTS_MODEL, "object": "model", "created": 0, "owned_by": "chobits-chii"}]}


@APP.post("/v1/audio/speech")
async def openai_audio_speech(request: Request):
    try:
        body = await request.json()
    except (ValueError, json.JSONDecodeError):
        return _openai_error(400, "请求体不是合法 JSON")
    text = str(body.get("input") or "").strip()
    if not text:
        return _openai_error(400, "input 不能为空")
    voice = str(body.get("voice") or "chii")
    if voice not in VOICES:
        return _openai_error(400, f"voice: {voice} 不存在, 可用: {sorted(VOICES)}")
    fmt = str(body.get("response_format") or "wav").lower()
    if fmt not in FORMAT_MAP:
        return _openai_error(400, f"response_format: {fmt} 暂不支持, 可用: {sorted(FORMAT_MAP)}")
    speed = body.get("speed", 1.0)
    if not isinstance(speed, (int, float)) or not 0.25 <= float(speed) <= 4.0:
        return _openai_error(400, "speed 须在 0.25~4.0 之间")

    v = VOICES[voice]
    # 引擎 /tts 全部字段有默认值, 只需传覆盖项 (见上游 TTS_Request);
    # wav 走流式 (streaming_mode=2: 首块 WAV 头 + 后续 raw PCM, 首字延迟低);
    # aac/opus 逐块编码会拼出损坏帧, 保持非流式一次性返回
    payload = {
        "text": text,
        "text_lang": TEXT_LANG,
        "ref_audio_path": v["ref_audio_path"],
        "prompt_text": v["prompt_text"],
        "prompt_lang": v["prompt_lang"],
        "media_type": FORMAT_MAP[fmt],
        "speed_factor": float(speed),
        "streaming_mode": 2 if FORMAT_MAP[fmt] == "wav" else False,
    }
    return await _proxy_to_engine("POST", json=payload)


@APP.api_route("/tts", methods=["GET", "POST"])
async def tts_passthrough(request: Request):
    if request.method == "GET":
        # 剥掉 api_key 再转发 (multi_items 保留 aux_ref_audio_paths 等重复参数)
        params = [(k, v) for k, v in request.query_params.multi_items() if k != "api_key"]
        return await _proxy_to_engine("GET", params=params)
    body = await request.body()
    content_type = request.headers.get("content-type", "application/json")
    return await _proxy_to_engine("POST", content=body, headers={"content-type": content_type})

# ------------------------------------------------------------------------


def _extract_key(request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.query_params.get("api_key", "")


@APP.middleware("http")
async def auth_and_rate_limit(request, call_next):
    if _extract_key(request) != API_KEY:
        return JSONResponse(status_code=401, content={"message": "invalid or missing api key"})

    if RATE_LIMIT > 0 and request.url.path in ("/tts", "/v1/audio/speech"):
        ip = request.client.host if request.client else "unknown"
        now = time.time()
        q = _hits[ip]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_LIMIT:
            return JSONResponse(status_code=429, content={"message": "rate limit exceeded"})
        q.append(now)

    return await call_next(request)


BIND = _getenv("BIND", "0.0.0.0")  # 生产走 Caddy 反代时绑 127.0.0.1

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9880
    uvicorn.run(
        app=APP,
        host=BIND,
        port=port,
        workers=1,
        ssl_certfile=SSL_CERTFILE or None,
        ssl_keyfile=SSL_KEYFILE or None,
    )
