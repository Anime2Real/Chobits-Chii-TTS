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

原生 /tts (GET/POST) 透传到引擎 (GET query / POST JSON)，但经参数白名单 + 数值钳制
+ ref_audio 路径前缀约束（CHII_TTS_REF_AUDIO_PREFIX，默认 /data/）——不再暴露
引擎全部参数面。引擎的 /control 与 /set_*_weights 不对外暴露.

资源防护（TTS 推理昂贵，不设防时单请求长文本即可独占 GPU 数分钟）:
  - 合成文本长度硬上限 CHII_TTS_MAX_TEXT_CHARS (默认 2000 字符, /tts 与
    /v1/audio/speech 均生效);
  - 全局在途并发上限 CHII_TTS_MAX_INFLIGHT (默认 8, 超出即 429).

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
# 资源防护：文本长度硬上限 + 全局在途并发上限（TTS 推理昂贵，单请求长文本
# 经切分后可独占 GPU 数分钟，请求数限流管不了单请求成本）
MAX_TEXT_CHARS = int(_getenv("MAX_TEXT_CHARS", "2000"))
MAX_INFLIGHT = int(_getenv("MAX_INFLIGHT", "8"))

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
_inflight = 0  # /tts 与 /v1/audio/speech 的在途请求数（全局并发上限保护 GPU）

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
    if len(text) > MAX_TEXT_CHARS:
        return _openai_error(400, f"input 超长（上限 {MAX_TEXT_CHARS} 字符）")
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


# --- /tts 透传参数白名单（安全加固：此前引擎全部参数面裸露——
# ref_audio_path 可指向容器内任意路径（文件存在性 oracle / /dev/urandom 挂死引擎），
# batch_size/super_sampling 等采样参数可放大 GPU 消耗，还可克隆容器内任意音频） ---
REF_AUDIO_PREFIX = _getenv("REF_AUDIO_PREFIX", "/data/")
TTS_ALLOWED_KEYS = {
    "text", "text_lang", "prompt_text", "prompt_lang", "media_type",
    "speed_factor", "streaming_mode", "text_split_method", "fragment_interval",
    "seed", "top_k", "top_p", "temperature", "repetition_penalty",
    "batch_size", "batch_threshold", "split_bucket", "return_fragment",
    "fixed_length_chunk", "parallel_infer", "sample_steps", "super_sampling",
}
TTS_MEDIA_TYPES = {"wav", "aac", "ogg", "mp3", "flac", "pcm"}
# 数值钳制区间（防 GPU 消耗放大）；不在表内的数值参数原样放行
TTS_NUMERIC_CLAMPS = {
    "batch_size": (1, 16),
    "batch_threshold": (1, 100),
    "sample_steps": (1, 64),
    "top_k": (1, 100),
    "top_p": (0.0, 1.0),
    "temperature": (0.0, 2.0),
    "repetition_penalty": (0.0, 2.0),
    "speed_factor": (0.25, 4.0),
    "fragment_interval": (0.01, 1.0),
}


def _sanitize_tts_params(items):
    """/tts 透传参数清洗：白名单过滤 + 数值钳制 + ref_audio 前缀约束。
    items 为 (key, value) 列表（GET 用 multi_items 保留重复参数；POST 展平 JSON）。
    返回 (payload, error_message)；payload 中 ref_audio 类键保留为列表由调用方还原。"""
    payload = {}
    ref_audio_paths = []
    aux_ref_audio_paths = []
    for key, value in items:
        if key == "ref_audio_path":
            ref_audio_paths.append(value)
            continue
        if key == "aux_ref_audio_paths":
            aux_ref_audio_paths.append(value)
            continue
        if key not in TTS_ALLOWED_KEYS:
            continue
        if key in TTS_NUMERIC_CLAMPS:
            lo, hi = TTS_NUMERIC_CLAMPS[key]
            try:
                value = min(max(float(value), lo), hi)
            except (TypeError, ValueError):
                continue  # 非法数值直接丢弃，用引擎默认值
        payload[key] = value
    for path in ref_audio_paths + aux_ref_audio_paths:
        if not str(path).startswith(REF_AUDIO_PREFIX):
            return None, f"ref_audio 路径须在 {REF_AUDIO_PREFIX} 前缀内"
    text = payload.get("text")
    if text is not None and len(str(text)) > MAX_TEXT_CHARS:
        return None, f"text 超长（上限 {MAX_TEXT_CHARS} 字符）"
    prompt_text = payload.get("prompt_text")
    if prompt_text is not None and len(str(prompt_text)) > 500:
        return None, "prompt_text 超长（上限 500 字符）"
    media_type = payload.get("media_type")
    if media_type is not None and str(media_type) not in TTS_MEDIA_TYPES:
        return None, f"media_type 非法，可用: {sorted(TTS_MEDIA_TYPES)}"
    if ref_audio_paths:
        payload["ref_audio_path"] = ref_audio_paths[-1]
    if aux_ref_audio_paths:
        payload["aux_ref_audio_paths"] = aux_ref_audio_paths
    return payload, None


@APP.api_route("/tts", methods=["GET", "POST"])
async def tts_passthrough(request: Request):
    if request.method == "GET":
        # 剥掉 api_key 再清洗转发 (multi_items 保留 aux_ref_audio_paths 等重复参数)
        items = [(k, v) for k, v in request.query_params.multi_items() if k != "api_key"]
        payload, error = _sanitize_tts_params(items)
        if error:
            return JSONResponse(status_code=400, content={"message": error})
        # GET 透传展开为 query 参数（aux_ref_audio_paths 还原为重复键）
        params = [(k, v) for k, v in payload.items() if k != "aux_ref_audio_paths"]
        params += [("aux_ref_audio_paths", v) for v in payload.get("aux_ref_audio_paths", [])]
        return await _proxy_to_engine("GET", params=params)
    try:
        body = await request.json()
    except (ValueError, json.JSONDecodeError):
        return JSONResponse(status_code=400, content={"message": "请求体不是合法 JSON"})
    if not isinstance(body, dict):
        return JSONResponse(status_code=400, content={"message": "请求体须为 JSON 对象"})
    items = []
    for key, value in body.items():
        if key == "aux_ref_audio_paths" and isinstance(value, list):
            items += [(key, item) for item in value]
        else:
            items.append((key, value))
    payload, error = _sanitize_tts_params(items)
    if error:
        return JSONResponse(status_code=400, content={"message": error})
    return await _proxy_to_engine("POST", json=payload)

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

    limited_path = request.url.path in ("/tts", "/v1/audio/speech")
    if RATE_LIMIT > 0 and limited_path:
        ip = request.client.host if request.client else "unknown"
        now = time.time()
        q = _hits[ip]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_LIMIT:
            return JSONResponse(status_code=429, content={"message": "rate limit exceeded"})
        q.append(now)

    # 全局在途并发上限：限流管请求数管不了单请求成本（长文本独占 GPU 数分钟），
    # 超出即拒，防并发大请求打满引擎
    global _inflight
    if limited_path and _inflight >= MAX_INFLIGHT:
        return JSONResponse(status_code=429, content={"message": "server busy, too many in-flight requests"})
    if limited_path:
        _inflight += 1
    try:
        return await call_next(request)
    finally:
        if limited_path:
            _inflight -= 1


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
