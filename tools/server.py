"""小叽 TTS HTTP 门面: 鉴权 + 限流 + OpenAI 垫片, 反向代理 Docker 容器内的 GPT-SoVITS 引擎.

架构 (与家族 ASR 服务一致): 推理引擎 (上游 api_v2, 无鉴权) 跑在 Docker 容器里,
只发布到宿主机 127.0.0.1 (见 docs/deployment.md); 本门面是唯一对外入口:
  - API Key: 环境变量 CHII_TTS_API_KEY (必填, 未设置则拒绝启动).
    客户端通过 `Authorization: Bearer <key>` 提供 (query 传 key 会进访问日志, 不支持).
  - 限流: /tts 与 /v1/audio/speech 每 IP 每分钟最多 CHII_TTS_RATE_LIMIT 次 (默认 60, 设 0 关闭).
  - TLS: 同时设置 CHII_TTS_SSL_CERTFILE / CHII_TTS_SSL_KEYFILE 时以 HTTPS 启动.
  - 引擎地址: CHII_TTS_ENGINE_URL (默认 http://127.0.0.1:9882).

OpenAI TTS 兼容垫片 (客户端 baseUrl 填 http(s)://<IP>:9880/v1):
  - GET  /healthz          → 200 (免鉴权浅探活，与 ASR 门面对齐；深探测 /healthz/deep 仍需 key)
  - GET  /v1/models        → 固定返回 chii-tts
  - POST /v1/audio/speech  → OpenAI TTS 协议: {"model", "input", "voice", "response_format"?, "speed"?}
    voice 映射引擎侧参考音频 (当前仅 "chii");
    response_format 支持 wav/aac/opus (默认 wav; mp3/flac/pcm 暂不支持, 返回 400);
    speed (0.25~4.0) 映射 speed_factor; text_lang 由服务端钉死 auto,
    其余采样参数取引擎默认值. wav 为流式输出 (边合成边推流, 首字延迟低);
    aac/opus 为合成完成后一次性返回.

wav 流式路径的门面侧加固 (引擎流式模式多句并行批推理会触发
"Sizes of tensors must match": 返回 200 但音频截断, 反复触发还会拖垮引擎
(t2s_model 变 None, 之后所有请求 200 空流)):
  - 按句串行化: 多句文本按日/中标点与换行切句, 逐句串行调引擎流式接口,
    多条 PCM 流合并成一条 WAV 流 (首句首块含 WAV 头原样下发, 后续句子剥头);
    流式请求的 batch_size 一律钉 1 (引擎内部还会把单句再切成片段——内部切分
    含逗号/顿号等, 门面切句管不到——批推理同样触发该 bug; 对单片段文本无影响,
    本就只有 1 个片段进批);
  - 上游即败: 合成开始前预读上游首块, 连接失败/非 200/空流时返回 502 JSON,
    不发 200 空流; 已在流中的失败只能中断连接.

健康检查:
  - GET /v1/models      → 轻量存活 (进程级, 固定响应);
  - GET /healthz/deep   → 深度检查: 用极短文本向引擎发一次真实合成
    (短超时 CHII_TTS_DEEP_PROBE_TIMEOUT, 默认 20s), 能发现"200 空流"变砖;
    结果缓存 CHII_TTS_DEEP_PROBE_TTL 秒 (默认 30) 避免高频探测烧 GPU;
    响应体带 ref_ready (启动时参考文本可读且非空则为 true) —— 参考文本缺失
    时 OpenAI 垫片静默退化为零样本提示, 引擎仍健康, 状态码不变, 靠该字段
    与启动 WARN 日志暴露降质; 与其余端点一样须带 API key.

原生 /tts (GET/POST) 透传到引擎 (GET query / POST JSON)，但经参数白名单 + 数值钳制
+ ref_audio 路径前缀约束（CHII_TTS_REF_AUDIO_PREFIX，默认 /data/；归一化后须
严格落在前缀内，.. 穿越与符号链接逃逸均拒绝）——不再暴露
引擎全部参数面。流式透传 (streaming_mode 为真) 的 batch_size 一律钉 1
（与 OpenAI 垫片 wav 流式同款规避，见上方说明）。引擎的 /control 与 /set_*_weights 不对外暴露.

资源防护（TTS 推理昂贵，不设防时单请求长文本即可独占 GPU 数分钟）:
  - 请求体大小硬上限 CHII_TTS_MAX_BODY_BYTES (默认 25MB, /tts POST 与
    /v1/audio/speech 均生效): Content-Length 超限直接 413, 无 Content-Length
    (chunked) 时按 request.stream() 累计兜底 —— uvicorn/Caddy 均无默认 cap,
    不设防时持 key 即可发超大 JSON 吃内存;
  - 合成文本长度硬上限 CHII_TTS_MAX_TEXT_CHARS (默认 2000 字符, /tts 与
    /v1/audio/speech 均生效);
  - 全局在途并发上限 CHII_TTS_MAX_INFLIGHT (默认 8)：超上限不立即拒绝，
    排队等待空位，CHII_TTS_QUEUE_TIMEOUT 秒 (默认 30) 内仍拿不到才 429。
    持有期覆盖真实 GPU 占用：并发控制在端点层执行，流式响应 (wav 流式与
    /tts 透传) 的信号量持有到推流结束/客户端断开（放中间件层时 call_next
    拿到响应头即返回，数分钟的流式合成会落在信号量外）。

引擎批推理：客户端未显式传 batch_size 时注入 CHII_TTS_BATCH_SIZE (默认 5)，
仅对非流式 (aac/opus) 生效——引擎 parallel_infer 批推理在流式模式下不启用
(见 GPT_SoVITS/TTS_infer_pack/TTS.py)。

启动: python tools/server.py [端口, 默认 9880]  (默认绑 127.0.0.1, 生产由 Caddy 反代;
                                     显式绑非回环地址须同时配 TLS, 否则拒绝启动)
"""

from __future__ import annotations  # 宿主机 Python 3.8 (Ubuntu 20.04) 兼容

import asyncio
import json
import array
import struct
import os
import posixpath
import re
import sys
import time

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from chii_facade_common import (
    ApiKeyAuth,
    EnvConfig,
    SlidingWindowRateLimiter,
    client_ip as _client_ip,
    engine_error_body,
    extract_bearer_token,
    filter_response_headers,
)

# 公共逻辑（env 容错解析 / XFF 真实 IP / key 校验 / 限流桶 / 错误通用化 / 响应头白名单）
# 源自共享库 chii_facade_common（CloudDeploy 仓库 tools/chii-facade-common，兄弟目录 editable 安装）：
# 安全加固只改共享库一处，两门面同步生效，勿在本地重建副本。

_env = EnvConfig("CHII_TTS_")
_getenv = _env.get
_getenv_int = _env.get_int
_getenv_float = _env.get_float


API_KEY = _getenv("API_KEY")
if not API_KEY:
    sys.exit("[错误] 未设置 CHII_TTS_API_KEY 环境变量, 拒绝以无鉴权方式启动")

_auth = ApiKeyAuth(API_KEY)
_key_ok = _auth.key_ok

RATE_LIMIT = _getenv_int("RATE_LIMIT", 60)
# 请求体大小硬上限（字节）：POST /tts 与 /v1/audio/speech 的 JSON 体上限，
# 与 ASR 批量上传上限同量级（25MB）。uvicorn/Caddy 均无默认 cap，
# 持 key 即可发超大 JSON 吃内存；chunked 可不带 Content-Length，故再按累计兜底
MAX_BODY_BYTES = _getenv_int("MAX_BODY_BYTES", 25 * 1024 * 1024)
# 资源防护：文本长度硬上限 + 全局在途并发上限（TTS 推理昂贵，单请求长文本
# 经切分后可独占 GPU 数分钟，请求数限流管不了单请求成本）
MAX_TEXT_CHARS = _getenv_int("MAX_TEXT_CHARS", 2000)
MAX_INFLIGHT = _getenv_int("MAX_INFLIGHT", 8)
# 并发排队等待超时（秒）：在途满 MAX_INFLIGHT 后排队，超时仍无空位才 429
QUEUE_TIMEOUT = _getenv_float("QUEUE_TIMEOUT", 30.0)
# 引擎批推理 batch_size 默认值：客户端未显式传时注入（仅非流式路径生效）
BATCH_SIZE = _getenv_int("BATCH_SIZE", 5)

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

# /tts 与 /v1/audio/speech 的每 IP 滑动窗口限流桶（共享库 SlidingWindowRateLimiter）；
# RATE_LIMIT 在检查路径上同步进 limiter，保持 monkeypatch 模块常量即时生效的行为不变
_rate_limiter = SlidingWindowRateLimiter(RATE_LIMIT)
# /tts 与 /v1/audio/speech 的全局在途并发信号量（保护 GPU）。限流管请求数管不了
# 单请求成本（长文本独占 GPU 数分钟）；并发控制在端点层执行而非中间件——
# BaseHTTPMiddleware 的 call_next 拿到响应头即返回，流式合成数分钟的 GPU 占用
# 会落在信号量外，上限名存实亡。
# Python 3.8 的 asyncio.Semaphore 创建时即绑定事件循环，模块级创建会绑错 loop，
# 故在 startup 钩子（运行中的 loop 内）初始化；端点里保留防御性懒创建兜底
_inflight_sem: asyncio.Semaphore | None = None


@APP.on_event("startup")
async def _init_inflight_sem():
    global _inflight_sem
    _inflight_sem = asyncio.Semaphore(MAX_INFLIGHT)


class _InflightHold:
    """在途信号量持有句柄：幂等释放，允许把持有期从端点函数延长到流式响应体写完。"""

    def __init__(self, sem: asyncio.Semaphore):
        self._sem: asyncio.Semaphore | None = sem

    def release(self) -> None:
        sem, self._sem = self._sem, None
        if sem is not None:
            sem.release()


async def _acquire_inflight() -> tuple[_InflightHold | None, JSONResponse | None]:
    """获取全局在途信号量：超上限排队等待空位，QUEUE_TIMEOUT 秒内仍拿不到返回 429 响应。
    成功时返回 (hold, None)，调用方须在响应生命周期结束时 hold.release()。"""
    global _inflight_sem
    if _inflight_sem is None:  # 防御兜底：协程内创建，绑定当前运行中的 loop
        _inflight_sem = asyncio.Semaphore(MAX_INFLIGHT)
    try:
        await asyncio.wait_for(_inflight_sem.acquire(), QUEUE_TIMEOUT)
    except asyncio.TimeoutError:
        return None, JSONResponse(status_code=429, content={"message": "server busy, queue wait timeout"})
    return _InflightHold(_inflight_sem), None


def _hold_through_response(response, hold: _InflightHold):
    """按响应类型决定信号量释放时机：非流式响应在端点返回前合成已完成，立即释放；
    流式响应（wav 流式 / /tts 透传）的合成贯穿整个 body 迭代，包裹迭代器把释放
    推迟到迭代结束/异常/客户端断开。"""
    if not isinstance(response, StreamingResponse):
        hold.release()
        return response
    iterator = response.body_iterator

    async def _guarded():
        try:
            async for chunk in iterator:
                yield chunk
        finally:
            hold.release()

    response.body_iterator = _guarded()
    return response

# --- OpenAI TTS 兼容垫片 -------------------------------------------------
# voice → 引擎侧参考音频映射. ref_audio_path 是引擎容器内路径 (客户端不可见),
# 用 CHII_TTS_REF_AUDIO 覆盖; prompt_text 由门面从宿主机文件读出后内联传给引擎,
# 用 CHII_TTS_REF_TEXT_FILE 覆盖 (默认取本仓库 models/ref_text.txt)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ref_text_file = _getenv("REF_TEXT_FILE", os.path.join(REPO_ROOT, "models", "ref_text.txt"))
try:
    with open(_ref_text_file, encoding="utf-8") as _f:
        _ref_text = _f.read().strip()
except OSError as _exc:
    _ref_text = ""
    # 静默置空会让 OpenAI 垫片退化为零样本提示（合成质量降质且 /healthz/deep 仍 200）：
    # 启动即 WARN，深度健康检查响应体带 ref_ready 字段暴露该状态（见 healthz_deep）
    print(f"[warn] 参考文本文件不可读: {_ref_text_file} ({_exc.__class__.__name__})，"
          f"OpenAI 垫片将退化为零样本提示，合成质量可能降质；"
          f"可用 CHII_TTS_REF_TEXT_FILE 指定正确路径", file=sys.stderr)
if not _ref_text:
    if os.path.exists(_ref_text_file):
        print(f"[warn] 参考文本文件为空: {_ref_text_file}，"
              f"OpenAI 垫片将退化为零样本提示，合成质量可能降质", file=sys.stderr)
    REF_READY = False
else:
    REF_READY = True

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

# --- wav 流式按句串行化 ---------------------------------------------------
# 引擎流式 (streaming_mode=2) 模式下, 多句文本触发内部并行批推理的
# "Sizes of tensors must match" 错误: 返回 200 但音频被截断, 反复触发还会拖垮
# 引擎 (t2s_model 变 None, 之后所有请求 200 空流). 门面侧规避: 按句切分后
# 逐句串行调引擎, 把多条 PCM 流合并成一条 WAV 流返回.
_SENTENCE_RE = re.compile(r"[^。！？!?．；;\r\n]+(?:[。！？!?．；;\r\n]+|$)")
# 句中至少含一个文字/数字才算可合成 (纯标点/引号句引擎切分后为空, 会得到 200 空流)
_WORD_RE = re.compile(r"\w")
# 剥后续句子 WAV 头时在前 N 字节内找 data 标记 (找不到回退 44 字节标准头长)
_WAV_HEADER_SCAN = 128


def _split_sentences(text: str) -> list[str]:
    """按日/中标点与换行切句 (标点保留在句尾), 过滤空白句与无文字句."""
    return [s for s in (t.strip() for t in _SENTENCE_RE.findall(text))
            if s and _WORD_RE.search(s)]


class _EngineFailure(Exception):
    """引擎在产出任何音频字节前失败. status 为回给客户端的状态码;
    异常文本仅进服务端日志 (引擎报错含容器内细节, 不外泄)."""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status


class _EngineEmptyStream(_EngineFailure):
    """引擎返回 200 空流: 多为无可合成内容的退化输入 (引擎切分后为空)."""


async def _engine_stream_first(payload: dict):
    """打开引擎流式请求并预读首个数据块; 上游即败 (连接失败/非 200/空流) 抛
    _EngineFailure, 调用方据此返回错误码而不是 200 空流."""
    try:
        req = _client.build_request("POST", "/tts", json=payload)
        upstream = await _client.send(req, stream=True)
    except httpx.HTTPError as exc:
        raise _EngineFailure(502, f"[tts] engine unavailable: {exc.__class__.__name__}: {exc}")
    if upstream.status_code != 200:
        body = await upstream.aread()
        await upstream.aclose()
        raise _EngineFailure(502, f"[tts] engine {upstream.status_code}: {body[:200]!r}")
    it = upstream.aiter_raw()
    try:
        first = await it.__anext__()
    except StopAsyncIteration:
        await upstream.aclose()
        raise _EngineEmptyStream(502, "[tts] engine returned empty stream")
    except httpx.HTTPError as exc:
        await upstream.aclose()
        raise _EngineFailure(
            502, f"[tts] engine stream failed before first byte: {exc.__class__.__name__}: {exc}")
    return upstream, it, first


def _strip_wav_header(head: bytes) -> bytes:
    """剥掉后续句子流首块的 WAV 头: 在前 _WAV_HEADER_SCAN 字节内找 data 标记
    (头长=偏移+8), 找不到回退 44 字节."""
    idx = head[:_WAV_HEADER_SCAN].find(b"data")
    cut = idx + 8 if idx >= 0 else 44
    return head[cut:] if cut < len(head) else b""


def _parse_wav_fmt(head: bytes):
    """从 WAV 头前 _WAV_HEADER_SCAN 字节解析 (rate, channels); 找不到返回 None."""
    idx = head[:_WAV_HEADER_SCAN].find(b"fmt ")
    if idx < 0 or len(head) < idx + 16:
        return None
    try:
        channels, rate = struct.unpack_from("<HI", head, idx + 10)
        return rate, channels
    except struct.error:
        return None


def _stream_wav_header(rate: int, channels: int, sampwidth: int = 2) -> bytes:
    """流式 WAV 头: RIFF/data 长度置 0xFFFFFFFF（ indefinit 约定），
    供流式客户端取 fmt；严格按头解析整包长度的客户端不应收到本模式."""
    byte_rate = rate * channels * sampwidth
    block_align = channels * sampwidth
    return (b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE"
            + b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels, rate,
                                    byte_rate, block_align, sampwidth * 8)
            + b"data" + struct.pack("<I", 0xFFFFFFFF))


def _gain_pcm(pcm: bytes, target_peak: int = 28000, max_gain: float = 8.0) -> bytes:
    """逐句响度归一: 峰值提至 ~85% 满幅, 增益上限防近静音段噪声被放大。
    引擎参考音频近静音时整句输出峰值常仅 ~8%（真机 54% 音量不可闻，实证），
    垫片侧整包归一覆盖不了流式响应，故门面流式模式逐句归一。
    纯 Python (array) 实现: 门面宿主 Python >= 3.13 无 audioop/numpy。
    单句 1-4s (3-13 万采样点), 峰值扫描 + 增益约 50-150ms, 相对秒级合成可忽略."""
    pcm = pcm[: len(pcm) // 2 * 2]
    if len(pcm) < 2:
        return pcm
    a = array.array("h")
    a.frombytes(pcm)
    peak = max(map(abs, a))
    if peak == 0 or peak >= target_peak:
        return pcm
    g = min(max_gain, target_peak / peak)
    out = array.array("h", (max(-32768, min(32767, int(round(s * g)))) for s in a))
    return out.tobytes()


async def _merged_wav_stream_normalized(first_state, sentences: list, payload: dict):
    """流式归一模式 (?stream=1): 与 _merged_wav_stream 相同的逐句串行合并,
    但每句整句缓冲后按峰值归一再下发, 首句剥掉引擎占位头、换发自带
    fmt 的流式头 (RIFF/data = 0xFFFFFFFF)。句间仍保持流式——前句播放时
    后句在合成, 长回复首声延迟从整包合成完成降到首句合成完成 (~1-2s).
    归一以句为单位: 句间响度一致性优于整包单次归一, 代价是句内逐块延迟
    让位于整句缓冲 (单句 1-4s, 可接受)."""
    header_sent = False
    for i, sentence in enumerate(sentences):
        try:
            if i == 0:
                upstream, it, head = first_state
            else:
                upstream, it, head = await _engine_stream_first(dict(payload, text=sentence))
        except _EngineEmptyStream:
            continue
        try:
            while len(head) < _WAV_HEADER_SCAN and b"data" not in head:
                try:
                    head += await it.__anext__()
                except StopAsyncIteration:
                    break
            if i == 0:
                fmt = _parse_wav_fmt(head) or (32000, 1)
                yield _stream_wav_header(*fmt)
                header_sent = True
            buf = bytearray(_strip_wav_header(head))
            async for chunk in it:
                buf += chunk
            gained = _gain_pcm(bytes(buf))
            if gained:
                yield gained
        finally:
            await upstream.aclose()
    if not header_sent:
        return


async def _merged_wav_stream(first_state, sentences: list[str], payload: dict):
    """首句流 (含 WAV 头的首块已预读) 原样下发; 后续句子逐句串行合成,
    剥掉各自 WAV 头后追加 raw PCM. 某句被引擎切成空 (退化句漏网) 时跳过该句;
    流中 (客户端已收到数据) 的其余上游失败只能让异常向上抛、中断连接."""
    upstream, it, first = first_state
    try:
        yield first
        async for chunk in it:
            yield chunk
    finally:
        await upstream.aclose()
    for sentence in sentences[1:]:
        try:
            upstream, it, head = await _engine_stream_first(dict(payload, text=sentence))
        except _EngineEmptyStream:
            continue  # 该句无可合成内容, 跳过不影响其余句子的音频
        try:
            # 首块可能不足一个完整头, 累积到能定位 data 标记或足够判定回退
            while len(head) < _WAV_HEADER_SCAN and b"data" not in head:
                try:
                    head += await it.__anext__()
                except StopAsyncIteration:
                    break
            pcm = _strip_wav_header(head)
            if pcm:
                yield pcm
            async for chunk in it:
                yield chunk
        finally:
            await upstream.aclose()


# --- 深度健康检查 -----------------------------------------------------------
# /v1/models 是进程级轻量存活; /healthz/deep 用极短文本向引擎发一次真实合成,
# 能发现"返回 200 空流"这类变砖状态. 结果缓存 DEEP_PROBE_TTL 秒,
# 避免监控高频刷接口烧 GPU. 鉴权语义与其余端点一致 (middleware 全局校验 key).
DEEP_PROBE_TTL = _getenv_float("DEEP_PROBE_TTL", 30.0)
DEEP_PROBE_TIMEOUT = _getenv_float("DEEP_PROBE_TIMEOUT", 20.0)
_deep_probe: dict = {"at": 0.0, "status": "degraded", "engine": "not probed yet"}


async def _probe_engine() -> dict:
    """向引擎发一次真实合成探测, 返回 {"status", "engine"} (engine 为状态简述)."""
    v = VOICES["chii"]
    payload = {
        "text": "テスト",
        "text_lang": TEXT_LANG,
        "ref_audio_path": v["ref_audio_path"],
        "prompt_text": v["prompt_text"],
        "prompt_lang": v["prompt_lang"],
        "media_type": "wav",
        "streaming_mode": False,
    }
    try:
        resp = await _client.post("/tts", json=payload, timeout=DEEP_PROBE_TIMEOUT)
    except httpx.HTTPError as exc:
        return {"status": "degraded", "engine": f"unreachable: {exc.__class__.__name__}"}
    if resp.status_code != 200:
        return {"status": "degraded", "engine": f"http_{resp.status_code}"}
    if not resp.content:
        return {"status": "degraded", "engine": "empty response"}
    return {"status": "ok", "engine": "ok"}


@APP.get("/healthz")
async def healthz():
    # 免鉴权浅探活（中间件对 /healthz 放行）；不回引擎字段，免鉴权端点不暴露指纹
    return {"status": "ok", "model": TTS_MODEL}


@APP.get("/healthz/deep")
async def healthz_deep():
    if time.time() - _deep_probe["at"] >= DEEP_PROBE_TTL:
        result = await _probe_engine()
        _deep_probe.update(result, at=time.time())
        if result["status"] != "ok":
            print(f"[healthz] deep probe degraded: {result['engine']}", file=sys.stderr)
    # ref_ready 不进状态码：参考文本缺失/为空是降质而非变砖，引擎健康仍回 200，
    # 避免惊动监控；运维应盯 ref_ready=false 的告警（启动时门面已打 WARN 日志）
    return JSONResponse(
        status_code=200 if _deep_probe["status"] == "ok" else 503,
        content={"status": _deep_probe["status"], "engine": _deep_probe["engine"],
                 "ref_ready": REF_READY})


def _openai_error(code: int, message: str, err_code: str | None = None) -> JSONResponse:
    return JSONResponse(status_code=code, content={
        "error": {"message": message, "type": "invalid_request_error", "param": None,
                  "code": err_code}})


def _payload_too_large() -> JSONResponse:
    """请求体超限的 413 错误体：与门面其余 4xx 一致脱敏，另带稳定 code 便于客户端分支。"""
    return JSONResponse(status_code=413, content={
        "message": f"请求体过大（上限 {MAX_BODY_BYTES // (1024 * 1024)}MB）",
        "code": "payload_too_large"})


async def _read_body_capped(request: Request) -> tuple[bytes | None, JSONResponse | None]:
    """读取请求体并施加 MAX_BODY_BYTES 上限。Content-Length 超限直接 413（不读体）；
    无/非法 Content-Length（chunked 可不带长度）按 request.stream() 累计兜底。
    返回 (body, error)，error 非 None 时 body 为 None。"""
    length = request.headers.get("content-length")
    if length:
        try:
            if int(length) > MAX_BODY_BYTES:
                return None, _payload_too_large()
        except ValueError:
            pass  # 非法 Content-Length 交给累计兜底
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        chunks.append(chunk)
        total += len(chunk)
        if total > MAX_BODY_BYTES:
            return None, _payload_too_large()
    return b"".join(chunks), None


async def _proxy_to_engine(method: str, **kwargs) -> StreamingResponse | JSONResponse:
    """流式转发到引擎 /tts: 状态码与 Content-Type 原样回传, 客户端断开时关闭上游连接。
    引擎错误体不原样回传（含容器内路径/栈细节，不外泄）：5xx 回写通用错误（502），
    4xx 保留引擎状态码语义但回写通用拒绝体；原文进门面日志排障。
    上游不可达回 502 通用文案（对齐 ASR 门面，不回 503、不含服务名指纹）。"""
    try:
        req = _client.build_request(method, "/tts", **kwargs)
        upstream = await _client.send(req, stream=True)
    except httpx.HTTPError as exc:
        print(f"[tts] engine unavailable: {exc.__class__.__name__}: {exc}", file=sys.stderr)
        return JSONResponse(status_code=502,
                            content={"error": f"upstream error: {exc.__class__.__name__}"})

    if upstream.status_code >= 400:
        body = await upstream.aread()
        print(f"[tts] engine {upstream.status_code}: {body[:200]!r}", file=sys.stderr)
        await upstream.aclose()
        if upstream.status_code >= 500:
            return JSONResponse(status_code=502, content=engine_error_body("tts", body_key="message"))
        return JSONResponse(status_code=upstream.status_code,
                            content={"message": "speech request rejected"})

    headers = filter_response_headers(upstream.headers)

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
    body_bytes, too_large = await _read_body_capped(request)
    if too_large is not None:
        return _openai_error(413, f"请求体过大（上限 {MAX_BODY_BYTES // (1024 * 1024)}MB）",
                             err_code="payload_too_large")
    try:
        body = json.loads(body_bytes)
    except ValueError:
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
    # aac/opus 逐块编码会拼出损坏帧, 保持非流式一次性返回 (此时批推理生效)
    payload = {
        "text": text,
        "text_lang": TEXT_LANG,
        "ref_audio_path": v["ref_audio_path"],
        "prompt_text": v["prompt_text"],
        "prompt_lang": v["prompt_lang"],
        "media_type": FORMAT_MAP[fmt],
        "speed_factor": float(speed),
        "batch_size": BATCH_SIZE,
        "streaming_mode": 2 if FORMAT_MAP[fmt] == "wav" else False,
    }
    # 全局在途并发上限：超上限排队等待空位，QUEUE_TIMEOUT 秒内仍拿不到才 429；
    # 持有期覆盖真实 GPU 占用（流式响应持有到推流结束，见 _hold_through_response）。
    # 注意不能用 try/finally 释放：端点 return 响应对象时流式 body 尚未开始迭代，
    # finally 会在推流前就把信号量放掉；异常路径在 except 里释放，正常路径由
    # _hold_through_response 按响应类型决定释放时机
    hold, busy = await _acquire_inflight()
    if busy is not None:
        return busy
    try:
        if FORMAT_MAP[fmt] != "wav":
            return _hold_through_response(await _proxy_to_engine("POST", json=payload), hold)
        # wav 流式: 多句按句串行化合并 (规避引擎多片段并行批推理 bug, 见上方说明);
        # 切不出多句时整句直发, 请求文本与现状一致
        sentences = _split_sentences(text)
        if len(sentences) <= 1:
            sentences = [text]
        # 引擎还会按自身规则把单句再切成片段 (内部切分含逗号/顿号等, 门面切句管不到),
        # batch_size>1 即触发同样的批推理 bug, 故流式请求的 batch_size 一律钉 1;
        # 对单片段文本无影响 (本就只有 1 个片段进批, 推理结果一致)
        payload = dict(payload, batch_size=1)
        try:
            # 合成开始前预读上游首块: 上游即败返回错误码, 不发 200 空流
            first_state = await _engine_stream_first(dict(payload, text=sentences[0]))
        except _EngineFailure as exc:
            print(str(exc), file=sys.stderr)
            return _hold_through_response(
                JSONResponse(status_code=exc.status, content={"message": "tts engine error"}), hold)
        headers = {}
        if ct := first_state[0].headers.get("content-type"):
            headers["content-type"] = ct
        merged = (_merged_wav_stream_normalized(first_state, sentences, payload)
                  if request.query_params.get("stream") == "1"
                  else _merged_wav_stream(first_state, sentences, payload))
        return _hold_through_response(StreamingResponse(
            merged, status_code=200, headers=headers), hold)
    except Exception:
        hold.release()
        raise


# --- /tts 透传参数白名单（安全加固：此前引擎全部参数面裸露——
# ref_audio_path 可指向容器内任意路径（文件存在性 oracle / /dev/urandom 挂死引擎），
# batch_size/super_sampling 等采样参数可放大 GPU 消耗，还可克隆容器内任意音频） ---
REF_AUDIO_PREFIX = _getenv("REF_AUDIO_PREFIX", "/data/")


def _ref_audio_path_allowed(path) -> bool:
    """ref_audio 白名单校验（值为引擎容器内 POSIX 路径，门面按字面判）：
    归一化后须严格位于 REF_AUDIO_PREFIX 目录内。仅 startswith 判前缀会被
    ``/data/../../etc/passwd`` 之类的 .. 路径绕过；normpath 先折叠 .. 与
    冗余分隔符再判前缀（\"/data/../x\" → \"/x\"，拒绝；\"/data/./a.wav\" → 放行）。
    路径在门面宿主机本地可见时（/data 卷同时挂载的场景）再 defense-in-depth：
    解析符号链接复核仍须落在前缀内且须为常规文件；容器路径在宿主机不可见时
    该复核自动跳过，文件存在性最终由引擎侧 4xx 兜底。"""
    prefix = posixpath.normpath(str(REF_AUDIO_PREFIX).replace("\\", "/"))
    norm = posixpath.normpath(str(path).replace("\\", "/"))
    if not norm.startswith(prefix + "/"):
        return False
    if os.path.exists(norm):  # 本地可见才复核（引擎容器路径通常不可见）
        real_prefix = os.path.realpath(prefix)
        real = os.path.realpath(norm)
        if not (real == real_prefix or real.startswith(real_prefix + os.sep)):
            return False
        if not os.path.isfile(norm):
            return False
    return True


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
    "batch_size": (1, 20),
    "batch_threshold": (0.0, 1.0),  # 引擎侧是 0~1 切分阈值（默认 0.75）
    "sample_steps": (1, 64),
    "top_k": (1, 100),
    "top_p": (0.0, 1.0),
    "temperature": (0.0, 2.0),
    "repetition_penalty": (0.0, 2.0),
    "speed_factor": (0.25, 4.0),
    "fragment_interval": (0.01, 1.0),
}
# 整型参数：钳制统一走 float()，钳完须还原 int——GET 展开成 query 后
# 引擎 pydantic 对 "5.0" 这类浮点字符串解析 int 会 422
TTS_INT_PARAMS = {"batch_size", "sample_steps", "top_k"}


def _is_streaming(value) -> bool:
    """streaming_mode 透传值判真：GET 走 query 全是字符串，"false"/"0" 也是非空串。"""
    if isinstance(value, str):
        return value.strip().lower() not in ("", "0", "false", "none", "off")
    return bool(value)


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
            if key in TTS_INT_PARAMS:
                value = int(value)  # 还原整型，否则 GET 透传 "5.0" 被引擎拒成 422
        payload[key] = value
    for path in ref_audio_paths + aux_ref_audio_paths:
        if not _ref_audio_path_allowed(path):
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
    # 客户端未显式传 batch_size 时注入门面默认值（引擎默认 1 不启用批推理）
    payload.setdefault("batch_size", BATCH_SIZE)
    # 引擎流式模式（streaming_mode=2 等）对多片段并行批推理有 bug：batch_size>1 即
    # "Sizes of tensors must match"——返回 200 但音频截断，反复触发还会拖垮引擎
    # （之后所有请求 200 空流，见模块 docstring）；引擎内部还会把单句再切成片段，
    # 客户端侧切句管不到。故流式请求的 batch_size 一律钉 1（覆盖显式传值并记日志），
    # 与 OpenAI 垫片 wav 流式路径同款规避；对单片段文本无影响
    if _is_streaming(payload.get("streaming_mode")) and payload["batch_size"] != 1:
        print(f"[tts] streaming_mode 开启，batch_size 由 {payload['batch_size']!r} 强制钉为 1",
              file=sys.stderr)
        payload["batch_size"] = 1
    return payload, None


@APP.api_route("/tts", methods=["GET", "POST"])
async def tts_passthrough(request: Request):
    # 全局在途并发上限（排队 QUEUE_TIMEOUT 秒，超时 429；流式响应持有到推流结束）。
    # 不能用 try/finally 释放（端点 return 时流式 body 尚未迭代，finally 会提前放锁），
    # 异常路径在 except 里释放，正常路径由 _hold_through_response 决定释放时机
    hold, busy = await _acquire_inflight()
    if busy is not None:
        return busy
    try:
        if request.method == "GET":
            # 剔出 api_key 再清洗转发：鉴权只认 Bearer 头，这里仅防止该参数透传给引擎
            # (multi_items 保留 aux_ref_audio_paths 等重复参数)
            items = [(k, v) for k, v in request.query_params.multi_items() if k != "api_key"]
            payload, error = _sanitize_tts_params(items)
            if error:
                return _hold_through_response(
                    JSONResponse(status_code=400, content={"message": error}), hold)
            # GET 透传展开为 query 参数（aux_ref_audio_paths 还原为重复键）
            params = [(k, v) for k, v in payload.items() if k != "aux_ref_audio_paths"]
            params += [("aux_ref_audio_paths", v) for v in payload.get("aux_ref_audio_paths", [])]
            return _hold_through_response(await _proxy_to_engine("GET", params=params), hold)
        body_bytes, too_large = await _read_body_capped(request)
        if too_large is not None:
            return _hold_through_response(too_large, hold)
        try:
            body = json.loads(body_bytes)
        except ValueError:
            return _hold_through_response(
                JSONResponse(status_code=400, content={"message": "请求体不是合法 JSON"}), hold)
        if not isinstance(body, dict):
            return _hold_through_response(
                JSONResponse(status_code=400, content={"message": "请求体须为 JSON 对象"}), hold)
        items = []
        for key, value in body.items():
            if key == "aux_ref_audio_paths" and isinstance(value, list):
                items += [(key, item) for item in value]
            else:
                items.append((key, value))
        payload, error = _sanitize_tts_params(items)
        if error:
            return _hold_through_response(
                JSONResponse(status_code=400, content={"message": error}), hold)
        return _hold_through_response(await _proxy_to_engine("POST", json=payload), hold)
    except Exception:
        hold.release()
        raise

# ------------------------------------------------------------------------


def _extract_key(request) -> str:
    # 只认 Authorization 头；不再接受 ?api_key=（query string 会进 uvicorn/反代访问日志）
    return extract_bearer_token(request.headers)


@APP.middleware("http")
async def auth_and_rate_limit(request, call_next):
    # 免鉴权浅探活：/healthz 直接放行（/healthz/deep 仍需 key——真实合成探测有 GPU 成本）
    if request.url.path == "/healthz":
        return await call_next(request)

    limited_path = request.url.path in ("/tts", "/v1/audio/speech")
    # 限流前置：鉴权失败也计桶（在线爆破有成本）；空桶即删键防海量 IP 驻留；
    # 单调时钟（系统时钟回拨不会把窗口拉长）
    if RATE_LIMIT > 0 and limited_path:
        ip = _client_ip(request.client.host if request.client else "unknown", request.headers)
        _rate_limiter.limit = RATE_LIMIT
        if not _rate_limiter.allow(ip):
            return JSONResponse(status_code=429, content={"message": "rate limit exceeded"})

    if not _key_ok(_extract_key(request)):
        return JSONResponse(status_code=401, content={"message": "invalid or missing api key"})

    if limited_path:
        # 审计日志（journald 自带时间戳）：key 哈希 + 客户端 IP + 请求体长度，
        # 不记明文内容；声音克隆滥用可事后追溯。
        # 在途并发上限不在此层执行（call_next 拿到响应头即返回，流式合成会落在
        # 信号量外），已下沉到端点层，见 _acquire_inflight / _hold_through_response
        print(f"[audit] {request.url.path} key={_auth.key_digest(_extract_key(request))}"
              f" ip={_client_ip(request.client.host if request.client else 'unknown', request.headers)}"
              f" len={request.headers.get('content-length', '?')}", file=sys.stderr)
    return await call_next(request)


BIND = _getenv("BIND", "127.0.0.1")  # 默认只绑本机（生产由 Caddy 反代）；显式改绑公网须配 TLS
_LOOPBACK = {"127.0.0.1", "localhost", "::1"}
if BIND not in _LOOPBACK and not SSL_CERTFILE:
    sys.exit(f"[错误] BIND={BIND} 为非回环地址但未配置 TLS（CHII_TTS_SSL_CERTFILE/KEYFILE），"
             "明文 HTTP 会泄露 API Key，拒绝启动；请配置 TLS 证书，或改绑 127.0.0.1 由 Caddy 反代终结 TLS")

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
