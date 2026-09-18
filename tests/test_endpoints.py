"""tools/server.py 端点级测试：TestClient + 内存假引擎（不触真实 :9882）。"""
import httpx
import pytest
from fastapi.testclient import TestClient

import server
from conftest import API_KEY

AUTH = {"Authorization": "Bearer " + API_KEY}


class FakeUpstream:
    """httpx 流式响应替身：_proxy_to_engine / _engine_stream_first 用到的最小面。"""

    def __init__(self, status_code=200, chunks=(b"RIFF-fake-wav-data",),
                 content_type="audio/wav"):
        self.status_code = status_code
        self._chunks = list(chunks)
        self.headers = {"content-type": content_type}
        self.closed = False

    @property
    def content(self):
        return b"".join(self._chunks)

    async def aiter_raw(self):
        for chunk in self._chunks:
            yield chunk

    async def aread(self):
        return b"".join(self._chunks)

    async def aclose(self):
        self.closed = True


class FakeEngineClient:
    """httpx.AsyncClient 替身：记录 build_request 参数供断言，send 返回假上游或抛异常。"""

    def __init__(self, upstream=None, exc=None):
        self.requests = []
        self.upstream = upstream or FakeUpstream()
        self.exc = exc

    def build_request(self, method, url, **kwargs):
        self.requests.append({"method": method, "url": url, "kwargs": kwargs})
        return self.requests[-1]

    async def send(self, req, stream=False):
        if self.exc is not None:
            raise self.exc
        return self.upstream

    async def post(self, url, **kwargs):
        self.requests.append({"method": "POST", "url": url, "kwargs": kwargs})
        return self.upstream


@pytest.fixture
def client():
    with TestClient(server.APP) as c:
        yield c


@pytest.fixture
def fake_engine(monkeypatch):
    engine = FakeEngineClient()
    monkeypatch.setattr(server, "_client", engine)
    return engine


# --- 鉴权 --------------------------------------------------------------------

def test_speech_no_key_401(client):
    resp = client.post("/v1/audio/speech", json={"input": "hi", "voice": "chii"})
    assert resp.status_code == 401


def test_speech_wrong_key_401(client):
    resp = client.post("/v1/audio/speech", json={"input": "hi", "voice": "chii"},
                       headers={"Authorization": "Bearer wrong-key"})
    assert resp.status_code == 401


def test_speech_query_api_key_no_longer_accepted(client):
    resp = client.post("/v1/audio/speech?api_key=" + API_KEY,
                       json={"input": "hi", "voice": "chii"})
    assert resp.status_code == 401


def test_tts_get_query_api_key_no_longer_accepted(client):
    resp = client.get("/tts", params={"text": "hi", "api_key": API_KEY})
    assert resp.status_code == 401


def test_models_requires_key(client):
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers=AUTH).status_code == 200


def test_healthz_is_open(client, fake_engine):
    # /healthz 免鉴权浅探活（与 ASR 门面对齐）：无 key / 错 key 都 200，不暴露引擎指纹
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model": "chii-tts"}
    assert client.get("/healthz", headers={"Authorization": "Bearer wrong"}).status_code == 200
    # 不触引擎（浅探活无 GPU 成本）
    assert not fake_engine.requests


def test_healthz_deep_requires_key(client, fake_engine):
    # 深探测是真实合成（有 GPU 成本），与其余端点一样须带 key
    assert client.get("/healthz/deep").status_code == 401
    assert client.get("/healthz/deep", headers=AUTH).status_code in (200, 503)


# --- 限流 --------------------------------------------------------------------

def test_rate_limit_429(client, monkeypatch):
    # 鉴权失败也计桶：小限额下前两次 401，第三次 429
    monkeypatch.setattr(server, "RATE_LIMIT", 2)
    assert client.post("/v1/audio/speech", json={"input": "hi"}).status_code == 401
    assert client.post("/v1/audio/speech", json={"input": "hi"}).status_code == 401
    resp = client.post("/v1/audio/speech", json={"input": "hi"})
    assert resp.status_code == 429
    assert resp.json()["message"] == "rate limit exceeded"


def test_rate_limit_does_not_count_other_paths(client, monkeypatch):
    monkeypatch.setattr(server, "RATE_LIMIT", 1)
    client.get("/v1/models")
    client.get("/v1/models")
    # /v1/models 不限流，只计 /tts 与 /v1/audio/speech
    assert client.post("/v1/audio/speech", json={"input": "hi"}).status_code == 401
    assert client.post("/v1/audio/speech", json={"input": "hi"}).status_code == 429


# --- 引擎错误通用化（与 ASR 门面对齐：错误体脱敏，细节只进服务端日志） -------------

def test_tts_engine_4xx_body_sanitized(client, monkeypatch):
    # 引擎 4xx 响应体原样透传会泄漏容器内路径等细节：改写为通用拒绝体，保留状态码语义
    upstream = FakeUpstream(status_code=422,
                            chunks=[b"engine detail: /container/path leaked"],
                            content_type="application/json")
    monkeypatch.setattr(server, "_client", FakeEngineClient(upstream=upstream))
    resp = client.get("/tts", params={"text": "hi"}, headers=AUTH)
    assert resp.status_code == 422  # 保留引擎 4xx 状态码语义
    assert resp.json() == {"message": "speech request rejected"}
    assert "container" not in resp.text  # 引擎内部细节不外泄


def test_tts_engine_5xx_becomes_generic_502(client, monkeypatch):
    upstream = FakeUpstream(status_code=500,
                            chunks=[b"Internal: /app/engine/secret.py traceback"],
                            content_type="application/json")
    monkeypatch.setattr(server, "_client", FakeEngineClient(upstream=upstream))
    resp = client.get("/tts", params={"text": "hi"}, headers=AUTH)
    assert resp.status_code == 502
    assert resp.json() == {"message": "tts engine error"}
    assert "secret" not in resp.text


def test_tts_engine_unreachable_502_generic(client, monkeypatch):
    # 上游不可达对齐 ASR：502 + 通用文案（此前 503 "tts engine unavailable" 含服务名指纹）
    monkeypatch.setattr(server, "_client", FakeEngineClient(exc=httpx.ConnectError("refused")))
    resp = client.get("/tts", params={"text": "hi"}, headers=AUTH)
    assert resp.status_code == 502
    assert resp.json() == {"error": "upstream error: ConnectError"}
    assert "tts" not in resp.text.lower()


def test_speech_wav_upstream_unreachable_502(client, monkeypatch):
    # wav 流式预读首块即上游不可达：状态码同样对齐 502（此前 503）
    monkeypatch.setattr(server, "_client", FakeEngineClient(exc=httpx.ConnectError("refused")))
    resp = client.post("/v1/audio/speech",
                       json={"input": "テストです", "voice": "chii", "response_format": "wav"},
                       headers=AUTH)
    assert resp.status_code == 502
    assert resp.json() == {"message": "tts engine error"}


# --- 端点 → 引擎参数断言 ------------------------------------------------------

def test_tts_get_streaming_forwards_batch_size_1(client, fake_engine):
    resp = client.get("/tts", params={"text": "こんにちは", "streaming_mode": "2"},
                      headers=AUTH)
    assert resp.status_code == 200
    assert len(fake_engine.requests) == 1
    req = fake_engine.requests[0]
    assert req["method"] == "GET" and req["url"] == "/tts"
    params = dict(req["kwargs"]["params"])
    assert params["batch_size"] == 1
    assert params["streaming_mode"] == "2"


def test_tts_get_non_streaming_forwards_default_batch_size(client, fake_engine):
    resp = client.get("/tts", params={"text": "こんにちは"}, headers=AUTH)
    assert resp.status_code == 200
    params = dict(fake_engine.requests[0]["kwargs"]["params"])
    assert params["batch_size"] == 5


def test_tts_get_api_key_param_not_forwarded(client, fake_engine):
    # 即使带了合法 Bearer，query 里的 api_key 也不透传给引擎
    resp = client.get("/tts", params={"text": "hi", "api_key": "whatever"}, headers=AUTH)
    assert resp.status_code == 200
    params = dict(fake_engine.requests[0]["kwargs"]["params"])
    assert "api_key" not in params


def test_tts_post_streaming_forwards_batch_size_1(client, fake_engine):
    resp = client.post("/tts", json={"text": "hi", "streaming_mode": 2, "batch_size": 12},
                       headers=AUTH)
    assert resp.status_code == 200
    payload = fake_engine.requests[0]["kwargs"]["json"]
    assert payload["batch_size"] == 1


def test_speech_wav_streams_with_batch_size_1(client, fake_engine):
    resp = client.post("/v1/audio/speech",
                       json={"input": "テストです", "voice": "chii", "response_format": "wav"},
                       headers=AUTH)
    assert resp.status_code == 200
    payload = fake_engine.requests[0]["kwargs"]["json"]
    assert payload["batch_size"] == 1
    assert payload["streaming_mode"] == 2
    assert payload["media_type"] == "wav"


def test_speech_aac_uses_default_batch_size(client, fake_engine):
    resp = client.post("/v1/audio/speech",
                       json={"input": "テストです", "voice": "chii", "response_format": "aac"},
                       headers=AUTH)
    assert resp.status_code == 200
    payload = fake_engine.requests[0]["kwargs"]["json"]
    assert payload["batch_size"] == 5
    assert payload["streaming_mode"] is False


def test_speech_validation_errors(client, fake_engine):
    assert client.post("/v1/audio/speech", json={"input": "", "voice": "chii"},
                       headers=AUTH).status_code == 400
    assert client.post("/v1/audio/speech", json={"input": "hi", "voice": "nobody"},
                       headers=AUTH).status_code == 400
    assert client.post("/v1/audio/speech",
                       json={"input": "hi", "voice": "chii", "response_format": "mp3"},
                       headers=AUTH).status_code == 400
    assert client.post("/v1/audio/speech",
                       json={"input": "hi", "voice": "chii", "speed": 9.0},
                       headers=AUTH).status_code == 400
    assert fake_engine.requests == []  # 校验失败不触引擎


# --- 请求体大小上限（413, 超限不触引擎） ----------------------------------------

def test_speech_body_too_large_413(client, fake_engine, monkeypatch):
    monkeypatch.setattr(server, "MAX_BODY_BYTES", 64)
    resp = client.post("/v1/audio/speech",
                       json={"input": "a" * 100, "voice": "chii"}, headers=AUTH)
    assert resp.status_code == 413
    # OpenAI 错误协议 + 稳定 code
    assert resp.json()["error"]["code"] == "payload_too_large"
    assert fake_engine.requests == []  # 超限在转发前拦截


def test_tts_post_body_too_large_413(client, fake_engine, monkeypatch):
    monkeypatch.setattr(server, "MAX_BODY_BYTES", 64)
    resp = client.post("/tts", json={"text": "a" * 100}, headers=AUTH)
    assert resp.status_code == 413
    assert resp.json()["code"] == "payload_too_large"
    assert fake_engine.requests == []


def test_speech_chunked_body_too_large_413(client, fake_engine, monkeypatch):
    # 无 Content-Length (chunked) 路径: 按流累计兜底, 同样 413
    monkeypatch.setattr(server, "MAX_BODY_BYTES", 64)

    def gen():
        yield b'{"input": "' + b"a" * 100 + b'"}'

    resp = client.post("/v1/audio/speech", content=gen(),
                       headers={**AUTH, "Content-Type": "application/json"})
    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "payload_too_large"
    assert fake_engine.requests == []


def test_speech_body_under_limit_unaffected(client, fake_engine, monkeypatch):
    # 上限内的正常请求不受影响 (此处 cap 远大于请求体)
    monkeypatch.setattr(server, "MAX_BODY_BYTES", 1024)
    resp = client.post("/v1/audio/speech",
                       json={"input": "テストです", "voice": "chii",
                             "response_format": "aac"}, headers=AUTH)
    assert resp.status_code == 200
    assert len(fake_engine.requests) == 1


# --- ref_audio 路径穿越（端点级） ------------------------------------------------

def test_tts_get_ref_audio_traversal_400(client, fake_engine):
    # 回归: /data/../../etc/passwd 曾可通过 startswith("/data/") 前缀检查
    resp = client.get("/tts", params={"text": "hi",
                                      "ref_audio_path": "/data/../../etc/passwd"},
                      headers=AUTH)
    assert resp.status_code == 400
    assert "ref_audio" in resp.json()["message"]
    assert fake_engine.requests == []


def test_tts_get_ref_audio_aux_traversal_400(client, fake_engine):
    resp = client.get("/tts",
                      params=[("text", "hi"), ("aux_ref_audio_paths", "/data/../x.wav")],
                      headers=AUTH)
    assert resp.status_code == 400
    assert fake_engine.requests == []


# --- /healthz/deep 的 ref_ready 字段（降质不体现在状态码） -----------------------

def test_healthz_deep_has_ref_ready_field(client, fake_engine, monkeypatch):
    monkeypatch.setattr(server, "REF_READY", True)
    resp = client.get("/healthz/deep", headers=AUTH)
    assert resp.status_code == 200  # 假引擎合成成功
    assert resp.json()["ref_ready"] is True
    # ref_ready 不进状态码: 引擎健康但参考文本缺失仍是 200 (降质非变砖, 不惊动监控)
    monkeypatch.setattr(server, "REF_READY", False)
    resp = client.get("/healthz/deep", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["ref_ready"] is False
