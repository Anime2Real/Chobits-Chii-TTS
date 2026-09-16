"""tools/server.py 纯逻辑单元测试（不触网络、不需要引擎）。"""
import asyncio
import json
import os
import sys
from types import SimpleNamespace

import pytest

import server


def _req(headers=None):
    """最小 Request 替身：_extract_key / _client_ip 只用 .headers.get。"""
    return SimpleNamespace(headers=headers or {})


# --- _getenv_int / _getenv_float -------------------------------------------

def test_getenv_int_valid(monkeypatch):
    monkeypatch.setenv("CHII_TTS_FOO", "42")
    assert server._getenv_int("FOO", 7) == 42


def test_getenv_int_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("CHII_TTS_FOO", "abc")
    assert server._getenv_int("FOO", 7) == 7


def test_getenv_int_empty_falls_back(monkeypatch):
    monkeypatch.delenv("CHII_TTS_FOO", raising=False)
    assert server._getenv_int("FOO", 7) == 7


def test_getenv_float_valid(monkeypatch):
    monkeypatch.setenv("CHII_TTS_BAR", "2.5")
    assert server._getenv_float("BAR", 1.0) == 2.5


def test_getenv_float_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("CHII_TTS_BAR", "1.5x")
    assert server._getenv_float("BAR", 1.0) == 1.0


# --- _client_ip -------------------------------------------------------------

def test_client_ip_loopback_trusts_xff_last_hop():
    # 反代把真实 IP 追加在尾部；第一跳可被客户端伪造，必须取末跳
    headers = {"x-forwarded-for": "1.1.1.1, 2.2.2.2, 203.0.113.9"}
    assert server._client_ip("127.0.0.1", headers) == "203.0.113.9"


def test_client_ip_loopback_without_xff():
    assert server._client_ip("127.0.0.1", {}) == "127.0.0.1"


def test_client_ip_direct_ignores_xff():
    headers = {"x-forwarded-for": "203.0.113.9"}
    assert server._client_ip("198.51.100.7", headers) == "198.51.100.7"


# --- _extract_key -----------------------------------------------------------

def test_extract_key_bearer():
    assert server._extract_key(_req({"authorization": "Bearer abc123"})) == "abc123"


def test_extract_key_bearer_case_insensitive():
    assert server._extract_key(_req({"authorization": "bEaReR abc123"})) == "abc123"


def test_extract_key_missing():
    assert server._extract_key(_req()) == ""


def test_extract_key_non_bearer_scheme_ignored():
    assert server._extract_key(_req({"authorization": "Basic abc123"})) == ""


def test_extract_key_never_reads_query():
    # query 里的 api_key 不应再生效：_extract_key 只认 Authorization 头
    req = SimpleNamespace(headers={}, query_params={"api_key": "abc123"})
    assert server._extract_key(req) == ""


# --- _is_streaming ----------------------------------------------------------

@pytest.mark.parametrize("value", [True, 1, 2, "2", "true", "TRUE", " true ", "yes"])
def test_is_streaming_truthy(value):
    assert server._is_streaming(value) is True


@pytest.mark.parametrize("value", [False, 0, None, "", "0", "false", "False", "none", "off", " "])
def test_is_streaming_falsy(value):
    assert server._is_streaming(value) is False


# --- _sanitize_tts_params ---------------------------------------------------

def test_sanitize_whitelist_drops_unknown_keys():
    payload, err = server._sanitize_tts_params([("text", "hi"), ("evil_param", "x")])
    assert err is None
    assert "evil_param" not in payload
    assert payload["text"] == "hi"


def test_sanitize_numeric_clamp():
    payload, err = server._sanitize_tts_params([("batch_size", "100"), ("top_p", "1.7")])
    assert err is None
    assert payload["batch_size"] == 20
    assert payload["top_p"] == 1.0


def test_sanitize_numeric_clamp_low():
    payload, err = server._sanitize_tts_params([("batch_size", "0"), ("speed_factor", "0.01")])
    assert err is None
    assert payload["batch_size"] == 1
    assert payload["speed_factor"] == 0.25


def test_sanitize_int_params_restored_to_int():
    # GET query 全是字符串；钳完必须还原 int，否则引擎 pydantic 对 "5.0" 422
    payload, err = server._sanitize_tts_params(
        [("batch_size", "7"), ("top_k", "30"), ("sample_steps", "16"), ("top_p", "0.9")])
    assert err is None
    assert payload["batch_size"] == 7 and isinstance(payload["batch_size"], int)
    assert payload["top_k"] == 30 and isinstance(payload["top_k"], int)
    assert payload["sample_steps"] == 16 and isinstance(payload["sample_steps"], int)
    assert payload["top_p"] == 0.9 and isinstance(payload["top_p"], float)


def test_sanitize_invalid_numeric_dropped():
    payload, err = server._sanitize_tts_params([("batch_size", "abc")])
    assert err is None
    # 非法数值丢弃后按未显式传处理：注入门面默认值
    assert payload["batch_size"] == 5


def test_sanitize_default_batch_size_injected():
    payload, err = server._sanitize_tts_params([("text", "hi")])
    assert err is None
    assert payload["batch_size"] == 5


def test_sanitize_explicit_batch_size_kept_when_not_streaming():
    payload, err = server._sanitize_tts_params([("batch_size", "9")])
    assert err is None
    assert payload["batch_size"] == 9


def test_sanitize_streaming_pins_batch_size_1():
    # 流式模式 batch_size>1 触发引擎并行批推理 bug：显式传值也被强制钉 1
    payload, err = server._sanitize_tts_params(
        [("text", "hi"), ("streaming_mode", "2"), ("batch_size", "10")])
    assert err is None
    assert payload["batch_size"] == 1


def test_sanitize_streaming_pins_default_batch_size_1():
    payload, err = server._sanitize_tts_params([("text", "hi"), ("streaming_mode", True)])
    assert err is None
    assert payload["batch_size"] == 1


@pytest.mark.parametrize("value", ["false", "0", "off", "none", ""])
def test_sanitize_streaming_string_falsy_not_pinned(value):
    # GET query 里 "false"/"0" 是非空字符串，必须按假处理，不能误钉 batch_size
    payload, err = server._sanitize_tts_params(
        [("text", "hi"), ("streaming_mode", value), ("batch_size", "8")])
    assert err is None
    assert payload["batch_size"] == 8


def test_sanitize_ref_audio_prefix_enforced():
    payload, err = server._sanitize_tts_params([("ref_audio_path", "/etc/passwd")])
    assert payload is None
    assert err is not None


def test_sanitize_ref_audio_prefix_ok_last_wins():
    payload, err = server._sanitize_tts_params(
        [("ref_audio_path", "/data/a.wav"), ("ref_audio_path", "/data/b.wav")])
    assert err is None
    assert payload["ref_audio_path"] == "/data/b.wav"


def test_sanitize_aux_ref_audio_paths_collected():
    payload, err = server._sanitize_tts_params(
        [("aux_ref_audio_paths", "/data/x.wav"), ("aux_ref_audio_paths", "/data/y.wav")])
    assert err is None
    assert payload["aux_ref_audio_paths"] == ["/data/x.wav", "/data/y.wav"]


def test_sanitize_text_too_long():
    payload, err = server._sanitize_tts_params([("text", "a" * 2001)])
    assert payload is None
    assert "超长" in err


def test_sanitize_media_type_invalid():
    payload, err = server._sanitize_tts_params([("media_type", "exe")])
    assert payload is None
    assert err is not None


# --- _InflightHold / _acquire_inflight --------------------------------------

def test_inflight_hold_release_idempotent():
    async def run():
        sem = asyncio.Semaphore(1)
        await sem.acquire()
        hold = server._InflightHold(sem)
        hold.release()
        hold.release()  # 第二次 release 必须是 no-op
        # 只释放了一次：能再拿一个槽位，但拿不到第二个
        await asyncio.wait_for(sem.acquire(), 0.1)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(sem.acquire(), 0.05)
    asyncio.run(run())


def test_inflight_hold_release_without_semaphore_is_noop():
    hold = server._InflightHold(None)
    hold.release()
    hold.release()


def test_acquire_inflight_success():
    async def run():
        server._inflight_sem = asyncio.Semaphore(2)
        hold, busy = await server._acquire_inflight()
        assert busy is None
        assert hold is not None
        hold.release()
    try:
        asyncio.run(run())
    finally:
        server._inflight_sem = None


def test_acquire_inflight_queue_timeout_429(monkeypatch):
    monkeypatch.setattr(server, "QUEUE_TIMEOUT", 0.05)

    async def run():
        server._inflight_sem = asyncio.Semaphore(1)
        await server._inflight_sem.acquire()  # 占满唯一槽位
        hold, busy = await server._acquire_inflight()
        assert hold is None
        assert busy is not None
        assert busy.status_code == 429
        assert json.loads(bytes(busy.body))["message"] == "server busy, queue wait timeout"
    try:
        asyncio.run(run())
    finally:
        server._inflight_sem = None
