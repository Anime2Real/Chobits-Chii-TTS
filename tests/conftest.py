"""pytest 公共夹具：导入 tools/server.py 前注入测试用环境变量。

server.py 在模块级读取 CHII_TTS_* 配置（未设置 CHII_TTS_API_KEY 会 sys.exit），
故环境变量必须在 import 之前就绪。引擎依赖不启动：端点测试用 monkeypatch
替换 server._client（httpx.AsyncClient）为内存假实现。
"""
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

os.environ.setdefault("CHII_TTS_API_KEY", "test-tts-key")

import server  # noqa: E402

API_KEY = os.environ["CHII_TTS_API_KEY"]


@pytest.fixture(autouse=True)
def reset_global_state():
    """每个测试隔离模块级全局状态（限流桶 / 在途信号量 / 深探测缓存）。

    信号量与深探测锁置 None 让端点在运行中的事件循环里重建（asyncio 同步原语
    首次使用时绑定所在 loop，跨 TestClient/asyncio.run 复用会报错）。"""
    server._rate_limiter.clear()
    server._inflight_sem = None
    server._deep_probe_lock = None
    server._deep_probe.update({"at": 0.0, "status": "degraded", "engine": "not probed yet"})
    yield
    server._rate_limiter.clear()
    server._inflight_sem = None
    server._deep_probe_lock = None
    server._deep_probe.update({"at": 0.0, "status": "degraded", "engine": "not probed yet"})


@pytest.fixture
def auth_headers():
    return {"Authorization": "Bearer " + API_KEY}
