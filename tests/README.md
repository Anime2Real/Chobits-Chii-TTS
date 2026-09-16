# 测试

门面的纯逻辑（鉴权 / 限流 / 参数清洗 / 并发控制）与端点行为的 pytest 套件。
引擎依赖全部 mock（内存假 httpx 客户端），不需要启动 :9882 推理引擎。

## 运行

```bash
.venv/bin/pip install pytest   # 一次性
.venv/bin/python -m pytest tests/ -v
```

## 覆盖范围

- `test_unit.py`：`_getenv_int/_getenv_float` 容错、`_client_ip` XFF 末跳逻辑、
  `_extract_key`（只认 Bearer）、`_is_streaming` 各输入形态、
  `_sanitize_tts_params`（白名单 / 数值钳制 / 整型还原 / 缺省 batch_size /
  流式钉 1）、`_InflightHold` 幂等释放、`_acquire_inflight` 排队超时 429。
- `test_endpoints.py`：TestClient 端点级——无 key / 错 key / query api_key 均 401，
  限流 429，`/tts` 与 `/v1/audio/speech` 转发给引擎的参数断言
  （流式 batch_size==1、非流式注入默认 5）。

注：本服务没有免鉴权的 `/healthz`；`/healthz/deep` 与其余端点一样须带 key
（测试按此现状断言）。
