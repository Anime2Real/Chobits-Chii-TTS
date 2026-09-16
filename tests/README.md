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
- `test_endpoints.py`：TestClient 端点级——`/healthz` 免鉴权（无 key / 错 key 均 200，
  不触引擎）、`/healthz/deep` 无 key 401，无 key / 错 key / query api_key 访问受保护
  端点均 401，限流 429，`/tts` 与 `/v1/audio/speech` 转发给引擎的参数断言
  （流式 batch_size==1、非流式注入默认 5）。
- `test_clean_dataset.py`：数据清洗纯函数——`text_dirty`（repeat/latin/short 及
  优先级）、`text_clean`（修复值四条件）、`normalize`、`parse_name`、`cps_exceeded`
  （含 15 字/秒边界与 dur<=0）、`index_transcripts`/`lookup_transcript`
  （最近命中、0.5s 边界落空、缺集数）、`decide`（keep/repaired/dropped 全路径，
  干净行不触发对照查找）；用例参数化自 `tests/fixtures/*.csv`。

注：`GET /healthz` 为免鉴权浅探活（与 ASR 门面对齐，无 key / 错 key 均 200，不暴露引擎指纹）；
深度检查 `/healthz/deep` 与其余端点一样须带 key（测试按此现状断言）。
