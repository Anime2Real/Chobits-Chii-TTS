"""tools/clean_dataset.py 数据清洗规则的单元测试（纯函数, 不读 wav、不触文件系统输出）。

用例主要来自 tests/fixtures/ 下的人工可复核 CSV（与 cleaning_report.csv
同风格）, 覆盖每条判定规则的 True/False 两路及规则间优先级。
"""
import csv
from pathlib import Path

import pytest

import clean_dataset

FIXTURES = Path(__file__).parent / "fixtures"


def _load_cases(filename):
    with open(FIXTURES / filename, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# --- text_dirty: 脏文本判定 (repeat > latin > short 优先级) -------------------

@pytest.mark.parametrize("case", _load_cases("dirty_cases.csv"), ids=lambda c: c["name"])
def test_text_dirty(case):
    assert clean_dataset.text_dirty(case["text"]) == (case["reason"] or None)


# --- text_clean: 对照文本可否作为修复值 ---------------------------------------

@pytest.mark.parametrize("case", _load_cases("clean_check_cases.csv"), ids=lambda c: c["name"])
def test_text_clean(case):
    # text_clean 对无假名文本返回 None (falsy), 只断言真假值
    assert bool(clean_dataset.text_clean(case["text"])) == (case["usable"] == "1")


# --- normalize ----------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("?", "？"),
    ("!", "！"),
    ("  チチ?!  ", "チチ？！"),
    ("チー!", "チー！"),
    ("チチ！", "チチ！"),  # 全角保持不动
    ("a  b", "a  b"),  # 内部空白不动
])
def test_normalize(raw, expected):
    assert clean_dataset.normalize(raw) == expected


# --- parse_name ---------------------------------------------------------------

def test_parse_name_ok():
    assert clean_dataset.parse_name("ep05_00667.42s") == ("ep05", 667.42)


def test_parse_name_multi_digit_ep():
    assert clean_dataset.parse_name("ep123_00001.00s") == ("ep123", 1.0)


@pytest.mark.parametrize("name", [
    "ep01_00100s",        # 缺少小数点
    "xx01_00100.00s",     # 非 ep 前缀
    "ep01_00100.00s.wav", # 多余后缀
    "ep01_00100.00",      # 缺少 s 结尾
])
def test_parse_name_invalid(name):
    with pytest.raises(ValueError):
        clean_dataset.parse_name(name)


# --- cps_exceeded: 字/秒 > 15 --------------------------------------------------

@pytest.mark.parametrize("text,dur,expected", [
    ("あ" * 16, 1.0, True),
    ("あ" * 15, 1.0, False),   # 恰好 15 不超限
    ("あ" * 16, 0.0, False),   # 时长未知不判定
    ("あ" * 16, -1.0, False),
    ("", 1.0, False),
])
def test_cps_exceeded(text, dur, expected):
    assert clean_dataset.cps_exceeded(text, dur) is expected


# --- index_transcripts / lookup_transcript: 对照文本查找 -----------------------

def _index():
    return clean_dataset.index_transcripts([
        {"ep": "ep01", "start": "100.00", "text": "あ"},
        {"ep": "ep01", "start": "100.40", "text": "い"},
        {"ep": "ep01", "start": "101.00", "text": "う"},
        {"ep": "ep02", "start": "200.00", "text": "え"},
    ])


def test_index_transcripts_groups_by_ep_in_order():
    by_ep = _index()
    assert list(by_ep) == ["ep01", "ep02"]
    assert [r["text"] for r in by_ep["ep01"]] == ["あ", "い", "う"]


def test_lookup_nearest_candidate_wins():
    assert clean_dataset.lookup_transcript(_index(), "ep01", 100.10) == "あ"
    assert clean_dataset.lookup_transcript(_index(), "ep01", 100.60) == "い"


def test_lookup_exact_match():
    assert clean_dataset.lookup_transcript(_index(), "ep02", 200.00) == "え"


def test_lookup_boundary_half_second_is_no_match():
    # 时间差 < 0.5s 才算命中; 恰好 0.5s 必须落空
    only = clean_dataset.index_transcripts([{"ep": "ep01", "start": "100.50", "text": "あ"}])
    assert clean_dataset.lookup_transcript(only, "ep01", 100.00) is None


def test_lookup_missing_ep_returns_none():
    assert clean_dataset.lookup_transcript(_index(), "ep99", 100.00) is None
    assert clean_dataset.lookup_transcript({}, "ep01", 100.00) is None


# --- decide: 单条清洗判定 (keep / repaired / dropped) --------------------------

@pytest.mark.parametrize("case", _load_cases("decide_cases.csv"), ids=lambda c: c["name"])
def test_decide(case):
    candidate = case["candidate"] or None
    result = clean_dataset.decide(
        case["raw_text"], float(case["dur"]), lambda: candidate)
    assert result == (case["action"], case["reason"], case["final"])


def test_decide_keep_does_not_call_lookup():
    def _boom():
        raise AssertionError("干净文本不应触发对照查找")
    assert clean_dataset.decide("こんにちは", 1.0, _boom) == ("keep", "", "こんにちは")


def test_decide_dirty_calls_lookup_once():
    calls = []

    def _lookup():
        calls.append(1)
        return None

    assert clean_dataset.decide("ああああああああ", 1.0, _lookup) == ("dropped", "repeat", "")
    assert len(calls) == 1
