"""清洗 Chobits-Chii-Voice 数据集, 输出 GPT-SoVITS 可直接使用的训练数据.

规则:
  1. 幻听长重复: 同一字符连续重复 >= 8 次 (如 '超火火火...', 'キィィ...') -> 修复或剔除
  2. 外语串扰: 含拉丁/西里尔字母 (Whisper 误转写, 如 'チatsächlich', 'Gパソコン') -> 修复或剔除
  3. 单字残句: 去除标点/空白后长度 <= 1 (如 '海', '着', 'G') -> 修复或剔除
  4. 字/秒异常: > 15 字/秒, 物理上不可能 -> 修复或剔除
  5. 标点归一化: 半角 '?' '!' -> 全角 '？' '！' (pyopenjtalk 友好)

修复策略: 被标记的条目按文件名中的 (集数, 起始秒) 到 transcripts.csv 查找同一句
(时间差 < 0.5s) 的对照文本, 若对照文本干净 (无外语字母、无长重复、含假名)
则替换, 否则剔除. 全部被标记条目记入 cleaning_report.csv 供人工复核.

用法:
  python tools/clean_dataset.py [--src DIR] [--dst DIR]
"""

import argparse
import csv
import re
import shutil
import wave
from collections.abc import Callable
from pathlib import Path

DEFAULT_SRC = Path.home() / "Github/Chobits-Chii-Voice/dataset"
DEFAULT_DST = Path(__file__).resolve().parent.parent / "data"

LATIN_RE = re.compile(r"[A-Za-zЀ-ӿÀ-ɏ]")
REPEAT_RE = re.compile(r"(.)\1{7,}")  # 同一字符 >= 8 连
KANA_RE = re.compile(r"[぀-ヿ]")  # 平假名或片假名
STRIP_CHARS = " \t?!?!…。、"


def text_dirty(text: str) -> str | None:
    """返回文本被标记的原因, 干净则返回 None."""
    if REPEAT_RE.search(text):
        return "repeat"
    if LATIN_RE.search(text):
        return "latin"
    if len(text.strip(STRIP_CHARS)) <= 1:
        return "short"
    return None


def text_clean(text: str) -> bool:
    """对照文本是否可作为修复值."""
    return (
        not LATIN_RE.search(text)
        and not REPEAT_RE.search(text)
        and KANA_RE.search(text)
        and len(text.strip(STRIP_CHARS)) >= 2
    )


def normalize(text: str) -> str:
    """半角标点转全角, 去首尾空白."""
    return text.strip().replace("?", "？").replace("!", "！")


def parse_name(name: str) -> tuple[str, float]:
    """ep05_00667.42s -> ('ep05', 667.42)"""
    m = re.match(r"(ep\d+)_(\d+\.\d+)s$", name)
    if not m:
        raise ValueError(f"无法解析文件名: {name}")
    return m.group(1), float(m.group(2))


def cps_exceeded(text: str, dur: float) -> bool:
    """字/秒 > 15 视为物理上不可能; 时长未知 (<= 0) 时不判定."""
    return dur > 0 and len(text) / dur > 15


def index_transcripts(rows: list[dict]) -> dict[str, list[dict]]:
    """transcripts.csv 行按集数分组, 供对照文本查找."""
    by_ep: dict[str, list[dict]] = {}
    for r in rows:
        by_ep.setdefault(r["ep"], []).append(r)
    return by_ep


def lookup_transcript(by_ep: dict[str, list[dict]], ep: str, start: float) -> str | None:
    """取 (集数, 起始秒) 时间差 < 0.5s 的最近对照文本, 无则 None."""
    cand = sorted(by_ep.get(ep, []), key=lambda r: abs(float(r["start"]) - start))
    if cand and abs(float(cand[0]["start"]) - start) < 0.5:
        return cand[0]["text"]
    return None


def decide(
    raw_text: str, dur: float, lookup: Callable[[], str | None]
) -> tuple[str, str, str]:
    """单条 metadata 的清洗判定, 返回 (action, reason, final).

    action: keep / repaired / dropped; reason 对 keep 为 "" (其余为
    repeat / latin / short / cps); final 为写出文本 (dropped 为 "").
    """
    text = normalize(raw_text)
    reason = text_dirty(text)
    if reason is None and cps_exceeded(text, dur):
        reason = "cps"

    if reason is None:
        return "keep", "", text
    candidate = lookup()
    if candidate is not None and text_clean(normalize(candidate)):
        return "repaired", reason, normalize(candidate)
    return "dropped", reason, ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC, help="Chobits-Chii-Voice dataset 目录")
    ap.add_argument("--dst", type=Path, default=DEFAULT_DST, help="输出目录 (默认 <repo>/data)")
    args = ap.parse_args()
    src, dst = args.src, args.dst

    meta = []
    with open(src / "metadata.csv", encoding="utf-8") as f:
        for line in f:
            name, text = line.rstrip("\n").split("|", 1)
            meta.append((name, text))

    transcripts = list(csv.DictReader(open(src / "transcripts.csv", encoding="utf-8")))
    by_ep = index_transcripts(transcripts)

    (dst / "wavs").mkdir(parents=True, exist_ok=True)
    report, kept = [], []

    for name, raw_text in meta:
        ep, start = parse_name(name)
        dur = 0.0
        with wave.open(str(src / "wavs" / f"{name}.wav")) as w:
            dur = w.getnframes() / w.getframerate()

        action, reason, final = decide(raw_text, dur, lambda: lookup_transcript(by_ep, ep, start))

        report.append({
            "file": name, "dur": f"{dur:.2f}", "action": action, "reason": reason,
            "original": raw_text, "final": final,
        })
        if action != "dropped":
            kept.append((name, final))
            shutil.copy2(src / "wavs" / f"{name}.wav", dst / "wavs" / f"{name}.wav")

    # metadata.csv (文件名|文本)
    with open(dst / "metadata.csv", "w", encoding="utf-8") as f:
        for name, text in kept:
            f.write(f"{name}|{text}\n")

    # GPT-SoVITS .list (绝对路径|说话人|语言|文本)
    with open(dst / "gpt_sovits.list", "w", encoding="utf-8") as f:
        for name, text in kept:
            f.write(f"{dst / 'wavs' / (name + '.wav')}|chii|ja|{text}\n")

    with open(dst / "cleaning_report.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "dur", "action", "reason", "original", "final"])
        w.writeheader()
        w.writerows(report)

    n_rep = sum(1 for r in report if r["action"] == "repaired")
    n_drop = sum(1 for r in report if r["action"] == "dropped")
    total = sum(float(r["dur"]) for r in report if r["action"] != "dropped")
    print(f"保留 {len(kept)} 条 (修复 {n_rep}, 剔除 {n_drop}), 总时长 {total / 60:.1f} min")
    print(f"输出: {dst}")
    print(f"  wavs/ x{len(kept)}, metadata.csv, gpt_sovits.list, cleaning_report.csv")


if __name__ == "__main__":
    main()
