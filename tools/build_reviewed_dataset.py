"""把自动切边结果与人工复核决定合并为最终训练集.

默认要求 review_queue.csv 中的每条都已在 manual/decisions.csv 给出结论:
  - keep: 使用人工保留版本;
  - edit: 使用人工剪辑版本;
  - drop: 不进入最终训练集;
  - 未进入复核队列的音频: 使用自动切边版本.

manual/text_review.csv 中的文本决定同样在构建时应用:
  - keep: 保留当前文本;
  - fix: 使用 verified_text;
  - drop: 从最终训练集排除;
  - 空 action: 尚未复核, 默认沿用当前文本 (--require-text-reviewed 可拒绝构建).

输出 reviewed/wavs/、metadata.csv、gpt_sovits.list 与 build_report.csv.
"""

import argparse
import csv
import os
import shutil
import wave
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "trimmed"
DEFAULT_DST_NAME = "reviewed"


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def audio_info(path: Path) -> tuple[int, int, int, float]:
    with wave.open(str(path), "rb") as w:
        if w.getcomptype() != "NONE":
            raise ValueError(f"{path}: 只支持 PCM WAV")
        return (
            w.getframerate(),
            w.getnchannels(),
            w.getsampwidth(),
            w.getnframes() / w.getframerate(),
        )


def load_decisions(path: Path) -> dict[str, dict]:
    decisions = {}
    for row in read_csv(path):
        name = row.get("file", "")
        if not name:
            raise ValueError(f"{path}: 存在缺少 file 的决定")
        if name in decisions:
            raise ValueError(f"{path}: {name} 存在重复决定")
        action = row.get("action", "")
        if action not in {"keep", "edit", "drop"}:
            raise ValueError(f"{path}: {name} 的 action 非法: {action!r}")
        decisions[name] = row
    return decisions


def load_text_reviews(path: Path) -> dict[str, dict]:
    reviews = {}
    for row in read_csv(path):
        name = row.get("file", "")
        if not name:
            raise ValueError(f"{path}: 存在缺少 file 的文本决定")
        if name in reviews:
            raise ValueError(f"{path}: {name} 存在重复文本决定")
        action = row.get("action", "")
        if action not in {"", "keep", "fix", "drop"}:
            raise ValueError(f"{path}: {name} 的文本 action 非法: {action!r}")
        verified = str(row.get("verified_text", "") or "").strip()
        if action == "fix":
            if not verified:
                raise ValueError(f"{path}: {name} 标记 fix 但 verified_text 为空")
            if any(ch in verified for ch in "\r\n|"):
                raise ValueError(f"{path}: {name} 的修正文本包含换行或 '|'")
            if verified == row.get("current_text", ""):
                raise ValueError(f"{path}: {name} 的修正文本与当前文本相同")
        reviews[name] = {**row, "action": action, "verified_text": verified}
    return reviews


def assemble(data_dir: Path, dst: Path, allow_pending: bool = False,
             require_text_reviewed: bool = False) -> dict:
    data_dir, dst = Path(data_dir), Path(dst)
    report = read_csv(data_dir / "trim_report.csv")
    if not report:
        raise ValueError(f"{data_dir / 'trim_report.csv'} 为空或不存在")
    names = [row["file"] for row in report]
    if len(names) != len(set(names)):
        raise ValueError("trim_report.csv 存在重复文件")

    queue = {row["file"]: row for row in read_csv(data_dir / "review_queue.csv")}
    decisions = load_decisions(data_dir / "manual" / "decisions.csv")
    text_reviews = load_text_reviews(data_dir / "manual" / "text_review.csv")
    unknown = sorted(set(decisions) - set(names))
    if unknown:
        raise ValueError(f"决定引用了未知音频: {unknown[:5]}")
    unknown_text = sorted(set(text_reviews) - set(names))
    if unknown_text:
        raise ValueError(f"文本决定引用了未知音频: {unknown_text[:5]}")
    pending = sorted(set(queue) - set(decisions))
    if pending and not allow_pending:
        raise ValueError(
            f"仍有 {len(pending)} 条复核队列未决定, 例如 {pending[:5]}; "
            "完成复核或显式传 --allow-pending"
        )
    text_pending = sorted(
        name for name, review in text_reviews.items()
        if not review["action"]
    )
    if text_pending and require_text_reviewed:
        raise ValueError(
            f"仍有 {len(text_pending)} 条文本未复核, 例如 {text_pending[:5]}"
        )

    temp = dst.parent / f".{dst.name}.tmp-{os.getpid()}"
    shutil.rmtree(temp, ignore_errors=True)
    wav_dir = temp / "wavs"
    wav_dir.mkdir(parents=True)
    build_rows = []
    source_counts = Counter()
    formats = set()
    total_s = 0.0

    try:
        for row in report:
            name = row["file"]
            decision = decisions.get(name)
            action = decision["action"] if decision else ""
            if action == "drop":
                source_counts["dropped"] += 1
                continue
            text_review = text_reviews.get(name)
            text_action = text_review["action"] if text_review else ""
            if text_action == "drop":
                source_counts["text_dropped"] += 1
                continue
            final_text = row["text"]
            if text_action == "fix":
                final_text = text_review["verified_text"]
            if action in {"keep", "edit"}:
                source = data_dir / "manual" / "wavs" / f"{name}.wav"
                if not source.is_file():
                    raise ValueError(f"缺少人工音频: {source}")
                source_kind = f"manual_{action}"
                expected_s = float(decision.get("output_s", 0) or 0)
            else:
                source = data_dir / "wavs" / f"{name}.wav"
                if not source.is_file():
                    raise ValueError(f"缺少自动切边音频: {source}")
                source_kind = "auto" if name not in queue else "auto_pending"
                expected_s = float(row["output_s"])

            sample_rate, channels, width, duration = audio_info(source)
            if expected_s and abs(duration - expected_s) > 0.05:
                raise ValueError(
                    f"{source}: 时长 {duration:.3f}s 与决定/报告 {expected_s:.3f}s 不一致"
                )
            formats.add((sample_rate, channels, width))
            shutil.copyfile(source, wav_dir / f"{name}.wav")
            build_rows.append({
                "file": name,
                "text": final_text,
                "review_action": action,
                "text_action": text_action,
                "source": source_kind,
                "duration_s": f"{duration:.3f}",
            })
            source_counts[source_kind] += 1
            total_s += duration

        if len(formats) != 1:
            raise ValueError(f"输出音频格式不一致: {sorted(formats)}")
        if not build_rows:
            raise ValueError("没有可输出的音频")

        with open(temp / "metadata.csv", "w", encoding="utf-8") as f:
            for row in build_rows:
                f.write(f"{row['file']}|{row['text']}\n")
        with open(temp / "gpt_sovits.list", "w", encoding="utf-8") as f:
            for row in build_rows:
                f.write(f"{dst / 'wavs' / (row['file'] + '.wav')}|chii|ja|{row['text']}\n")
        with open(temp / "build_report.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["file", "text", "review_action", "text_action",
                               "source", "duration_s"]
            )
            writer.writeheader()
            writer.writerows(build_rows)

        old = dst.parent / f".{dst.name}.old-{os.getpid()}"
        shutil.rmtree(old, ignore_errors=True)
        if dst.exists():
            os.replace(dst, old)
        os.replace(temp, dst)
        shutil.rmtree(old, ignore_errors=True)
    finally:
        shutil.rmtree(temp, ignore_errors=True)

    return {
        "output_clips": len(build_rows),
        "dropped": source_counts.get("dropped", 0),
        "text_dropped": source_counts.get("text_dropped", 0),
        "text_pending": len(text_pending),
        "manual": source_counts.get("manual_keep", 0) + source_counts.get("manual_edit", 0),
        "auto": source_counts.get("auto", 0) + source_counts.get("auto_pending", 0),
        "duration_s": total_s,
        "format": next(iter(formats)),
        "dst": str(dst),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="合并自动切边与人工复核结果为最终训练集")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                        help="trim_dataset.py 输出目录")
    parser.add_argument("--dst", default=None,
                        help="最终输出目录 (默认 <data-dir>/reviewed)")
    parser.add_argument("--allow-pending", action="store_true",
                        help="允许复核队列中仍有未决定条目")
    parser.add_argument("--require-text-reviewed", action="store_true",
                        help="要求 text_review.csv 中不再有空白 action")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir)
    dst = Path(args.dst) if args.dst else data_dir / DEFAULT_DST_NAME
    summary = assemble(data_dir, dst, args.allow_pending, args.require_text_reviewed)
    sample_rate, channels, width = summary["format"]
    print(
        f"完成: {summary['output_clips']} 条 / {summary['duration_s']:.1f}s "
        f"({sample_rate}Hz {channels}ch {width * 8}bit)"
    )
    print(
        f"  人工 {summary['manual']} 条, 自动 {summary['auto']} 条, "
        f"音频排除 {summary['dropped']} 条, 文本排除 {summary['text_dropped']} 条"
    )
    if summary["text_pending"]:
        print(f"  注意: 仍有 {summary['text_pending']} 条文本未复核")
    print(f"输出: {summary['dst']}")
    print("  wavs/、metadata.csv、gpt_sovits.list、build_report.csv")


if __name__ == "__main__":
    main()
