"""保守裁剪训练音频的首尾近静音, 并输出句中长停顿的人工复核清单.

默认规则:
  - 20ms RMS 低于 -50 dBFS 视为近静音;
  - 首尾各保留 120ms 缓冲;
  - 句中连续近静音 >= 0.8s 只记录, 不自动删除;
  - P90 帧音量 < -40 dBFS 标记 low_level, 输出 < 0.5s 标记 very_short;
  - 全片近静音不裁切, 标记为 near_silent 待人工处理.

输出目录内的 wavs/ 是派生音频, manual/ 若已存在则原样保留. 原始 data/wavs/
与 data/metadata.csv 不会被修改.
"""

import argparse
import csv
import json
import math
import shutil
import sys
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = REPO_ROOT / "data" / "wavs"
DEFAULT_METADATA = REPO_ROOT / "data" / "metadata.csv"
DEFAULT_DST = REPO_ROOT / "data" / "trimmed"


@dataclass(frozen=True)
class TrimPlan:
    start: int
    end: int
    speech_start: int
    speech_end: int
    near_silent: bool


@dataclass(frozen=True)
class SilenceRun:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def read_wav(path: Path) -> tuple[int, array]:
    with wave.open(str(path), "rb") as w:
        if w.getcomptype() != "NONE":
            raise ValueError(f"{path}: 只支持 PCM WAV")
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path}: 只支持单声道 16-bit WAV")
        sample_rate = w.getframerate()
        samples = array("h")
        samples.frombytes(w.readframes(w.getnframes()))
    if sys.byteorder != "little":
        samples.byteswap()
    return sample_rate, samples


def write_wav(path: Path, samples: array, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = array("h", samples)
    if sys.byteorder != "little":
        payload.byteswap()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(payload.tobytes())


def frame_rms_db(samples: array, frame_samples: int) -> list[float]:
    levels = []
    for offset in range(0, len(samples), frame_samples):
        frame = samples[offset:offset + frame_samples]
        if not frame:
            continue
        mean_square = sum(v * v for v in frame) / len(frame)
        levels.append(10 * math.log10(max(mean_square, 1) / (32768.0 ** 2)))
    return levels


def percentile_db(levels: list[float], fraction: float) -> float:
    if not levels:
        return -120.0
    ordered = sorted(levels)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def peak_level(samples: array) -> float:
    return max((abs(v) for v in samples), default=0) / 32768


def plan_edge_trim(
    samples: array,
    sample_rate: int,
    threshold_db: float,
    frame_ms: int,
    edge_buffer_s: float,
) -> TrimPlan:
    frame_samples = max(1, round(sample_rate * frame_ms / 1000))
    levels = frame_rms_db(samples, frame_samples)
    active = [i for i, db in enumerate(levels) if db >= threshold_db]
    if not active:
        return TrimPlan(0, len(samples), 0, len(samples), True)

    speech_start = active[0] * frame_samples
    speech_end = min(len(samples), (active[-1] + 1) * frame_samples)
    pad = round(sample_rate * edge_buffer_s)
    start = max(0, speech_start - pad)
    end = min(len(samples), speech_end + pad)
    return TrimPlan(start, end, speech_start, speech_end, False)


def find_silence_runs(
    samples: array,
    sample_rate: int,
    threshold_db: float,
    frame_ms: int,
    min_duration_s: float,
) -> list[SilenceRun]:
    frame_samples = max(1, round(sample_rate * frame_ms / 1000))
    levels = frame_rms_db(samples, frame_samples)
    runs = []
    i = 0
    while i < len(levels):
        if levels[i] >= threshold_db:
            i += 1
            continue
        j = i + 1
        while j < len(levels) and levels[j] < threshold_db:
            j += 1
        start = i * frame_ms / 1000
        end = min(len(samples), j * frame_samples) / sample_rate
        if end - start >= min_duration_s:
            runs.append(SilenceRun(round(start, 3), round(end, 3)))
        i = j
    return runs


def load_metadata(path: Path) -> list[tuple[str, str]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if "|" in line:
                name, text = line.rstrip("\n").split("|", 1)
                rows.append((name, text))
    return rows


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def trim_dataset(args) -> dict:
    src, dst = Path(args.src), Path(args.dst)
    metadata_path = Path(args.metadata)
    entries = load_metadata(metadata_path)
    if not entries:
        raise SystemExit(f"没有从 {metadata_path} 读到任何标注")

    missing = [name for name, _ in entries if not (src / f"{name}.wav").is_file()]
    if missing:
        raise SystemExit(f"缺少 {len(missing)} 个音频, 例如: {missing[:3]}")

    wav_dir = dst / "wavs"
    shutil.rmtree(wav_dir, ignore_errors=True)
    wav_dir.mkdir(parents=True)

    report_rows, review_rows = [], []
    total_source = total_output = total_head = total_tail = 0.0
    n_trimmed = n_unchanged = n_near_silent = 0

    for name, text in entries:
        sample_rate, samples = read_wav(src / f"{name}.wav")
        plan = plan_edge_trim(
            samples, sample_rate, args.silence_db, args.frame_ms, args.edge_buffer
        )
        if plan.near_silent:
            output = samples
            status = "near_silent"
            internal = []
            n_near_silent += 1
        else:
            output = samples[plan.start:plan.end]
            internal = find_silence_runs(
                output, sample_rate, args.silence_db, args.frame_ms, args.review_internal
            )
            if plan.start or plan.end != len(samples):
                status = "trimmed"
                n_trimmed += 1
            else:
                status = "unchanged"
                n_unchanged += 1

        write_wav(wav_dir / f"{name}.wav", output, sample_rate)
        source_s = len(samples) / sample_rate
        output_s = len(output) / sample_rate
        head_removed = plan.start / sample_rate
        tail_removed = (len(samples) - plan.end) / sample_rate
        longest_internal = max((run.duration for run in internal), default=0.0)
        intervals = [
            {"start": run.start, "end": run.end, "duration": round(run.duration, 3)}
            for run in internal
        ]
        frame_samples = max(1, round(sample_rate * args.frame_ms / 1000))
        p90_db = percentile_db(frame_rms_db(output, frame_samples), 0.9)
        peak = peak_level(output)

        report_rows.append({
            "file": name,
            "text": text,
            "status": status,
            "source_s": f"{source_s:.3f}",
            "output_s": f"{output_s:.3f}",
            "head_removed_s": f"{head_removed:.3f}",
            "tail_removed_s": f"{tail_removed:.3f}",
            "p90_dbfs": f"{p90_db:.1f}",
            "peak": f"{peak:.3f}",
            "longest_internal_s": f"{longest_internal:.3f}",
            "internal_intervals": json.dumps(intervals, ensure_ascii=False, separators=(",", ":")),
        })

        reasons = []
        if status == "near_silent":
            reasons.append("near_silent")
        elif p90_db < args.low_level_db:
            reasons.append("low_level")
        if output_s < args.min_duration:
            reasons.append("very_short")
        if internal:
            reasons.append("internal_silence")
        if reasons:
            priority = "P0" if (
                status == "near_silent" or p90_db < -45 or output_s < 0.3
            ) else "P1"
            review_rows.append({
                "priority": priority,
                "file": name,
                "reasons": "+".join(reasons),
                "text": text,
                "output_s": f"{output_s:.3f}",
                "internal_intervals": json.dumps(intervals, ensure_ascii=False, separators=(",", ":")),
            })

        total_source += source_s
        total_output += output_s
        total_head += head_removed
        total_tail += tail_removed

    with open(dst / "metadata.csv", "w", encoding="utf-8") as f:
        for name, text in entries:
            f.write(f"{name}|{text}\n")
    with open(dst / "gpt_sovits.list", "w", encoding="utf-8") as f:
        for name, text in entries:
            f.write(f"{wav_dir / (name + '.wav')}|chii|ja|{text}\n")

    _write_csv(
        dst / "trim_report.csv",
        ["file", "text", "status", "source_s", "output_s", "head_removed_s",
         "tail_removed_s", "p90_dbfs", "peak", "longest_internal_s", "internal_intervals"],
        report_rows,
    )
    review_rows.sort(key=lambda row: (row["priority"], row["file"]))
    _write_csv(
        dst / "review_queue.csv",
        ["priority", "file", "reasons", "text", "output_s", "internal_intervals"],
        review_rows,
    )

    summary = {
        "clips": len(entries),
        "trimmed": n_trimmed,
        "unchanged": n_unchanged,
        "near_silent": n_near_silent,
        "review": len(review_rows),
        "source_seconds": total_source,
        "output_seconds": total_output,
        "head_removed_seconds": total_head,
        "tail_removed_seconds": total_tail,
        "dst": str(dst),
    }
    print(
        f"完成 {summary['clips']} 条: 切边 {n_trimmed}, 无需切边 {n_unchanged}, "
        f"全片近静音 {n_near_silent}, 待人工复核 {len(review_rows)}"
    )
    print(
        f"时长 {total_source:.1f}s -> {total_output:.1f}s "
        f"(片头 -{total_head:.1f}s, 片尾 -{total_tail:.1f}s)"
    )
    print(f"输出: {dst}")
    print(f"  wavs/、metadata.csv、gpt_sovits.list、trim_report.csv、review_queue.csv")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", default=str(DEFAULT_SRC), help="原始 WAV 目录")
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA), help="文件名|文本 标注")
    parser.add_argument("--dst", default=str(DEFAULT_DST), help="派生输出目录")
    parser.add_argument("--silence-db", type=float, default=-50.0,
                        help="近静音 RMS 阈值 dBFS (默认 -50)")
    parser.add_argument("--frame-ms", type=int, default=20, help="RMS 帧长毫秒 (默认 20)")
    parser.add_argument("--edge-buffer", type=float, default=0.12,
                        help="首尾保留缓冲秒数 (默认 0.12)")
    parser.add_argument("--review-internal", type=float, default=0.8,
                        help="进入人工复核的句中近静音秒数 (默认 0.8)")
    parser.add_argument("--low-level-db", type=float, default=-40.0,
                        help="P90 帧音量低于该值时标记 low_level (默认 -40 dBFS)")
    parser.add_argument("--min-duration", type=float, default=0.5,
                        help="输出短于该秒数时标记 very_short (默认 0.5)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.frame_ms <= 0:
        raise SystemExit("--frame-ms 必须大于 0")
    if args.edge_buffer < 0:
        raise SystemExit("--edge-buffer 不能为负数")
    if args.review_internal <= 0:
        raise SystemExit("--review-internal 必须大于 0")
    if args.min_duration < 0:
        raise SystemExit("--min-duration 不能为负数")
    trim_dataset(args)


if __name__ == "__main__":
    main()
