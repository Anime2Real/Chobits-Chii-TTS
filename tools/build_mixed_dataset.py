"""构建真实+合成混合训练集, 按时长控制合成数据占比.

输入:
  - real-dir: 已构建的真实混合目录 (含 wavs/ 与 metadata.csv, 真实文件名为 ep*.wav);
  - fixed-synthetic: 已人工确认的合成 selection.tsv (可多次传入);
  - candidate-synthetic: 新筛选出的候选 selection.tsv (可多次传入).

新合成片段按 score 从高到低选取, 直到合成总时长达到
--target-synthetic-duration. 输出 wavs/、metadata.csv、gpt_sovits.list、
manifest.json 与可选的试听页.
"""

import argparse
import csv
import html
import json
import os
import re
import shutil
import wave
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_FILE = re.compile(r"^ep\d+_.*\.wav$")


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def read_selection(path):
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for row in rows:
        if not row.get("candidate_id") or not row.get("selected_wav"):
            raise ValueError(f"{path}: 缺少 candidate_id 或 selected_wav 列")
    return rows


def read_metadata(path):
    texts = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if "|" in line:
                name, text = line.split("|", 1)
                texts[name] = text
    return texts


def read_exclusions(path):
    if not path:
        return set()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as f:
        first = f.readline()
        f.seek(0)
        if "candidate_id" in first:
            return {
                (row.get("candidate_id") or "").strip()
                for row in csv.DictReader(f, delimiter="\t")
                if (row.get("candidate_id") or "").strip()
            }
        return {
            line.strip().split("\t", 1)[0]
            for line in f
            if line.strip() and not line.lstrip().startswith("#")
        }


def wav_duration(path):
    with wave.open(str(path), "rb") as w:
        if w.getcomptype() != "NONE":
            raise ValueError(f"{path}: 只支持 PCM WAV")
        return w.getnframes() / w.getframerate()


def candidate_sort_key(row):
    try:
        score = float(row.get("score") or 0)
    except ValueError:
        score = 0.0
    try:
        seed = int(row.get("seed") or 0)
    except ValueError:
        seed = 0
    seed_rank = 2 if seed == 44 else 1 if seed == 42 else 0
    return (score, seed_rank)


def copy_row(row, output_wavs, filename):
    src = resolve_path(row["selected_wav"])
    if not src.is_file():
        raise FileNotFoundError(f"缺少音频: {src}")
    dst = output_wavs / filename
    if dst.exists():
        raise ValueError(f"输出重名: {dst}")
    shutil.copyfile(src, dst)
    return {
        "name": filename,
        "selected_wav": str(dst),
        "text": (row.get("text") or "").strip(),
        "src": src,
        "duration": wav_duration(dst),
        "candidate_id": row.get("candidate_id", ""),
        "prompt_id": row.get("prompt_id", ""),
        "seed": row.get("seed", ""),
        "score": row.get("score", ""),
        "qwen_similarity": row.get("qwen_similarity", ""),
        "whisper_similarity": row.get("whisper_similarity", ""),
        "asr_agreement": row.get("asr_agreement", ""),
        "speaker_similarity": row.get("speaker_similarity", ""),
        "p90_dbfs": row.get("p90_dbfs", ""),
        "kind": "synthetic",
    }


def build_review_html(rows, output_path, wav_dir, real_duration, synthetic_duration, target):
    ordered = sorted(rows, key=lambda row: float(row.get("score") or 0))
    cards = []
    for i, row in enumerate(ordered, 1):
        src = os.path.relpath(wav_dir / row["name"], output_path.parent)
        cards.append(
            "<article class=card>"
            f"<div class=head><b>{i}. {html.escape(row['candidate_id'])}</b>"
            f"<span>score {html.escape(row['score'])} · 时长 {row['duration']:.2f}s · "
            f"seed {html.escape(row['seed'])}</span></div>"
            f"<p class=text>{html.escape(row['text'])}</p>"
            f"<p class=metrics>Qwen {html.escape(row['qwen_similarity'])} · "
            f"Whisper {html.escape(row['whisper_similarity'])} · "
            f"ASR一致 {html.escape(row['asr_agreement'])} · "
            f"声纹 {html.escape(row['speaker_similarity'])} · "
            f"P90 {html.escape(row['p90_dbfs'])} dBFS</p>"
            f"<audio controls preload=none src=\"{html.escape(src)}\"></audio>"
            "</article>"
        )
    ratio = synthetic_duration / (real_duration + synthetic_duration) if real_duration + synthetic_duration else 0
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>synth_v2 新合成片段试听</title><style>
body{{margin:0;background:#111512;color:#e8efe9;font:15px/1.55 system-ui,"Noto Sans SC",sans-serif}}
main{{max-width:950px;margin:auto;padding:28px 18px 60px}}h1{{font-size:22px}}
.intro{{color:#9aab9f;max-width:800px}}.card{{background:#18201b;border:1px solid #34443a;border-radius:10px;padding:14px 16px;margin:14px 0}}
.head{{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;font:12px ui-monospace,monospace;color:#94a69b}}
.text{{font-size:18px;margin:9px 0}}.metrics{{color:#e2a84f;font-size:13px}}audio{{width:100%;margin-top:5px}}
</style></head><body><main><h1>synth_v2 新合成片段试听</h1>
<p class=intro>共 {len(ordered)} 条新合成片段，按筛选分数从低到高排列，方便优先试听风险项。
当前真实 {real_duration:.1f}s + 合成 {synthetic_duration:.1f}s，合成占比 {ratio:.1%}（目标 {target:.1f}s）。
请记录需要排除的 candidate_id（例如 p123_s44），通过 --exclude 重新构建即可。</p>
{''.join(cards)}</main></body></html>"""
    output_path.write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="构建真实+合成混合训练集")
    parser.add_argument("--real-dir", required=True)
    parser.add_argument("--fixed-synthetic", action="append", default=[])
    parser.add_argument("--candidate-synthetic", action="append", default=[])
    parser.add_argument("--target-synthetic-duration", type=float, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--exclude", default=None)
    parser.add_argument("--max-new-clips", type=int, default=0)
    parser.add_argument("--review-html", default=None)
    parser.add_argument("--generator-fixed", default="chii-e15 + cut0 + seed42/43")
    parser.add_argument("--generator-new", default="chii_mix-e15 + cut5 + seed42/43/44")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    real_dir = resolve_path(args.real_dir)
    output_dir = resolve_path(args.output_dir)
    if output_dir.exists() and not args.force:
        raise SystemExit(f"输出目录已存在: {output_dir} (加 --force 覆盖)")
    exclusions = read_exclusions(args.exclude)

    real_meta = read_metadata(real_dir / "metadata.csv")
    real_files = sorted(p for p in (real_dir / "wavs").glob("*.wav") if REAL_FILE.match(p.name))
    if not real_files:
        raise SystemExit(f"真实音频为空: {real_dir / 'wavs'}")

    fixed_rows = []
    fixed_prompts = set()
    for selection in args.fixed_synthetic:
        for row in read_selection(resolve_path(selection)):
            if row["candidate_id"] in exclusions:
                continue
            if row["prompt_id"] in fixed_prompts:
                continue
            fixed_prompts.add(row["prompt_id"])
            fixed_rows.append(row)

    candidates = []
    seen_candidates = set()
    for selection in args.candidate_synthetic:
        for row in read_selection(resolve_path(selection)):
            if row["candidate_id"] in exclusions or row["candidate_id"] in seen_candidates:
                continue
            if row["prompt_id"] in fixed_prompts:
                continue
            seen_candidates.add(row["candidate_id"])
            candidates.append(row)
    candidates.sort(key=candidate_sort_key, reverse=True)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_wavs = output_dir / "wavs"
    output_wavs.mkdir(parents=True)

    records = []
    real_duration = 0.0
    for src in real_files:
        name = src.name
        key = name[:-4] if name.endswith(".wav") else name
        if key not in real_meta:
            raise ValueError(f"metadata.csv 缺少 {key}")
        dst = output_wavs / name
        shutil.copyfile(src, dst)
        duration = wav_duration(dst)
        real_duration += duration
        records.append({
            "name": name, "text": real_meta[key], "src": src, "duration": duration,
            "candidate_id": "", "prompt_id": "", "seed": "", "score": "",
            "qwen_similarity": "", "whisper_similarity": "", "asr_agreement": "",
            "speaker_similarity": "", "p90_dbfs": "", "kind": "real",
        })

    fixed_duration = 0.0
    fixed_records = []
    for row in fixed_rows:
        name = Path(row["selected_wav"]).name
        record = copy_row(row, output_wavs, name)
        fixed_duration += record["duration"]
        fixed_records.append(record)

    remaining = max(0.0, args.target_synthetic_duration - fixed_duration)
    new_records = []
    seen_prompts = set(fixed_prompts)
    if remaining <= 0:
        candidates = []
    for row in candidates:
        if row["prompt_id"] in seen_prompts:
            continue
        if args.max_new_clips and len(new_records) >= args.max_new_clips:
            break
        name = f"syn2_{row['prompt_id']}_s{row['seed']}.wav"
        record = copy_row(row, output_wavs, name)
        new_records.append(record)
        seen_prompts.add(row["prompt_id"])
        if sum(r["duration"] for r in new_records) >= remaining:
            break

    synth_records = fixed_records + new_records
    synthetic_duration = sum(r["duration"] for r in synth_records)
    records.extend(fixed_records)
    records.extend(new_records)
    records.sort(key=lambda r: r["name"])

    with open(output_dir / "metadata.csv", "w", encoding="utf-8", newline="") as f:
        for row in records:
            if "|" in row["text"]:
                raise ValueError(f"文本含 |: {row['name']}")
            f.write(f"{row['name']}|{row['text']}\n")

    with open(output_dir / "gpt_sovits.list", "w", encoding="utf-8", newline="") as f:
        for row in records:
            f.write(f"{output_wavs / row['name']}|chii|ja|{row['text']}\n")

    seed_counts = Counter(row["seed"] for row in new_records)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "real": {"clips": len(real_files), "duration_s": round(real_duration, 3)},
        "synthetic": {
            "clips": len(synth_records),
            "duration_s": round(synthetic_duration, 3),
            "ratio_by_duration": round(synthetic_duration / (real_duration + synthetic_duration), 6),
            "fixed": {"clips": len(fixed_records), "duration_s": round(fixed_duration, 3)},
            "new": {
                "clips": len(new_records),
                "duration_s": round(sum(r["duration"] for r in new_records), 3),
                "seed_counts": dict(sorted(seed_counts.items())),
            },
        },
        "total": {
            "clips": len(records),
            "duration_s": round(real_duration + synthetic_duration, 3),
        },
        "target_synthetic_duration_s": args.target_synthetic_duration,
        "generator": {
            "fixed": args.generator_fixed,
            "new": args.generator_new,
        },
        "excluded": sorted(exclusions),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with open(output_dir / "selection_fixed.tsv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fixed_rows[0].keys()) if fixed_rows else ["candidate_id"],
                                delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(fixed_rows)
    with open(output_dir / "selection_new.tsv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(new_records[0].keys()) if new_records else ["candidate_id"],
                                delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(new_records)

    if args.review_html:
        review_path = resolve_path(args.review_html)
        review_path.parent.mkdir(parents=True, exist_ok=True)
        build_review_html(new_records, review_path, output_wavs, real_duration,
                          synthetic_duration, args.target_synthetic_duration)

    print(f"real={len(real_files)} clips / {real_duration:.1f}s")
    print(f"fixed synthetic={len(fixed_records)} clips / {fixed_duration:.1f}s")
    print(f"new synthetic={len(new_records)} clips / {sum(r['duration'] for r in new_records):.1f}s")
    print(f"synthetic total={len(synth_records)} clips / {synthetic_duration:.1f}s "
          f"({manifest['synthetic']['ratio_by_duration']:.1%})")
    print(f"total={len(records)} clips / {real_duration + synthetic_duration:.1f}s")
    if synthetic_duration + 0.5 < args.target_synthetic_duration:
        print(f"WARN: 合成时长未达到目标 {args.target_synthetic_duration:.1f}s")
    print("output:", output_dir)


if __name__ == "__main__":
    main()
