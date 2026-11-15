"""对固定模型/切分的多 seed 长文生成做质量回归评分.

评分优先级:
  1. 缺失从句总数 (窗口回译匹配, 越低越好);
  2. 低质量样本数 (max ASR similarity < 0.70);
  3. 平均 ASR 文本一致度;
  4. 最低样本一致度。

输出 seed_scores.csv、seed_details.csv 和 seed_sweep_summary.json。
"""

import argparse
import csv
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))
import asr_consensus as ac  # noqa: E402


def norm(text):
    text = unicodedata.normalize("NFKC", text or "").casefold()
    return "".join(ch for ch in text
                   if not ch.isspace()
                   and not unicodedata.category(ch).startswith(("P", "S")))


def similarity(a, b):
    a, b = norm(a), norm(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (ca != cb),
            ))
        previous = current
    return 1.0 - previous[-1] / max(len(a), len(b))


def best_window_similarity(needle, haystack):
    n = len(norm(needle))
    hay = norm(haystack)
    if not n or not hay:
        return 0.0
    best = 0.0
    for size in range(max(2, n - 4), min(len(hay), n + 7) + 1):
        for start in range(0, len(hay) - size + 1):
            best = max(best, similarity(needle, hay[start:start + size]))
    return best


def read_prompts(path):
    with open(path, encoding="utf-8", newline="") as f:
        return {r["id"]: r for r in csv.DictReader(f, delimiter="\t")}


def collect(root: Path, prompts: dict):
    rows = []
    for d in sorted(root.glob("cut5_s*")):
        seed = int(d.name.removeprefix("cut5_s"))
        for wav in sorted(d.glob("*.wav")):
            ident = wav.stem.split("-", 1)[0]
            if ident not in prompts:
                continue
            rows.append({"seed": seed, "id": ident, "file": f"{seed}_{wav.stem}",
                         "wav": wav, "expected": prompts[ident]["text"]})
    return rows


def missing_clauses(expected, qwen, whisper):
    clauses = [x for x in re.split(r"[。？！!?]", expected) if len(norm(x)) >= 4]
    missing = 0
    for clause in clauses:
        score = max(best_window_similarity(clause, qwen),
                    best_window_similarity(clause, whisper))
        if score < 0.55:
            missing += 1
    return missing, len(clauses)


def main():
    parser = argparse.ArgumentParser(description="长文多 seed 质量评分")
    parser.add_argument("--root", default="outputs/eval/seed_sweep")
    parser.add_argument("--prompts", default="examples/longform_prompts.tsv")
    parser.add_argument("--qwen-model", default="models/asr/Qwen3-ASR-1.7B")
    parser.add_argument("--whisper-model", default="models/asr/whisper-large-v3")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reuse-asr", action="store_true")
    parser.add_argument("--low-quality", type=float, default=0.70)
    args = parser.parse_args()

    root = Path(args.root)
    prompts = read_prompts(Path(args.prompts))
    rows = collect(root, prompts)
    if not rows:
        raise SystemExit("没有找到 cut5_s*/wav")

    cache = root / "asr_raw.json"
    if args.reuse_asr and cache.is_file():
        cached = json.loads(cache.read_text(encoding="utf-8"))
        qwen, whisper = cached.get("qwen", {}), cached.get("whisper", {})
    else:
        link_dir = root / ".asr_links"
        if link_dir.exists():
            for p in link_dir.glob("*.wav"):
                p.unlink()
        link_dir.mkdir(exist_ok=True)
        records = []
        for r in rows:
            os.symlink(r["wav"].resolve(), link_dir / f"{r['file']}.wav")
            records.append({"file": r["file"], "priority": "", "current_text": r["expected"]})
        qwen = ac.transcribe_qwen(
            records, link_dir, Path(args.qwen_model), args.device, 8, "")
        cache.write_text(json.dumps({"qwen": qwen, "whisper": {}},
                                    ensure_ascii=False, indent=1), encoding="utf-8")
        whisper = ac.transcribe_whisper(
            records, link_dir, Path(args.whisper_model), args.device)
        cache.write_text(json.dumps({"qwen": qwen, "whisper": whisper},
                                    ensure_ascii=False, indent=1), encoding="utf-8")

    details = []
    for r in rows:
        q = qwen.get(r["file"], "")
        w = whisper.get(r["file"], "")
        sq, sw = similarity(r["expected"], q), similarity(r["expected"], w)
        avg = (sq + sw) / 2
        missing, total_clauses = missing_clauses(r["expected"], q, w)
        details.append({
            "seed": r["seed"], "id": r["id"], "file": r["file"],
            "wav": str(r["wav"]), "expected": r["expected"],
            "qwen_text": q, "whisper_text": w,
            "qwen_similarity": f"{sq:.3f}", "whisper_similarity": f"{sw:.3f}",
            "avg_similarity": f"{avg:.3f}", "missing_clauses": missing,
            "total_clauses": total_clauses,
        })

    by_seed = defaultdict(list)
    for row in details:
        by_seed[row["seed"]].append(row)
    scores = []
    for seed, seed_rows in sorted(by_seed.items()):
        avgs = [float(r["avg_similarity"]) for r in seed_rows]
        missing = sum(r["missing_clauses"] for r in seed_rows)
        low = sum(_max_sim(r) < args.low_quality for r in seed_rows)
        complete = sum(r["missing_clauses"] == 0 for r in seed_rows)
        scores.append({
            "seed": seed,
            "samples": len(seed_rows),
            "missing_clauses": missing,
            "complete_samples": complete,
            "low_quality": low,
            "mean_similarity": round(sum(avgs) / len(avgs), 4),
            "min_similarity": round(min(avgs), 4),
            "avg_chars_per_sec": round(sum(
                len(r["expected"]) / _duration(r) for r in seed_rows) / len(seed_rows), 2),
        })
    # Prioritize completeness, then robustness, then average quality.
    scores.sort(key=lambda r: (r["missing_clauses"], r["low_quality"],
                               -r["mean_similarity"], -r["min_similarity"]))
    for rank, row in enumerate(scores, 1):
        row["rank"] = rank

    with open(root / "seed_scores.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scores[0]))
        writer.writeheader()
        writer.writerows(scores)
    with open(root / "seed_details.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    summary = {
        "recommended_seed": scores[0]["seed"],
        "ranking_rule": "missing_clauses, low_quality, -mean_similarity, -min_similarity",
        "seed_scores": scores,
        "details": str(root / "seed_details.csv"),
    }
    (root / "seed_sweep_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("rank seed missing low_quality mean min chars/s")
    for r in scores:
        print(r["rank"], r["seed"], r["missing_clauses"], r["low_quality"],
              r["mean_similarity"], r["min_similarity"], r["avg_chars_per_sec"])
    print("recommended seed:", scores[0]["seed"])


def _max_sim(row):
    return max(float(row["qwen_similarity"]), float(row["whisper_similarity"]))


def _duration(row):
    import wave
    with wave.open(str(Path(row["wav"])), "rb") as w:
        return w.getnframes() / w.getframerate()


if __name__ == "__main__":
    main()
