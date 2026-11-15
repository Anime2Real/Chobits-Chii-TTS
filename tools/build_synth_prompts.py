"""从 Chobits 台词索引挑选合成训练用 prompt.

筛选规则:
  - 排除已进入真实训练集、试点 prompt、评测 prompt 的文本;
  - 只保留 5-40 字、含假名、无拉丁字母/数字/括号/省略号等的台词;
  - 排除男性第一人称、明显片段结尾和重复短语;
  - 按 chi_prob 排序, 限制每话条数, 输出 GPT-SoVITS 生成用 TSV.

文本只做最小归一化: 秀吉->秀樹、きぃ->ちぃ. チー/ちー/ちい/ひでき
等真实数据中已有的写法保持不变, 避免把训练分布收窄成单一表记.
"""

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRANSCRIPTS = REPO_ROOT.parent / "Chobits-Chii-Voice/dataset/transcripts.csv"
DEFAULT_REVIEWED = REPO_ROOT / "data/trimmed/reviewed/metadata.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data/trimmed/synth_v2/generation_prompts.tsv"
DEFAULT_EXCLUDES = (
    REPO_ROOT / "data/trimmed/pilot/generation_prompts.tsv",
    REPO_ROOT / "examples/eval_prompts.tsv",
    REPO_ROOT / "examples/longform_prompts.tsv",
)

BAD_CHARS = re.compile(r"[A-Za-z0-9０-９\[\]\(\)\{\}「」『』【】…‥・♪★☆→←/\\:;：；]")
MALE_SPEECH = re.compile(
    r"俺|おれ|僕|ぼく|お前|てめえ|あいつ|野郎|俺様|じゃねえ|ねえよ|だぜ|だぞ|かよ|っす|やがる"
)
FRAGMENT_END = re.compile(
    r"(は|が|を|に|で|と|も|へ|や|の|けど|けれど|から|ので|のに|って|たり|たら|なら|では|には|とは|ってば)$"
)
SAME_CHAR_REPEAT = re.compile(r"(.)\1{3,}")
PHRASE_REPEAT = re.compile(r"(.{2,4})\1{3,}")
ADULT_TERMS = re.compile(r"みだら|人妻|裸|エロ|セックス|オナニー|レイプ|痴漢|おっぱい")


def normalize_text(text):
    text = re.sub(r"\s+", " ", text.strip())
    return text.replace("秀吉", "秀樹").replace("きぃ", "ちぃ")


def is_valid_text(text, min_chars=5, max_chars=40):
    if not (min_chars <= len(text) <= max_chars):
        return False
    if BAD_CHARS.search(text):
        return False
    if SAME_CHAR_REPEAT.search(text) or PHRASE_REPEAT.search(text):
        return False
    if not re.search(r"[ぁ-んァ-ヶ]", text):
        return False
    if FRAGMENT_END.search(text):
        return False
    if MALE_SPEECH.search(text):
        return False
    if ADULT_TERMS.search(text):
        return False
    return True


def read_reviewed_texts(path):
    texts = set()
    if not Path(path).is_file():
        return texts
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if "|" in line:
                texts.add(line.split("|", 1)[1].strip())
    return texts


def read_prompt_texts(paths):
    texts = set()
    for path in paths:
        path = Path(path)
        if not path.is_file():
            continue
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                text = (row.get("text") or "").strip()
                if text:
                    texts.add(text)
    return texts


def load_candidates(transcripts_path, excluded_texts):
    with open(transcripts_path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    best = {}
    for row in rows:
        if (row.get("in_dataset") or "").strip() != "0":
            continue
        text = normalize_text(row.get("text") or "")
        if not is_valid_text(text):
            continue
        if text in excluded_texts:
            continue
        try:
            chi = float(row.get("chi_prob") or 0)
        except ValueError:
            chi = 0.0
        current = best.get(text)
        if current is None or chi > current["chi_prob"]:
            best[text] = {
                "text": text,
                "ep": (row.get("ep") or "").strip(),
                "chi_prob": chi,
                "source_dur": (row.get("dur") or "").strip(),
            }
    return list(best.values())


def length_fit(length):
    if 8 <= length <= 18:
        return 1.0
    if 5 <= length <= 25:
        return 0.6
    return 0.2


def select_prompts(candidates, target, per_episode_cap):
    ordered = sorted(
        candidates,
        key=lambda row: (row["chi_prob"], length_fit(len(row["text"])), -len(row["text"])),
        reverse=True,
    )
    selected = []
    counts = Counter()
    for row in ordered:
        if counts[row["ep"]] >= per_episode_cap:
            continue
        selected.append(row)
        counts[row["ep"]] += 1
        if len(selected) >= target:
            break
    return selected, counts


def main():
    parser = argparse.ArgumentParser(description="挑选合成训练用 prompt")
    parser.add_argument("--transcripts", default=str(DEFAULT_TRANSCRIPTS))
    parser.add_argument("--reviewed", default=str(DEFAULT_REVIEWED))
    parser.add_argument("--exclude", nargs="*", default=[str(p) for p in DEFAULT_EXCLUDES])
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--target", type=int, default=400)
    parser.add_argument("--per-episode-cap", type=int, default=30)
    parser.add_argument("--id-start", type=int, default=101)
    args = parser.parse_args()

    transcripts = Path(args.transcripts)
    if not transcripts.is_file():
        raise SystemExit(f"找不到台词索引: {transcripts}")

    excluded = set()
    for text in read_reviewed_texts(args.reviewed):
        excluded.add(text)
        excluded.add(normalize_text(text))
    for text in read_prompt_texts(args.exclude):
        excluded.add(text)
        excluded.add(normalize_text(text))

    candidates = load_candidates(transcripts, excluded)
    selected, counts = select_prompts(candidates, args.target, args.per_episode_cap)
    if len(selected) < args.target:
        raise SystemExit(f"候选不足: 需要 {args.target}, 仅选出 {len(selected)}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["id", "category", "text", "ep", "chi_prob", "source_dur"])
        for i, row in enumerate(selected, args.id_start):
            writer.writerow([
                f"p{i:03d}",
                f"transcript_{row['ep']}",
                row["text"],
                row["ep"],
                f"{row['chi_prob']:.3f}",
                row["source_dur"],
            ])

    lengths = [len(row["text"]) for row in selected]
    print(f"candidates={len(candidates)} selected={len(selected)} episodes={len(counts)}")
    print(f"chi cutoff={selected[-1]['chi_prob']:.3f} "
          f"chi>=0.2={sum(1 for row in selected if row['chi_prob'] >= 0.2)}")
    print(f"chars min/median/max={min(lengths)}/{sorted(lengths)[len(lengths) // 2]}/{max(lengths)}")
    print("episode counts:", " ".join(f"{ep}:{n}" for ep, n in sorted(counts.items())))
    print("output:", output)


if __name__ == "__main__":
    main()
