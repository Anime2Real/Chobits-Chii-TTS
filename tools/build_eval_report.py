"""把两组 TTS 评测输出合成一个本地并排试听 HTML."""

import argparse
import csv
import html
import os
from pathlib import Path


def read_prompts(path: Path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def audio_block(path: Path, label: str, base_dir: Path):
    if not path.is_file():
        return f"<div class=missing>{label}: 缺失 {html.escape(path.name)}</div>"
    src = html.escape(os.path.relpath(path, base_dir))
    return (
        f"<div class=audio><b>{html.escape(label)}</b>"
        f"<audio controls preload=none src=\"{src}\"></audio></div>"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--prompts", default="examples/eval_prompts.tsv")
    parser.add_argument("--output", default="outputs/eval/compare.html")
    parser.add_argument("--baseline-name", default="旧模型 e10")
    parser.add_argument("--candidate-name", default="新模型")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    prompts = read_prompts(Path(args.prompts).resolve())
    if args.limit:
        prompts = prompts[:args.limit]
    baseline = Path(args.baseline_dir).resolve()
    candidate = Path(args.candidate_dir).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    cards = []
    for prompt in prompts:
        filename = f"{prompt['id']}-{prompt['category']}.wav"
        cards.append(
            "<article class=card>"
            f"<div class=meta>{html.escape(prompt['id'])} · "
            f"{html.escape(prompt['category'])}</div>"
            f"<p>{html.escape(prompt['text'])}</p>"
            + audio_block(baseline / filename, args.baseline_name, output.parent)
            + audio_block(candidate / filename, args.candidate_name, output.parent)
            + "</article>"
        )
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>小叽 TTS 对比试听</title>
<style>
body{{margin:0;background:#111512;color:#e8efe9;font:15px/1.55 system-ui,"Noto Sans SC",sans-serif}}
main{{max-width:1000px;margin:auto;padding:28px 18px 70px}}h1{{font-size:22px}}
.intro{{color:#9aab9f}}.card{{background:#18201b;border:1px solid #34443a;border-radius:10px;padding:15px 18px;margin:14px 0}}
.meta{{font:12px ui-monospace,monospace;color:#94a69b}}p{{font-size:18px;margin:8px 0 14px}}
.audio{{margin:10px 0}}.audio b{{display:block;color:#a9c7b4;font-size:13px;margin-bottom:4px}}
audio{{width:100%}}.missing{{color:#e17979;font-size:13px;margin:8px 0}}
code{{color:#94a69b}}
</style></head><body><main><h1>小叽 TTS 对比试听</h1>
<p class="intro">固定参考音频、seed 和评测句集。左侧/上方为基线，右侧/下方为候选。页面仅用于人工盲听和记录，不自动判定优劣。</p>
{''.join(cards)}</main></body></html>"""
    output.write_text(page, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
