"""用固定评测句集生成 GPT-SoVITS 对比音频.

每次加载一对 GPT/SoVITS 权重, 使用统一参考音频、参数和 seed, 输出 wav + manifest.
必须在 GPT-SoVITS 源码根目录的 Python 环境中运行 (依赖 TTS_infer_pack)。
"""

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

import soundfile as sf
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
GS_ROOT = REPO_ROOT / "GPT-SoVITS"
DEFAULT_PROMPTS = REPO_ROOT / "examples" / "eval_prompts.tsv"
DEFAULT_REF_AUDIO = REPO_ROOT / "models" / "ref_audio.wav"
DEFAULT_REF_TEXT = REPO_ROOT / "models" / "ref_text.txt"


def read_prompts(path: Path):
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows or not {"id", "category", "text"} <= set(rows[0]):
        raise ValueError(f"{path}: 需要 id/category/text 列")
    return rows


def manifest_entry(prompt, name, sample_rate):
    return {
        "id": prompt["id"],
        "category": prompt["category"],
        "text": prompt["text"],
        "wav": name,
        "sample_rate": sample_rate,
    }


def main():
    parser = argparse.ArgumentParser(description="批量生成固定评测集的 TTS 音频")
    parser.add_argument("--gpt", required=True)
    parser.add_argument("--sovits", required=True)
    parser.add_argument("--ref-audio", default=str(DEFAULT_REF_AUDIO))
    parser.add_argument("--ref-text-file", default=str(DEFAULT_REF_TEXT))
    parser.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.35)
    parser.add_argument("--split-method", default="cut0",
                        help="cut0=整段不切, cut1=凑四句, cut5=逐标点")
    args = parser.parse_args()
    gpt_path = Path(args.gpt).resolve()
    sovits_path = Path(args.sovits).resolve()
    ref_audio = Path(args.ref_audio).resolve()
    ref_text_path = Path(args.ref_text_file).resolve()
    prompts_path = Path(args.prompts).resolve()
    out_dir = Path(args.output_dir).resolve()

    sys.path.insert(0, str(GS_ROOT))
    sys.path.insert(0, str(GS_ROOT / "GPT_SoVITS"))
    os.chdir(GS_ROOT)
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

    prompts = read_prompts(prompts_path)
    if args.limit:
        prompts = prompts[:args.limit]
    ref_text = ref_text_path.read_text(encoding="utf-8").strip()
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "custom": {
            "bert_base_path": str(GS_ROOT / "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"),
            "cnhuhbert_base_path": str(GS_ROOT / "GPT_SoVITS/pretrained_models/chinese-hubert-base"),
            "device": args.device,
            "is_half": args.device.startswith("cuda"),
            "t2s_weights_path": str(gpt_path),
            "version": "v2Pro",
            "vits_weights_path": str(sovits_path),
        }
    }
    fd, config_path = tempfile.mkstemp(prefix="chii-eval-", suffix=".yaml", dir=GS_ROOT / "TEMP")
    os.close(fd)
    try:
        Path(config_path).write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
        tts = TTS(TTS_Config(config_path))
        manifest = []
        for i, prompt in enumerate(prompts, 1):
            inputs = {
                "text": prompt["text"],
                "text_lang": "ja",
                "ref_audio_path": str(ref_audio),
                "prompt_text": ref_text,
                "prompt_lang": "ja",
                "top_k": args.top_k,
                "top_p": args.top_p,
                "temperature": args.temperature,
                "text_split_method": args.split_method,
                "batch_size": 1,
                "speed_factor": 1.0,
                "fragment_interval": 0.05,
                "seed": args.seed,
                "parallel_infer": False,
                "repetition_penalty": args.repetition_penalty,
                "streaming_mode": False,
            }
            sr, audio = next(tts.run(inputs))
            name = f"{prompt['id']}-{prompt['category']}.wav"
            sf.write(out_dir / name, audio, sr, subtype="PCM_16")
            manifest.append(manifest_entry(prompt, name, sr))
            print(f"[{i}/{len(prompts)}] {prompt['id']} {prompt['category']}", flush=True)
        with open(out_dir / "manifest.tsv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["id", "category", "text", "wav", "sample_rate"],
                delimiter="\t", lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(manifest)
    finally:
        try:
            os.unlink(config_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
