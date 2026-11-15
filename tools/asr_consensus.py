"""用 Qwen3-ASR 与 Whisper large-v3 复核待修正的日文文本.

流程:
  1. 读取 data/trimmed/manual/text_review.csv 及 reviewed/wavs/;
  2. Qwen3-ASR-1.7B 与 Whisper large-v3 分别转写;
  3. 可选 NLLB-200 将当前文本、Qwen 结果、Whisper 结果翻译成中文;
  4. 输出 asr_candidates.csv, 供 review_text.py 在复核页展示.

模型均从本地目录加载, 推理失败会保留 asr_raw.json 断点, 便于重跑翻译或后续模型。
"""

import argparse
import csv
import gc
import json
import math
import re
import sys
import unicodedata
import wave
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "trimmed"
DEFAULT_QWEN = REPO_ROOT / "models" / "asr" / "Qwen3-ASR-1.7B"
DEFAULT_WHISPER = REPO_ROOT / "models" / "asr" / "whisper-large-v3"
DEFAULT_TRANSLATOR = REPO_ROOT / "models" / "asr" / "Qwen3-4B-Instruct-2507"
DEFAULT_NLLB = REPO_ROOT / "models" / "asr" / "nllb-200-distilled-600M"
DEFAULT_CONTEXT = ""
FIELDS = [
    "file", "priority", "current_text", "current_zh",
    "qwen_text", "qwen_zh", "whisper_text", "whisper_zh",
    "agreement", "confidence", "consensus_text", "notes",
]


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def normalize_ja(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").casefold()
    chars = []
    for ch in text:
        cat = unicodedata.category(ch)
        if ch.isspace() or cat.startswith("P") or cat.startswith("S"):
            continue
        chars.append(ch)
    return "".join(chars)


def text_similarity(a: str, b: str) -> float:
    a, b = normalize_ja(a), normalize_ja(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    # 编辑距离 / 较长串长度; 对短句比 SequenceMatcher 更严格
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


def choose_consensus(qwen: str, whisper: str) -> tuple[str, float, str]:
    similarity = text_similarity(qwen, whisper)
    qn, wn = normalize_ja(qwen), normalize_ja(whisper)
    if qn and qn == wn:
        return qwen.strip(), 1.0, "high"
    if not qn:
        return whisper.strip(), similarity, "low"
    if not wn:
        return qwen.strip(), similarity, "medium"
    if similarity >= 0.85:
        return qwen.strip(), similarity, "high"
    if similarity >= 0.60:
        return qwen.strip(), similarity, "medium"
    return qwen.strip(), similarity, "low"


def audio_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def load_torch(device: str):
    import torch
    resolved = device
    if device == "auto":
        resolved = "cuda:0" if torch.cuda.is_available() else "cpu"
    if resolved.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA 但 torch.cuda.is_available() 为 False")
    return torch, resolved


def release():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def transcribe_qwen(records, audio_dir: Path, model_dir: Path, device: str,
                    batch_size: int, context: str):
    from qwen_asr import Qwen3ASRModel
    torch, device = load_torch(device)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    print(f"[Qwen3-ASR] loading {model_dir} on {device}", flush=True)
    model = Qwen3ASRModel.from_pretrained(
        str(model_dir),
        dtype=dtype,
        device_map=device,
        max_inference_batch_size=max(1, batch_size),
        max_new_tokens=256,
    )
    results = {}
    paths = [str(audio_dir / f"{r['file']}.wav") for r in records]
    for start in range(0, len(records), batch_size):
        batch_paths = paths[start:start + batch_size]
        batch_names = [r["file"] for r in records[start:start + batch_size]]
        print(f"[Qwen3-ASR] {start + 1}-{start + len(batch_paths)}/{len(records)}", flush=True)
        outputs = model.transcribe(
            audio=batch_paths,
            context=[context] * len(batch_paths) if context else "",
            language=["Japanese"] * len(batch_paths),
        )
        for name, result in zip(batch_names, outputs):
            results[name] = (getattr(result, "text", "") or "").strip()
    del model
    release()
    return results


def transcribe_whisper(records, audio_dir: Path, model_dir: Path, device: str):
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    torch, device = load_torch(device)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    print(f"[Whisper] loading {model_dir} on {device}", flush=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        str(model_dir), torch_dtype=dtype, low_cpu_mem_usage=True,
        use_safetensors=True, attn_implementation="sdpa",
    ).to(device)
    processor = AutoProcessor.from_pretrained(str(model_dir))
    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=device,
    )
    results = {}
    for i, record in enumerate(records, 1):
        path = str(audio_dir / f"{record['file']}.wav")
        try:
            output = asr(
                path,
                return_timestamps=False,
                generate_kwargs={
                    "language": "ja",
                    "task": "transcribe",
                    "temperature": 0.0,
                    "condition_on_prev_tokens": False,
                },
            )
            text = (output.get("text") or "").strip() if isinstance(output, dict) else ""
        except Exception as exc:
            print(f"[Whisper] {record['file']} failed: {exc}", file=sys.stderr, flush=True)
            text = ""
        results[record["file"]] = text
        if i % 20 == 0 or i == len(records):
            print(f"[Whisper] {i}/{len(records)}", flush=True)
    del asr, model, processor
    release()
    return results


def translate_qwen3(texts: list[str], model_dir: Path, device: str, batch_size: int):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch, device = load_torch(device)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    print(f"[Qwen3 translator] loading {model_dir} on {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), dtype=dtype, low_cpu_mem_usage=True,
    ).to(device)
    result = {}
    unique = list(dict.fromkeys(t for t in texts if t and t.strip()))
    system = (
        "你是专业的日语到简体中文翻译助手。忠实翻译用户给出的日文语音转写，"
        "保留语气、称呼、口癖和角色感；原文不完整或无意义时尽量直译；"
        "固定译名：ちぃ/チー=小叽，ひでき/秀樹=秀树。"
        "只输出中文译文，不要解释。"
    )
    for start in range(0, len(unique), batch_size):
        batch = unique[start:start + batch_size]
        prompts = []
        for text in batch:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ]
            try:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)
        encoded = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=256,
        ).to(device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=128,
                do_sample=False,
            )
        decoded = tokenizer.batch_decode(
            generated[:, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
        for source, translated in zip(batch, decoded):
            result[source] = translated.strip()
        print(f"[Qwen3 translator] {min(start + len(batch), len(unique))}/{len(unique)}",
              flush=True)
    del model, tokenizer
    release()
    return result


def translate_nllb(texts: list[str], model_dir: Path, device: str, batch_size: int):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    torch, device = load_torch(device)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    print(f"[NLLB] loading {model_dir} on {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), src_lang="jpn_Jpan")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(model_dir), torch_dtype=dtype, low_cpu_mem_usage=True,
    ).to(device)
    target_id = tokenizer.convert_tokens_to_ids("zho_Hans")
    if target_id is None or target_id == tokenizer.unk_token_id:
        raise RuntimeError("NLLB tokenizer 缺少 zho_Hans 目标语言 token")
    result = {}
    unique = list(dict.fromkeys(t for t in texts if t and t.strip()))
    for start in range(0, len(unique), batch_size):
        batch = unique[start:start + batch_size]
        encoded = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=128,
        ).to(device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                forced_bos_token_id=target_id,
                max_new_tokens=128,
                num_beams=4,
            )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for source, translated in zip(batch, decoded):
            result[source] = translated.strip()
        print(f"[NLLB] {min(start + len(batch), len(unique))}/{len(unique)}", flush=True)
    del model, tokenizer
    release()
    return result


def build_rows(queue, audio_dir, qwen_texts, whisper_texts, translations):
    rows = []
    for item in queue:
        name = item["file"]
        current = item.get("current_text", "")
        qwen = qwen_texts.get(name, "")
        whisper = whisper_texts.get(name, "")
        consensus, similarity, confidence = choose_consensus(qwen, whisper)
        notes = []
        if not qwen:
            notes.append("qwen_empty")
        if not whisper:
            notes.append("whisper_empty")
        if qwen and whisper and normalize_ja(qwen) != normalize_ja(whisper):
            notes.append("models_disagree")
        if confidence == "low":
            notes.append("low_confidence")
        rows.append({
            "file": name,
            "priority": item.get("priority", ""),
            "current_text": current,
            "current_zh": translations.get(current, ""),
            "qwen_text": qwen,
            "qwen_zh": translations.get(qwen, ""),
            "whisper_text": whisper,
            "whisper_zh": translations.get(whisper, ""),
            "agreement": f"{similarity:.3f}",
            "confidence": confidence,
            "consensus_text": consensus,
            "notes": "+".join(notes),
        })
    return rows


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, qwen, whisper):
    path.write_text(json.dumps({"qwen": qwen, "whisper": whisper},
                               ensure_ascii=False, indent=1), encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(description="Qwen3-ASR + Whisper 双模型转写与中文辅助翻译")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--qwen-model", default=str(DEFAULT_QWEN))
    parser.add_argument("--whisper-model", default=str(DEFAULT_WHISPER))
    parser.add_argument("--translator", choices=("qwen3", "nllb", "none"), default="qwen3",
                        help="中文辅助翻译后端 (默认 Qwen3-4B-Instruct)")
    parser.add_argument("--translator-model", default=str(DEFAULT_TRANSLATOR),
                        help="翻译模型目录; nllb 时默认需显式传 NLLB 目录")
    parser.add_argument("--nllb-model", default=str(DEFAULT_NLLB), help=argparse.SUPPRESS)
    parser.add_argument("--output", default=None,
                        help="默认 <data-dir>/manual/asr_candidates.csv")
    parser.add_argument("--device", default="auto", help="auto / cuda:0 / cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 条, 用于试跑")
    parser.add_argument("--context", default=DEFAULT_CONTEXT,
                        help="Qwen 的 system context; 默认关闭，避免短音频复述提示词")
    parser.add_argument("--skip-translation", action="store_true")
    parser.add_argument("--skip-qwen", action="store_true")
    parser.add_argument("--skip-whisper", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir)
    queue_path = data_dir / "manual" / "text_review.csv"
    audio_dir = data_dir / "reviewed" / "wavs"
    raw_path = data_dir / "manual" / "asr_raw.json"
    output = Path(args.output) if args.output else data_dir / "manual" / "asr_candidates.csv"

    queue = read_csv(queue_path)
    if args.limit:
        queue = queue[:args.limit]
    missing = [r["file"] for r in queue if not (audio_dir / f"{r['file']}.wav").is_file()]
    if missing:
        raise SystemExit(f"缺少音频: {missing[:5]}")
    if not queue:
        raise SystemExit("文本复核队列为空")

    cached_qwen, cached_whisper = {}, {}
    if raw_path.is_file():
        try:
            cached = json.loads(raw_path.read_text(encoding="utf-8"))
            cached_qwen = cached.get("qwen", {})
            cached_whisper = cached.get("whisper", {})
        except ValueError:
            pass

    qwen = cached_qwen if args.skip_qwen else transcribe_qwen(
        queue, audio_dir, Path(args.qwen_model), args.device, args.batch_size,
        args.context)
    write_json(raw_path, {r["file"]: qwen.get(r["file"], "") for r in queue},
               {r["file"]: cached_whisper.get(r["file"], "") for r in queue})
    whisper = cached_whisper if args.skip_whisper else transcribe_whisper(
        queue, audio_dir, Path(args.whisper_model), args.device)
    write_json(raw_path, {r["file"]: qwen.get(r["file"], "") for r in queue},
               {r["file"]: whisper.get(r["file"], "") for r in queue})

    translations = {}
    translator = "none" if args.skip_translation else args.translator
    if translator != "none":
        texts = []
        for item in queue:
            texts.extend([
                item.get("current_text", ""),
                qwen.get(item["file"], ""),
                whisper.get(item["file"], ""),
            ])
        if translator == "qwen3":
            model_dir = Path(args.translator_model)
            translations = translate_qwen3(
                texts, model_dir, args.device, args.batch_size)
        else:
            model_dir = Path(args.translator_model)
            if model_dir == DEFAULT_TRANSLATOR:
                model_dir = Path(args.nllb_model)
            translations = translate_nllb(
                texts, model_dir, args.device, args.batch_size)

    rows = build_rows(queue, audio_dir, qwen, whisper, translations)
    write_csv(output, rows)
    confidence = Counter(row["confidence"] for row in rows)
    notes = Counter(note for row in rows for note in row["notes"].split("+") if note)
    print(f"完成: {len(rows)} 条 -> {output}")
    print("  confidence:", dict(confidence), " notes:", dict(notes))
    if translator != "none":
        print(f"  中文翻译列: current_zh / qwen_zh / whisper_zh ({translator})")


if __name__ == "__main__":
    main()
