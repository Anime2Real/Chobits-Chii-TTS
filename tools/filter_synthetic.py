"""筛选 e15 生成的合成语音, 输出可进入混合训练的候选集.

筛选依据:
  - 时长、P90 音量、峰值;
  - Qwen3-ASR / Whisper 对生成语音的回译与目标文本一致度;
  - 两个 ASR 之间的一致度;
  - WeSpeaker 与真实小叽嵌入质心的余弦相似度.

每个 prompt 至多选一条, 默认目标 100 条. 需要 .venv-asr 环境运行.
"""

import argparse
import csv
import json
import math
import os
import shutil
import sys
import wave
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
import kaldi_native_fbank as knf
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))
import asr_consensus as ac  # noqa: E402
import trim_dataset as td  # noqa: E402

DEFAULT_PROMPTS = REPO_ROOT / "data/trimmed/pilot/generation_prompts.tsv"
DEFAULT_ROOT = REPO_ROOT / "data/trimmed/pilot"
DEFAULT_QWEN = REPO_ROOT / "models/asr/Qwen3-ASR-1.7B"
DEFAULT_WHISPER = REPO_ROOT / "models/asr/whisper-large-v3"
DEFAULT_SPEAKER_MODEL = REPO_ROOT / "data/trimmed/audit_model/voxceleb_resnet34.onnx"
DEFAULT_REAL_EMB = REPO_ROOT / "data/trimmed/audit_model/wespeaker_embeddings.npz"
DEFAULT_FINAL_REPORT = REPO_ROOT / "data/trimmed/reviewed/build_report.csv"

FIELDS = [
    "candidate_id", "prompt_id", "seed", "text", "wav", "duration_s",
    "p90_dbfs", "peak", "qwen_text", "whisper_text", "qwen_similarity",
    "whisper_similarity", "asr_agreement", "speaker_similarity", "score",
    "passed", "reject_reasons",
]


def read_prompts(path):
    with open(path, encoding="utf-8", newline="") as f:
        return {r["id"]: r for r in csv.DictReader(f, delimiter="\t")}


def collect_candidates(root: Path, prompts: dict):
    rows = []
    for seed_dir in sorted(root.glob("seed*")):
        try:
            seed = int(seed_dir.name.removeprefix("seed"))
        except ValueError:
            continue
        for wav in sorted(seed_dir.glob("*.wav")):
            prompt_id = wav.stem.split("-", 1)[0]
            if prompt_id not in prompts:
                continue
            rows.append({
                "candidate_id": f"{prompt_id}_s{seed}",
                "prompt_id": prompt_id,
                "seed": seed,
                "text": prompts[prompt_id]["text"],
                "wav": wav,
            })
    return rows


def make_flat_links(candidates, root: Path):
    link_dir = root / ".asr_links"
    shutil.rmtree(link_dir, ignore_errors=True)
    link_dir.mkdir(parents=True)
    records = []
    for c in candidates:
        target = link_dir / f"{c['candidate_id']}.wav"
        os.symlink(c["wav"].resolve(), target)
        records.append({"file": c["candidate_id"], "priority": "", "current_text": c["text"]})
    return link_dir, records


def load_or_run_asr(candidates, root, qwen_model, whisper_model, device, reuse):
    cache = root / "asr_raw.json"
    if reuse and cache.is_file():
        data = json.loads(cache.read_text(encoding="utf-8"))
        return data.get("qwen", {}), data.get("whisper", {})
    link_dir, records = make_flat_links(candidates, root)
    qwen = ac.transcribe_qwen(records, link_dir, qwen_model, device, 8, "")
    cache.write_text(json.dumps({"qwen": qwen, "whisper": {}}, ensure_ascii=False, indent=1),
                     encoding="utf-8")
    whisper = ac.transcribe_whisper(records, link_dir, whisper_model, device)
    cache.write_text(json.dumps({"qwen": qwen, "whisper": whisper},
                                ensure_ascii=False, indent=1), encoding="utf-8")
    return qwen, whisper


def fbank_embedding(samples, sample_rate, session, opts):
    if sample_rate != 16000:
        from math import gcd
        g = gcd(16000, sample_rate)
        samples = resample_poly(samples, 16000 // g, sample_rate // g).astype(np.float32)
    else:
        samples = samples.astype(np.float32)
    extractor = knf.OnlineFbank(opts)
    extractor.accept_waveform(16000, (samples * 32768).tolist())
    extractor.input_finished()
    feats = np.stack([extractor.get_frame(i) for i in range(extractor.num_frames_ready)]).astype(np.float32)
    feats = feats - feats.mean(axis=0, keepdims=True)
    if feats.shape[0] < 100:
        feats = np.pad(feats, ((0, 100 - feats.shape[0]), (0, 0)))
    emb = session.run(["embs"], {"feats": feats[None, :, :]})[0][0]
    return emb / (np.linalg.norm(emb) + 1e-12)


def build_speaker_centroid(real_emb_path, final_report, final_names):
    data = np.load(real_emb_path, allow_pickle=True)
    files = list(data["files"])
    embs = data["embs"]
    selected = [embs[i] for i, f in enumerate(files) if f in final_names]
    if not selected:
        raise ValueError("真实小叽嵌入为空")
    centroid = np.mean(selected, axis=0)
    return centroid / (np.linalg.norm(centroid) + 1e-12)


def percentile_db(levels, fraction):
    return sorted(levels)[min(len(levels) - 1, round((len(levels) - 1) * fraction))]


def main():
    parser = argparse.ArgumentParser(description="筛选 e15 合成试点数据")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    parser.add_argument("--output", default=None)
    parser.add_argument("--selected-dir", default=None)
    parser.add_argument("--qwen-model", default=str(DEFAULT_QWEN))
    parser.add_argument("--whisper-model", default=str(DEFAULT_WHISPER))
    parser.add_argument("--speaker-model", default=str(DEFAULT_SPEAKER_MODEL))
    parser.add_argument("--real-embeddings", default=str(DEFAULT_REAL_EMB))
    parser.add_argument("--final-report", default=str(DEFAULT_FINAL_REPORT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reuse-asr", action="store_true")
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--min-duration", type=float, default=0.6)
    parser.add_argument("--max-duration", type=float, default=8.0)
    parser.add_argument("--min-p90", type=float, default=-45.0)
    parser.add_argument("--min-expected", type=float, default=0.70)
    parser.add_argument("--min-agreement", type=float, default=0.30)
    parser.add_argument("--min-speaker", type=float, default=0.20)
    args = parser.parse_args()

    root = Path(args.root)
    prompts = read_prompts(Path(args.prompts))
    candidates = collect_candidates(root / "generated", prompts)
    if not candidates:
        raise SystemExit("没有找到生成音频")
    qwen, whisper = load_or_run_asr(
        candidates, root, Path(args.qwen_model), Path(args.whisper_model),
        args.device, args.reuse_asr)

    final_names = {
        r["file"] for r in csv.DictReader(open(args.final_report, encoding="utf-8", newline=""))
    }
    centroid = build_speaker_centroid(
        Path(args.real_embeddings), Path(args.final_report), final_names)

    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = 16000
    opts.frame_opts.frame_length_ms = 25
    opts.frame_opts.frame_shift_ms = 10
    opts.frame_opts.dither = 0
    opts.frame_opts.window_type = "hamming"
    opts.frame_opts.snip_edges = True
    opts.frame_opts.preemph_coeff = 0.97
    opts.mel_opts.num_bins = 80
    opts.mel_opts.low_freq = 20
    opts.mel_opts.high_freq = 0
    opts.use_energy = False
    session = ort.InferenceSession(args.speaker_model, providers=["CPUExecutionProvider"])

    rows = []
    for c in candidates:
        sr, samples = td.read_wav(c["wav"])
        duration = len(samples) / sr
        levels = td.frame_rms_db(samples, max(1, round(sr * 0.02)))
        p90 = percentile_db(levels, 0.9)
        peak = td.peak_level(samples)
        q = qwen.get(c["candidate_id"], "")
        w = whisper.get(c["candidate_id"], "")
        sq = ac.text_similarity(c["text"], q)
        sw = ac.text_similarity(c["text"], w)
        agreement = ac.text_similarity(q, w)
        speaker = float(np.dot(fbank_embedding(samples, sr, session, opts), centroid))
        reasons = []
        if duration < args.min_duration:
            reasons.append("too_short")
        if duration > args.max_duration:
            reasons.append("too_long")
        if p90 < args.min_p90:
            reasons.append("too_quiet")
        if peak >= 0.99:
            reasons.append("clipped")
        if max(sq, sw) < args.min_expected:
            reasons.append("asr_text_mismatch")
        if agreement < args.min_agreement:
            reasons.append("asr_disagreement")
        if speaker < args.min_speaker:
            reasons.append("speaker_outlier")
        duration_fit = math.exp(-((duration - 3.0) / 2.5) ** 2)
        score = (
            0.40 * max(sq, sw)
            + 0.20 * ((sq + sw) / 2)
            + 0.15 * agreement
            + 0.20 * max(0.0, min(1.0, (speaker + 1) / 2))
            + 0.05 * duration_fit
        )
        rows.append({
            "candidate_id": c["candidate_id"],
            "prompt_id": c["prompt_id"],
            "seed": c["seed"],
            "text": c["text"],
            "wav": str(c["wav"]),
            "duration_s": f"{duration:.3f}",
            "p90_dbfs": f"{p90:.1f}",
            "peak": f"{peak:.3f}",
            "qwen_text": q,
            "whisper_text": w,
            "qwen_similarity": f"{sq:.3f}",
            "whisper_similarity": f"{sw:.3f}",
            "asr_agreement": f"{agreement:.3f}",
            "speaker_similarity": f"{speaker:.3f}",
            "score": f"{score:.3f}",
            "passed": "no" if reasons else "yes",
            "reject_reasons": "+".join(reasons),
        })

    output = Path(args.output) if args.output else root / "synthetic_candidates.csv"
    with open(output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    passed = [r for r in rows if r["passed"] == "yes"]
    best = {}
    for row in passed:
        current = best.get(row["prompt_id"])
        if current is None or float(row["score"]) > float(current["score"]):
            best[row["prompt_id"]] = row
    selected = sorted(best.values(), key=lambda r: float(r["score"]), reverse=True)[:args.target]

    selected_dir = Path(args.selected_dir) if args.selected_dir else root / "selected"
    shutil.rmtree(selected_dir / "wavs", ignore_errors=True)
    (selected_dir / "wavs").mkdir(parents=True)
    selected_rows = []
    for row in selected:
        target = selected_dir / "wavs" / f"syn_{row['prompt_id']}_s{row['seed']}.wav"
        shutil.copyfile(row["wav"], target)
        selected_rows.append({**row, "selected_wav": str(target)})
    with open(selected_dir / "selection.tsv", "w", encoding="utf-8", newline="") as f:
        fields = FIELDS + ["selected_wav"]
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(selected_rows)

    scores = [float(r["score"]) for r in rows]
    print(f"candidates={len(rows)} passed={len(passed)} selected={len(selected)}")
    print("score p5/p50/p95:", np.percentile(scores, [5, 50, 95]).round(3))
    for key in ("qwen_similarity", "whisper_similarity", "asr_agreement", "speaker_similarity"):
        values = [float(r[key]) for r in rows]
        print(key, "p5/p50/p95:", np.percentile(values, [5, 50, 95]).round(3))
    print("output:", output)
    print("selection:", selected_dir / "selection.tsv")


if __name__ == "__main__":
    main()
