"""批量生成 CosyVoice 评测音频，参数与生产 shim 一致（假名化 + instruct 前缀 + ep18 参考）。"""

import argparse
import csv
import os
import sys
from pathlib import Path

import pyopenjtalk
import soundfile as sf
import torch
import torchaudio

INSTRUCT = 'You are a helpful assistant.<|endofprompt|>'


def to_katakana(text):
    parts = []
    for node in pyopenjtalk.run_frontend(text.strip()):
        parts.append(node.get('pron') or node['surface'])
    return ' '.join(parts).replace("'", '').replace('’', '')


def read_prompts(path):
    with open(path, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--prompts', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--ref-audio', required=True)
    parser.add_argument('--ref-text-file', required=True)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    from cosyvoice.cli.cosyvoice import AutoModel

    prompts = read_prompts(args.prompts)
    if args.limit:
        prompts = prompts[:args.limit]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_text = Path(args.ref_text_file).read_text(encoding='utf-8').strip()
    prompt_text = INSTRUCT + to_katakana(ref_text)

    model = AutoModel(model_dir=args.model_dir, fp16=torch.cuda.is_available())
    sr = model.sample_rate
    resampler = None if sr == 32000 else torchaudio.transforms.Resample(sr, 32000)

    with torch.inference_mode():
        for i, prompt in enumerate(prompts, 1):
            text = to_katakana(prompt['text'])
            chunks = []
            for out in model.inference_zero_shot(text, prompt_text, args.ref_audio,
                                                 stream=False, speed=1.0):
                chunks.append(out['tts_speech'])
            speech = torch.cat(chunks, dim=1)
            if resampler is not None:
                speech = resampler(speech.cpu())
            wav = speech.squeeze(0).float().cpu().numpy()
            name = f"{prompt['id']}-{prompt['category']}.wav"
            sf.write(out_dir / name, wav, 32000, subtype='PCM_16')
            print(f'[{i}/{len(prompts)}] {prompt["id"]} {prompt["category"]}', flush=True)


if __name__ == '__main__':
    main()
