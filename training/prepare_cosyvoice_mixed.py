"""构建 mixed_v3 的 CosyVoice SFT 数据目录.

默认读取兄弟仓库 Chobits-Chii-TTS 的 data/trimmed/synth_v3/mixed:
  451 真实 + 89 v1 + 281 v2
输出到 CosyVoice/examples/libritts/cosyvoice3/data/chii_mixed_{train,dev}.
文本假名化方式与推理引擎一致 (pyopenjtalk NJD -> 空格分隔片假名).
"""

import argparse
import os
import random
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_MIXED = REPO.parent / "Chobits-Chii-TTS/data/trimmed/synth_v3/mixed"
DEFAULT_DES_BASE = REPO / "CosyVoice/examples/libritts/cosyvoice3/data"
INSTRUCT = 'You are a helpful assistant.<|endofprompt|>'


def to_katakana(text):
    import pyopenjtalk
    parts = []
    for node in pyopenjtalk.run_frontend(text.strip()):
        parts.append(node.get('pron') or node['surface'])
    return ' '.join(parts).replace("'", '').replace('’', '')


def write_split(des_base, name, rows):
    d = des_base / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    with open(d / 'wav.scp', 'w') as fw, open(d / 'text', 'w') as ft, \
         open(d / 'utt2spk', 'w') as fu, open(d / 'spk2utt', 'w') as fs, \
         open(d / 'instruct', 'w') as fi:
        utts = []
        for utt, text, wav in rows:
            fw.write(f'{utt} {wav}\n')
            ft.write(f'{utt} {text}\n')
            fu.write(f'{utt} chii\n')
            fi.write(f'{utt} {INSTRUCT}\n')
            utts.append(utt)
        fs.write('chii ' + ' '.join(utts) + '\n')
    print(f'{name}: {len(rows)} utts -> {d}')


def main():
    parser = argparse.ArgumentParser(description='构建 CosyVoice mixed_v3 SFT 数据')
    parser.add_argument('--mixed-dir', default=str(DEFAULT_MIXED),
                        help='mixed_v3 目录（含 metadata.csv 与 wavs/）')
    parser.add_argument('--des-base', default=str(DEFAULT_DES_BASE),
                        help='CosyVoice examples/libritts/cosyvoice3/data 目录')
    parser.add_argument('--dev-size', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1986)
    args = parser.parse_args()

    mixed = Path(args.mixed_dir)
    des_base = Path(args.des_base)
    rows = []
    failed = []
    with open(mixed / 'metadata.csv', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line or '|' not in line:
                continue
            utt, text = line.split('|', 1)
            utt = utt[:-4] if utt.endswith('.wav') else utt
            wav = mixed / 'wavs' / f'{utt}.wav'
            if not wav.is_file():
                failed.append((utt, 'missing_wav'))
                continue
            try:
                kana = to_katakana(text)
            except Exception as exc:
                failed.append((utt, repr(exc)))
                continue
            if not kana.strip():
                failed.append((utt, 'empty_kana'))
                continue
            rows.append((utt, kana, str(wav)))

    print(f'rows={len(rows)} failed={len(failed)}')
    if failed:
        print('first failures:', failed[:10])
    if not rows:
        raise SystemExit('没有可用数据')

    random.seed(args.seed)
    random.shuffle(rows)
    dev, train = rows[:args.dev_size], rows[args.dev_size:]
    write_split(des_base, 'chii_mixed_train', train)
    write_split(des_base, 'chii_mixed_dev', dev)
    print('sample train:', train[0])
    print('sample dev:', dev[0])


if __name__ == '__main__':
    main()
