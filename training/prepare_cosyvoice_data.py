# 构建 chii_train / chii_dev 数据目录: wav.scp/text(假名化)/utt2spk/spk2utt/instruct
# 用法: python training/prepare_cosyvoice_data.py [wav_dir]
#   wav_dir 默认 /tmp/chii-voice-hf/wavs (git clone https://hf-mirror.com/datasets/chenxin199305/Chobits-Chii-Voice /tmp/chii-voice-hf)
# 输出到 CosyVoice/examples/libritts/cosyvoice3/data/{chii_train,chii_dev} (seed=1986, dev 10 条)
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WAV_DIR = sys.argv[1] if len(sys.argv) > 1 else '/tmp/chii-voice-hf/wavs'
DES_BASE = os.path.join(REPO, 'CosyVoice/examples/libritts/cosyvoice3/data')
INSTRUCT = 'You are a helpful assistant.<|endofprompt|>'


def to_katakana(text):
    import pyopenjtalk
    parts = []
    for node in pyopenjtalk.run_frontend(text.strip()):
        pron = node.get('pron') or node['surface']
        parts.append(pron)
    return ' '.join(parts).replace("'", '').replace('’', '')


def write_split(name, rows):
    d = os.path.join(DES_BASE, name)
    os.makedirs(d, exist_ok=True)
    with open(f'{d}/wav.scp', 'w') as fw, open(f'{d}/text', 'w') as ft, \
         open(f'{d}/utt2spk', 'w') as fu, open(f'{d}/spk2utt', 'w') as fs, \
         open(f'{d}/instruct', 'w') as fi:
        utts = []
        for utt, text in rows:
            fw.write(f'{utt} {os.path.join(WAV_DIR, utt)}.wav\n')
            ft.write(f'{utt} {text}\n')
            fu.write(f'{utt} chii\n')
            fi.write(f'{utt} {INSTRUCT}\n')
            utts.append(utt)
        fs.write('chii ' + ' '.join(utts) + '\n')
    print(f'{name}: {len(rows)} utts -> {d}')


rows = []
missing = []
with open(os.path.join(REPO, 'data/metadata.csv'), encoding='utf-8') as f:
    for line in f:
        line = line.rstrip('\n')
        if not line or '|' not in line:
            continue
        utt, text = line.split('|', 1)
        if not os.path.exists(os.path.join(WAV_DIR, utt + '.wav')):
            missing.append(utt)
            continue
        rows.append((utt, to_katakana(text)))

print(f'metadata rows with wav: {len(rows)}, missing wav: {len(missing)} {missing[:5]}')
random.seed(1986)
random.shuffle(rows)
n_dev = 10
dev, train = rows[:n_dev], rows[n_dev:]
write_split('chii_train', train)
write_split('chii_dev', dev)
print('sample train text:', train[0])
print('sample dev text:', dev[0])
