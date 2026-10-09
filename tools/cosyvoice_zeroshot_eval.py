# CosyVoice 验证脚本 (cosy-voice 分支): Fun-CosyVoice3-0.5B zero-shot 克隆小叽声线
# - 日语文本先经 pyopenjtalk 转为分词片假名 (官方 example.py 的日语用法要求, 同时绕过
#   contains_chinese() 把含汉字日语文本路由到中文前端的 bug, 见 docs/cosyvoice-feasibility.md §3)
# - 记录: 非流式 RTF (warmup + 计时)、流式首包延迟、torch 峰值显存、nvidia-smi 峰值显存
# 用法 (conda activate CosyVoice, 仓库根目录):
#   python tools/cosyvoice_zeroshot_eval.py \
#       --model_dir CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B \
#       --ref_audio models/ref_audio.wav --ref_text models/ref_text.txt \
#       --target_text examples/target_text.txt --out_dir outputs/cosyvoice
import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COSYVOICE_DIR = REPO_ROOT / 'CosyVoice'
sys.path.insert(0, str(COSYVOICE_DIR))
sys.path.insert(0, str(COSYVOICE_DIR / 'third_party/Matcha-TTS'))

import torch
import torchaudio

INSTRUCT_PREFIX = 'You are a helpful assistant.<|endofprompt|>'


def to_katakana(text):
    """日语文本 -> 空格分隔的片假名读音序列 (pyopenjtalk NJD; 助词は->ワ等按实际发音)"""
    import pyopenjtalk
    parts = []
    for node in pyopenjtalk.run_frontend(text.strip()):
        pron = node.get('pron') or node['surface']
        parts.append(pron)
    return ' '.join(parts).replace("'", '').replace('’', '')


class VRamSampler:
    """后台线程采样 nvidia-smi 显存占用 (含其他进程, 反映整卡压力)"""

    def __init__(self, interval=0.1):
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                    text=True).strip()
                self.peak = max(self.peak, int(out.splitlines()[0]))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--ref_audio', required=True)
    ap.add_argument('--ref_text', required=True)
    ap.add_argument('--target_text', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--raw', action='store_true', help='不做假名化, 直接送原文 (用于对照 contains_chinese bug)')
    args = ap.parse_args()

    from cosyvoice.cli.cosyvoice import AutoModel

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_text = Path(args.ref_text).read_text(encoding='utf-8').strip()
    target_text = Path(args.target_text).read_text(encoding='utf-8').strip()

    if args.raw:
        prompt_text, tts_text = ref_text, target_text
    else:
        prompt_text = INSTRUCT_PREFIX + to_katakana(ref_text)
        tts_text = to_katakana(target_text)
    print(f'[prompt_text] {prompt_text}\n[tts_text] {tts_text}')

    torch.cuda.reset_peak_memory_stats()
    with VRamSampler() as smi:
        t0 = time.perf_counter()
        cosyvoice = AutoModel(model_dir=args.model_dir, fp16=True)
        load_s = time.perf_counter() - t0
        print(f'[load] {load_s:.1f}s, sample_rate={cosyvoice.sample_rate}')

        result = {'raw_mode': args.raw, 'load_s': round(load_s, 2),
                  'sample_rate': cosyvoice.sample_rate}

        # 非流式: warmup 一次, 再计时一次
        for tag, measure in [('warmup', False), ('nonstream', True)]:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            audio = None
            for out in cosyvoice.inference_zero_shot(tts_text, prompt_text, args.ref_audio, stream=False):
                audio = out['tts_speech']
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            dur = audio.shape[1] / cosyvoice.sample_rate
            if measure:
                path = out_dir / ('zero_shot_raw.wav' if args.raw else 'zero_shot.wav')
                torchaudio.save(str(path), audio, cosyvoice.sample_rate)
                result['nonstream'] = {'elapsed_s': round(elapsed, 3), 'audio_s': round(dur, 3),
                                       'rtf': round(elapsed / dur, 3), 'wav': str(path)}
                print(f'[nonstream] {elapsed:.2f}s / {dur:.2f}s audio, RTF={elapsed / dur:.3f} -> {path}')

        if not args.raw:
            # 流式: 首包延迟 + 总耗时
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            first_chunk_s = None
            chunks = []
            for out in cosyvoice.inference_zero_shot(tts_text, prompt_text, args.ref_audio, stream=True):
                if first_chunk_s is None:
                    first_chunk_s = time.perf_counter() - t0
                chunks.append(out['tts_speech'])
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            audio = torch.cat(chunks, dim=1)
            dur = audio.shape[1] / cosyvoice.sample_rate
            path = out_dir / 'zero_shot_stream.wav'
            torchaudio.save(str(path), audio, cosyvoice.sample_rate)
            result['stream'] = {'first_chunk_ms': round(first_chunk_s * 1000), 'elapsed_s': round(elapsed, 3),
                                'audio_s': round(dur, 3), 'rtf': round(elapsed / dur, 3),
                                'chunks': len(chunks), 'wav': str(path)}
            print(f'[stream] first chunk {first_chunk_s * 1000:.0f}ms, total {elapsed:.2f}s / '
                  f'{dur:.2f}s audio, RTF={elapsed / dur:.3f}, {len(chunks)} chunks -> {path}')

        result['peak_vram'] = {
            'torch_max_allocated_mib': round(torch.cuda.max_memory_allocated() / 2**20),
            'nvidia_smi_peak_mib': smi.peak,
        }
        print(f'[vram] {result["peak_vram"]}')

    metrics_path = out_dir / ('metrics_raw.json' if args.raw else 'metrics.json')
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[metrics] -> {metrics_path}')


if __name__ == '__main__':
    main()
