# CosyVoice3 推理引擎 shim: 与 GPT-SoVITS api_v2 相同的 /tts 契约, 供门面 tools/server.py 无缝反代
# - 日语假名化在引擎内部完成 (text 与 prompt_text 均转, 复用 tools/cosyvoice_zeroshot_eval.py 的
#   to_katakana 逻辑: pyopenjtalk NJD -> 空格分隔片假名, 剔除撇号; 绕过 contains_chinese 把含汉字
#   日语文本路由到中文前端的 bug, 见 docs/cosyvoice-feasibility.md §3)
# - prompt_text 前加 instruct 前缀 (与 SFT 训练数据的 instruct 字段一致)
# - streaming_mode 为真时: 首块为含 fmt 的 WAV 头 (24kHz mono 16bit, RIFF/data 长度 0xFFFFFFFF
#   占位约定, 与门面 _stream_wav_header 一致), 后续块为裸 PCM
# - 引擎无鉴权, 只发布到宿主机 127.0.0.1; 鉴权/限流/路径白名单由门面负责, 本 shim 不做路径白名单
# 启动: uvicorn engine_server:app --host 0.0.0.0 --port $ENGINE_PORT (见 entrypoint.sh)
# 环境变量: MODEL_DIR (默认 /data/model), COSYVOICE_DIR (默认 /opt/CosyVoice),
#           ENGINE_PORT (默认 9880, 仅供裸跑 python engine_server.py 时读取)
import itertools
import logging
import os
import struct
import subprocess
import sys
import threading
from contextlib import asynccontextmanager

COSYVOICE_DIR = os.environ.get('COSYVOICE_DIR', '/opt/CosyVoice')
sys.path.insert(0, COSYVOICE_DIR)
sys.path.insert(0, os.path.join(COSYVOICE_DIR, 'third_party/Matcha-TTS'))

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

MODEL_DIR = os.environ.get('MODEL_DIR', '/data/model')
ENGINE_PORT = int(os.environ.get('ENGINE_PORT', '9880'))

INSTRUCT_PREFIX = 'You are a helpful assistant.<|endofprompt|>'
DEFAULT_REF_AUDIO = '/data/models/ref_audio.wav'
SAMPLE_RATE = 24000  # Fun-CosyVoice3-0.5B 固定 24kHz, 加载后以模型实测值校准

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('cosyvoice-engine')

_model = None
# GPU 推理全局串行化: 门面允许 8 路在途并发, CosyVoice 无批推理接口, 逐请求排队执行
_gpu_lock = threading.Lock()


def to_katakana(text):
    """日语文本 -> 空格分隔的片假名读音序列 (pyopenjtalk NJD; 助词は->ワ等按实际发音)"""
    import pyopenjtalk
    parts = []
    for node in pyopenjtalk.run_frontend(text.strip()):
        pron = node.get('pron') or node['surface']
        parts.append(pron)
    return ' '.join(parts).replace("'", '').replace('’', '')


def get_model():
    global _model, SAMPLE_RATE
    if _model is None:
        from cosyvoice.cli.cosyvoice import AutoModel
        log.info('loading model: %s', MODEL_DIR)
        _model = AutoModel(model_dir=MODEL_DIR, fp16=torch.cuda.is_available())
        SAMPLE_RATE = _model.sample_rate
        log.info('model loaded, sample_rate=%d, cuda=%s', SAMPLE_RATE, torch.cuda.is_available())
    return _model


def _to_pcm16(tts_speech):
    """模型输出 float tensor (1, T) [-1,1] -> s16le 裸 PCM 字节"""
    pcm = torch.round(tts_speech.clamp(-1.0, 1.0) * 32767).to(torch.int16)
    return pcm.cpu().numpy().tobytes()


def _wav_header(data_len):
    """完整 WAV 头 (非流式, 长度写实); data_len 为 PCM 字节数"""
    byte_rate = SAMPLE_RATE * 2
    return (b'RIFF' + struct.pack('<I', 36 + data_len) + b'WAVE'
            + b'fmt ' + struct.pack('<IHHIIHH', 16, 1, 1, SAMPLE_RATE, byte_rate, 2, 16)
            + b'data' + struct.pack('<I', data_len))


def _stream_wav_header():
    """流式 WAV 头: RIFF/data 长度置 0xFFFFFFFF (indefinite 约定), 与门面 _stream_wav_header 一致"""
    byte_rate = SAMPLE_RATE * 2
    return (b'RIFF' + struct.pack('<I', 0xFFFFFFFF) + b'WAVE'
            + b'fmt ' + struct.pack('<IHHIIHH', 16, 1, 1, SAMPLE_RATE, byte_rate, 2, 16)
            + b'data' + struct.pack('<I', 0xFFFFFFFF))


def _pcm_stream(tts_text, prompt_text, ref_audio_path, speed):
    """持锁逐块生成裸 PCM (锁持有到流结束, 保证 GPU 推理串行)"""
    with _gpu_lock:
        for out in get_model().inference_zero_shot(tts_text, prompt_text, ref_audio_path,
                                                   stream=True, speed=speed):
            yield _to_pcm16(out['tts_speech'])


def _encode_ffmpeg(pcm, media_type):
    """裸 PCM -> aac (adts) / ogg (libopus 封装), 容器内 ffmpeg 7"""
    codec, muxer = ('aac', 'adts') if media_type == 'aac' else ('libopus', 'ogg')
    cmd = ['ffmpeg', '-v', 'error',
           '-f', 's16le', '-ar', str(SAMPLE_RATE), '-ac', '1', '-i', 'pipe:0',
           '-c:a', codec, '-f', muxer, 'pipe:1']
    proc = subprocess.run(cmd, input=pcm, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError('ffmpeg encode failed: %s' % proc.stderr[:200])
    return proc.stdout


def _err(status, message):
    return JSONResponse(status_code=status, content={'message': message})


def _is_streaming(value):
    """streaming_mode 判真: GET query 全是字符串, "false"/"0" 也是非空串 (与门面同款)"""
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in ('', '0', 'false', 'none', 'off')
    return bool(value)


def _handle_tts(params):
    """/tts 契约实现 (GET query 与 POST JSON 共用); params 为 dict.
    契约白名单参数 (batch_size/top_k/top_p/temperature/seed/text_lang/prompt_lang 等)
    由门面清洗后透传, 本 shim 接受但忽略——FastAPI 不校验多余字段, 天然兼容."""
    text = str(params.get('text') or '').strip()
    if not text:
        return _err(400, 'text 不能为空')
    media_type = str(params.get('media_type') or 'wav').lower()
    if media_type not in ('wav', 'aac', 'ogg'):
        return _err(400, 'media_type 非法, 可用: wav/aac/ogg')
    ref_audio_path = str(params.get('ref_audio_path') or DEFAULT_REF_AUDIO)
    if not os.path.isfile(ref_audio_path):
        return _err(400, 'ref_audio 不可读: %s' % ref_audio_path)
    prompt_text = str(params.get('prompt_text') or '')
    try:
        speed = float(params.get('speed_factor') or 1.0)
    except (TypeError, ValueError):
        speed = 1.0
    speed = min(max(speed, 0.25), 4.0)
    streaming = _is_streaming(params.get('streaming_mode'))

    # 日语前端: text/prompt_text 假名化 + instruct 前缀 (与 SFT 训练序列一致)
    tts_text = to_katakana(text)
    prompt_full = INSTRUCT_PREFIX + to_katakana(prompt_text)
    log.info('[tts] text=%s -> %s | stream=%s speed=%.2f media=%s',
             text[:30], tts_text[:50], streaming, speed, media_type)

    gen = _pcm_stream(tts_text, prompt_full, ref_audio_path, speed)
    try:
        # 预读首块: 推理在进入响应前失败 -> 5xx; 合法输入绝不可返回 200 空流
        first = next(gen)
    except StopIteration:
        log.error('[tts] 推理未产出任何音频块: %s', tts_text[:50])
        return _err(500, 'synthesis produced no audio')
    except Exception as exc:
        log.exception('[tts] 推理失败')
        return _err(500, 'synthesis failed: %s' % exc.__class__.__name__)

    if streaming and media_type == 'wav':
        return StreamingResponse(
            itertools.chain([_stream_wav_header(), first], gen),
            media_type='audio/wav')

    try:
        pcm = first + b''.join(gen)
    except Exception as exc:
        log.exception('[tts] 推理失败')
        return _err(500, 'synthesis failed: %s' % exc.__class__.__name__)
    if media_type == 'wav':
        return Response(_wav_header(len(pcm)) + pcm, media_type='audio/wav')
    try:
        payload = _encode_ffmpeg(pcm, media_type)
    except RuntimeError as exc:
        log.error('[tts] %s', exc)
        return _err(500, 'audio encode failed')
    return Response(payload, media_type='audio/%s' % media_type)


@asynccontextmanager
async def _lifespan(app):
    # 启动即加载模型 (失败则进程退出, entrypoint 非零退出暴露配置/挂载问题)
    get_model()
    yield


app = FastAPI(title='chobits-chii cosyvoice engine', lifespan=_lifespan)


@app.get('/healthz')
async def healthz():
    return {'status': 'ok', 'engine': 'cosyvoice3'}


@app.get('/tts')
def tts_get(request: Request):
    return _handle_tts(dict(request.query_params))


@app.post('/tts')
async def tts_post(request: Request):
    try:
        body = await request.json()
    except ValueError:
        return _err(400, '请求体不是合法 JSON')
    if not isinstance(body, dict):
        return _err(400, '请求体须为 JSON 对象')
    return _handle_tts(body)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host=os.environ.get('ENGINE_HOST', '0.0.0.0'), port=ENGINE_PORT, workers=1)
