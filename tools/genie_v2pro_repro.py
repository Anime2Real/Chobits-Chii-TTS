#!/usr/bin/env python3
"""Genie-TTS 引入验证一键复现 (gpt-sovits-genie 分支, 2026-10-11, genie-tts 2.0.2)。

用法:
    pip install genie-tts "torch==2.9.1"   # GenieData 首次 import 会交互下载, 用 yes y | 喂 y
    hf download chenxin199305/Chobits-Chii-TTS-GPTSoVITS --local-dir models/
    yes y | python3 tools/genie_v2pro_repro.py models/

预期结果: 两次 ONNX 转换均报「成功」, 但 load_character 均失败
(vits_fp32.onnx initializer buffer size mismatch) —— v2Pro (VITS 512 通道)
不被 genie-tts 2.0.2 支持 (仅 v2 / v2ProPlus)。详见 docs/genie-feasibility.md。
"""

import io
import os
import sys

import torch


def load_sovits_pth(pth_path: str):
    """HF 发布的权重被削去 ZIP 魔数头 "PK" (见 docs/genie-feasibility.md §3.5),
    与 genie_tts/Converter/load_state_dict.py 相同手法修复后再 torch.load。"""
    f = open(pth_path, 'rb')
    if f.read(2) != b'PK':
        return torch.load(io.BytesIO(b'PK' + f.read()), map_location='cpu', weights_only=False)
    return torch.load(pth_path, map_location='cpu', weights_only=False)


def print_key_shapes(models_dir: str) -> None:
    import genie_tts  # noqa: F401  # 让 pickle 能找到其附带的 utils 模块路径
    sys.path.append(os.path.join(os.path.dirname(genie_tts.__file__), 'Converter'))
    pth = load_sovits_pth(os.path.join(models_dir, 'chii_e10_s1210.pth'))
    sd = pth.get('weight', pth)
    print('== 小叽 v2Pro pth 关键层形状 (Genie v2ProPlus 资源图期望 768 通道) ==')
    for k in ('dec.conv_pre.weight', 'dec.ups.0.weight_g', 'dec.ups.0.weight_v'):
        print(f'  {k}: {tuple(sd[k].shape)}   <- 实际为 {sd[k].shape[0]} 通道')


def try_convert_and_load(models_dir: str, out_dir: str, force_v2pp: bool) -> None:
    import genie_tts as genie
    from genie_tts.ModelManager import model_manager
    if force_v2pp:
        from genie_tts.Converter.v2ProPlus.Converter import convert
        label = 'v2ProPlus 转换器 (强制)'
    else:
        convert = genie.convert_to_onnx
        label = '默认路由 (按 .pth 大小, 134.9MB < 150MB → v2)'
    print(f'== 转换: {label} ==')
    convert(torch_ckpt_path=os.path.join(models_dir, 'chii-e10.ckpt'),
            torch_pth_path=os.path.join(models_dir, 'chii_e10_s1210.pth'),
            output_dir=out_dir)
    # 注意: load_character 加载失败只记日志不抛异常, 须查注册表确认
    genie.load_character(character_name='chii-repro', onnx_model_dir=out_dir, language='Japanese')
    if 'chii-repro' in model_manager.character_to_model:
        print('  加载成功 (意外之喜: 上游已支持 v2Pro?)')
    else:
        print('  加载失败 (预期): vits_fp32.onnx initializer 维度不符, 角色未注册')


if __name__ == '__main__':
    models_dir = sys.argv[1] if len(sys.argv) > 1 else 'models/'
    print_key_shapes(models_dir)
    try_convert_and_load(models_dir, os.path.join(models_dir, 'onnx-v2-route'), force_v2pp=False)
    try_convert_and_load(models_dir, os.path.join(models_dir, 'onnx-v2pp-route'), force_v2pp=True)
