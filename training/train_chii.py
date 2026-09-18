"""小叽 TTS 训练流水线驱动脚本 (GPT-SoVITS v2Pro, 命令行版, 复刻 webui.py 的调用方式).

步骤:
  1. 1-get-text.py          文本 -> 音素 (ja 不需要 BERT 特征, 但脚本框架一致)
  2. 2-get-hubert-wav32k.py HuBERT SSL 特征 + wav 重采样到 32k
  3. 2-get-sv.py            说话人嵌入 (v2Pro 需要)
  4. 3-get-semantic.py      语义 token (VQ)
  5. s2_train.py            SoVITS 声学模型微调 (全量, v2Pro)
  6. s1_train.py            GPT 语义模型微调

用法 (在 GPT-SoVITS 目录下运行, 或任意目录, 脚本会自行定位):
  conda activate GPTSoVits
  python training/train_chii.py [--skip-preprocess] [--skip-s2] [--skip-s1] [--force]

幂等语义: 预处理各步以 .done-{step}.json 标记 (记录输入内容哈希) 判定完成,
标记与产物齐全才跳过; --force 忽略标记重跑预处理。

注: EXP_NAME 已由 "chi" 改为 "chii" (角色官方罗马字 Chii), 改动后新训练产物写入
GPT-SoVITS/logs/chii/, 权重文件名为 chii_e*.pth / chii-e*.ckpt;
旧的 logs/chi/ 与 chi-* 历史产物保持不动, 如需续跑旧实验请自行改回 EXP_NAME.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Chobits-Chii-TTS/
GS_ROOT = os.path.join(REPO_ROOT, "GPT-SoVITS")

EXP_NAME = "chii"
VERSION = "v2Pro"
LIST_PATH = os.path.join(REPO_ROOT, "data", "gpt_sovits.list")
WAV_DIR = os.path.join(REPO_ROOT, "data", "wavs")
OPT_DIR = os.path.join(GS_ROOT, "logs", EXP_NAME)
TMP_DIR = os.path.join(GS_ROOT, "TEMP")

PRETRAINED = {
    "bert": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
    "hubert": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
    "sv": "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt",
    "s2G": "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth",
    "s2D": "GPT_SoVITS/pretrained_models/v2Pro/s2Dv2Pro.pth",
    "s1": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
    "s2config": "GPT_SoVITS/configs/s2v2Pro.json",
}

# 8GB 显存 (4060 Laptop, 桌面约占 1.6G) + 483 条/21.4min 小数据集
S2_BATCH_SIZE = 4
S2_EPOCHS = 15
S2_SAVE_EVERY = 5
S1_BATCH_SIZE = 4
S1_EPOCHS = 15
S1_SAVE_EVERY = 5

BASE_ENV = {
    **os.environ,
    # webui.py 靠 users.pth 注入这些路径, 命令行直接用 PYTHONPATH 显式指定
    "PYTHONPATH": os.pathsep.join(
        [GS_ROOT, os.path.join(GS_ROOT, "GPT_SoVITS"), os.path.join(GS_ROOT, "GPT_SoVITS", "BigVGAN")]
        + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])
    ),
    # torchcodec 的 libtorchcodec_core*.so 依赖 pip 包 nvidia-npp-cu12 提供的
    # libnppicc.so.12, 但该目录不在默认搜索路径, 需显式加入
    "LD_LIBRARY_PATH": os.pathsep.join(
        [
            os.path.join(
                os.path.dirname(sys.executable),
                "..",
                "lib",
                f"python{sys.version_info.major}.{sys.version_info.minor}",
                "site-packages",
                "nvidia",
                "npp",
                "lib",
            )
        ]
        + ([os.environ["LD_LIBRARY_PATH"]] if os.environ.get("LD_LIBRARY_PATH") else [])
    ),
    "version": VERSION,
    "is_half": "True",
    "i_part": "0",
    "all_parts": "1",
    "_CUDA_VISIBLE_DEVICES": "0",
}


def run(script: str, extra_env: dict, desc: str) -> None:
    env = {**BASE_ENV, **extra_env}
    print(f"\n{'=' * 60}\n[{desc}]\n{'=' * 60}", flush=True)
    p = subprocess.run(
        [sys.executable, "-s", script],
        cwd=GS_ROOT,
        env=env,
    )
    if p.returncode != 0:
        raise SystemExit(f"[失败] {desc} 退出码 {p.returncode}")


# --- 步骤完整性标记 ------------------------------------------------------------
# 此前以"产物文件存在/目录非空"为准跳过：中断留下的半成品会被静默当成完成，
# 重训结果不可信。改为每步落 .done-{step}.json（记录输入内容哈希），
# 标记存在 + 输入哈希一致 + 产物齐全三者同时成立才跳过；输入变更、产物被删、
# 标记损坏/缺失都会重跑该步。--force 忽略标记重跑全部预处理（跑完仍落标记）。

def _inputs_hash(paths) -> str:
    """输入文件/目录的内容哈希：文件 hash 内容，目录按 文件名+大小 排序累加。
    用于 .done.json 标记判断预处理步骤是否对当前输入完整跑完过。"""
    h = hashlib.sha256()
    for p in paths:
        if os.path.isdir(p):
            h.update(b"dir\0")
            for name in sorted(os.listdir(p)):
                fp = os.path.join(p, name)
                if os.path.isfile(fp):
                    h.update(f"{name}:{os.path.getsize(fp)}\0".encode("utf-8"))
        elif os.path.isfile(p):
            h.update(b"file\0")
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
        else:
            h.update(f"missing:{p}\0".encode("utf-8"))
    return h.hexdigest()


def _marker_path(step: str) -> str:
    return os.path.join(OPT_DIR, f".done-{step}.json")


def _step_done(step: str, inputs, outputs) -> bool:
    """步骤算完成：标记存在 + 记录的输入哈希与当前一致 + 产物齐全。
    半成品（有产物无标记/哈希不符）不再被静默跳过。"""
    try:
        with open(_marker_path(step), encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return False
    if data.get("inputs_sha256") != _inputs_hash(inputs):
        return False
    return all(os.path.exists(p) for p in outputs)


def _mark_step_done(step: str, inputs) -> None:
    os.makedirs(OPT_DIR, exist_ok=True)
    with open(_marker_path(step), "w", encoding="utf-8") as f:
        json.dump({"step": step, "inputs_sha256": _inputs_hash(inputs),
                   "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")},
                  f, ensure_ascii=False, indent=2)


def preprocess(force: bool = False, runner=run) -> None:
    # 预处理各步输入一致：标注列表 + wav 目录（语义 token 步另含 s2G 权重与配置）
    wav_inputs = [LIST_PATH, WAV_DIR]
    common = {
        "inp_text": LIST_PATH,
        "inp_wav_dir": WAV_DIR,
        "exp_name": EXP_NAME,
        "opt_dir": OPT_DIR,
    }
    name2text = os.path.join(OPT_DIR, "2-name2text.txt")
    if not force and _step_done("1-text", wav_inputs, [name2text]):
        print("[跳过] 1/6 文本 -> 音素 (已完成)")
    else:
        runner(
            "GPT_SoVITS/prepare_datasets/1-get-text.py",
            {**common, "bert_pretrained_dir": PRETRAINED["bert"]},
            "1/6 文本 -> 音素",
        )
        # 单进程结果合并 (webui 中多 GPU 分片后的合并逻辑); os.replace 原子覆盖,
        # 重跑 (force/标记失效) 时旧文件不碍事 (shutil.move 对已存在 dst 会失败)
        os.replace(os.path.join(OPT_DIR, "2-name2text-0.txt"), name2text)
        _mark_step_done("1-text", wav_inputs)

    hubert_dir = os.path.join(OPT_DIR, "4-cnhubert")
    if not force and _step_done("2-hubert", wav_inputs, [hubert_dir]):
        print("[跳过] 2/6 HuBERT SSL 特征 (已完成)")
    else:
        runner(
            "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py",
            {**common, "cnhubert_base_dir": PRETRAINED["hubert"]},
            "2/6 HuBERT SSL 特征",
        )
        _mark_step_done("2-hubert", wav_inputs)

    sv_dir = os.path.join(OPT_DIR, "7-sv_cn")
    if not force and _step_done("3-sv", wav_inputs, [sv_dir]):
        print("[跳过] 3/6 说话人嵌入 (已完成)")
    else:
        runner(
            "GPT_SoVITS/prepare_datasets/2-get-sv.py",
            {**common, "cnhubert_base_dir": PRETRAINED["hubert"], "sv_path": PRETRAINED["sv"]},
            "3/6 说话人嵌入 (v2Pro)",
        )
        _mark_step_done("3-sv", wav_inputs)

    semantic_tsv = os.path.join(OPT_DIR, "6-name2semantic.tsv")
    semantic_inputs = wav_inputs + [
        os.path.join(GS_ROOT, PRETRAINED["s2G"]),
        os.path.join(GS_ROOT, PRETRAINED["s2config"]),
    ]
    if not force and _step_done("4-semantic", semantic_inputs, [semantic_tsv]):
        print("[跳过] 4/6 语义 token (已完成)")
    else:
        runner(
            "GPT_SoVITS/prepare_datasets/3-get-semantic.py",
            {
                "inp_text": LIST_PATH,
                "exp_name": EXP_NAME,
                "opt_dir": OPT_DIR,
                "pretrained_s2G": PRETRAINED["s2G"],
                "s2config_path": PRETRAINED["s2config"],
            },
            "4/6 语义 token",
        )
        with open(os.path.join(OPT_DIR, "6-name2semantic-0.tsv"), encoding="utf-8") as f:
            body = f.read().strip("\n")
        with open(semantic_tsv, "w", encoding="utf-8") as f:
            f.write("item_name\tsemantic_audio\n" + body + "\n")
        os.remove(os.path.join(OPT_DIR, "6-name2semantic-0.tsv"))
        _mark_step_done("4-semantic", semantic_inputs)


def train_s2() -> None:
    with open(os.path.join(GS_ROOT, PRETRAINED["s2config"]), encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["train"].update(
        {
            "batch_size": S2_BATCH_SIZE,
            "epochs": S2_EPOCHS,
            "text_low_lr_rate": 0.4,
            "pretrained_s2G": PRETRAINED["s2G"],
            "pretrained_s2D": PRETRAINED["s2D"],
            "if_save_latest": True,
            "if_save_every_weights": True,
            "save_every_epoch": S2_SAVE_EVERY,
            "gpu_numbers": "0",
            "grad_ckpt": False,
            "lora_rank": "32",
        }
    )
    cfg["model"]["version"] = VERSION
    cfg["data"]["exp_dir"] = cfg["s2_ckpt_dir"] = OPT_DIR
    cfg["save_weight_dir"] = "SoVITS_weights_v2Pro"
    cfg["name"] = EXP_NAME
    cfg["version"] = VERSION
    os.makedirs(os.path.join(OPT_DIR, "logs_s2_v2Pro"), exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)
    tmp_cfg = os.path.join(TMP_DIR, "tmp_s2.json")
    with open(tmp_cfg, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    print(f"\n{'=' * 60}\n[5/6 SoVITS 训练]\n{'=' * 60}", flush=True)
    p = subprocess.run(
        [sys.executable, "-s", "GPT_SoVITS/s2_train.py", "--config", tmp_cfg],
        cwd=GS_ROOT,
        env=BASE_ENV,
    )
    if p.returncode != 0:
        raise SystemExit(f"[失败] SoVITS 训练退出码 {p.returncode}")


def train_s1() -> None:
    # 懒加载: 门面测试 venv 不装 pyyaml, 仅训练步骤需要（conda 环境已装）
    import yaml
    with open(os.path.join(GS_ROOT, "GPT_SoVITS/configs/s1longer-v2.yaml"), encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg["train"].update(
        {
            "batch_size": S1_BATCH_SIZE,
            "epochs": S1_EPOCHS,
            "save_every_n_epoch": S1_SAVE_EVERY,
            "if_save_every_weights": True,
            "if_save_latest": True,
            "if_dpo": False,
            "half_weights_save_dir": "GPT_weights_v2Pro",
            "exp_name": EXP_NAME,
        }
    )
    cfg["pretrained_s1"] = PRETRAINED["s1"]
    cfg["train_semantic_path"] = os.path.join(OPT_DIR, "6-name2semantic.tsv")
    cfg["train_phoneme_path"] = os.path.join(OPT_DIR, "2-name2text.txt")
    cfg["output_dir"] = os.path.join(OPT_DIR, "logs_s1_v2Pro")
    os.makedirs(cfg["output_dir"], exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)
    tmp_cfg = os.path.join(TMP_DIR, "tmp_s1.yaml")
    with open(tmp_cfg, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    env = {**BASE_ENV, "hz": "25hz"}
    print(f"\n{'=' * 60}\n[6/6 GPT 训练]\n{'=' * 60}", flush=True)
    p = subprocess.run(
        [sys.executable, "-s", "GPT_SoVITS/s1_train.py", "--config_file", tmp_cfg],
        cwd=GS_ROOT,
        env=env,
    )
    if p.returncode != 0:
        raise SystemExit(f"[失败] GPT 训练退出码 {p.returncode}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-preprocess", action="store_true")
    ap.add_argument("--skip-s2", action="store_true")
    ap.add_argument("--skip-s1", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="忽略 .done 完成标记, 重跑全部预处理步骤 (训练步本就无跳过)")
    args = ap.parse_args()

    for k, v in PRETRAINED.items():
        p = os.path.join(GS_ROOT, v)
        if not os.path.exists(p):
            raise SystemExit(f"预训练文件缺失: {p}")

    if not args.skip_preprocess:
        preprocess(force=args.force)
    if not args.skip_s2:
        train_s2()
    if not args.skip_s1:
        train_s1()
    print("\n[完成] 权重输出: GPT-SoVITS/SoVITS_weights_v2Pro/ 与 GPT-SoVITS/GPT_weights_v2Pro/")


if __name__ == "__main__":
    main()
