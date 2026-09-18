"""training/train_chii.py 步骤完整性标记测试（合成临时目录, 不跑真实训练脚本）。

回归目标: 此前以"产物存在/目录非空"跳过预处理, 中断留下的半成品会被静默
当成完成。现在以 .done-{step}.json 标记 + 输入哈希 + 产物齐全三重判定,
这些测试覆盖: 完整跑完幂等跳过 / 半成品(有产物无标记)重跑 / 标记损坏重跑 /
输入变更使标记失效 / 产物被删重跑 / --force 全量重跑。
"""
import json
import os
import sys
from types import SimpleNamespace

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "training"))

import train_chii  # noqa: E402

SCRIPT_1 = "GPT_SoVITS/prepare_datasets/1-get-text.py"
SCRIPT_2 = "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py"
SCRIPT_3 = "GPT_SoVITS/prepare_datasets/2-get-sv.py"
SCRIPT_4 = "GPT_SoVITS/prepare_datasets/3-get-semantic.py"
ALL_SCRIPTS = [SCRIPT_1, SCRIPT_2, SCRIPT_3, SCRIPT_4]


class FakeRunner:
    """run() 替身：记录调用, 并按真实脚本的最小副作用产出对应产物文件。"""

    def __init__(self):
        self.calls = []

    def __call__(self, script, extra_env, desc):
        self.calls.append(script)
        opt = train_chii.OPT_DIR
        os.makedirs(opt, exist_ok=True)
        if script == SCRIPT_1:
            with open(os.path.join(opt, "2-name2text-0.txt"), "w", encoding="utf-8") as f:
                f.write("fake\tja\t0\twavs/a.wav\n")
        elif script == SCRIPT_2:
            os.makedirs(os.path.join(opt, "4-cnhubert"), exist_ok=True)
            with open(os.path.join(opt, "4-cnhubert", "a.wav.pt"), "w") as f:
                f.write("hubert")
        elif script == SCRIPT_3:
            os.makedirs(os.path.join(opt, "7-sv_cn"), exist_ok=True)
            with open(os.path.join(opt, "7-sv_cn", "a.wav.pt"), "w") as f:
                f.write("sv")
        elif script == SCRIPT_4:
            with open(os.path.join(opt, "6-name2semantic-0.tsv"), "w", encoding="utf-8") as f:
                f.write("wavs/a.wav\t[1,2,3]")
        return None


@pytest.fixture
def synth(monkeypatch, tmp_path):
    """合成最小输入目录 (标注列表 + wavs) 并劫持 train_chii 的路径常量。"""
    list_path = tmp_path / "data" / "gpt_sovits.list"
    list_path.parent.mkdir(parents=True)
    list_path.write_text("wavs/a.wav\tfake\tja\t0\n", encoding="utf-8")
    wav_dir = tmp_path / "data" / "wavs"
    wav_dir.mkdir(parents=True)
    (wav_dir / "a.wav").write_bytes(b"RIFF-fake-wav")
    opt_dir = tmp_path / "GPT-SoVITS" / "logs" / "chii"
    monkeypatch.setattr(train_chii, "LIST_PATH", str(list_path))
    monkeypatch.setattr(train_chii, "WAV_DIR", str(wav_dir))
    monkeypatch.setattr(train_chii, "OPT_DIR", str(opt_dir))
    return SimpleNamespace(opt_dir=str(opt_dir), runner=FakeRunner())


def test_full_run_then_idempotent_skip(synth):
    train_chii.preprocess(runner=synth.runner)
    assert synth.runner.calls == ALL_SCRIPTS
    # 标记 + 产物齐全: 第二次运行全部跳过, 幂等
    second = FakeRunner()
    train_chii.preprocess(runner=second)
    assert second.calls == []


def test_half_done_artifact_no_longer_skipped(synth):
    # 回归核心: 旧逻辑见 4-cnhubert 非空即跳过; 新逻辑无标记必须重跑
    train_chii.preprocess(runner=synth.runner)
    marker = os.path.join(synth.opt_dir, ".done-2-hubert.json")
    os.remove(marker)
    assert os.listdir(os.path.join(synth.opt_dir, "4-cnhubert"))  # 产物仍在 (半成品状态)
    second = FakeRunner()
    train_chii.preprocess(runner=second)
    assert second.calls == [SCRIPT_2]


def test_corrupt_marker_reruns_step(synth):
    train_chii.preprocess(runner=synth.runner)
    marker = os.path.join(synth.opt_dir, ".done-1-text.json")
    with open(marker, "w", encoding="utf-8") as f:
        f.write("not-json{{{")
    second = FakeRunner()
    train_chii.preprocess(runner=second)
    assert second.calls == [SCRIPT_1]


def test_input_change_invalidates_markers(synth):
    train_chii.preprocess(runner=synth.runner)
    # 换内容同大小的 wav 不敏感 (目录按 文件名+大小 哈希), 增删文件必须失效
    new_wav = os.path.join(train_chii.WAV_DIR, "b.wav")
    with open(new_wav, "wb") as f:
        f.write(b"RIFF-another")
    second = FakeRunner()
    train_chii.preprocess(runner=second)
    assert second.calls == ALL_SCRIPTS


def test_deleted_output_reruns_step(synth):
    train_chii.preprocess(runner=synth.runner)
    os.remove(os.path.join(synth.opt_dir, "6-name2semantic.tsv"))
    second = FakeRunner()
    train_chii.preprocess(runner=second)
    assert second.calls == [SCRIPT_4]


def test_force_reruns_all_and_refreshes_markers(synth):
    train_chii.preprocess(runner=synth.runner)
    forced = FakeRunner()
    train_chii.preprocess(force=True, runner=forced)
    assert forced.calls == ALL_SCRIPTS
    # force 跑完仍落标记: 之后恢复幂等跳过
    third = FakeRunner()
    train_chii.preprocess(runner=third)
    assert third.calls == []


def test_marker_records_input_hash(synth):
    train_chii.preprocess(runner=synth.runner)
    with open(os.path.join(synth.opt_dir, ".done-1-text.json"), encoding="utf-8") as f:
        data = json.load(f)
    assert data["inputs_sha256"] == train_chii._inputs_hash(
        [train_chii.LIST_PATH, train_chii.WAV_DIR])
    assert data["step"] == "1-text"


def test_step_done_requires_all_three(synth):
    inputs = [train_chii.LIST_PATH, train_chii.WAV_DIR]
    outputs = [os.path.join(synth.opt_dir, "2-name2text.txt")]
    train_chii._mark_step_done("unit", inputs)
    # 标记在, 但产物不存在 → 未完成
    assert not train_chii._step_done("unit", inputs, outputs)
    with open(outputs[0], "w", encoding="utf-8") as f:
        f.write("x")
    # 标记 + 产物 + 哈希一致 → 完成
    assert train_chii._step_done("unit", inputs, outputs)
    # 哈希不一致 (输入变更) → 未完成
    other = inputs + [os.path.join(synth.opt_dir, "nope")]
    assert not train_chii._step_done("unit", other, outputs)
