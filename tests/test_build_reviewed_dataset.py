"""tools/build_reviewed_dataset.py 的最终训练集合成测试."""

import contextlib
import csv
import io
import sys
import tempfile
import unittest
import wave
from array import array
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import build_reviewed_dataset
import review_clips
import trim_dataset


SR = 1000
TONE = array("h", [8000, -8000] * 200)
LOW_TONE = array("h", [300, -300] * 200)


def silence(seconds):
    return array("h", [0] * round(SR * seconds))


def tone(seconds, samples=TONE):
    count = round(SR * seconds)
    return array("h", (samples[i % len(samples)] for i in range(count)))


class BuildReviewedDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.src = self.root / "wavs"
        self.src.mkdir()
        samples = {
            "normal": silence(0.5) + tone(0.5) + silence(1.0) + tone(0.5) + silence(0.4),
            "quiet": silence(2.0),
            "low": tone(0.5, LOW_TONE),
            "clean": tone(1.0),
        }
        for name, pcm in samples.items():
            trim_dataset.write_wav(self.src / f"{name}.wav", pcm, SR)
        metadata = self.root / "metadata.csv"
        metadata.write_text(
            "normal|こんにちは\nquiet|テスト\nlow|ささやき\nclean|おはよう\n",
            encoding="utf-8",
        )
        self.data_dir = self.root / "trimmed"
        args = SimpleNamespace(
            src=str(self.src), metadata=str(metadata), dst=str(self.data_dir),
            silence_db=-50.0, frame_ms=20, edge_buffer=0.12, review_internal=0.8,
            low_level_db=-40.0, min_duration=0.5,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            trim_dataset.trim_dataset(args)
        self.store = review_clips.ReviewStore(self.data_dir)

    def tearDown(self):
        self.temp.cleanup()

    def _review_all(self):
        normal = self.store.records["normal"]
        self.store.save("normal", "edit", 0, normal["duration"], normal["internal"])
        self.store.save("quiet", "drop", 0, 0, [])
        self.store.save("low", "keep", 0, 0, [])

    def test_merge_uses_manual_overrides_drops_and_auto_fallback(self):
        self._review_all()
        dst = self.data_dir / "reviewed"
        summary = build_reviewed_dataset.assemble(self.data_dir, dst)
        self.assertEqual(summary["output_clips"], 3)
        self.assertEqual(summary["manual"], 2)
        self.assertEqual(summary["auto"], 1)
        self.assertEqual(summary["dropped"], 1)

        with open(dst / "metadata.csv", encoding="utf-8") as f:
            metadata = [line.rstrip("\n") for line in f]
        self.assertEqual(metadata, ["normal|こんにちは", "low|ささやき", "clean|おはよう"])
        with open(dst / "build_report.csv", encoding="utf-8", newline="") as f:
            report = {row["file"]: row for row in csv.DictReader(f)}
        self.assertEqual(report["normal"]["source"], "manual_edit")
        self.assertEqual(report["low"]["source"], "manual_keep")
        self.assertEqual(report["clean"]["source"], "auto")

        with open(dst / "gpt_sovits.list", encoding="utf-8") as f:
            listing = [line.rstrip("\n") for line in f]
        self.assertEqual(len(listing), 3)
        for line in listing:
            path = Path(line.split("|", 1)[0])
            self.assertTrue(path.is_file(), path)
            self.assertEqual(path.parent, dst / "wavs")
            with wave.open(str(path), "rb") as w:
                self.assertEqual(w.getframerate(), SR)

        with wave.open(str(dst / "wavs" / "normal.wav"), "rb") as w:
            normal_duration = w.getnframes() / w.getframerate()
        self.assertLess(normal_duration, self.store.records["normal"]["duration"])

    def _write_text_review(self, rows):
        path = self.data_dir / "manual" / "text_review.csv"
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["priority", "file", "current_text", "reason",
                               "verified_text", "action", "reviewed_at"])
            writer.writeheader()
            writer.writerows(rows)

    def test_text_fix_keep_and_drop_are_applied(self):
        self._review_all()
        self._write_text_review([
            {"priority": "P0", "file": "normal", "current_text": "こんにちは",
             "reason": "x", "verified_text": "こんにちは、ちぃ", "action": "fix",
             "reviewed_at": "now"},
            {"priority": "P1", "file": "low", "current_text": "ささやき",
             "reason": "x", "verified_text": "ちいさなこえ", "action": "drop",
             "reviewed_at": "now"},
            {"priority": "P1", "file": "clean", "current_text": "おはよう",
             "reason": "x", "verified_text": "", "action": "keep",
             "reviewed_at": "now"},
        ])
        dst = self.data_dir / "reviewed"
        summary = build_reviewed_dataset.assemble(self.data_dir, dst)
        self.assertEqual(summary["output_clips"], 2)
        self.assertEqual(summary["text_dropped"], 1)
        self.assertEqual(summary["text_pending"], 0)
        with open(dst / "metadata.csv", encoding="utf-8") as f:
            metadata = [line.rstrip("\n") for line in f]
        self.assertEqual(metadata, ["normal|こんにちは、ちぃ", "clean|おはよう"])
        with open(dst / "gpt_sovits.list", encoding="utf-8") as f:
            listing = [line.rstrip("\n") for line in f]
        self.assertIn("|chii|ja|こんにちは、ちぃ", listing[0])
        with open(dst / "build_report.csv", encoding="utf-8", newline="") as f:
            report = {row["file"]: row for row in csv.DictReader(f)}
        self.assertEqual(report["normal"]["text_action"], "fix")
        self.assertEqual(report["clean"]["text_action"], "keep")

    def test_require_text_reviewed_blocks_blank_action(self):
        self._review_all()
        self._write_text_review([
            {"priority": "P0", "file": "normal", "current_text": "こんにちは",
             "reason": "x", "verified_text": "", "action": "",
             "reviewed_at": ""},
        ])
        with self.assertRaisesRegex(ValueError, "文本未复核"):
            build_reviewed_dataset.assemble(
                self.data_dir, self.data_dir / "reviewed", require_text_reviewed=True)

    def test_pending_review_blocks_by_default(self):
        with self.assertRaisesRegex(ValueError, "未决定"):
            build_reviewed_dataset.assemble(self.data_dir, self.data_dir / "reviewed")
        summary = build_reviewed_dataset.assemble(
            self.data_dir, self.data_dir / "reviewed", allow_pending=True
        )
        self.assertEqual(summary["auto"], 4)

    def test_missing_manual_audio_blocks(self):
        self._review_all()
        (self.data_dir / "manual" / "wavs" / "normal.wav").unlink()
        with self.assertRaisesRegex(ValueError, "缺少人工音频"):
            build_reviewed_dataset.assemble(self.data_dir, self.data_dir / "reviewed")

    def test_unknown_decision_blocks(self):
        self._review_all()
        path = self.data_dir / "manual" / "decisions.csv"
        with open(path, "a", encoding="utf-8") as f:
            f.write("missing.wav,keep,0,0,[],0,\n")
        with self.assertRaisesRegex(ValueError, "未知音频"):
            build_reviewed_dataset.assemble(self.data_dir, self.data_dir / "reviewed")


if __name__ == "__main__":
    unittest.main()
