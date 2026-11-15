"""tools/trim_dataset.py 的保守切边、静音检测与导出测试."""

import json
import sys
import tempfile
import unittest
from array import array
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import trim_dataset


SR = 1000
TONE = array("h", [8000, -8000] * 200)
LOW_TONE = array("h", [300, -300] * 200)


def silence(seconds):
    return array("h", [0] * round(SR * seconds))


def tone(seconds, samples=TONE):
    count = round(SR * seconds)
    return array("h", (samples[i % len(samples)] for i in range(count)))


class TrimDatasetTest(unittest.TestCase):
    def test_edge_trim_keeps_buffer_and_reports_internal_silence(self):
        samples = silence(0.5) + tone(0.5) + silence(1.0) + tone(0.5) + silence(0.4)
        plan = trim_dataset.plan_edge_trim(samples, SR, -50.0, 20, 0.12)
        self.assertFalse(plan.near_silent)
        self.assertAlmostEqual(plan.start / SR, 0.38, places=2)
        self.assertAlmostEqual(plan.end / SR, 2.62, places=2)

        clipped = samples[plan.start:plan.end]
        runs = trim_dataset.find_silence_runs(clipped, SR, -50.0, 20, 0.8)
        self.assertEqual(len(runs), 1)
        self.assertAlmostEqual(runs[0].start, 0.62, places=2)
        self.assertAlmostEqual(runs[0].duration, 1.0, places=2)

    def test_all_silent_is_flagged_without_trimming(self):
        samples = silence(2.0)
        plan = trim_dataset.plan_edge_trim(samples, SR, -50.0, 20, 0.12)
        self.assertTrue(plan.near_silent)
        self.assertEqual((plan.start, plan.end), (0, len(samples)))

    def test_short_pause_is_not_reported(self):
        samples = tone(0.5) + silence(0.4) + tone(0.5)
        runs = trim_dataset.find_silence_runs(samples, SR, -50.0, 20, 0.8)
        self.assertEqual(runs, [])

    def test_trim_dataset_writes_derived_files_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "wavs"
            dst = root / "trimmed"
            src.mkdir()
            for name, samples in {
                "normal": silence(0.5) + tone(0.5) + silence(1.0) + tone(0.5) + silence(0.4),
                "quiet": silence(2.0),
                "low": tone(0.5, LOW_TONE),
            }.items():
                trim_dataset.write_wav(src / f"{name}.wav", samples, SR)
            original = {p.name: p.read_bytes() for p in src.glob("*.wav")}
            metadata = root / "metadata.csv"
            metadata.write_text("normal|こんにちは\nquiet|テスト\nlow|ささやき\n", encoding="utf-8")

            args = SimpleNamespace(
                src=str(src), metadata=str(metadata), dst=str(dst),
                silence_db=-50.0, frame_ms=20, edge_buffer=0.12, review_internal=0.8,
                low_level_db=-40.0, min_duration=0.5,
            )
            summary = trim_dataset.trim_dataset(args)

            self.assertEqual(summary["clips"], 3)
            self.assertEqual(summary["near_silent"], 1)
            self.assertEqual(summary["review"], 3)
            self.assertEqual({p.name: p.read_bytes() for p in src.glob("*.wav")}, original)
            self.assertTrue((dst / "wavs" / "normal.wav").is_file())
            self.assertTrue((dst / "wavs" / "quiet.wav").is_file())
            self.assertEqual(
                (dst / "metadata.csv").read_text(encoding="utf-8"),
                "normal|こんにちは\nquiet|テスト\nlow|ささやき\n",
            )
            listing = (dst / "gpt_sovits.list").read_text(encoding="utf-8").splitlines()
            self.assertTrue(listing[0].startswith(f"{(dst / 'wavs').resolve()}"))
            self.assertIn("|chii|ja|こんにちは", listing[0])

            with open(dst / "review_queue.csv", encoding="utf-8") as f:
                import csv
                review = list(csv.DictReader(f))
            by_file = {row["file"]: row for row in review}
            self.assertEqual(by_file["quiet"]["priority"], "P0")
            self.assertIn("near_silent", by_file["quiet"]["reasons"])
            self.assertIn("low_level", by_file["low"]["reasons"])
            intervals = json.loads(by_file["normal"]["internal_intervals"])
            self.assertEqual(len(intervals), 1)
            self.assertAlmostEqual(intervals[0]["duration"], 1.0, places=2)

            first_run = trim_dataset.trim_dataset(args)
            self.assertEqual(first_run["clips"], 3)


if __name__ == "__main__":
    unittest.main()
