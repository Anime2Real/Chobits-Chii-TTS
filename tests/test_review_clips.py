"""tools/review_clips.py 的剪辑、决策落盘与本地 HTTP 端点测试."""

import contextlib
import csv
import io
import json
import sys
import tempfile
import threading
import unittest
import wave
from array import array
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import review_clips
import trim_dataset


SR = 1000
TONE = array("h", [8000, -8000] * 200)


def silence(seconds):
    return array("h", [0] * round(SR * seconds))


def tone(seconds):
    return array("h", (TONE[i % len(TONE)] for i in range(round(SR * seconds))))


def wav_info(payload):
    with wave.open(io.BytesIO(payload), "rb") as w:
        return w.getframerate(), w.getnframes()


class ReviewStoreTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.src = self.root / "wavs"
        self.src.mkdir()
        trim_dataset.write_wav(
            self.src / "normal.wav",
            silence(0.5) + tone(0.5) + silence(1.0) + tone(0.5) + silence(0.4),
            SR,
        )
        trim_dataset.write_wav(self.src / "quiet.wav", silence(2.0), SR)
        metadata = self.root / "metadata.csv"
        metadata.write_text("normal|こんにちは\nquiet|テスト\n", encoding="utf-8")
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

    def test_state_marks_both_review_rows(self):
        state = self.store.state()
        self.assertEqual(state["stats"]["needs_review"], 2)
        self.assertEqual(state["stats"]["pending"], 2)
        normal = next(item for item in state["records"] if item["file"] == "normal")
        quiet = next(item for item in state["records"] if item["file"] == "quiet")
        self.assertEqual(normal["priority"], "P1")
        self.assertEqual(len(normal["internal"]), 1)
        self.assertEqual(quiet["priority"], "P0")

    def test_preview_removes_checked_pause_with_padding(self):
        record = self.store.records["normal"]
        payload = self.store.preview("normal", 0, record["duration"], record["internal"])
        sample_rate, frames = wav_info(payload)
        self.assertEqual(sample_rate, SR)
        expected = record["duration"] - (record["internal"][0]["duration"] - 0.15)
        self.assertAlmostEqual(frames / sample_rate, expected, places=2)

    def test_save_edit_persists_audio_and_decision(self):
        record = self.store.records["normal"]
        start, end = 0.1, record["duration"] - 0.1
        decision = self.store.save("normal", "edit", start, end, [record["internal"][0]])
        self.assertEqual(decision["action"], "edit")
        self.assertTrue((self.data_dir / "manual" / "wavs" / "normal.wav").is_file())
        with open(self.data_dir / "manual" / "decisions.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(rows[0]["file"], "normal")
        self.assertAlmostEqual(json.loads(rows[0]["removed_intervals"])[0]["start"], 0.62, places=2)

        reloaded = review_clips.ReviewStore(self.data_dir)
        self.assertEqual(reloaded.state()["stats"]["reviewed"], 1)
        self.assertEqual(reloaded.state()["stats"]["pending"], 1)

    def test_keep_drop_and_reset(self):
        self.store.save("normal", "keep", 0, 0, [])
        self.assertTrue((self.data_dir / "manual" / "wavs" / "normal.wav").is_file())
        self.store.save("quiet", "drop", 0, 0, [])
        self.assertFalse((self.data_dir / "manual" / "wavs" / "quiet.wav").exists())
        self.assertEqual(self.store.decisions["quiet"]["action"], "drop")

        self.store.reset("normal")
        self.assertNotIn("normal", self.store.decisions)
        self.assertFalse((self.data_dir / "manual" / "wavs" / "normal.wav").exists())

    def test_invalid_edit_is_rejected(self):
        with self.assertRaises(ValueError):
            self.store.preview("normal", 0.5, 0.2, [])
        with self.assertRaises(KeyError):
            self.store.preview("missing.wav", 0, 1, [])

    def test_http_state_and_range_audio(self):
        review_clips.ReviewHandler.store = self.store
        review_clips.ReviewHandler.ui_bytes = b"<html>ok</html>"
        server = ThreadingHTTPServer(("127.0.0.1", 0), review_clips.ReviewHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(base + "/api/state", timeout=5) as response:
                payload = json.load(response)
            self.assertEqual(payload["stats"]["needs_review"], 2)

            request = Request(base + "/api/audio/normal", headers={"Range": "bytes=0-99"})
            with urlopen(request, timeout=5) as response:
                self.assertEqual(response.status, 206)
                self.assertEqual(len(response.read()), 100)
                self.assertIn("Content-Range", response.headers)

            request = Request(base + "/api/audio/normal", headers={"Range": "bytes=-100"})
            with urlopen(request, timeout=5) as response:
                self.assertEqual(response.status, 206)
                self.assertEqual(len(response.read()), 100)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
