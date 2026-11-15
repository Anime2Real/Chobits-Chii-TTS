"""tools/review_text.py 的文本修正、落盘与本地端点测试."""

import csv
import json
import sys
import tempfile
import threading
import unittest
from array import array
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import review_text
import trim_dataset


class TextReviewStoreTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp.name) / "trimmed"
        (self.data_dir / "manual").mkdir(parents=True)
        (self.data_dir / "reviewed" / "wavs").mkdir(parents=True)
        tone = array("h", [4000, -4000] * 1000)
        for name in ("one", "two"):
            trim_dataset.write_wav(
                self.data_dir / "reviewed" / "wavs" / f"{name}.wav", tone, 1000)
        self.queue_path = self.data_dir / "manual" / "text_review.csv"
        with open(self.queue_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["priority", "file", "current_text", "reason",
                               "verified_text", "action", "reviewed_at"])
            writer.writeheader()
            writer.writerows([
                {"priority": "P0", "file": "one", "current_text": "ふ六",
                 "reason": "异常", "verified_text": "", "action": "", "reviewed_at": ""},
                {"priority": "P1", "file": "two", "current_text": "チー",
                 "reason": "称呼", "verified_text": "", "action": "", "reviewed_at": ""},
            ])
        self.store = review_text.TextReviewStore(self.data_dir)

    def tearDown(self):
        self.temp.cleanup()

    def test_state_and_fix_keep_drop_reset(self):
        self.assertEqual(self.store.state()["stats"]["pending"], 2)

        row = self.store.save("one", "fix", "ふぅ")
        self.assertEqual(row["action"], "fix")
        self.assertEqual(row["verified_text"], "ふぅ")
        self.assertTrue(row["reviewed_at"])
        self.store.save("two", "keep", "")
        self.assertEqual(self.store.state()["stats"]["fix"], 1)
        self.assertEqual(self.store.state()["stats"]["keep"], 1)
        self.assertEqual(self.store.state()["stats"]["pending"], 0)

        reloaded = review_text.TextReviewStore(self.data_dir)
        self.assertEqual(reloaded.by_file["one"]["verified_text"], "ふぅ")

        self.store.reset("one")
        self.assertFalse(self.store.by_file["one"]["action"])
        self.assertEqual(self.store.state()["stats"]["pending"], 1)

        self.store.save("one", "drop", "")
        self.assertEqual(self.store.by_file["one"]["action"], "drop")

    def test_reviewed_audio_falls_back_for_excluded_clip(self):
        reviewed = self.data_dir / "reviewed" / "wavs" / "one.wav"
        fallback = self.data_dir / "manual" / "wavs" / "one.wav"
        fallback.parent.mkdir(parents=True, exist_ok=True)
        reviewed.replace(fallback)
        store = review_text.TextReviewStore(self.data_dir)
        self.assertEqual(store.state()["stats"]["total"], 2)
        self.assertEqual(store._audio_path("one"), fallback)

    def test_asr_candidates_are_exposed_in_state(self):
        path = self.data_dir / "manual" / "asr_candidates.csv"
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["file", "priority", "current_text", "current_zh",
                               "qwen_text", "qwen_zh", "whisper_text", "whisper_zh",
                               "agreement", "confidence", "consensus_text", "notes"])
            writer.writeheader()
            writer.writerow({
                "file": "one", "priority": "P0", "current_text": "ふ六",
                "current_zh": "不对", "qwen_text": "ふぅ", "qwen_zh": "呼",
                "whisper_text": "ふう", "whisper_zh": "呼", "agreement": "0.67",
                "confidence": "medium", "consensus_text": "ふぅ",
                "notes": "models_disagree",
            })
        store = review_text.TextReviewStore(self.data_dir)
        rows = {r["file"]: r for r in store.state()["records"]}
        self.assertEqual(rows["one"]["asr"]["qwen_text"], "ふぅ")
        self.assertEqual(rows["one"]["asr"]["qwen_zh"], "呼")
        self.assertIsNone(rows["two"]["asr"])

    def test_invalid_fix_text_is_rejected(self):
        with self.assertRaises(ValueError):
            self.store.save("one", "fix", "")
        with self.assertRaises(ValueError):
            self.store.save("one", "fix", "ふ六")
        with self.assertRaises(ValueError):
            self.store.save("one", "fix", "ふ|六")
        with self.assertRaises(KeyError):
            self.store.save("missing", "keep", "")

    def test_http_state_and_range_audio(self):
        review_text.TextReviewHandler.store = self.store
        review_text.TextReviewHandler.ui_bytes = b"<html>ok</html>"
        server = ThreadingHTTPServer(("127.0.0.1", 0), review_text.TextReviewHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(base + "/api/state", timeout=5) as response:
                payload = json.load(response)
            self.assertEqual(payload["stats"]["total"], 2)
            request = Request(base + "/api/audio/one", headers={"Range": "bytes=0-99"})
            with urlopen(request, timeout=5) as response:
                self.assertEqual(response.status, 206)
                self.assertEqual(len(response.read()), 100)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
