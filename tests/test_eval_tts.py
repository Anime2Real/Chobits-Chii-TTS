"""tools/eval_tts.py 的 manifest 字段测试."""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

try:
    import eval_tts
except ModuleNotFoundError:
    eval_tts = None


@unittest.skipUnless(eval_tts, "eval_tts 依赖 soundfile/yaml (.venv-train)")
class EvalTtsTest(unittest.TestCase):
    def test_manifest_entry_drops_extra_prompt_columns(self):
        prompt = {
            "id": "p101",
            "category": "transcript_ep16",
            "text": "わからない。",
            "ep": "ep16",
            "chi_prob": "0.917",
            "source_dur": "2.0",
        }
        self.assertEqual(
            eval_tts.manifest_entry(prompt, "p101-transcript_ep16.wav", 32000),
            {
                "id": "p101",
                "category": "transcript_ep16",
                "text": "わからない。",
                "wav": "p101-transcript_ep16.wav",
                "sample_rate": 32000,
            },
        )


if __name__ == "__main__":
    unittest.main()
