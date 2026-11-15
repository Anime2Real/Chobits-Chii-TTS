"""tools/build_synth_prompts.py 的文本过滤与挑选测试."""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import build_synth_prompts


class BuildSynthPromptsTest(unittest.TestCase):
    def test_normalize_only_fixes_known_transcription_errors(self):
        self.assertEqual(build_synth_prompts.normalize_text("秀吉とチー"), "秀樹とチー")
        self.assertEqual(build_synth_prompts.normalize_text("きぃ、おはよう"), "ちぃ、おはよう")
        self.assertEqual(build_synth_prompts.normalize_text("ひでき  元気"), "ひでき 元気")

    def test_rejects_male_speech_fragments_and_bad_chars(self):
        for text in ("俺の名前は秀樹", "これは", "ABC", "うおおおお", "あっちあっちあっちあっち", "夜のみだらな人妻"):
            self.assertFalse(build_synth_prompts.is_valid_text(text), text)
        for text in ("ひでき、どうしたの?", "私は旅に出る", "ちぃ、バイト行ってくる"):
            self.assertTrue(build_synth_prompts.is_valid_text(text), text)

    def test_selection_respects_episode_cap_and_orders_by_chi(self):
        candidates = [
            {"text": "おはようございます", "ep": "ep01", "chi_prob": 0.9, "source_dur": "1"},
            {"text": "いってきます", "ep": "ep01", "chi_prob": 0.8, "source_dur": "1"},
            {"text": "ただいま帰りました", "ep": "ep02", "chi_prob": 0.7, "source_dur": "1"},
            {"text": "今日はいい天気", "ep": "ep03", "chi_prob": 0.6, "source_dur": "1"},
        ]
        selected, counts = build_synth_prompts.select_prompts(candidates, target=3, per_episode_cap=1)
        self.assertEqual([row["ep"] for row in selected], ["ep01", "ep02", "ep03"])
        self.assertEqual(counts, {"ep01": 1, "ep02": 1, "ep03": 1})


if __name__ == "__main__":
    unittest.main()
