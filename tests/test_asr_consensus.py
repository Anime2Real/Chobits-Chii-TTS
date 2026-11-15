"""tools/asr_consensus.py 的纯逻辑测试 (不加载模型)."""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import asr_consensus


class AsrConsensusTest(unittest.TestCase):
    def test_normalize_ja_ignores_width_case_and_punctuation(self):
        self.assertEqual(
            asr_consensus.normalize_ja("チー！　Chii。"),
            "チーchii",
        )

    def test_similarity_exact_and_partial(self):
        self.assertEqual(asr_consensus.text_similarity("ちぃ、好き", "ちぃ好き"), 1.0)
        self.assertLess(asr_consensus.text_similarity("ちぃ", "秀樹"), 0.3)
        self.assertAlmostEqual(
            asr_consensus.text_similarity("あいう", "あいえ"), 2 / 3, places=4)

    def test_choose_consensus_uses_qwen_as_primary(self):
        text, similarity, confidence = asr_consensus.choose_consensus(
            "ちぃ、秀樹のこと好き", "ちぃ、秀樹のことが好き")
        self.assertEqual(text, "ちぃ、秀樹のこと好き")
        self.assertGreater(similarity, 0.7)
        self.assertIn(confidence, {"high", "medium"})

    def test_choose_consensus_handles_missing_model(self):
        text, similarity, confidence = asr_consensus.choose_consensus("", "ひでき")
        self.assertEqual((text, similarity, confidence), ("ひでき", 0.0, "low"))

    def test_build_rows_adds_notes_and_translation(self):
        queue = [{"file": "a", "priority": "P0", "current_text": "ふ六"}]
        translations = {"ふ六": "不对", "ふぅ": "呼", "ふう": "呼"}
        rows = asr_consensus.build_rows(
            queue, Path("."), {"a": "ふぅ"}, {"a": "ふう"}, translations)
        row = rows[0]
        self.assertEqual(row["confidence"], "low")
        self.assertIn("models_disagree", row["notes"])
        self.assertEqual(row["qwen_zh"], "呼")
        self.assertEqual(row["current_zh"], "不对")
        self.assertEqual(row["consensus_text"], "ふぅ")


if __name__ == "__main__":
    unittest.main()
