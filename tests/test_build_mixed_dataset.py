"""tools/build_mixed_dataset.py 的排序与排除列表测试."""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import build_mixed_dataset


class BuildMixedDatasetTest(unittest.TestCase):
    def test_candidate_sort_prefers_score_then_seed44(self):
        rows = [
            {"candidate_id": "a", "score": "0.90", "seed": "43"},
            {"candidate_id": "b", "score": "0.90", "seed": "44"},
            {"candidate_id": "c", "score": "0.95", "seed": "42"},
        ]
        ordered = sorted(rows, key=build_mixed_dataset.candidate_sort_key, reverse=True)
        self.assertEqual([row["candidate_id"] for row in ordered], ["c", "b", "a"])

    def test_read_exclusions_accepts_header_and_plain_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            header = root / "header.tsv"
            with open(header, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, delimiter="\t", lineterminator="\n")
                writer.writerow(["candidate_id", "reason"])
                writer.writerow(["p123_s44", "bad"])
            self.assertEqual(build_mixed_dataset.read_exclusions(header), {"p123_s44"})

            plain = root / "plain.txt"
            plain.write_text("# comment\np124_s42\np125_s43\n", encoding="utf-8")
            self.assertEqual(
                build_mixed_dataset.read_exclusions(plain), {"p124_s42", "p125_s43"}
            )


if __name__ == "__main__":
    unittest.main()
