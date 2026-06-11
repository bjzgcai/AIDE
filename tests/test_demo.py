import tempfile
import unittest
from pathlib import Path

from aide.demo import run_demo


class DemoTests(unittest.TestCase):
    def test_offline_demo_writes_expected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            summary = run_demo(tmp_path)

            self.assertEqual(summary["collection_candidates"], 2)
            self.assertEqual(summary["selected_datasets"], 1)
            self.assertEqual(summary["rag_items"], 2)
            self.assertEqual(summary["instruction_items"], 2)
            self.assertTrue((tmp_path / "collection" / "candidates.json").exists())
            self.assertTrue(
                (tmp_path / "selection" / "selected_datasets.json").exists()
            )
            self.assertTrue((tmp_path / "organization" / "rag_corpus.jsonl").exists())
            self.assertTrue(
                (tmp_path / "organization" / "instruction_data.jsonl").exists()
            )


if __name__ == "__main__":
    unittest.main()
