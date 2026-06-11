import json
import tempfile
import unittest
from pathlib import Path

from aide.dedup import (
    collect_instruction_pairs,
    dedup_datasets,
    write_instruction_jsonl,
)


class DedupTests(unittest.TestCase):
    def test_dedup_skips_non_dataset_artifacts_and_exports_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "log.txt").write_text("not a dataset", encoding="utf-8")
            (root / "cache").mkdir()
            (root / "cache" / "metadata").mkdir()

            task_dir = root / "Dataset_A" / "task_one"
            task_dir.mkdir(parents=True)
            (task_dir / "train.json").write_text(
                json.dumps(["Question one?\tAnswer one.", "not a pair"]),
                encoding="utf-8",
            )

            summary_path = root / "dedup_summary.txt"
            dedup_datasets(str(root), str(summary_path))

            self.assertTrue((root / "Dataset_A" / "dedup.json").exists())
            self.assertFalse((root / "cache" / "dedup.json").exists())

            sentences = collect_instruction_pairs(str(root))
            self.assertEqual(sentences, ["Question one?\tAnswer one."])

            train_output = root / "instruction_train.jsonl"
            val_output = root / "instruction_val.jsonl"
            write_instruction_jsonl(
                sentences,
                train_output_path=str(train_output),
                val_output_path=str(val_output),
                val_ratio=0,
                seed=0,
            )

            self.assertEqual(
                json.loads(train_output.read_text(encoding="utf-8").strip()),
                {"instruction": "Question one?", "output": "Answer one."},
            )
            self.assertEqual(val_output.read_text(encoding="utf-8"), "")

    def test_force_rebuilds_stale_dataset_dedup_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "Dataset_A"
            task_dir = dataset_dir / "task_one"
            task_dir.mkdir(parents=True)
            (task_dir / "train.json").write_text(
                json.dumps(["Fresh question?\tFresh answer."]),
                encoding="utf-8",
            )
            (dataset_dir / "dedup.json").write_text(
                json.dumps(["Stale question?\tStale answer."]),
                encoding="utf-8",
            )

            dedup_datasets(
                str(root), str(root / "dedup_summary.txt"), threshold=0.7, force=True
            )

            self.assertEqual(
                json.loads((dataset_dir / "dedup.json").read_text(encoding="utf-8")),
                ["Fresh question?\tFresh answer."],
            )


if __name__ == "__main__":
    unittest.main()
