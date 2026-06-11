import unittest
from unittest.mock import patch

from aide import DatasetInfo
from aide.selection import load_dataset_safe


def _dataset_info() -> DatasetInfo:
    return DatasetInfo(
        name="demo/science",
        subset="default",
        number_of_rows=2,
        columns=["question", "answer"],
        description="Toy scientific QA dataset",
        hf_tags=["science"],
        modalities=["text"],
        license="mit",
        n_likes=1,
        n_downloads_last_month=1,
    )


class SelectionTests(unittest.TestCase):
    def test_fallback_dataset_loading_uses_current_datasets_api(self) -> None:
        call_kwargs = {}

        def fake_load_dataset(**kwargs):
            call_kwargs.update(kwargs)
            return []

        with patch("aide.selection.load_dataset", new=fake_load_dataset):
            load_dataset_safe(_dataset_info(), "cache/huggingface")

        self.assertEqual(
            {
                "path",
                "name",
                "split",
                "cache_dir",
                "streaming",
            },
            set(call_kwargs),
        )
        self.assertTrue(call_kwargs["streaming"])


if __name__ == "__main__":
    unittest.main()
