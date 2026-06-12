import tempfile
import unittest

from aide import AIDEConfig, DatasetInfo
from aide.pipeline import AIDEPipeline
from aide.utils import save_json


def _dataset(index: int) -> DatasetInfo:
    return DatasetInfo(
        name=f"demo/dataset-{index}",
        subset="default",
        number_of_rows=10,
        columns=["text"],
        description="Demo dataset",
        hf_tags=[],
        modalities=["text"],
        license="mit",
        n_likes=0,
        n_downloads_last_month=0,
    )


class PipelineTests(unittest.TestCase):
    def test_cached_dataset_and_selection_lists_respect_current_max(self) -> None:
        candidates = [_dataset(i) for i in range(5)]

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = f"{tmpdir}/cache"
            save_json(
                [dataset.__dict__ for dataset in candidates],
                f"{cache_dir}/datasets.json",
            )
            save_json(
                [dataset.__dict__ for dataset in candidates],
                f"{cache_dir}/selection.json",
            )

            config = AIDEConfig(
                prompt="demo",
                output_dir=tmpdir,
                cache_dir=f"{tmpdir}/hf-cache",
                max_datasets=2,
                enable_stage_metrics=False,
            )
            pipeline = AIDEPipeline(config)
            pipeline.keyword_generator.generate_keywords = lambda _query: ["demo"]

            selected_input_sizes = []

            def fake_select(datasets, _user_query):
                selected_input_sizes.append(len(datasets))
                return datasets

            processed = []
            pipeline.select_datasets = fake_select

            def fake_organize(dataset):
                processed.append(dataset)
                return True

            pipeline.organize_dataset = fake_organize

            pipeline.run()

        self.assertEqual([2], selected_input_sizes)
        self.assertEqual(
            ["demo/dataset-0", "demo/dataset-1"],
            [dataset.name for dataset in processed],
        )


if __name__ == "__main__":
    unittest.main()
