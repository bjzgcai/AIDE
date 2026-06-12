import unittest
from types import SimpleNamespace
from unittest.mock import patch

from aide.collection_search import HFDatasetCollector


def _hub_dataset_info(dataset_id: str):
    return SimpleNamespace(
        id=dataset_id,
        description=f"{dataset_id} description",
        cardData={},
        likes=0,
        downloads=0,
    )


def _builder():
    return SimpleNamespace(
        info=SimpleNamespace(
            splits={"train": SimpleNamespace(num_examples=5)},
            features={"text": object()},
        )
    )


class CollectionSearchTests(unittest.TestCase):
    def test_global_max_datasets_stops_before_next_keyword(self) -> None:
        list_calls = []

        def fake_list_datasets(search, **_kwargs):
            list_calls.append(search)
            return [SimpleNamespace(id=f"demo/{search}")]

        with (
            patch(
                "aide.collection_search._safe_list_datasets",
                new=fake_list_datasets,
            ),
            patch(
                "aide.collection_search._safe_dataset_info",
                new=lambda dataset_id, **_kwargs: _hub_dataset_info(dataset_id),
            ),
            patch(
                "aide.collection_search._safe_get_dataset_config_names",
                new=lambda **_kwargs: ["default"],
            ),
            patch(
                "aide.collection_search._safe_load_dataset_builder",
                new=lambda *_args, **_kwargs: _builder(),
            ),
        ):
            results = HFDatasetCollector(cache_dir="cache").search(
                ["alpha", "beta"],
                cache_dir="cache",
                max_results=1,
                max_results_per_term=1,
            )

        self.assertEqual(1, len(results))
        self.assertEqual("demo/alpha", results[0].name)
        self.assertEqual(["alpha"], list_calls)

    def test_process_dataset_config_normalizes_missing_metadata(self) -> None:
        hub_info = _hub_dataset_info("demo/null-description")
        hub_info.description = None
        hub_info.cardData = {
            "license": None,
            "tags": None,
            "modalities": "text",
        }

        with patch(
            "aide.collection_search._safe_load_dataset_builder",
            new=lambda *_args, **_kwargs: _builder(),
        ):
            result = HFDatasetCollector(cache_dir="cache")._process_dataset_config(
                hub_info, "default"
            )

        self.assertIsNotNone(result)
        self.assertEqual("", result.description)
        self.assertEqual("N/A", result.license)
        self.assertEqual([], result.hf_tags)
        self.assertEqual(["text"], result.modalities)


if __name__ == "__main__":
    unittest.main()
