import os

from datasets import get_dataset_config_names, load_dataset_builder
from huggingface_hub import DatasetInfo as HfHubDatasetInfo
from huggingface_hub import dataset_info, list_datasets
from loguru import logger

from .models import DatasetInfo
from .utils import make_safe_call


def _is_non_retryable_dataset_structure_error(exc: Exception) -> bool:
    error_text = str(exc).lower()
    non_retryable_patterns = [
        "no (supported) data files found",
        "couldn't infer the same data file format for all splits",
        "must be called with a dataclass type or instance",
        "couldn't find any data file",
        "doesn't contain any data files",
        "does not contain any data files",
        "dataset scripts are no longer supported",
        "feature type",
        "not found. available feature types",
    ]
    return any(pattern in error_text for pattern in non_retryable_patterns)


# Wrapper functions with the decorator applied
@make_safe_call(
    allow_failure=True,
    giveup=_is_non_retryable_dataset_structure_error,
)
def _safe_load_dataset_builder(*args, **kwargs):
    return load_dataset_builder(*args, **kwargs)


@make_safe_call(allow_failure=True, max_tries=3)
def _safe_list_datasets(*args, **kwargs):
    return list_datasets(*args, **kwargs)


@make_safe_call(allow_failure=True)
def _safe_dataset_info(*args, **kwargs):
    return dataset_info(*args, **kwargs)


@make_safe_call(
    allow_failure=True,
    giveup=_is_non_retryable_dataset_structure_error,
)
def _safe_get_dataset_config_names(*args, **kwargs):
    return get_dataset_config_names(*args, **kwargs)


class HFDatasetCollector:
    """
    Search the dataset in HF Hub.
    """

    def __init__(self, cache_dir: str = ""):
        self.cache_dir = cache_dir
        self.hf_token = os.getenv("HF_TOKEN") or None
        if not self.hf_token:
            logger.warning("HF_TOKEN is not set.")

    def _process_dataset_config(
        self, hf_hub_ds_info: HfHubDatasetInfo, config: str, split: str = "train"
    ) -> DatasetInfo | None:
        """
        Process a single dataset configuration and return its DatasetInfo.

        Args:
            ds_info: Dataset info from HuggingFace Hub
            config: Dataset configuration name

        Returns:
            DatasetInfo object or None if processing fails
        """
        ds_builder = _safe_load_dataset_builder(
            hf_hub_ds_info.id,
            name=config,
            cache_dir=self.cache_dir,
            token=self.hf_token,
        )
        if not ds_builder:
            logger.warning(
                f"Failed to load dataset builder for {hf_hub_ds_info.id} - {config}"
            )
            return None

        ds_info = ds_builder.info

        # Get number of examples (from dataset info if available, otherwise use API or set to 0)
        if ds_info.splits and split in ds_info.splits:
            num_examples = ds_info.splits[split].num_examples
        elif ds_info.splits:
            logger.info(
                f"Dataset {hf_hub_ds_info.id} - {config} does not provide split {split}. Skipping."
            )
            return None
        else:
            num_examples = "N/A"
            logger.info(
                f"Number of examples not available for {hf_hub_ds_info.id} - {config}, setting to {num_examples}"
            )

        card_data = hf_hub_ds_info.cardData or {}
        hf_tags = card_data.get("tags") or []
        if not isinstance(hf_tags, list):
            hf_tags = [str(hf_tags)]
        modalities = card_data.get("modalities") or []
        if not isinstance(modalities, list):
            modalities = [str(modalities)]

        return DatasetInfo(
            name=hf_hub_ds_info.id,
            subset=config,
            description=str(getattr(hf_hub_ds_info, "description", "") or ""),
            license=str(card_data.get("license") or "N/A"),
            hf_tags=hf_tags,
            modalities=modalities,
            n_likes=int(hf_hub_ds_info.likes) if hf_hub_ds_info.likes else 0,
            n_downloads_last_month=(
                int(hf_hub_ds_info.downloads) if hf_hub_ds_info.downloads else 0
            ),
            number_of_rows=num_examples,
            columns=list(ds_info.features.keys()) if ds_info.features else [],
            # dataset=dataset,
        )

    def search(
        self,
        search_terms: list[str] | str,
        cache_dir: str = "",
        max_results: int = -1,
        max_results_per_term: int = -1,
    ) -> list[DatasetInfo]:
        """
        Search the dataset in HF Hub.

        Args:
            max_results_per_term: Maximum number of datasets to return per search term. Default is unlimited.
        """
        if not cache_dir:
            cache_dir = self.cache_dir
        if not cache_dir:
            raise ValueError("cache_dir is not set.")

        if isinstance(search_terms, str):
            search_terms = [search_terms]

        ret = []
        visited_ds = set()
        for keyword in search_terms:
            per_term_count = 0
            logger.info(f"Searching datasets for keyword: {keyword}")
            ds_infos = _safe_list_datasets(
                search=keyword,
                sort="downloads",
                gated=False,
                token=self.hf_token,  # Use access key if available
            )
            if ds_infos is None:
                logger.warning(f"Skipping keyword {keyword} due to HF search failure.")
                continue
            for item in ds_infos:
                if item.id in visited_ds:
                    logger.info(
                        f"Dataset {item.id} has already been visited. Skipping."
                    )
                    continue
                ds_info = _safe_dataset_info(item.id, token=self.hf_token)
                if not ds_info:
                    logger.warning(f"Failed to fetch dataset info for {item.id}")
                    continue
                if ds_info.cardData is None:
                    ds_info.cardData = {}

                configs = _safe_get_dataset_config_names(
                    path=ds_info.id,
                    cache_dir=cache_dir,
                    token=self.hf_token,
                )

                if not configs:
                    logger.warning(f"Dataset {ds_info.id} does not have any config.")
                    continue

                for config in configs:
                    dataset_info_obj = self._process_dataset_config(ds_info, config)
                    if dataset_info_obj:
                        logger.info(
                            f"Dataset {dataset_info_obj.name} - {dataset_info_obj.subset} was collected from Hugging Face"
                        )
                        ret.append(dataset_info_obj)
                        per_term_count += 1
                        if max_results > 0 and len(ret) >= max_results:
                            return ret
                        # limit per-term results
                        if (
                            max_results_per_term > 0
                            and per_term_count >= max_results_per_term
                        ):
                            break
                # stop if per-term limit reached for this keyword
                if max_results_per_term > 0 and per_term_count >= max_results_per_term:
                    break

                visited_ds.add(item.id)
        return ret
