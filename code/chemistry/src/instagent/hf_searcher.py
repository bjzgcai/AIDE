from typing import Optional
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from datasets import get_dataset_config_names, load_dataset_builder
from huggingface_hub import DatasetInfo as HfHubDatasetInfo
from huggingface_hub import dataset_info, list_datasets
from loguru import logger

from .dataset_info import DatasetInfo
from .network_config import safe_request_with_retry, get_connect_timeout, get_read_timeout

#from huggingface_hub import login
#login(token="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")


def create_session_with_retry():
    """Create a requests session with retry mechanism"""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


class HFSearcher:
    """
    Search the dataset in HF Hub.
    """

    def __init__(self, cache_dir: str = ""):
        self.cache_dir = cache_dir
        self.session = create_session_with_retry()

    def _process_dataset_config(
        self, hf_hub_ds_info: HfHubDatasetInfo, config: str, split: str = "train"
    ) -> Optional[DatasetInfo]:
        """
        Process a single dataset configuration and return its DatasetInfo.

        Args:
            ds_info: Dataset info from HuggingFace Hub
            config: Dataset configuration name

        Returns:
            DatasetInfo object or None if processing fails
        """
        try:
            ds_builder = load_dataset_builder(
                hf_hub_ds_info.id,
                name=config,
                cache_dir=self.cache_dir,
                trust_remote_code=False,
            )

            ds_info = ds_builder.info

            if ds_info.splits is None:
                logger.warning(
                    f"Dataset {hf_hub_ds_info.id} - {config} does not have any split."
                )
                return None

            if split not in ds_info.splits:
                logger.warning(
                    f"Dataset {hf_hub_ds_info.id} - {config} does not have {split} split."
                )
                return None

            return DatasetInfo(
                name=hf_hub_ds_info.id,
                subset=config,
                description=hf_hub_ds_info.cardData.get("description", "N/A"),
                license=str(hf_hub_ds_info.cardData.get("license", "N/A")),
                hf_tags=hf_hub_ds_info.cardData.get("tags", []),
                modalities=hf_hub_ds_info.cardData.get("modalities", []),
                n_likes=hf_hub_ds_info.likes if hf_hub_ds_info.likes else 0,
                n_downloads_last_month=(
                    hf_hub_ds_info.downloads if hf_hub_ds_info.downloads else 0
                ),
                number_of_rows=ds_info.splits[split].num_examples,
                columns=list(ds_info.features.keys()) if ds_info.features else [],
                # dataset=dataset,
            )
        except Exception as e:
            logger.error(
                f"Failed to process dataset {hf_hub_ds_info.id} with config {config}: {e}"
            )
            return None

    def _safe_list_datasets(self, keyword: str):
        """Safely call list_datasets with retry mechanism"""
        try:
            logger.info(f"Searching for keyword: '{keyword}'")
            result = safe_request_with_retry(
                list_datasets,
                search=keyword, 
                sort="downloads", 
                gated=False
            )
            if result is not None:
                return [item for item in result]
            else:
                logger.error(f"Failed to search for keyword '{keyword}' after all retries")
                return []
        except Exception as e:
            logger.error(f"Unexpected error while searching for keyword '{keyword}': {e}")
            return []

    def _safe_dataset_info(self, item_id: str):
        """Safely call dataset_info with retry mechanism"""
        try:
            result = safe_request_with_retry(dataset_info, item_id)
            if result is not None:
                if result.cardData is None:
                    result.cardData = {}
                return result
            else:
                logger.error(f"Failed to get dataset info for {item_id} after all retries")
                return None
        except Exception as e:
            logger.error(f"Unexpected error while getting dataset info for {item_id}: {e}")
            return None

    def _safe_get_config_names(self, ds_info, cache_dir: str):
        """Safely call get_dataset_config_names with retry mechanism"""
        try:
            result = safe_request_with_retry(
                get_dataset_config_names,
                path=ds_info.id,
                cache_dir=cache_dir,
                trust_remote_code=False,
            )
            if result is not None:
                return result
            else:
                logger.error(f"Failed to get config names for {ds_info.id} after all retries")
                return []
        except Exception as e:
            logger.error(f"Unexpected error while getting config names for {ds_info.id}: {e}")
            return []

    def search(
        self,
        search_terms: list[str] | str,
        cache_dir: str = "",
        max_results: int = -1,
        max_results_per_term: int = -1,
        results_threshold_per_term: int = 500,
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
            logger.info(f"Searching for keyword: {keyword}")
            
            # Use safe method with retry
            ds_info_list = self._safe_list_datasets(keyword)
            
            if len(ds_info_list) == 0:
                logger.warning(f"No datasets found for keyword '{keyword}' after all retries")
                continue
                
            if len(ds_info_list) > results_threshold_per_term:
                logger.info(
                    f"Found {len(ds_info_list)} datasets for keyword '{keyword}', which exceeds the threshold of {results_threshold_per_term}. the keyword is too general."
                )
                continue
                
            for item in ds_info_list:
                if item.id in visited_ds:
                    continue
                visited_ds.add(item.id)
                
                # Use safe method with retry
                ds_info = self._safe_dataset_info(item.id)
                if ds_info is None:
                    continue

                # Use safe method with retry
                configs = self._safe_get_config_names(ds_info, cache_dir)
                
                if len(configs) == 0:
                    logger.warning(f"Dataset {ds_info.id} does not have any config.")
                    continue

                for config in configs:
                    dataset_info_obj = self._process_dataset_config(ds_info, config)
                    if dataset_info_obj:
                        logger.info(
                            f"Dataset {dataset_info_obj.name} - {dataset_info_obj.subset} has been used by hf_searcher"
                        )
                        ret.append(dataset_info_obj)
                        per_term_count += 1
                        # limit per-term results
                        if (
                            max_results_per_term > 0
                            and per_term_count >= max_results_per_term
                        ):
                            break
                        if len(ret) >= max_results > 0:
                            return ret
                # stop if per-term limit reached for this keyword
                if max_results_per_term > 0 and per_term_count >= max_results_per_term:
                    break
        return ret


if __name__ == "__main__":
    searcher = HFSearcher()
    datasets = searcher.search("biology", cache_dir="/tmp/")
    print(datasets)
