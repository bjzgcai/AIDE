import os
import time
import traceback
from dataclasses import asdict

from loguru import logger
from tqdm import tqdm

from .collection_keywords import KeywordGenerator
from .collection_search import HFDatasetCollector
from .config import AIDEConfig
from .metrics import (
    StageCostTracker,
    activate_metrics_tracker,
    metrics_scope,
    record_cache_event,
    record_counter,
)
from .models import DatasetInfo
from .organization_instruction import InstructionOrganizer
from .organization_rag import RAGOrganizer
from .selection import DatasetSelector
from .utils import load_json, save_json


def cache_step(cache_path, serialize_fn=None, deserialize_fn=None, step_name=None):
    def decorator(fn):
        def wrapper(self, *args, **kwargs):
            if os.path.exists(cache_path):
                data = load_json(cache_path)
            else:
                data = None

            if data is None or (serialize_fn and not isinstance(data, list)):
                result = fn(self, *args, **kwargs)
                to_save = serialize_fn(result) if serialize_fn else result
                save_json(to_save, cache_path)
                record_cache_event(False)
                logger.info(
                    f"[{step_name}] fresh: {getattr(result, '__len__', lambda: 'n/a')()} items"
                    if hasattr(result, "__len__")
                    else f"[{step_name}] fresh: {result}"
                )
                return result

            result = deserialize_fn(data) if deserialize_fn else data
            record_cache_event(True)
            logger.info(
                f"[{step_name}] cache: {getattr(result, '__len__', lambda: 'n/a')()} items"
                if hasattr(result, "__len__")
                else f"[{step_name}] cache: {result}"
            )
            return result

        return wrapper

    return decorator


class AIDEPipeline:
    """
    End-to-end AIDE pipeline coordinating collection, selection, and organization.
    """

    def __init__(self, config: AIDEConfig):
        self.config = config
        self.keyword_generator = KeywordGenerator(config)
        self.collection_searcher = HFDatasetCollector()
        self.dataset_selector = DatasetSelector(config)
        self.instruction_organizer = InstructionOrganizer(config)
        self.rag_organizer = RAGOrganizer(config)

    def search_dataset(self, prompt: list[str]):
        """
        Conduct dataset search in the Hugging Face Hub.
        """
        return self.collection_searcher.search(
            prompt,
            cache_dir=self.config.cache_dir,
            max_results=self.config.max_datasets,
            max_results_per_term=self.config.max_results_per_term,
        )

    def select_datasets(
        self, datasets: list[DatasetInfo], user_query: str
    ) -> list[DatasetInfo]:
        selection_state_path = os.path.join(
            getattr(self.config, "output_dir", "outputs"),
            "cache",
            "selection_state.json",
        )
        loaded_state = load_json(selection_state_path)
        if not isinstance(loaded_state, dict):
            loaded_state = {}
        decisions = loaded_state.get("decisions")
        if not isinstance(decisions, dict):
            decisions = {}

        def dataset_key(dataset: DatasetInfo) -> str:
            return f"{dataset.name}::{dataset.subset}"

        def persist_decision(
            dataset: DatasetInfo,
            *,
            status: str,
            score: int | None = None,
            report: str = "",
            error: str = "",
        ) -> None:
            decisions[dataset_key(dataset)] = {
                "status": status,
                "score": score,
                "report": report,
                "error": error,
                "dataset": asdict(dataset),
            }
            save_json(
                {
                    "version": 1,
                    "user_query": user_query,
                    "decisions": decisions,
                },
                selection_state_path,
            )

        high_quality_datasets = []
        for dataset in datasets:
            key = dataset_key(dataset)
            cached_decision = decisions.get(key)
            if isinstance(cached_decision, dict):
                status = cached_decision.get("status")
                if status == "kept":
                    logger.info(
                        f"Dataset {dataset.name} - {dataset.subset} selection resumed as kept."
                    )
                    high_quality_datasets.append(dataset)
                    record_counter("selection_resumed_kept", 1)
                    continue
                if status in {"removed", "error"}:
                    logger.info(
                        f"Dataset {dataset.name} - {dataset.subset} selection resumed as {status}."
                    )
                    record_counter(f"selection_resumed_{status}", 1)
                    continue

            try:
                result = self.dataset_selector.analyze(dataset, user_query)
            except Exception as e:
                logger.error(f"Error in selection for dataset {dataset.name}: {e}")
                persist_decision(dataset, status="error", error=str(e))
                record_counter("selection_errors", 1)
                continue
            logger.info(
                f"Dataset {dataset.name} - {dataset.subset} analysis result: {result}"
            )
            if result.score < self.config.selection_score_threshold:
                logger.info(
                    f"Dataset {dataset.name} - {dataset.subset} is removed due to low score."
                )
                persist_decision(
                    dataset,
                    status="removed",
                    score=result.score,
                    report=result.report,
                )
                record_counter("selection_removed", 1)
                continue
            logger.info(f"Dataset {dataset.name} - {dataset.subset} is kept.")
            high_quality_datasets.append(dataset)
            persist_decision(
                dataset,
                status="kept",
                score=result.score,
                report=result.report,
            )
            record_counter("selection_kept", 1)
        return high_quality_datasets

    def organize_dataset(self, dataset_info: DatasetInfo):
        if self.config.organization_format == "instruction":
            return self.instruction_organizer.process(dataset_info)
        return self.rag_organizer.process(dataset_info)

    def _build_tracker(self, output_dir: str) -> StageCostTracker | None:
        if not getattr(self.config, "enable_stage_metrics", True):
            return None
        metrics_path = os.path.join(output_dir, "stage_metrics.json")
        return StageCostTracker(
            metrics_path,
            pricing_path=getattr(self.config, "model_pricing_path", ""),
        )

    def run(self):
        user_query = self.config.prompt
        output_dir = getattr(self.config, "output_dir", "outputs")
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "user_prompt.txt"), "w") as f:
            f.write(user_query)

        cache_path = os.path.join(output_dir, "cache")
        keywords_path = os.path.join(cache_path, "keywords.json")
        datasets_path = os.path.join(cache_path, "datasets.json")
        selection_path = os.path.join(cache_path, "selection.json")

        tracker = self._build_tracker(output_dir)

        with activate_metrics_tracker(tracker):
            collection_keywords_start_time = time.time()

            @cache_step(keywords_path, step_name="keywords")
            def get_keywords(self, user_query):
                return self.keyword_generator.generate_keywords(user_query)

            with metrics_scope(
                "keywords",
                metadata={
                    "cache_path": keywords_path,
                    "model": self.config.collection_keyword_model,
                },
            ):
                keywords = get_keywords(self, user_query)
            collection_keywords_end_time = time.time()
            if tracker:
                tracker.persist(force=True)

            search_start_time = time.time()

            @cache_step(
                datasets_path,
                serialize_fn=lambda ds: [d.__dict__ for d in ds],
                deserialize_fn=lambda ds: [DatasetInfo(**d) for d in ds],
                step_name="collection_search",
            )
            def get_datasets(self, keywords):
                return self.search_dataset(keywords)

            with metrics_scope(
                "collection_search",
                metadata={
                    "cache_path": datasets_path,
                    "max_datasets": self.config.max_datasets,
                    "max_results_per_term": self.config.max_results_per_term,
                },
            ):
                datasets = get_datasets(self, keywords)
            search_end_time = time.time()
            all_datasets = len(datasets)
            if tracker:
                tracker.persist(force=True)

            selection_start_time = time.time()

            @cache_step(
                selection_path,
                serialize_fn=lambda ds: [d.__dict__ for d in ds],
                deserialize_fn=lambda ds: [DatasetInfo(**d) for d in ds],
                step_name="selection",
            )
            def get_selected_datasets(self, datasets, user_query):
                return self.select_datasets(datasets, user_query)

            with metrics_scope(
                "selection",
                metadata={
                    "cache_path": selection_path,
                    "model": self.config.selection_model,
                },
            ):
                datasets = get_selected_datasets(self, datasets, user_query)
            selection_end_time = time.time()
            selected_datasets = len(datasets)
            if tracker:
                tracker.persist(force=True)

            for ds in datasets:
                logger.info(ds.name)

            organization_start_time = time.time()
            organized_datasets = 0
            with metrics_scope(
                "organization",
                metadata={
                    "organization_format": self.config.organization_format,
                    "model": self.config.organization_model,
                },
            ):
                for dataset in tqdm(datasets):
                    record_counter("datasets_attempted", 1)
                    try:
                        success = self.organize_dataset(dataset)
                    except Exception as e:
                        logger.error(
                            f"Data organization crashed for dataset {dataset.name} - {dataset.subset}: {e}"
                        )
                        logger.debug(traceback.format_exc())
                        success = False
                    if success:
                        organized_datasets += 1
                        record_counter("datasets_completed", 1)
                    else:
                        record_counter("datasets_failed", 1)
                    if tracker:
                        tracker.persist(force=True)
            organization_end_time = time.time()

        with open(os.path.join(output_dir, "log.txt"), "w") as f:
            f.write(
                f"Data collection keyword generation time: {collection_keywords_end_time - collection_keywords_start_time}\n"
            )
            f.write(
                f"Data collection search time: {search_end_time - search_start_time}\n"
            )
            f.write(
                f"Data selection time: {selection_end_time - selection_start_time}\n"
            )
            f.write(
                f"Data organization time: {organization_end_time - organization_start_time}\n"
            )
            f.write(
                f"Found {all_datasets} datasets, {selected_datasets} datasets after selection, {organized_datasets} datasets after organization."
            )

        if tracker:
            tracker.persist(force=True)
        logger.success("All done!")
