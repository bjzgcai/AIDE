import os
import time

from loguru import logger
from tqdm import tqdm

from AIDE.hf_keywords_generator import HFKeywordsGenerator

from .advanced_processor import AdvancedProcessor
from .advanced_processor_rag import AdvancedProcessorRAG
from .basic_analyzer import BasicAnalyzer
from .configs import InstAgentConfig
from .dataset_info import DatasetInfo
from .hf_searcher import HFSearcher
from .utils import load_json, save_json


def cache_step(cache_path, serialize_fn=None, deserialize_fn=None, step_name=None):
    def decorator(fn):
        def wrapper(self, *args, **kwargs):
            if os.path.exists(cache_path):
                data = load_json(cache_path)
            else:
                data = None

            if not data or (serialize_fn and not isinstance(data, list)):
                result = fn(self, *args, **kwargs)
                to_save = serialize_fn(result) if serialize_fn else result
                save_json(to_save, cache_path)
                logger.info(
                    f"[{step_name}] fresh: {getattr(result, '__len__', lambda: 'n/a')()} items"
                    if hasattr(result, "__len__")
                    else f"[{step_name}] fresh: {result}"
                )
                return result
            else:
                result = deserialize_fn(data) if deserialize_fn else data
                logger.info(
                    f"[{step_name}] cache: {getattr(result, '__len__', lambda: 'n/a')()} items"
                    if hasattr(result, "__len__")
                    else f"[{step_name}] cache: {result}"
                )
                return result

        return wrapper

    return decorator


class InstAgent:
    """
    The main class of InstAgent.
    """

    def __init__(self, config: InstAgentConfig):
        self.config = config
        self.hf_keywords_generator = HFKeywordsGenerator(config)
        self.searcher = HFSearcher()
        self.basic_analyzer = BasicAnalyzer(config)
        self.advanced_processor = AdvancedProcessor(config)
        self.advanced_processor_rag = AdvancedProcessorRAG(config)

    def search_dataset(self, prompt: list[str]):
        """
        Conduct dataset search in HF Hub.
        """
        return self.searcher.search(
            prompt,
            cache_dir=self.config.cache_dir,
            max_results=self.config.max_datasets,
            max_results_per_term=self.config.max_results_per_term,
        )

    def basic_analysis(
        self, datasets: list[DatasetInfo], user_query: str
    ) -> list[DatasetInfo]:
        high_quality_datasets = []
        for dataset in datasets:
            try:
                result = self.basic_analyzer.analyze(dataset, user_query)
            except Exception as e:
                logger.error(f"Error in basic analysis for dataset {dataset.name}: {e}")
                continue
            logger.info(
                f"Dataset {dataset.name} - {dataset.subset} analysis result: {result}"
            )
            if result.score < 5:
                logger.info(
                    f"Dataset {dataset.name} - {dataset.subset} is removed due to low score."
                )
                continue
            else:
                logger.info(f"Dataset {dataset.name} - {dataset.subset} is kept.")
            high_quality_datasets.append(dataset)
        return high_quality_datasets

    def advanced_processing(self, dataset_info: DatasetInfo):
        if self.config.advanced_process_type == "inst":
            return self.advanced_processor.process(dataset_info)
        else:
            return self.advanced_processor_rag.process(dataset_info)

    def run(self):
        user_query = self.config.prompt
        output_dir = getattr(self.config, "output_dir", "outputs")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        # save user prompt to a text file
        with open(os.path.join(output_dir, "user_prompt.txt"), "w") as f:
            f.write(user_query)

        # Cache file paths
        cache_path = os.path.join(output_dir, "cache")
        keywords_path = os.path.join(cache_path, "keywords.json")
        datasets_path = os.path.join(cache_path, "datasets.json")
        basic_analysis_path = os.path.join(cache_path, "basic_analysis.json")

        # Step 0: HF keywords generation (cache)
        key_words_start_time = time.time()

        @cache_step(keywords_path, step_name="keywords")
        def get_keywords(self, user_query):
            return self.hf_keywords_generator.generate_keywords(user_query)

        keywords = get_keywords(self, user_query)
        keywords_end_time = time.time()

        # Step 1: Conduct dataset search in HF Hub (cache)
        search_start_time = time.time()

        @cache_step(
            datasets_path,
            serialize_fn=lambda ds: [d.__dict__ for d in ds],
            deserialize_fn=lambda ds: [DatasetInfo(**d) for d in ds],
            step_name="datasets",
        )
        def get_datasets(self, keywords):
            return self.search_dataset(keywords)

        datasets = get_datasets(self, keywords)
        search_end_time = time.time()
        all_datasets = len(datasets)

        # Step 2: for each dataset, do the basic analysis (cache)
        basic_start_time = time.time()

        @cache_step(
            basic_analysis_path,
            serialize_fn=lambda ds: [d.__dict__ for d in ds],
            deserialize_fn=lambda ds: [DatasetInfo(**d) for d in ds],
            step_name="basic_analysis",
        )
        def get_basic_analysis(self, datasets, user_query):
            return self.basic_analysis(datasets, user_query)

        datasets = get_basic_analysis(self, datasets, user_query)
        basic_end_time = time.time()
        basic_datasets = len(datasets)

        for ds in datasets:
            logger.info(ds.name)

        # Step 3: for each dataset, do the advanced processing
        advanced_start_time = time.time()
        advanced_datasets = 0
        for dataset in tqdm(datasets):
            success = self.advanced_processing(dataset)
            if success:
                advanced_datasets += 1
        advanced_end_time = time.time()
        with open("log.txt", "w") as f:
            f.write(
                f"Keywords generation time: {keywords_end_time - key_words_start_time}\n"
            )
            f.write(f"Search time: {search_end_time - search_start_time}\n")
            f.write(f"Basic analysis time: {basic_end_time - basic_start_time}\n")
            f.write(
                f"Advanced processing time: {advanced_end_time - advanced_start_time}\n"
            )
            f.write(
                f"Found {all_datasets} datasets, {basic_datasets} datasets after basic analysis, {advanced_datasets} datasets after advanced processing."
            )

        logger.success("All done!")
