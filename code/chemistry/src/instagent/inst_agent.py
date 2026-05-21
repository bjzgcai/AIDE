import time

from loguru import logger
from tqdm import tqdm

from instagent.hf_keywords_generator import HFKeywordsGenerator
import glob
import shutil

from .advanced_processor import AdvancedProcessor
from .advanced_processor_rag import AdvancedProcessorRAG
from .basic_analyzer import BasicAnalyzer
from .configs import InstAgentConfig
from .dataset_info import DatasetInfo
from .hf_searcher import HFSearcher


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
            results_threshold_per_term=self.config.results_threshold_per_term,
        )

    def basic_analysis(
        self, datasets: list[DatasetInfo], user_query: str
    ) -> list[DatasetInfo]:
        high_quality_datasets = []
        fw = open("high_quality_datasets.txt", "w")
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
            fw.write(f"{dataset.name}\t{dataset.subset}\t{result.score}\n")
        fw.close()
        return high_quality_datasets

    def advanced_processing(self, dataset_info: DatasetInfo):
        if self.config.advanced_process_type == "inst":
            return self.advanced_processor.process(dataset_info)
        else:
            return self.advanced_processor_rag.process(dataset_info)

    def run(self):
        user_query = self.config.prompt

        # Step 0: HF keywords generation
        key_words_start_time = time.time()
        keywords = self.hf_keywords_generator.generate_keywords(user_query)
        keywords_end_time = time.time()
        logger.info(f"Generated keywords: {keywords}")

        # Step 1: Conduct dataset search in HF Hub
        search_start_time = time.time()
        datasets = self.search_dataset(keywords)
        search_end_time = time.time()
        all_datasets = len(datasets)
        logger.info(f"Found {len(datasets)} datasets.")

        # Step 2: for each dataset, do the basic analysis
        # and remove the dataset that does not meet the requirement
        basic_start_time = time.time()
        try:
            datasets = self.basic_analysis(datasets, user_query)
        except Exception as e:
            logger.error(f"Unknown Error in basic analysis: {e}")
            datasets = []
        basic_end_time = time.time()
        basic_datasets = len(datasets)

        logger.info(f"Found {len(datasets)} datasets after filtering.")
        for ds in datasets:
            logger.info(ds.name)

        # Step 3: for each dataset, do the advanced processing
        # brain storming data usage, generate the code snippet
        # running the extration and validation, taging etc.
        # Note we may generate multiple datasets for one dataset, e.g., mol->prop, prop->mol
        advanced_start_time = time.time()
        advanced_datasets = 0
        for dataset in tqdm(datasets):
            try:
                success = self.advanced_processing(dataset)
            except Exception as e:
                logger.error(
                    f"Unknown Error in advanced processing for dataset {dataset.name}: {e}"
                )
                success = False
            if success:
                advanced_datasets += 1
            for folder in glob.glob(f"{self.config.cache_dir}/datasets*"):
                shutil.rmtree(folder, ignore_errors=True)
                print(f"Remove cache folder {folder}")
        advanced_end_time = time.time()
        with open("log_new.txt", "w") as f:
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
