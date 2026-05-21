import time
import json
from datasets import load_dataset, get_dataset_config_info
from loguru import logger
from pydantic import BaseModel

from .configs import InstAgentConfig
from .dataset_info import DatasetInfo
from .utils import OpenAIClient

system_prompt = """
You are an data scientist working in a AI4Science research lab. You are asked to analyze a dataset and judge the quality of the dataset.
The dataset will be later used in to run instruction fine-tuning or RAG on scientific LLM models about natural science. The nature science should be a inclusive term that includes but not limited to physics, chemistry, biology, medical, and so on.

Especially, you are asked to check the following aspects from both metadata and data samples:
1. The correctness and quality of the dataset; only the text columns matters, not other modalities like image or embedding.
2. The relevance of the dataset content and the target domain {{topic}}.
3. The language of the dataset should ONLY be English; Otherwise you should give a score of 0.
4. The dataset should not be too large or too small. Especially, you should reject the dataset with > 10M rows.
5. The dataset should be widely used and popular.
6. There is some possibility that the text in the dataset can be used to run instruction fine-tuning or RAG on scientific LLM models about natural science after careful processing.

Your output should contains two parts:
1. A brief report that contains a item list checking the above aspects.
2. An overall score of the dataset in [0-9] where 0 is the worst and 9 is the best.
""".strip()


class AnalyzerResult(BaseModel):
    report: str
    score: int


class BasicAnalyzer:
    def __init__(self, config: InstAgentConfig):
        self.client = OpenAIClient(
            base_url=config.base_url,
            api_key=config.api_key,
            client_type="key",
        )
        self.config = config

    def analyze(self, dataset_info: DatasetInfo, user_goal: str) -> AnalyzerResult:
        # make user message
        user_message = f"""
        Check if the dataset can be used to "{user_goal.replace('"', '\"')}".
        Below is the information of the dataset:
        Name: {dataset_info.name}
        Description: {dataset_info.description}
        License: {dataset_info.license}
        Tags: {dataset_info.hf_tags}
        Modalities: {dataset_info.modalities}
        n_likes: {dataset_info.n_likes}
        n_downloads_last_month: {dataset_info.n_downloads_last_month}
        Number of rows: {dataset_info.number_of_rows}
        Columns: {dataset_info.columns}
        """.strip()
        
        # take 3 samples and append to the user message
        # Automatically detect available splits
        dataset = None
        try:
            # Get dataset configuration info to find available splits
            config_info = get_dataset_config_info(
                dataset_info.name, 
                config_name=dataset_info.subset if dataset_info.subset else None
            )
            available_splits = list(config_info.splits.keys())
            
            if not available_splits:
                raise ValueError("No splits available in the dataset")
            
            # Prefer certain splits in order: train > test > validation > dev > others
            preferred_order = ["train", "test", "validation", "dev"]
            split_to_use = None
            
            for preferred in preferred_order:
                if preferred in available_splits:
                    split_to_use = preferred
                    break
            
            # If no preferred split found, use the first available one
            if split_to_use is None:
                split_to_use = available_splits[0]
            
            # Load the dataset with the selected split
            dataset = load_dataset(
                path=dataset_info.name,
                name=dataset_info.subset,
                split=split_to_use,
                cache_dir=self.config.cache_dir,
                streaming=True,
                trust_remote_code=False,
            )
            print(f"Dataset loaded successfully with split: {split_to_use}")
            logger.info(f"Available splits: {available_splits}, using: {split_to_use}")
            
        except Exception as e:
            logger.error(f"Error in loading dataset {dataset_info.name}: {e}")
            return AnalyzerResult(report="Failed to load the dataset.", score=0)
        
        if dataset is None:
            logger.error(f"Error in loading dataset {dataset_info.name}: Failed to load dataset")
            return AnalyzerResult(report="Failed to load the dataset.", score=0)
        #print("dataset loaded")
        sample_num = min(2, dataset_info.number_of_rows)
        if sample_num == 0:
            return AnalyzerResult(report="No samples found in the dataset.", score=0)

        # take first 'sample_num' samples
        samples = []
        for i, sample in enumerate(dataset):
            if i >= sample_num:
                break
            samples.append(sample)

        # samples = dataset.select(list(range(sample_num)))
        # samples = samples.to_pandas()
        # samples = samples.to_dict(orient="records")
        user_message += "\n\nBelow are data samples:\n"
        for sample in samples:
            user_message += str(sample) + "\n"
        #print(user_message)
        
        # compose the AOIA message
        messages = [
            {"role": "system", "content": system_prompt.replace("{{topic}}", self.config.prompt)},
            {"role": "user", "content": user_message},
        ]

        resp = None
        for attempt in range(self.config.ba_max_retries):
            try:
                resp = self.client.client.beta.chat.completions.parse(
                    model=self.config.ba_model,
                    messages=messages,
                    response_format=AnalyzerResult,
                    temperature=self.config.ba_temperature,
                    top_p=self.config.ba_top_p,
                    max_tokens=self.config.ba_max_tokens,
                )
                break
            except Exception as e:
                if attempt < self.config.ba_max_retries - 1:
                    logger.error(f"Failed to analyze the dataset: {e}. Retrying...")
                    #time.sleep(2**attempt)  # Exponential backoff
                    time.sleep(60)
                else:
                    raise e
        if resp is None:
            logger.error(f"Failed to analyze the dataset {dataset_info.name}.")
            return AnalyzerResult(report="Failed to analyze the dataset.", score=0)
        resp_message = resp.choices[0].message
        if not resp_message.parsed:
            logger.error(f"Failed to parse the response: {resp_message}")
            return AnalyzerResult(report="Failed to parse the response.", score=0)
        return resp_message.parsed


# debug
if __name__ == "__main__":
    config = InstAgentConfig(prompt="")
    
    dataset_info = DatasetInfo(
        name="biomedical-translator/pubmed2024_sentence_embeddings",
        subset="default",
        number_of_rows=185_145_351,
        columns=["PMID", "sentence", "embedding"],
        description="",
        hf_tags=[],
        modalities=["text"],
        license="",
        n_likes=0,
        n_downloads_last_month=715,
    )
    analyzer = BasicAnalyzer(config)
    result = analyzer.analyze(dataset_info, "test user goal")
    print(result)