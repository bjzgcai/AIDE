from datasets import load_dataset
from loguru import logger
from pydantic import BaseModel

from .configs import InstAgentConfig
from .dataset_info import DatasetInfo
from .prompts import basic_analyzer_system_prompt
from .utils import OpenAIClient, make_safe_call


class AnalyzerResult(BaseModel):
    report: str
    score: int


@make_safe_call(allow_failure=True)
def load_dataset_safe(dataset_info: DatasetInfo, cache_dir: str):
    return load_dataset(
        path=dataset_info.name,
        name=dataset_info.subset,
        split="train",
        cache_dir=cache_dir,
        streaming=True,
    )


class BasicAnalyzer:
    def __init__(self, config: InstAgentConfig):
        self.client = OpenAIClient(
            system_message=basic_analyzer_system_prompt,
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
        # take 10 samples and append to the user message
        dataset = load_dataset_safe(dataset_info, self.config.cache_dir)

        if isinstance(dataset_info.number_of_rows, str):
            sample_num = 10
        else:
            sample_num = min(10, dataset_info.number_of_rows)
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

        resp = self.client.chat(
            user_message,
            model=self.config.ba_model,
            response_format=AnalyzerResult,
        )
        if not resp:
            logger.error(
                f"Failed to analyze the dataset: {dataset_info.name} - {dataset_info.subset}"
            )
            return AnalyzerResult(
                report="Failed to analyze the dataset.",
                score=0,
            )
        return resp


# debug
if __name__ == "__main__":
    config = InstAgentConfig(prompt="")

    # dataset_info = DatasetInfo(
    #     name="FreedomIntelligence/medical-o1-reasoning-SFT",
    #     subset="zh",
    #     number_of_rows=20_200,
    #     columns=["Question", "Complex_CoT", "Response"],
    #     description="A dataset for medical reasoning.",
    #     hf_tags=[],
    #     modalities=["text"],
    #     license="apache-2.0",
    #     n_likes=707,
    #     n_downloads_last_month=9845,
    # )

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
