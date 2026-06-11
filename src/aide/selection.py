import json
import os
import urllib.error
import urllib.parse
import urllib.request

from datasets import load_dataset
from loguru import logger
from pydantic import BaseModel

from .config import AIDEConfig
from .models import DatasetInfo
from .prompts import selection_system_prompt
from .utils import OpenAIClient, make_safe_call, sanitize_item_for_prompt


class AnalyzerResult(BaseModel):
    report: str
    score: int


HF_DATASET_PREVIEW_API_URL = "https://datasets-server.huggingface.co/first-rows"


def _is_non_retryable_dataset_load_error(exc: Exception) -> bool:
    error_text = str(exc).lower()
    non_retryable_patterns = [
        "bad split: train",
        "no (supported) data files found",
        "couldn't infer the same data file format for all splits",
        "dataset scripts are no longer supported",
        "must be called with a dataclass type or instance",
        "directory not empty",
        "the tar archives of the dataset should be in webdataset format",
        "don't share the same prefix or the same types",
        "feature type",
        "not found. available feature types",
        "doesn't contain any data files",
        "does not contain any data files",
    ]
    return any(pattern in error_text for pattern in non_retryable_patterns)


def _is_non_retryable_preview_api_error(exc: Exception) -> bool:
    return (
        isinstance(exc, urllib.error.HTTPError)
        and 400 <= exc.code < 500
        and exc.code not in {408, 429}
    )


@make_safe_call(allow_failure=True, giveup=_is_non_retryable_dataset_load_error)
def load_dataset_safe(dataset_info: DatasetInfo, cache_dir: str):
    return load_dataset(
        path=dataset_info.name,
        name=dataset_info.subset,
        split="train",
        cache_dir=cache_dir,
        streaming=True,
    )


@make_safe_call(
    allow_failure=True,
    giveup=_is_non_retryable_preview_api_error,
    max_tries=3,
)
def fetch_dataset_preview_rows(
    dataset_info: DatasetInfo,
    split: str = "train",
    api_url: str = HF_DATASET_PREVIEW_API_URL,
    timeout: int = 30,
) -> list[dict] | None:
    params = {
        "dataset": dataset_info.name,
        "split": split,
    }
    if dataset_info.subset:
        params["config"] = dataset_info.subset

    url = f"{api_url}?{urllib.parse.urlencode(params)}"
    headers = {
        "Accept": "application/json",
        "User-Agent": "aide-curator/0.1",
    }
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))

    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"Unexpected dataset preview response for {dataset_info.name}")

    samples: list[dict] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        row = item.get("row")
        if isinstance(row, dict):
            samples.append(row)

    if payload.get("truncated"):
        logger.info(
            f"Dataset preview response for {dataset_info.name} - {dataset_info.subset} was truncated."
        )
    return samples


def _sample_from_streaming_dataset(
    dataset_info: DatasetInfo, cache_dir: str, sample_num: int
) -> list[dict]:
    dataset = load_dataset_safe(dataset_info, cache_dir)
    if not dataset:
        return []

    samples = []
    for i, sample in enumerate(dataset):
        if i >= sample_num:
            break
        sanitized = sanitize_item_for_prompt(sample)
        if sanitized:
            samples.append(sanitized)
    return samples


def load_dataset_samples(
    dataset_info: DatasetInfo, cache_dir: str, sample_num: int
) -> list[dict]:
    preview_rows = fetch_dataset_preview_rows(dataset_info)
    if preview_rows is not None:
        samples = []
        for row in preview_rows[:sample_num]:
            sanitized = sanitize_item_for_prompt(row)
            if sanitized:
                samples.append(sanitized)
        return samples

    logger.warning(
        f"Falling back to streaming load_dataset samples for {dataset_info.name} - {dataset_info.subset}."
    )
    return _sample_from_streaming_dataset(dataset_info, cache_dir, sample_num)


class DatasetSelector:
    def __init__(self, config: AIDEConfig):
        self.client = OpenAIClient(
            system_message=selection_system_prompt,
        )

        self.config = config

    def analyze(self, dataset_info: DatasetInfo, user_goal: str) -> AnalyzerResult:
        description = dataset_info.description
        if len(description) > 4000:
            description = description[:4000] + "...(omitted)"

        safe_user_goal = user_goal.replace('"', '\\"')
        user_message = "\n".join(
            [
                f'Check if the dataset can be used to "{safe_user_goal}".',
                "Below is the information of the dataset:",
                f"Name: {dataset_info.name}",
                f"Description: {description}",
                f"License: {dataset_info.license}",
                f"Tags: {dataset_info.hf_tags}",
                f"Modalities: {dataset_info.modalities}",
                f"n_likes: {dataset_info.n_likes}",
                f"n_downloads_last_month: {dataset_info.n_downloads_last_month}",
                f"Number of rows: {dataset_info.number_of_rows}",
                f"Columns: {dataset_info.columns}",
            ]
        )

        if isinstance(dataset_info.number_of_rows, str):
            sample_num = 10
        else:
            sample_num = min(10, dataset_info.number_of_rows)
        if sample_num == 0:
            return AnalyzerResult(report="No samples found in the dataset.", score=0)

        samples = load_dataset_samples(dataset_info, self.config.cache_dir, sample_num)
        if not samples:
            return AnalyzerResult(report="No samples found in the dataset.", score=0)

        # samples = dataset.select(list(range(sample_num)))
        # samples = samples.to_pandas()
        # samples = samples.to_dict(orient="records")
        user_message += "\n\nBelow are data samples:\n"
        for sample in samples:
            user_message += json.dumps(sample, ensure_ascii=False) + "\n"

        resp = self.client.chat(
            user_message,
            model=self.config.selection_model,
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
