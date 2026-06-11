from dataclasses import dataclass
from typing import Literal


@dataclass
class AIDEConfig:
    """
    Configuration for the AIDE data-curation pipeline.
    """

    prompt: str
    output_dir: str = "outputs"
    cache_dir: str = "cache/huggingface"

    # Stage 1: data collection.
    use_llm_collection_keywords: bool = False
    collection_keyword_model: str = "openai/gpt-4o-mini"
    collection_keyword_attempts: int = 3
    max_datasets: int = -1
    max_results_per_term: int = -1

    # Stage 2: data selection.
    selection_model: str = "openai/gpt-4o-mini"
    selection_score_threshold: int = 5

    # Shared endpoint/system prompt configuration.
    system_message: str = "You are a helpful AI assistant."

    # Stage 3: data organization.
    organization_format: Literal["instruction", "rag"] = "instruction"
    organization_model: str = "openai/gpt-4o-mini"
    instruction_max_retries: int = 3
    instruction_sample_size: int = 5
    instruction_quality_check: bool = True
    instruction_max_items_per_dataset: int = 0
    rag_dataset_split: str = "train"
    rag_max_items_per_dataset: int = 0
    rag_sample_size: int = 15
    rag_max_retries: int = 3
    rag_streaming: bool = True

    # Metrics.
    enable_stage_metrics: bool = True
    model_pricing_path: str = ""
