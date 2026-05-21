from dataclasses import dataclass
from typing import Literal


@dataclass
class InstAgentConfig:
    """
    The configuration class of InstAgent.
    """

    prompt: str = "chemistry text for instruction tuning"  # the prompt for the dataset search
    output_dir: str = "/tos-mlp-zgci/liuzequn/instagent/processed_data"  # output directory
    cache_dir: str = "/root/.cache/huggingface/hub"  # cache director

    ## hf search config
    use_hf_keywords_generation: bool = (
        True  # Use keywords generation or direct search with prompt
    )
    hf_keywords_model: str = "gpt-4o"  # the model for keywords generation
    hf_keywords_temperature: float = 0.5  # the temperature for keywords generation
    hf_keywords_top_p: float = 0.9
    hf_keywords_max_tokens: int = 4096
    max_datasets: int = -1  # overall maximum number of datasets to return
    max_results_per_term: int = (
        200
    )  # maximum datasets to return per search term (keyword)
    results_threshold_per_term: int = (
        500  # minimum number of datasets to return per search term (keyword)
    )

    ## basic analysis config
    ba_model: str = "gpt-4o"  # the model for basic analysis
    ba_temperature: float = 0.5  # the temperature for basic analysis
    ba_top_p: float = 0.9  # the top_p for basic analysis
    ba_max_tokens: int = 4096  # the max_tokens for basic analysis
    ba_max_retries: int = 3  # the max_retries for basic analysis

    # endpoint config
    api_version: str = "2025-03-01-preview"  # api version
    resource_name: str = "ds-afdbmgnify-gpt-japaneast"  # resource name
    system_message: str = "You are a helpful AI assistant."  # system message
    base_url: str = "your OPENAI_API_BASE_URL"
    api_key: str = "your OPENAI_API_KEY"


    # config for advanced analysis
    advanced_process_type: Literal["inst", "rag"] = (
        "inst"  # the type of advanced process
    )
