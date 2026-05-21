from dataclasses import dataclass
from typing import Literal


@dataclass
class InstAgentConfig:
    """
    The configuration class of InstAgent.
    """

    prompt: str  # the prompt for the dataset search
    output_dir: str = "./outputs/"  # output directory
    cache_dir: str = "./cache"  # reviewer-friendly default cache directory

    ## hf search config
    use_hf_keywords_generation: bool = (
        False  # Use keywords generation or direct search with prompt
    )
    hf_keywords_model: str = "openai/gpt-4o"  # the model for keywords generation
    hf_keywords_generation_attempts: int = (
        3  # number of times to call chat for keyword generation
    )
    max_datasets: int = -1  # overall maximum number of datasets to return
    max_results_per_term: int = (
        -1
    )  # maximum datasets to return per search term (keyword)

    ## basic analysis config
    ba_model: str = "openai/gpt-4o"  # the model for basic analysis

    # endpoint config
    system_message: str = "You are a helpful AI assistant."  # system message

    # config for advanced analysis
    advanced_process_type: Literal["inst", "rag"] = (
        "inst"  # the type of advanced process
    )
    advanced_process_model: str = "openai/gpt-4o"  # the model for advanced process
