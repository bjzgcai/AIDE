# Configuration Reference

AIDE uses `jsonargparse`, so every field in `AIDEConfig` can be supplied from a
YAML config file or as a CLI argument.

## Core

AIDE reads LLM credentials from a single OpenAI-compatible endpoint contract:

| Environment variable | Required | Description |
| --- | --- | --- |
| `LLM_API_KEY` | yes | API key for the configured OpenAI-compatible endpoint. |
| `LLM_API_BASE` | no | Endpoint base URL. Defaults to the OpenAI SDK default when empty. |
| `LLM_EXTRA_BODY_JSON` | no | JSON object forwarded as `extra_body` on chat requests. Useful for provider-specific options without adding provider-specific env vars. |
| `HF_TOKEN` | no | Hugging Face token for private datasets or higher public Hub rate limits. |

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `prompt` | string | required | Natural-language data requirement. |
| `output_dir` | string | `outputs` | Root directory for prompt-scoped outputs. |
| `cache_dir` | string | `cache/huggingface` | Local Hugging Face dataset cache. |
| `system_message` | string | `You are a helpful AI assistant.` | Shared system message for LLM clients. |

## Data Collection

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `use_llm_collection_keywords` | bool | `false` | Expand the request into multiple search keywords before querying the Hub. |
| `collection_keyword_model` | string | `openai/gpt-4o-mini` | Model for collection keyword generation. |
| `collection_keyword_attempts` | int | `3` | Number of independent keyword-generation attempts to merge. |
| `max_results_per_term` | int | `-1` | Maximum Hub search results per keyword. `-1` means unbounded. |
| `max_datasets` | int | `-1` | Maximum candidate datasets to keep. `-1` means unbounded. |

## Data Selection

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `selection_model` | string | `openai/gpt-4o-mini` | Model for LLM-based dataset quality scoring. |
| `selection_score_threshold` | int | `5` | Minimum score required for organization. |

## Data Organization

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `organization_format` | `instruction` or `rag` | `instruction` | Target organized-data format. |
| `organization_model` | string | `openai/gpt-4o-mini` | Model for processor generation, checker feedback, and validation. |
| `instruction_max_retries` | int | `3` | Maximum coder-checker attempts for instruction processors. |
| `instruction_sample_size` | int | `5` | Sample rows used to validate generated instruction processors. |
| `instruction_quality_check` | bool | `true` | Run LLM quality checks before accepting instruction processors. |
| `instruction_max_items_per_dataset` | int | `0` | Instruction item cap per dataset. `0` means no cap. |
| `rag_dataset_split` | string | `train` | Dataset split used for RAG organization. |
| `rag_max_items_per_dataset` | int | `0` | RAG item cap per dataset. `0` means no cap. |
| `rag_sample_size` | int | `15` | Sample rows shown to the model when generating a RAG processor. |
| `rag_max_retries` | int | `3` | Maximum coder-checker attempts for RAG processors. |
| `rag_streaming` | bool | `true` | Stream RAG datasets from Hugging Face. Set to `false` for tiny smoke tests that should load a finite split slice and exit cleanly. |

## Metrics

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `enable_stage_metrics` | bool | `true` | Write stage timing and LLM usage metrics. |
| `model_pricing_path` | string | empty | Optional model-pricing file for cost estimates. |
