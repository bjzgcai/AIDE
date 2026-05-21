import json
import os
import re
import sys
import traceback
from typing import Callable, List, Optional, Tuple, TypeAlias

from datasets import load_dataset
from loguru import logger
from pydantic import BaseModel
from tqdm import tqdm

from .configs import InstAgentConfig
from .dataset_info import DatasetInfo
from .prompts import prompt_for_rag_process, prompt_for_rag_process_validator
from .utils import (
    OpenAIClient,
    extract_code,
    make_safe_call,
    remove_non_serializable_fields,
    truncate_fields,
)

ProcessDataFn: TypeAlias = Callable[[str], list[dict[str, str]]]


class RagValidationResult(BaseModel):
    thinking: str
    errors: list[str]
    is_correct: bool


class ProcessingResult:
    """Result of processing a dataset."""

    def __init__(
        self,
        success: bool,
        chunk_count: int = 0,
        code_str: Optional[str] = None,
        chat_history: Optional[List[dict]] = None,
    ):
        self.success = success
        self.chunk_count = chunk_count  # Changed from chunks list to count
        self.code_str = code_str
        self.chat_history = chat_history or []


class SingleItemResult:
    """Result of processing a single item."""

    def __init__(
        self,
        success: bool,
        items: List[dict[str, str]],
        validated: bool,
        error_message: Optional[str] = None,
    ):
        self.success = success
        self.items = items
        self.validated = validated
        self.error_message = error_message


class AdvancedProcessorRAG:
    def __init__(self, config: InstAgentConfig):
        self.config = config
        self.max_retries = 3

    def process(self, dataset_info: DatasetInfo) -> bool:
        """Main processing method - now simplified and focused."""
        logger.info(
            f"RAG processing: dataset {dataset_info.name} - {dataset_info.subset}"
        )

        # Early return if output already exists
        if self._should_skip_processing(dataset_info):
            return True

        # Initialize components
        gpt4 = OpenAIClient(system_message=self.config.system_message)
        ds = self._load_dataset(dataset_info)
        if not ds:
            # The ds is None even after retries
            logger.warning(
                f"Failed to load dataset {dataset_info.name} - {dataset_info.subset}"
            )
            return False

        chat_history = self._initialize_chat_history(ds)

        # Attempt processing with retries
        result = self._process_with_retries(gpt4, ds, chat_history, dataset_info)

        logger.info(f"Processing result: {result.success}")
        if result.success:
            logger.info(f"Processed {result.chunk_count} chunks successfully")
            # Save the final successful code
            if result.code_str:
                self._write_code_output(dataset_info, result.code_str)
            return True

        # Try best-effort processing if all retries failed
        best_effort_result = self._try_best_effort_processing(
            ds, dataset_info, result.code_str
        )
        logger.info(f"Best effort processing result: {best_effort_result.success}")

        if best_effort_result.success:
            logger.info(
                f"Best effort processing completed with {best_effort_result.chunk_count} chunks"
            )
            # Save the code used in best effort processing
            if best_effort_result.code_str:
                self._write_code_output(dataset_info, best_effort_result.code_str)
            return True

        return False

    def _should_skip_processing(self, dataset_info: DatasetInfo) -> bool:
        """Check if processing should be skipped because output already exists."""
        out_file, _ = self._get_output_paths(dataset_info)
        if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
            logger.info(
                f"Output file {out_file} already exists and is non-empty. Skipping processing."
            )
            return True
        return False

    def _initialize_chat_history(self, ds) -> List[dict]:
        """Initialize chat history with system prompt and sample data."""
        sample_str = self._sample_examples(ds)
        prompt = prompt_for_rag_process.replace(
            "{{user_query}}", self.config.prompt
        ).replace("{{sample_data}}", sample_str)
        return [{"role": "system", "content": prompt}]

    def _process_with_retries(
        self,
        gpt4: OpenAIClient,
        ds,
        chat_history: List[dict],
        dataset_info: DatasetInfo,
    ) -> ProcessingResult:
        """Process dataset with retry logic."""
        process_data = None
        code_str = None
        numbered_code = None

        for attempt in range(self.max_retries):
            logger.info(f"Processing attempt {attempt + 1}/{self.max_retries}")

            # Generate or update process_data function
            try:
                process_data, chat_history, code_str, numbered_code = (
                    self._get_or_update_process_function(
                        gpt4,
                        chat_history,
                        code_str,
                        numbered_code,
                        dataset_info,
                        attempt + 1,
                    )
                )
            except Exception as e:
                logger.error(
                    f"Failed to generate process_data function on attempt {attempt + 1}: {e}"
                )
                continue

            if process_data is None:
                logger.error(
                    f"Failed to generate process_data function on attempt {attempt + 1}"
                )
                continue

            # Try processing all items
            result = self._process_all_items(
                process_data,
                ds,
                gpt4,
                chat_history,
                dataset_info,
                numbered_code,
                code_str,
            )

            if result.success:
                return ProcessingResult(
                    True, result.chunk_count, code_str, chat_history
                )

            # Update chat history with error for next attempt
            chat_history = result.chat_history
            logger.error(
                "Attempt failed, retrying with refined process_data function..."
            )

        logger.error("All retries failed for RAG processing.")
        return ProcessingResult(False, 0, code_str, chat_history)

    def _get_or_update_process_function(
        self,
        gpt4: OpenAIClient,
        chat_history: List[dict],
        current_code: Optional[str],
        current_numbered_code: Optional[str],
        dataset_info: DatasetInfo,
        attempt: int,
    ) -> Tuple[Optional[ProcessDataFn], List[dict], Optional[str], Optional[str]]:
        """Generate or update the process_data function."""
        process_data, chat_history, new_code_str, new_numbered_code = (
            self._generate_process_data_function(
                gpt4,
                chat_history,
                return_raw_code=True,
                dataset_info=dataset_info,
                attempt=attempt,
            )
        )

        # Update code strings if we got new ones
        code_str = new_code_str if new_code_str is not None else current_code
        numbered_code = (
            new_numbered_code
            if new_numbered_code is not None
            else current_numbered_code
        )

        return process_data, chat_history, code_str, numbered_code

    def _process_all_items(
        self,
        process_data: ProcessDataFn,
        ds,
        gpt4: OpenAIClient,
        chat_history: List[dict],
        dataset_info: DatasetInfo,
        numbered_code: Optional[str],
        code_str: Optional[str] = None,
    ) -> ProcessingResult:
        """Process all items in the dataset."""
        return self._process_items(
            process_data=process_data,
            ds=ds,
            dataset_info=dataset_info,
            gpt4=gpt4,
            chat_history=chat_history,
            numbered_code=numbered_code,
            code_str=code_str,
            best_effort=False,
        )

    def _try_best_effort_processing(
        self, ds, dataset_info: DatasetInfo, code_str: Optional[str]
    ) -> ProcessingResult:
        """Try best-effort processing with last available process_data function."""
        if "process_data" not in globals():
            logger.error(
                "No process_data function found in the response during best efforts."
            )
            return ProcessingResult(False, 0, code_str)

        logger.info(
            "Attempting best-effort processing with last generated process_data, ignoring failed items."
        )

        process_data = globals()["process_data"]
        result = self._process_items(
            process_data=process_data,
            ds=ds,
            dataset_info=dataset_info,
            gpt4=None,
            chat_history=None,
            numbered_code=None,
            code_str=code_str,
            best_effort=True,
        )

        # Return the result with the code_str for saving
        return ProcessingResult(
            result.success, result.chunk_count, code_str, result.chat_history
        )

    def _process_items(
        self,
        process_data: ProcessDataFn,
        ds,
        dataset_info: DatasetInfo,
        gpt4: Optional[OpenAIClient],
        chat_history: Optional[List[dict]],
        numbered_code: Optional[str],
        code_str: Optional[str] = None,
        best_effort: bool = False,
    ) -> ProcessingResult:
        """Unified method to process items with different strategies."""
        output_has_been_validated = False
        skipped_count = 0
        chunk_count = 0

        # Get output file path and open for streaming write
        out_file, _ = self._get_output_paths(dataset_info)

        # Set up description based on mode
        desc = (
            "Best-effort processing"
            if best_effort
            else f"Processing {dataset_info.name} - {dataset_info.subset}"
        )

        ds_id = f"{dataset_info.name}_{dataset_info.subset}"
        try:
            with open(out_file, "w", encoding="utf-8") as f:
                for raw_item in tqdm(ds, desc=desc):
                    if best_effort:
                        # Best effort mode: skip failed items and continue
                        success, processed_items = self._process_item_best_effort(
                            raw_item, process_data
                        )
                        if success:
                            # Write items directly to file
                            for processed_item in processed_items:
                                processed_item["id"] = f'{ds_id}:{processed_item["id"]}'
                                f.write(
                                    f"{json.dumps(processed_item, ensure_ascii=False)}\n"
                                )
                                chunk_count += 1
                        else:
                            skipped_count += 1
                    else:
                        # Normal mode: validate and break on first error
                        # These should never be None in normal mode
                        assert (
                            gpt4 is not None
                        ), "gpt4 is required for normal processing mode"
                        assert (
                            chat_history is not None
                        ), "chat_history is required for normal processing mode"

                        result = self._process_single_item(
                            raw_item,
                            process_data,
                            gpt4,
                            output_has_been_validated,
                            numbered_code,
                            code_str,
                        )

                        if not result.success:
                            # Add error to chat history and break
                            chat_history.append(
                                {"role": "user", "content": result.error_message}
                            )
                            return ProcessingResult(
                                False, chunk_count, None, chat_history
                            )

                        # Write items directly to file
                        for processed_item in result.items:
                            processed_item["id"] = f'{ds_id}:{processed_item["id"]}'
                            f.write(
                                f"{json.dumps(processed_item, ensure_ascii=False)}\n"
                            )
                            chunk_count += 1

                        output_has_been_validated = result.validated

        except Exception as e:
            logger.error(f"Dataset iteration failed: {e}")
            return ProcessingResult(False, chunk_count, None, chat_history)

        # Log results based on mode
        if best_effort:
            logger.info(
                f"Best-effort: Found {chunk_count} chunks from dataset {dataset_info.name} - {dataset_info.subset}. Skipped {skipped_count} items."
            )
        else:
            logger.info(
                f"Found {chunk_count} chunks from dataset {dataset_info.name} - {dataset_info.subset}"
            )

        logger.info(f"RAG processing saved to {out_file}")
        return ProcessingResult(True, chunk_count, None, chat_history)

    def _process_single_item(
        self,
        item,
        process_data: ProcessDataFn,
        gpt4: OpenAIClient,
        output_validated: bool,
        numbered_code: Optional[str],
        raw_code: Optional[str] = None,
    ) -> SingleItemResult:
        """Process a single item and handle validation/errors."""
        item_str = json.dumps(item, ensure_ascii=False)

        try:
            processed_items = process_data(item_str)

            # Validate output if not already validated
            if not output_validated:
                validation_result = self._validate_generated_data(
                    gpt4, item, processed_items, raw_code
                )
                if not validation_result.is_correct:
                    error_msg = self._create_validation_error_message(
                        item_str, processed_items, validation_result
                    )
                    return SingleItemResult(False, [], False, error_msg)
                output_validated = True

            return SingleItemResult(True, processed_items, output_validated, None)

        except Exception as e:
            error_msg = self._create_processing_error_message(
                item_str, e, numbered_code
            )
            return SingleItemResult(False, [], output_validated, error_msg)

    def _create_validation_error_message(
        self, item_str: str, processed_items, validation_result: RagValidationResult
    ) -> str:
        """Create error message for validation failure."""
        logger.error(
            f"Validation failed for item: {item_str}\nErrors: {validation_result.errors}"
        )
        return f"I don't think the output is correct for Item: {item_str}\nProcessed: {processed_items}\nErrors: {validation_result.errors}"

    def _create_processing_error_message(
        self, item_str: str, exception: Exception, numbered_code: Optional[str]
    ) -> str:
        """Create error message for processing exception."""
        tb = traceback.format_exc()
        tb_obj = sys.exc_info()[2]
        failing_line_info = self._get_failing_line_info(tb_obj, numbered_code)

        error_msg = (
            f"Error processing item for RAG:\n"
            f"Item: {item_str}\n"
            f"Exception: {str(exception)}\n"
            f"Traceback:\n{tb}\n"
            f"Please refine the process_data function to handle this case.{failing_line_info}"
        )
        logger.error(error_msg)
        return error_msg

    def _get_failing_line_info(self, tb_obj, numbered_code: Optional[str]) -> str:
        """Extract failing line information from traceback."""
        failing_lineno = None
        if tb_obj is not None:
            tb_pd = tb_obj.tb_next
            if tb_pd is not None:
                failing_lineno = tb_pd.tb_lineno

        if failing_lineno and numbered_code:
            code_lines = numbered_code.splitlines()
            context = code_lines[max(0, failing_lineno - 3) : failing_lineno + 2]
            context_str = "\n".join(context)
            return f"\n[process_data failed at generated code line: {failing_lineno}]\nContext:\n{context_str}"
        elif failing_lineno:
            return f"\n[process_data failed at generated code line: {failing_lineno}]"

        return ""

    def _process_item_best_effort(
        self, raw_item, process_data: ProcessDataFn
    ) -> Tuple[bool, List[dict[str, str]]]:
        """Process a single item in best-effort mode (no validation, skip errors)."""
        try:
            item_str = json.dumps(raw_item, ensure_ascii=False)
            processed_items = process_data(item_str)

            if processed_items is not None and hasattr(processed_items, "__iter__"):
                return True, processed_items
            else:
                return False, []

        except Exception:
            return False, []

    def _parse_validation_response(self, response: str) -> RagValidationResult:
        """Parse unstructured validation response into RagValidationResult."""
        try:
            # Initialize defaults
            thinking = "No thinking section found"
            errors = []
            is_correct = False

            # Extract thinking section
            thinking_match = re.search(
                r"## THINKING:\s*\n(.*?)(?=## ERRORS:|## RESULT:|$)",
                response,
                re.DOTALL,
            )
            if thinking_match:
                thinking = thinking_match.group(1).strip()

            # Extract errors section
            errors_match = re.search(
                r"## ERRORS:\s*\n(.*?)(?=## RESULT:|$)", response, re.DOTALL
            )
            if errors_match:
                errors_text = errors_match.group(1).strip()
                if errors_text.lower() != "none":
                    # Split by lines and extract items that start with "- "
                    error_lines = [
                        line.strip() for line in errors_text.split("\n") if line.strip()
                    ]
                    errors = [
                        line[2:].strip()
                        for line in error_lines
                        if line.startswith("- ")
                    ]

            # Extract result section
            result_match = re.search(
                r"## RESULT:\s*\n(.*?)(?=\n|$)", response, re.DOTALL
            )
            if result_match:
                result_text = result_match.group(1).strip().upper()
                is_correct = result_text == "CORRECT"

            return RagValidationResult(
                thinking=thinking, errors=errors, is_correct=is_correct
            )

        except Exception as e:
            logger.error(f"Failed to parse validation response: {e}")
            return RagValidationResult(
                thinking="Failed to parse validation response",
                errors=[f"Response parsing error: {str(e)}"],
                is_correct=False,
            )

    def _validate_generated_data(
        self,
        gpt4: OpenAIClient,
        raw_item,
        processed_items,
        process_code: Optional[str] = None,
    ) -> RagValidationResult:
        """Validate both process code and generated data using GPT-4."""

        # check if the key 'id' and 'contents' are in the processed_items
        for item in processed_items:
            if not isinstance(item, dict):
                return RagValidationResult(
                    thinking="Processed items are not dictionaries",
                    errors=[f"Processed item {item} is not a dictionary"],
                    is_correct=False,
                )
            if "id" not in item or not isinstance(item["id"], str):
                return RagValidationResult(
                    thinking="Processed items do not contain valid 'id'",
                    errors=[f"Processed item {item} does not contain valid 'id'"],
                    is_correct=False,
                )
            if "contents" not in item or not isinstance(item["contents"], str):
                return RagValidationResult(
                    thinking="Processed items do not contain valid 'contents'",
                    errors=[f"Processed item {item} does not contain valid 'contents'"],
                    is_correct=False,
                )

        system_prompt = prompt_for_rag_process_validator.replace(
            "{{user_query}}", self.config.prompt
        )
        raw_item = remove_non_serializable_fields(raw_item)
        raw_item = truncate_fields(raw_item)
        processed_items = [truncate_fields(item) for item in processed_items]

        user_text = (
            f"Raw item:\n{json.dumps(raw_item, indent=2, ensure_ascii=False)}\n\n"
            f"Processed items:\n{json.dumps(processed_items, indent=2, ensure_ascii=False)}\n\n"
        )

        if process_code:
            user_text += (
                f"Process_data function code:\n```python\n{process_code}\n```\n\n"
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]

        # Remove response_format parameter since it's not supported
        response = gpt4.chat(
            message=messages,
            model=self.config.advanced_process_model,
        )

        # Handle the case where gpt4.chat might return None
        if not response:
            return RagValidationResult(
                thinking="Failed to validate due to no response from LLM",
                errors=["Validation service returned no response"],
                is_correct=False,
            )

        # Parse the unstructured response
        return self._parse_validation_response(response)

    def _get_output_paths(self, dataset_info: DatasetInfo) -> Tuple[str, str]:
        """Get the output file paths for RAG data and code files."""
        if not os.path.exists(self.config.output_dir):
            os.makedirs(self.config.output_dir, exist_ok=True)
        ds_name = dataset_info.name.replace("/", "_").replace(" ", "_")
        out_file = os.path.join(
            self.config.output_dir,
            f"{ds_name}_{dataset_info.subset}_rag.jsonl",
        )
        code_file = os.path.join(
            self.config.output_dir,
            f"{ds_name}_{dataset_info.subset}_process_data.py",
        )
        return out_file, code_file

    def _save_code_revision(
        self, dataset_info: DatasetInfo, code_str: str, attempt: int, revision: int = 0
    ):
        """Save code revision for debugging purposes."""
        if not os.path.exists(self.config.output_dir):
            os.makedirs(self.config.output_dir, exist_ok=True)
        ds_name = dataset_info.name.replace("/", "_").replace(" ", "_")
        revision_file = os.path.join(
            self.config.output_dir,
            f"{ds_name}_{dataset_info.subset}_code_attempt{attempt}_rev{revision}.py",
        )
        with open(revision_file, "w") as f:
            f.write(f"# Code revision for attempt {attempt}, revision {revision}\n")
            f.write(f"# Dataset: {dataset_info.name} - {dataset_info.subset}\n\n")
            f.write(code_str)
        logger.info(f"Code revision saved to {revision_file}")

    def _write_code_output(
        self,
        dataset_info: DatasetInfo,
        code_str: str,
    ):
        """Write the final process_data code to output file."""
        _, code_file = self._get_output_paths(dataset_info)
        with open(code_file, "w") as cf:
            cf.write(code_str)
        logger.info(f"process_data code saved to {code_file}")

    def _generate_process_data_function(
        self,
        gpt4: OpenAIClient,
        chat_history: List[dict],
        return_raw_code: bool = False,
        dataset_info: Optional[DatasetInfo] = None,
        attempt: int = 0,
    ) -> Tuple[Optional[ProcessDataFn], List[dict], Optional[str], Optional[str]]:
        """Generate or update the process_data function."""
        response = gpt4.chat(chat_history, model=self.config.advanced_process_model)
        if not response:
            logger.error("No response from OpenAI for RAG processing.")
            return None, chat_history, None, None

        chat_history = chat_history + [{"role": "assistant", "content": response}]
        code_str = extract_code(response)
        numbered_code = None

        if code_str:
            numbered_code = "\n".join(
                f"{i+1:4}: {line}" for i, line in enumerate(code_str.splitlines())
            )

            # Save code revision for debugging
            if dataset_info:
                self._save_code_revision(dataset_info, code_str, attempt)

        try:
            exec(code_str, globals())
        except Exception as exec_e:
            logger.error(f"Error defining process_data: {exec_e}")
            refine_msg = (
                f"Error occurred when executing process_data:\n"
                f"{numbered_code if numbered_code else code_str}\n"
                f"Exception:\n"
                f"{exec_e}\n"
                f"Please refine the process_data function to fix the error. "
                f"Ensure your output ends with a python code block containing only the function."
            )
            chat_history = chat_history + [{"role": "user", "content": refine_msg}]
            return (
                None,
                chat_history,
                code_str if return_raw_code else None,
                numbered_code,
            )

        if "process_data" not in globals():
            logger.error("No process function generated for RAG processing.")
            refine_msg = (
                "No process_data function found in the response. "
                "Please refine the process_data function to fix the error. "
                "Ensure your output ends with a python code block containing only the function."
            )
            chat_history = chat_history + [{"role": "user", "content": refine_msg}]
            return (
                None,
                chat_history,
                code_str if return_raw_code else None,
                numbered_code,
            )

        return (
            globals()["process_data"],
            chat_history,
            code_str if return_raw_code else None,
            numbered_code,
        )

    def _sample_examples(self, ds, n: int = 15, max_length: int = 10000) -> str:
        """Sample examples from the dataset for prompting."""
        sample_list = []

        for example in ds:
            example = remove_non_serializable_fields(example)
            example = truncate_fields(example)
            sample_list.append(json.dumps(example, ensure_ascii=False))
            if len(sample_list) >= n:
                break

        sample_str = ""
        for idx, sample in enumerate(sample_list):
            sample_str += f"Sample {idx + 1}:\n{sample}\n"
            if len(sample_str) >= max_length:
                break
        return sample_str

    @make_safe_call(allow_failure=True)
    def _load_dataset(self, dataset_info: DatasetInfo):
        """Load dataset from HuggingFace."""
        split = "train"
        return load_dataset(
            path=dataset_info.name,
            name=dataset_info.subset,
            split=split,
            cache_dir=self.config.cache_dir,
            trust_remote_code=False,
            streaming=False,  # No streaming as the iter may fail due to network issues
        )


if __name__ == "__main__":
    config = InstAgentConfig("medical", advanced_process_model="Qwen/Qwen3-32B")
    processor = AdvancedProcessorRAG(config=config)
    dataset_info = DatasetInfo(
        name="MedRag/pubmed",
        subset="default",
        number_of_rows=2209839,
        columns=["id", "title", "content", "contents", "PMID"],
        description="dataset",
        hf_tags=["medical", "qa", "disease", "treatment"],
        modalities=["text"],
        license="cc-by-nc-4.0",
        n_likes=0,
        n_downloads_last_month=0,
    )
    processor.process(dataset_info)
