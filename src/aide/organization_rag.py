import json
import os
import re
import shutil
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeAlias

from datasets import load_dataset
from datasets.exceptions import DatasetGenerationCastError
from datasets.naming import camelcase_to_snakecase
from loguru import logger
from pydantic import BaseModel
from tqdm import tqdm

from .config import AIDEConfig
from .metrics import (
    DatasetRunStateStore,
    make_dataset_scope_slug,
    metrics_scope,
    record_counter,
    record_metadata,
    record_named_duration,
)
from .models import DatasetInfo
from .prompts import prompt_for_rag_process, prompt_for_rag_process_validator
from .utils import (
    OpenAIClient,
    extract_code,
    make_safe_call,
    remove_non_serializable_fields,
    truncate_fields,
)

ProcessDataFn: TypeAlias = Callable[[str], list[dict[str, str]]]

_DATASET_LOAD_TIMEOUT_SEC = float(
    os.environ.get("DATASET_LOAD_TIMEOUT_SEC", "300") or 300
)


def _is_non_retryable_dataset_load_error(exc: Exception) -> bool:
    if isinstance(exc, DatasetGenerationCastError):
        return True

    error_text = str(exc).lower()
    non_retryable_patterns = [
        "all the data files must have the same columns",
        "column names don't match",
        "bad split: train",
        "no (supported) data files found",
        "couldn't infer the same data file format for all splits",
        "dataset scripts are no longer supported",
        "must be called with a dataclass type or instance",
    ]
    return any(pattern in error_text for pattern in non_retryable_patterns)


def _dataset_cache_root(cache_dir: str, dataset_name: str) -> Path:
    parts = dataset_name.split("/")
    parts[-1] = camelcase_to_snakecase(parts[-1])
    return Path(cache_dir) / "___".join(parts)


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    try:
        for p in path.rglob("*"):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except OSError:
                continue
    except OSError:
        return 0
    return total


def _call_with_timeout(
    fn: Callable[..., Any],
    *,
    args: tuple = (),
    kwargs: dict | None = None,
    timeout_sec: float = 300.0,
    description: str = "call",
) -> Any:
    """Run ``fn(*args, **kwargs)`` in a daemon thread with a wall-clock cap.

    On timeout we raise ``TimeoutError`` so the surrounding retry/backoff
    decorator (``make_safe_call``) decides whether to retry. The runaway
    thread is left as a daemon and will be reaped on interpreter exit; for
    HF dataset loads, ``_cleanup_stale_incomplete_dirs`` then deletes any
    ``_builder.lock`` it left behind so the next attempt does not block on
    the dead lock.
    """
    if kwargs is None:
        kwargs = {}
    box: dict[str, Any] = {"result": None, "exc": None, "done": False}

    def _runner() -> None:
        try:
            box["result"] = fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001
            box["exc"] = exc
        finally:
            box["done"] = True

    t = threading.Thread(target=_runner, name=f"timeout:{description}", daemon=True)
    t.start()
    t.join(timeout_sec)
    if not box["done"]:
        raise TimeoutError(
            f"{description} did not finish within {timeout_sec:.0f}s; abandoning thread"
        )
    if box["exc"] is not None:
        raise box["exc"]
    return box["result"]


def _cleanup_stale_incomplete_dirs(cache_dir: str, dataset_info: DatasetInfo) -> int:
    if not cache_dir:
        return 0

    cache_root = _dataset_cache_root(cache_dir, dataset_info.name)
    if not cache_root.exists():
        return 0

    removed = 0
    patterns = [f"{dataset_info.subset}/*/*.incomplete", "*/*/*.incomplete"]
    seen_paths = set()
    for pattern in patterns:
        for incomplete_dir in cache_root.glob(pattern):
            resolved = str(incomplete_dir)
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            shutil.rmtree(incomplete_dir, ignore_errors=True)
            removed += 1

    # Also drop any orphan _builder.lock files. If a previous attempt's thread
    # was abandoned (e.g., wall-clock timeout), it leaves a held lock that the
    # next call would block on forever. Since we are single-process, removing
    # these is safe.
    lock_removed = 0
    for lock_pattern in (
        f"{dataset_info.subset}/*/*_builder.lock",
        "*/*/*_builder.lock",
        f"{dataset_info.subset}/*/*.lock",
        "*/*/*.lock",
    ):
        for lock_path in cache_root.glob(lock_pattern):
            try:
                lock_path.unlink()
                lock_removed += 1
            except OSError:
                continue

    if removed or lock_removed:
        logger.warning(
            f"Cleaned cache for {dataset_info.name} - {dataset_info.subset}: "
            f"{removed} stale incomplete dirs, {lock_removed} orphan lock files"
        )
    return removed + lock_removed


def _truncate_error_text(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...(truncated)"


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
        code_str: str | None = None,
        chat_history: list[dict] | None = None,
        process_data: ProcessDataFn | None = None,
    ):
        self.success = success
        self.chunk_count = chunk_count
        self.code_str = code_str
        self.chat_history = chat_history or []
        self.process_data = process_data


class SingleItemResult:
    """Result of processing a single item."""

    def __init__(
        self,
        success: bool,
        items: list[dict[str, str]],
        validated: bool,
        error_message: str | None = None,
    ):
        self.success = success
        self.items = items
        self.validated = validated
        self.error_message = error_message


class RAGOrganizer:
    def __init__(self, config: AIDEConfig):
        self.config = config
        self.max_retries = config.rag_max_retries
        self._current_dataset_scope: str | None = None

    def _get_state_store(self) -> DatasetRunStateStore:
        return DatasetRunStateStore(
            os.path.join(self.config.output_dir, "final_stage_dataset_state.json")
        )

    def _stage(self, base_name: str) -> str:
        # Per-dataset attribution: "<dataset_scope>.<base_name>" when inside
        # a dataset, otherwise the bare base_name. Letting each dataset have
        # its own bucket means we can compare apples-to-apples and attribute
        # HF load / codegen / file_write time to specific datasets, instead
        # of having a single global aggregate.
        if self._current_dataset_scope:
            return f"{self._current_dataset_scope}.{base_name}"
        return base_name

    def process(self, dataset_info: DatasetInfo) -> bool:
        logger.info(
            f"RAG processing: dataset {dataset_info.name} - {dataset_info.subset}"
        )

        state_store = self._get_state_store()
        out_file, code_file = self._get_output_paths(dataset_info)
        dataset_scope = f"organization.dataset::{make_dataset_scope_slug(dataset_info)}"
        dataset_metadata = {
            "dataset_name": dataset_info.name,
            "subset": dataset_info.subset,
            "mode": "rag",
            "output_path": out_file,
            "code_path": code_file,
        }
        prev_dataset_scope = self._current_dataset_scope
        self._current_dataset_scope = dataset_scope

        try:
            with metrics_scope(dataset_scope, metadata=dataset_metadata):
                state_store.update(
                    dataset_info,
                    scope_name=dataset_scope,
                    output_path=out_file,
                    code_path=code_file,
                )
                if self._should_skip_processing(dataset_info, state_store):
                    existing_state = state_store.get(dataset_info) or {}
                    state_store.update(
                        dataset_info,
                        status=existing_state.get("status", "completed"),
                        last_error="",
                        output_path=out_file,
                        code_path=code_file,
                    )
                    record_counter("skipped_existing_output", 1)
                    return True

                state_store.increment(dataset_info, "attempts", 1)
                state_store.update(
                    dataset_info,
                    status="running",
                    last_error="",
                    output_path=out_file,
                    code_path=code_file,
                )

                gpt4 = OpenAIClient(system_message=self.config.system_message)
                with metrics_scope(
                    self._stage("dataset_load"), metadata=dataset_metadata
                ):
                    ds = self._load_dataset(dataset_info)
                if ds is None:
                    logger.warning(
                        f"Failed to load dataset {dataset_info.name} - {dataset_info.subset}"
                    )
                    state_store.update(
                        dataset_info,
                        status="load_failed",
                        last_error="Failed to load dataset after retries.",
                    )
                    return False

                with metrics_scope(
                    self._stage("sample_examples"), metadata=dataset_metadata
                ):
                    sample_str = self._sample_examples(
                        ds, n=self.config.rag_sample_size
                    )
                chat_history = self._initialize_chat_history(sample_str)

                result = self._process_with_retries(
                    gpt4, ds, chat_history, dataset_info
                )

                logger.info(f"Processing result: {result.success}")
                if result.success:
                    logger.info(f"Processed {result.chunk_count} chunks successfully")
                    if result.code_str:
                        self._write_code_output(dataset_info, result.code_str)
                    state_store.update(
                        dataset_info,
                        status="completed",
                        chunk_count=result.chunk_count,
                        best_effort_used=False,
                        last_error="",
                    )
                    return True

                best_effort_result = self._try_best_effort_processing(
                    ds, dataset_info, result.code_str, result.process_data
                )
                logger.info(
                    f"Best effort processing result: {best_effort_result.success}"
                )

                if best_effort_result.success:
                    logger.info(
                        f"Best effort processing completed with {best_effort_result.chunk_count} chunks"
                    )
                    if best_effort_result.code_str:
                        self._write_code_output(
                            dataset_info, best_effort_result.code_str
                        )
                    state_store.update(
                        dataset_info,
                        status="completed_best_effort",
                        chunk_count=best_effort_result.chunk_count,
                        best_effort_used=True,
                        last_error="",
                    )
                    return True

                last_error = self._extract_last_error(result.chat_history)
                state_store.update(
                    dataset_info,
                    status="failed",
                    chunk_count=result.chunk_count,
                    best_effort_used=False,
                    last_error=last_error,
                )
                return False
        finally:
            self._current_dataset_scope = prev_dataset_scope

    def _extract_last_error(self, chat_history: list[dict]) -> str:
        if not chat_history:
            return "All retries failed for RAG processing."
        last_message = chat_history[-1].get("content", "")
        return _truncate_error_text(str(last_message))

    def _should_skip_processing(
        self, dataset_info: DatasetInfo, state_store: DatasetRunStateStore
    ) -> bool:
        out_file, _ = self._get_output_paths(dataset_info)
        state = state_store.get(dataset_info) or {}
        status = state.get("status", "")
        if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
            if status in {"completed", "completed_best_effort"}:
                logger.info(
                    f"Output file {out_file} already exists and has completed state. Skipping processing."
                )
                return True
            logger.warning(
                f"Output file {out_file} exists without a completed state ({status or 'missing'}). Rerunning to avoid treating a partial file as complete."
            )
        return False

    def _initialize_chat_history(self, sample_str: str) -> list[dict]:
        prompt = prompt_for_rag_process.replace(
            "{{user_query}}", self.config.prompt
        ).replace("{{sample_data}}", sample_str)
        return [{"role": "system", "content": prompt}]

    def _process_with_retries(
        self,
        gpt4: OpenAIClient,
        ds,
        chat_history: list[dict],
        dataset_info: DatasetInfo,
    ) -> ProcessingResult:
        process_data = None
        code_str = None
        numbered_code = None

        for attempt in range(self.max_retries):
            logger.info(f"Processing attempt {attempt + 1}/{self.max_retries}")
            record_counter("retry_attempts", 1)

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
                    True, result.chunk_count, code_str, chat_history, process_data
                )

            chat_history = result.chat_history
            logger.error(
                "Attempt failed, retrying with refined process_data function..."
            )

        logger.error("All retries failed for RAG processing.")
        return ProcessingResult(False, 0, code_str, chat_history, process_data)

    def _get_or_update_process_function(
        self,
        gpt4: OpenAIClient,
        chat_history: list[dict],
        current_code: str | None,
        current_numbered_code: str | None,
        dataset_info: DatasetInfo,
        attempt: int,
    ) -> tuple[ProcessDataFn | None, list[dict], str | None, str | None]:
        process_data, chat_history, new_code_str, new_numbered_code = (
            self._generate_process_data_function(
                gpt4,
                chat_history,
                return_raw_code=True,
                dataset_info=dataset_info,
                attempt=attempt,
            )
        )

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
        chat_history: list[dict],
        dataset_info: DatasetInfo,
        numbered_code: str | None,
        code_str: str | None = None,
    ) -> ProcessingResult:
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
        self,
        ds,
        dataset_info: DatasetInfo,
        code_str: str | None,
        process_data: ProcessDataFn | None,
    ) -> ProcessingResult:
        if process_data is None:
            logger.error(
                "No process_data function found in the response during best efforts."
            )
            return ProcessingResult(False, 0, code_str)

        logger.info(
            "Attempting best-effort processing with last generated process_data, ignoring failed items."
        )
        record_counter("best_effort_attempts", 1)

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

        return ProcessingResult(
            result.success, result.chunk_count, code_str, result.chat_history
        )

    def _process_items(
        self,
        process_data: ProcessDataFn,
        ds,
        dataset_info: DatasetInfo,
        gpt4: OpenAIClient | None,
        chat_history: list[dict] | None,
        numbered_code: str | None,
        code_str: str | None = None,
        best_effort: bool = False,
    ) -> ProcessingResult:
        output_has_been_validated = False
        skipped_count = 0
        chunk_count = 0
        items_seen = 0
        file_write_time_sec = 0.0

        out_file, _ = self._get_output_paths(dataset_info)
        # Stream to a temp file and only atomically replace `out_file` once
        # we have a successful (or best-effort with chunks) result. This way a
        # failed retry never wipes out a previously-good output that took hours
        # to produce.
        tmp_out = f"{out_file}.tmp.{os.getpid()}.{uuid.uuid4().hex}"

        def _drop_tmp() -> None:
            try:
                if os.path.exists(tmp_out):
                    os.remove(tmp_out)
            except OSError:
                pass

        def _commit_tmp() -> None:
            # Atomic rename so concurrent readers never see a half-written file.
            try:
                os.replace(tmp_out, out_file)
            except OSError as exc:
                logger.error(f"Failed to commit output {tmp_out} -> {out_file}: {exc}")
                _drop_tmp()

        desc = (
            "Best-effort processing"
            if best_effort
            else f"Processing {dataset_info.name} - {dataset_info.subset}"
        )

        ds_id = f"{dataset_info.name}_{dataset_info.subset}"

        def finalize_stats() -> None:
            record_counter("items_seen", items_seen)
            if skipped_count:
                record_counter("items_skipped", skipped_count)
            if chunk_count:
                record_counter("chunks_written", chunk_count)
            record_named_duration(self._stage("file_write"), file_write_time_sec)

        try:
            with open(tmp_out, "w", encoding="utf-8") as f:
                for raw_item in tqdm(ds, desc=desc):
                    if (
                        self.config.rag_max_items_per_dataset > 0
                        and items_seen >= self.config.rag_max_items_per_dataset
                    ):
                        break
                    items_seen += 1
                    if best_effort:
                        success, processed_items = self._process_item_best_effort(
                            raw_item, process_data
                        )
                        if success:
                            for processed_item in processed_items:
                                processed_item["id"] = f"{ds_id}:{processed_item['id']}"
                                write_start = time.perf_counter()
                                f.write(
                                    f"{json.dumps(processed_item, ensure_ascii=False)}\n"
                                )
                                file_write_time_sec += time.perf_counter() - write_start
                                chunk_count += 1
                        else:
                            skipped_count += 1
                    else:
                        assert gpt4 is not None, (
                            "gpt4 is required for normal processing mode"
                        )
                        assert chat_history is not None, (
                            "chat_history is required for normal processing mode"
                        )

                        result = self._process_single_item(
                            raw_item,
                            process_data,
                            gpt4,
                            output_has_been_validated,
                            numbered_code,
                            code_str,
                        )

                        if not result.success:
                            finalize_stats()
                            # Single-item failure means we'll regenerate
                            # process_data and retry; the partial temp is
                            # written by stale code, so discard it. Previous
                            # good `out_file` (if any) is preserved untouched.
                            _drop_tmp()
                            chat_history.append(
                                {"role": "user", "content": result.error_message}
                            )
                            return ProcessingResult(
                                False, chunk_count, None, chat_history
                            )

                        for processed_item in result.items:
                            processed_item["id"] = f"{ds_id}:{processed_item['id']}"
                            write_start = time.perf_counter()
                            f.write(
                                f"{json.dumps(processed_item, ensure_ascii=False)}\n"
                            )
                            file_write_time_sec += time.perf_counter() - write_start
                            chunk_count += 1

                        output_has_been_validated = result.validated

        except Exception as e:
            finalize_stats()
            logger.error(f"Dataset iteration failed: {e}")
            # Iteration crash: in best-effort mode keep what we got if any
            # chunks were written, otherwise discard so prior good output
            # survives.
            if best_effort and chunk_count > 0:
                _commit_tmp()
            else:
                _drop_tmp()
            return ProcessingResult(False, chunk_count, None, chat_history)

        finalize_stats()
        # Successful full iteration: replace the previous output atomically.
        _commit_tmp()

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
        numbered_code: str | None,
        raw_code: str | None = None,
    ) -> SingleItemResult:
        item_str = json.dumps(item, ensure_ascii=False)

        process_start = time.perf_counter()
        try:
            processed_items = process_data(item_str)
        except Exception as e:
            record_named_duration(
                self._stage("local_process_data"),
                time.perf_counter() - process_start,
            )
            error_msg = self._create_processing_error_message(
                item_str, e, numbered_code
            )
            return SingleItemResult(False, [], output_validated, error_msg)
        record_named_duration(
            self._stage("local_process_data"),
            time.perf_counter() - process_start,
        )

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

    def _create_validation_error_message(
        self, item_str: str, processed_items, validation_result: RagValidationResult
    ) -> str:
        logger.error(
            f"Validation failed for item: {item_str}\nErrors: {validation_result.errors}"
        )
        return (
            f"I don't think the output is correct for Item: {item_str}\n"
            f"Processed: {processed_items}\n"
            f"Errors: {validation_result.errors}"
        )

    def _create_processing_error_message(
        self, item_str: str, exception: Exception, numbered_code: str | None
    ) -> str:
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

    def _get_failing_line_info(self, tb_obj, numbered_code: str | None) -> str:
        failing_lineno = None
        if tb_obj is not None:
            tb_pd = tb_obj.tb_next
            if tb_pd is not None:
                failing_lineno = tb_pd.tb_lineno

        if failing_lineno and numbered_code:
            code_lines = numbered_code.splitlines()
            context = code_lines[max(0, failing_lineno - 3) : failing_lineno + 2]
            context_str = "\n".join(context)
            return (
                f"\n[process_data failed at generated code line: {failing_lineno}]\n"
                f"Context:\n{context_str}"
            )
        if failing_lineno:
            return f"\n[process_data failed at generated code line: {failing_lineno}]"

        return ""

    def _process_item_best_effort(
        self, raw_item, process_data: ProcessDataFn
    ) -> tuple[bool, list[dict[str, str]]]:
        item_str = json.dumps(raw_item, ensure_ascii=False)
        process_start = time.perf_counter()
        try:
            processed_items = process_data(item_str)
            record_named_duration(
                self._stage("local_process_data"),
                time.perf_counter() - process_start,
            )
            if processed_items is not None and hasattr(processed_items, "__iter__"):
                return True, processed_items
            return False, []
        except Exception:
            record_named_duration(
                self._stage("local_process_data"),
                time.perf_counter() - process_start,
            )
            return False, []

    def _parse_validation_response(self, response: str) -> RagValidationResult:
        try:
            thinking = "No thinking section found"
            errors = []
            is_correct = False

            thinking_match = re.search(
                r"## THINKING:\s*\n(.*?)(?=## ERRORS:|## RESULT:|$)",
                response,
                re.DOTALL,
            )
            if thinking_match:
                thinking = thinking_match.group(1).strip()

            errors_match = re.search(
                r"## ERRORS:\s*\n(.*?)(?=## RESULT:|$)", response, re.DOTALL
            )
            if errors_match:
                errors_text = errors_match.group(1).strip()
                if errors_text.lower() != "none":
                    error_lines = [
                        line.strip() for line in errors_text.split("\n") if line.strip()
                    ]
                    errors = [
                        line[2:].strip()
                        for line in error_lines
                        if line.startswith("- ")
                    ]

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
        process_code: str | None = None,
    ) -> RagValidationResult:
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

        with metrics_scope(self._stage("validation_llm")):
            response = gpt4.chat(
                message=messages,
                model=self.config.organization_model,
            )

        if not response:
            return RagValidationResult(
                thinking="Failed to validate due to no response from LLM",
                errors=["Validation service returned no response"],
                is_correct=False,
            )

        return self._parse_validation_response(response)

    def _get_output_paths(self, dataset_info: DatasetInfo) -> tuple[str, str]:
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
        if not os.path.exists(self.config.output_dir):
            os.makedirs(self.config.output_dir, exist_ok=True)
        ds_name = dataset_info.name.replace("/", "_").replace(" ", "_")
        revision_file = os.path.join(
            self.config.output_dir,
            f"{ds_name}_{dataset_info.subset}_code_attempt{attempt}_rev{revision}.py",
        )
        with open(revision_file, "w", encoding="utf-8") as f:
            f.write(f"# Code revision for attempt {attempt}, revision {revision}\n")
            f.write(f"# Dataset: {dataset_info.name} - {dataset_info.subset}\n\n")
            f.write(code_str)
        logger.info(f"Code revision saved to {revision_file}")

    def _write_code_output(
        self,
        dataset_info: DatasetInfo,
        code_str: str,
    ):
        _, code_file = self._get_output_paths(dataset_info)
        with open(code_file, "w", encoding="utf-8") as cf:
            cf.write(code_str)
        logger.info(f"process_data code saved to {code_file}")

    def _generate_process_data_function(
        self,
        gpt4: OpenAIClient,
        chat_history: list[dict],
        return_raw_code: bool = False,
        dataset_info: DatasetInfo | None = None,
        attempt: int = 0,
    ) -> tuple[ProcessDataFn | None, list[dict], str | None, str | None]:
        with metrics_scope(self._stage("codegen_llm")):
            response = gpt4.chat(chat_history, model=self.config.organization_model)
        if not response:
            logger.error("No response from LLM for RAG processing.")
            return None, chat_history, None, None

        chat_history = chat_history + [{"role": "assistant", "content": response}]
        code_str = extract_code(response)
        numbered_code = None

        if code_str:
            numbered_code = "\n".join(
                f"{i + 1:4}: {line}" for i, line in enumerate(code_str.splitlines())
            )
            if dataset_info:
                self._save_code_revision(dataset_info, code_str, attempt)

        exec_start = time.perf_counter()
        try:
            namespace = {
                "__builtins__": __builtins__,
                "json": json,
                "os": os,
                "re": re,
                "uuid": uuid,
            }
            exec(code_str, namespace)
            record_named_duration(
                self._stage("generated_code_exec"),
                time.perf_counter() - exec_start,
            )
        except Exception as exec_e:
            record_named_duration(
                self._stage("generated_code_exec"),
                time.perf_counter() - exec_start,
            )
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

        process_data = namespace.get("process_data")
        if not callable(process_data):
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
            process_data,
            chat_history,
            code_str if return_raw_code else None,
            numbered_code,
        )

    def _sample_examples(self, ds, n: int = 15, max_length: int = 10000) -> str:
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

    @make_safe_call(
        allow_failure=True,
        giveup=_is_non_retryable_dataset_load_error,
    )
    def _load_dataset(self, dataset_info: DatasetInfo):
        split = self.config.rag_dataset_split
        streaming = self.config.rag_streaming
        if not streaming and self.config.rag_max_items_per_dataset > 0:
            slice_size = max(
                self.config.rag_max_items_per_dataset,
                self.config.rag_sample_size,
            )
            split = f"{split}[:{slice_size}]"
        _cleanup_stale_incomplete_dirs(self.config.cache_dir, dataset_info)

        # Snapshot cache state before the call so we can attribute time
        # to "downloaded" vs "cached" and report bytes pulled from HF.
        cache_root = _dataset_cache_root(self.config.cache_dir, dataset_info.name)
        was_cached = cache_root.exists() and any(cache_root.rglob("*.parquet"))
        bytes_before = _dir_size_bytes(cache_root)
        download_start = time.perf_counter()

        try:
            ds = _call_with_timeout(
                load_dataset,
                kwargs=dict(
                    path=dataset_info.name,
                    name=dataset_info.subset,
                    split=split,
                    cache_dir=self.config.cache_dir,
                    streaming=streaming,
                ),
                timeout_sec=_DATASET_LOAD_TIMEOUT_SEC,
                description=(
                    f"load_dataset({dataset_info.name}, {dataset_info.subset})"
                ),
            )
        except OSError as exc:
            if "directory not empty" in str(exc).lower():
                _cleanup_stale_incomplete_dirs(self.config.cache_dir, dataset_info)
            raise

        load_elapsed = time.perf_counter() - download_start
        bytes_after = _dir_size_bytes(cache_root)
        bytes_downloaded = max(0, bytes_after - bytes_before)
        # If we started cold, classify as "download"; otherwise "cache_load".
        # We log both numbers so the export script can split per-dataset
        # download time from per-dataset cache load time.
        record_metadata(
            {
                "hf_load_was_cached": was_cached,
                "hf_load_bytes_downloaded": bytes_downloaded,
                "hf_cache_bytes_after": bytes_after,
            }
        )
        record_counter("hf_load_calls", 1)
        if was_cached:
            record_counter("hf_load_cached", 1)
            record_named_duration(self._stage("hf_cache_load"), load_elapsed)
        else:
            record_counter("hf_load_downloaded", 1)
            record_named_duration(self._stage("hf_download"), load_elapsed)
            if bytes_downloaded:
                record_counter("hf_bytes_downloaded", bytes_downloaded)
        return ds
