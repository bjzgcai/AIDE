import json
import os
import random
import re
import sys
import time
import traceback
from collections.abc import Callable
from itertools import islice
from typing import Any, TypeAlias

from datasets import load_dataset
from loguru import logger
from pydantic import BaseModel
from tqdm import tqdm

from .config import AIDEConfig
from .metrics import make_dataset_scope_slug, metrics_scope, record_named_duration
from .models import DatasetInfo
from .prompts import (
    prompt_for_extract_code_generation,
    prompt_for_extract_code_generation_template,
    prompt_for_instruction_task_discovery,
    prompt_for_template_generation,
    prompts_for_instruction_quality_check,
)
from .utils import OpenAIClient, extract_code, extract_json, save_json

InstructionProcessDataFn: TypeAlias = Callable[..., list[str]]


class InstructionValidationResult(BaseModel):
    is_correct: bool
    thinking: str
    data: str = ""


class GeneratedInstructionProcessor(BaseModel):
    process_data: Any
    chat_history: list[dict]
    code_str: str | None
    numbered_code: str | None


class InstructionOrganizer:
    def __init__(self, config: AIDEConfig):
        self.config = config

    def process(self, dataset_info: DatasetInfo) -> bool:
        logger.info(
            f"Instruction organization: dataset {dataset_info.name} - {dataset_info.subset}"
        )
        dataset_scope = f"organization.dataset::{make_dataset_scope_slug(dataset_info)}"
        dataset_metadata = {
            "dataset_name": dataset_info.name,
            "subset": dataset_info.subset,
            "mode": "instruction",
        }

        with metrics_scope(dataset_scope, metadata=dataset_metadata):
            gpt4 = OpenAIClient(system_message=self.config.system_message)
            split = "train"

            try:
                with metrics_scope(
                    "organization.dataset_load", metadata=dataset_metadata
                ):
                    ds = self._load_dataset(dataset_info, split)
                sample_rows, examples = self._sample_examples(ds)
            except Exception as e:
                logger.error(
                    f"Error loading instruction dataset {dataset_info.name}: {e}"
                )
                return False

            try:
                tasks = self._discover_tasks(gpt4, dataset_info, examples)
            except Exception as e:
                logger.error("Error in instruction task discovery output: " + str(e))
                return False

            logger.info(f"Possible tasks: {[task.get('task') for task in tasks]}")
            success = False

            for task in tasks:
                if not self._is_valid_task(task, dataset_info):
                    continue

                try:
                    prompts_list, extract_input = self._prepare_codegen_prompt(
                        gpt4, dataset_info, task, examples
                    )
                except Exception as e:
                    logger.error(f"Error generating prompts for task {task}: {e}")
                    continue

                chat_history = [{"role": "system", "content": extract_input}]
                processor = self._build_validated_processor(
                    gpt4=gpt4,
                    chat_history=chat_history,
                    sample_rows=sample_rows,
                    prompts_list=prompts_list,
                    task=task,
                )
                if processor is None:
                    logger.error(
                        f"Failed to validate process_data for task '{task['task']}' "
                        f"after {self.config.instruction_max_retries} attempts."
                    )
                    continue

                try:
                    data = self._run_full_processing(
                        processor.process_data,
                        dataset_info,
                        split,
                        prompts_list,
                    )
                    self._write_task_output(
                        dataset_info=dataset_info,
                        split=split,
                        task=task,
                        prompts_list=prompts_list,
                        code_str=processor.code_str or "",
                        data=data,
                    )
                except Exception as e:
                    logger.error(
                        f"Error in instruction organization task '{task['task']}': {e}"
                    )
                    continue

                logger.info(f"Instruction organization task '{task['task']}' done")
                success = True

            return success

    def _load_dataset(self, dataset_info: DatasetInfo, split: str):
        return load_dataset(
            path=dataset_info.name,
            name=dataset_info.subset,
            split=split,
            cache_dir=self.config.cache_dir,
            streaming=True,
        )

    def _sample_examples(self, ds) -> tuple[list[dict], str]:
        sample_rows: list[dict] = []
        for example in ds:
            if isinstance(example, dict):
                sample_rows.append(example)
            if len(sample_rows) >= self.config.instruction_sample_size:
                break

        examples = "\n".join(
            json.dumps(example, ensure_ascii=False) for example in sample_rows
        )
        return sample_rows, examples

    def _discover_tasks(
        self, gpt4: OpenAIClient, dataset_info: DatasetInfo, examples: str
    ) -> list[dict]:
        columns = ", ".join(dataset_info.columns)
        task_discovery_input = prompt_for_instruction_task_discovery.replace(
            "{{dataset_name}}", dataset_info.name
        )
        task_discovery_input = task_discovery_input.replace(
            "{{description}}", dataset_info.description
        )
        task_discovery_input = task_discovery_input.replace("{{columns}}", columns)
        task_discovery_input = task_discovery_input.replace("{{sample_data}}", examples)

        with metrics_scope("organization.task_discovery_llm"):
            task_discovery_output = gpt4.chat(
                task_discovery_input,
                model=self.config.organization_model,
            )

        tasks = extract_json(task_discovery_output)
        if not isinstance(tasks, list):
            raise ValueError("Instruction task discovery output must be a JSON list.")
        return tasks

    def _is_valid_task(self, task: dict, dataset_info: DatasetInfo) -> bool:
        task_name = task.get("task")
        input_columns = task.get("input columns")
        output_columns = task.get("output columns")
        if not isinstance(task_name, str) or not task_name:
            logger.error(f"Skipping malformed task without task name: {task}")
            return False
        if not isinstance(input_columns, list) or not isinstance(output_columns, list):
            logger.error(f"Skipping malformed task without column lists: {task}")
            return False

        missing_input = [
            col for col in input_columns if col not in dataset_info.columns
        ]
        missing_output = [
            col for col in output_columns if col not in dataset_info.columns
        ]
        if missing_input or missing_output:
            logger.error(
                f"Skipping task '{task_name}' due to missing columns. "
                f"missing_input={missing_input}, missing_output={missing_output}, "
                f"available={dataset_info.columns}"
            )
            return False
        return True

    def _prepare_codegen_prompt(
        self,
        gpt4: OpenAIClient,
        dataset_info: DatasetInfo,
        task: dict,
        examples: str,
    ) -> tuple[dict, str]:
        prompt_input = prompt_for_template_generation.replace(
            "{{task_name}}", task["task"]
        )
        prompt_input = prompt_input.replace(
            "{{input_modality}}", ", ".join(task["input columns"])
        )
        prompt_input = prompt_input.replace(
            "{{output_modality}}", ", ".join(task["output columns"])
        )
        prompt_input = prompt_input.replace("{{examples}}", examples)

        with metrics_scope("organization.template_generation_llm"):
            prompts = gpt4.chat(
                prompt_input,
                model=self.config.organization_model,
            )
        prompts_list = extract_json(prompts)
        if not isinstance(prompts_list, dict):
            raise ValueError("Template generation output must be a JSON object.")
        if "need_template" not in prompts_list:
            raise ValueError("Template generation output missing 'need_template'.")

        extract_input = self._build_extract_prompt(
            dataset_info, task, examples, prompts_list
        )
        return prompts_list, extract_input

    def _build_extract_prompt(
        self,
        dataset_info: DatasetInfo,
        task: dict,
        examples: str,
        prompts_list: dict,
    ) -> str:
        columns = ", ".join(dataset_info.columns)
        if prompts_list["need_template"]:
            extract_input = prompt_for_extract_code_generation_template.replace(
                "{{dataset_name}}", dataset_info.name
            )
            extract_input = extract_input.replace(
                "{{prompt_list}}", str(prompts_list.get("templates", []))
            )
        else:
            extract_input = prompt_for_extract_code_generation.replace(
                "{{dataset_name}}", dataset_info.name
            )

        replacements = {
            "{{columns}}": columns,
            "{{sample_data}}": examples,
            "{{task_name}}": str(task["task"]),
            "{{input_modality}}": ", ".join(task["input columns"]),
            "{{output_modality}}": ", ".join(task["output columns"]),
        }
        for key, value in replacements.items():
            extract_input = extract_input.replace(key, value)
        return extract_input

    def _build_validated_processor(
        self,
        *,
        gpt4: OpenAIClient,
        chat_history: list[dict],
        sample_rows: list[dict],
        prompts_list: dict,
        task: dict,
    ) -> GeneratedInstructionProcessor | None:
        process_data: InstructionProcessDataFn | None = None
        code_str = None
        numbered_code = None

        for attempt in range(self.config.instruction_max_retries):
            logger.info(
                f"Instruction process_data attempt {attempt + 1}/"
                f"{self.config.instruction_max_retries}"
            )
            generated = self._generate_process_data_function(gpt4, chat_history)
            chat_history = generated.chat_history
            if generated.code_str is not None:
                code_str = generated.code_str
            if generated.numbered_code is not None:
                numbered_code = generated.numbered_code
            if generated.process_data is not None:
                process_data = generated.process_data

            if process_data is None:
                logger.error(
                    f"Failed to generate process_data for task '{task['task']}' "
                    f"on attempt {attempt + 1}."
                )
                continue

            try:
                data = self._run_process_data(process_data, sample_rows, prompts_list)
            except Exception as e:
                error_msg = self._processing_error_message(e, numbered_code)
                logger.error(error_msg)
                chat_history = chat_history + [{"role": "user", "content": error_msg}]
                continue

            validation = self._validate_process_data(gpt4, data)
            if validation.is_correct:
                logger.info(
                    f"Process_data function validated successfully on attempt {attempt + 1}."
                )
                return GeneratedInstructionProcessor(
                    process_data=process_data,
                    chat_history=chat_history,
                    code_str=code_str,
                    numbered_code=numbered_code,
                )

            feedback = (
                "The generated instruction data did not pass validation. "
                f"Reason: {validation.thinking}\n"
                f"Example data:\n{validation.data}\n"
                "Please regenerate the process_data function."
            )
            logger.error(feedback)
            chat_history = chat_history + [{"role": "user", "content": feedback}]

        return None

    def _generate_process_data_function(
        self,
        gpt4: OpenAIClient,
        chat_history: list[dict],
    ) -> GeneratedInstructionProcessor:
        with metrics_scope("organization.codegen_llm"):
            response = gpt4.chat(
                chat_history,
                model=self.config.organization_model,
            )
        if not response:
            logger.error("No response from LLM for instruction code generation.")
            return GeneratedInstructionProcessor(
                process_data=None,
                chat_history=chat_history,
                code_str=None,
                numbered_code=None,
            )

        chat_history = chat_history + [{"role": "assistant", "content": response}]
        code_str = extract_code(response)
        numbered_code = "\n".join(
            f"{i + 1:4}: {line}" for i, line in enumerate(code_str.splitlines())
        )

        namespace: dict[str, Any] = {
            "__builtins__": __builtins__,
            "json": json,
            "os": os,
            "random": random,
            "re": re,
            "tqdm": tqdm,
        }
        exec_start = time.perf_counter()
        try:
            exec(code_str, namespace)
            record_named_duration(
                "organization.generated_code_exec",
                time.perf_counter() - exec_start,
            )
        except Exception as exec_e:
            record_named_duration(
                "organization.generated_code_exec",
                time.perf_counter() - exec_start,
            )
            logger.error(f"Error defining process_data: {exec_e}")
            refine_msg = (
                f"Error occurred when executing process_data:\n{numbered_code}\n"
                f"Exception:\n{exec_e}\n"
                "Please refine the process_data function to fix the error. "
                "Ensure your output ends with a python code block containing only the function."
            )
            return GeneratedInstructionProcessor(
                process_data=None,
                chat_history=chat_history + [{"role": "user", "content": refine_msg}],
                code_str=code_str,
                numbered_code=numbered_code,
            )

        process_data = namespace.get("process_data")
        if not callable(process_data):
            refine_msg = (
                "No callable process_data function was found in the response. "
                "Please return a python code block containing only the function."
            )
            return GeneratedInstructionProcessor(
                process_data=None,
                chat_history=chat_history + [{"role": "user", "content": refine_msg}],
                code_str=code_str,
                numbered_code=numbered_code,
            )

        return GeneratedInstructionProcessor(
            process_data=process_data,
            chat_history=chat_history,
            code_str=code_str,
            numbered_code=numbered_code,
        )

    def _run_process_data(
        self,
        process_data: InstructionProcessDataFn,
        rows,
        prompts_list: dict,
    ) -> list[str]:
        process_start = time.perf_counter()
        if prompts_list["need_template"]:
            data = process_data(rows, prompts_list.get("templates", []))
        else:
            data = process_data(rows)
        record_named_duration(
            "organization.local_process_data",
            time.perf_counter() - process_start,
        )
        if not isinstance(data, list):
            raise TypeError("process_data must return a list of instruction strings.")
        return data

    def _validate_process_data(
        self,
        gpt4: OpenAIClient,
        data: list[str],
    ) -> InstructionValidationResult:
        data_dicts = []
        for item in data:
            if not isinstance(item, str):
                continue
            normalized = item.replace("\\t", "\t").replace("\n", " ").strip()
            if normalized.count("\t") != 1:
                continue
            question, answer = [part.strip() for part in normalized.split("\t", 1)]
            if not question or not answer:
                continue
            data_dicts.append({"question": question, "answer": answer})

        if not data_dicts:
            return InstructionValidationResult(
                is_correct=False,
                thinking=(
                    "process_data did not return any valid instruction strings with "
                    "exactly one tab separator."
                ),
                data="",
            )

        data_samples = "\n".join(
            json.dumps(item, ensure_ascii=False) for item in data_dicts
        )
        if not self.config.instruction_quality_check:
            return InstructionValidationResult(
                is_correct=True,
                thinking="local schema validation passed; LLM quality check disabled.",
                data=data_samples,
            )

        for prompt_for_quality_check in prompts_for_instruction_quality_check:
            prompt_quality = prompt_for_quality_check.replace("{{input}}", data_samples)
            prompt_quality = prompt_quality.replace("{{topic}}", self.config.prompt)
            try:
                with metrics_scope("organization.quality_check_llm"):
                    quality = gpt4.chat(
                        prompt_quality,
                        model=self.config.organization_model,
                    )
                quality_dict = extract_json(quality)
            except Exception as e:
                return InstructionValidationResult(
                    is_correct=False,
                    thinking=f"Quality check failed to return valid JSON: {e}",
                    data=data_samples,
                )

            if isinstance(quality_dict, list):
                quality_dict = quality_dict[0] if quality_dict else {}
            answer = (
                quality_dict.get("answer") if isinstance(quality_dict, dict) else None
            )
            if isinstance(answer, str):
                answer = answer.strip().lower() == "true"
            if answer is not True:
                reason = (
                    quality_dict.get("reason", "No reason provided")
                    if isinstance(quality_dict, dict)
                    else "Malformed quality response"
                )
                return InstructionValidationResult(
                    is_correct=False,
                    thinking=str(reason),
                    data=data_samples,
                )

        return InstructionValidationResult(
            is_correct=True,
            thinking="process_data passed schema and quality checks.",
            data=data_samples,
        )

    def _processing_error_message(
        self, exception: Exception, numbered_code: str | None
    ) -> str:
        tb = traceback.format_exc()
        tb_obj = sys.exc_info()[2]
        failing_lineno = None
        if tb_obj is not None:
            tb_pd = tb_obj.tb_next
            if tb_pd is not None:
                failing_lineno = tb_pd.tb_lineno

        failing_line_info = ""
        if failing_lineno and numbered_code:
            code_lines = numbered_code.splitlines()
            context = code_lines[max(0, failing_lineno - 3) : failing_lineno + 2]
            failing_line_info = (
                f"\n[process_data failed at generated code line: {failing_lineno}]"
                f"\nContext:\n{chr(10).join(context)}"
            )
        elif failing_lineno:
            failing_line_info = (
                f"\n[process_data failed at generated code line: {failing_lineno}]"
            )

        return (
            "Error processing sample data:\n"
            f"Exception: {exception}\n"
            f"Traceback:\n{tb}\n"
            "Please refine the process_data function to handle this case."
            f"{failing_line_info}"
        )

    def _run_full_processing(
        self,
        process_data: InstructionProcessDataFn,
        dataset_info: DatasetInfo,
        split: str,
        prompts_list: dict,
    ) -> list[str]:
        ds = self._load_dataset(dataset_info, split)
        if self.config.instruction_max_items_per_dataset > 0:
            ds = islice(ds, self.config.instruction_max_items_per_dataset)
        return self._run_process_data(process_data, ds, prompts_list)

    def _write_task_output(
        self,
        *,
        dataset_info: DatasetInfo,
        split: str,
        task: dict,
        prompts_list: dict,
        code_str: str,
        data: list[str],
    ) -> None:
        ds_name = dataset_info.name.replace("/", "_").replace(" ", "_")
        task_name = task["task"].replace("/", "_").replace(" ", "_")
        base_dir = os.path.join(self.config.output_dir, ds_name, task_name)
        raw_dir = os.path.join(base_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)

        write_start = time.perf_counter()
        save_json(task, os.path.join(raw_dir, "task.json"))
        save_json(prompts_list, os.path.join(raw_dir, "prompts.json"))
        with open(
            os.path.join(raw_dir, "extract_code.py"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(code_str)
        save_json(data, os.path.join(base_dir, f"{split}.json"))
        record_named_duration(
            "organization.file_write", time.perf_counter() - write_start
        )
