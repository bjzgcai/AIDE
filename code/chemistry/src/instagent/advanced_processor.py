import json
import os
import traceback
import sys
from typing import Callable, TypeAlias

from datasets import load_dataset
from loguru import logger

from .configs import InstAgentConfig
from .dataset_info import DatasetInfo
from .prompts import (
    prompt_for_extract_code_generation,
    prompt_for_extract_code_generation_template,
    prompt_for_template_generation,
    prompt_for_usage_brain_storming,
    prompts_for_quality_check,
)
from .utils import OpenAIClient, extract_code, extract_json
from pydantic import BaseModel
import random

ProcessDataFn: TypeAlias = Callable[[str], list[tuple[str, str]]]

class ValidationResult(BaseModel):
    is_correct: bool
    thinking: str
    data: str


class AdvancedProcessor:
    def __init__(self, config: InstAgentConfig):
        self.config = config

    def process(self, dataset_info: DatasetInfo) -> bool:
        logger.info(
            f"Advanced processing: dataset {dataset_info.name} - {dataset_info.subset}"
        )
        # Step 1: Brain storming the data usage
        gpt4 = OpenAIClient(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            client_type="key",
        )
        split = "train"
        try:
            ds = load_dataset(
                path=dataset_info.name,
                name=dataset_info.subset,
                split=split,
                cache_dir=self.config.cache_dir,
                trust_remote_code=False,
                streaming=True
            )
        except Exception as e:
            logger.error(f"Error in loading dataset {dataset_info.name}: {e}")
            return False
        try:
            example_data = []
            for example in ds:
                example_data.append(json.dumps(example))
                if len(example_data) >= 10000:
                    break
            example_data = random.sample(example_data, min(5, len(example_data)))
            examples = "\n".join(example_data)
            columns = ", ".join(dataset_info.columns)
            brain_storming_input = prompt_for_usage_brain_storming.replace(
                "{{dataset_name}}", dataset_info.name
            )
            brain_storming_input = brain_storming_input.replace(
                "{{description}}", dataset_info.description
            )
            brain_storming_input = brain_storming_input.replace("{{columns}}", columns)
            brain_storming_input = brain_storming_input.replace(
                "{{sample_data}}", examples
            )
            brain_storming_output = gpt4.chat(brain_storming_input)
        except Exception as e:
            logger.error("Error in getting inputs for brain storming " + str(e))
            return False
        try:
            output_dict = extract_json(brain_storming_output)
        except Exception as e:
            logger.error("Error in brain storming output" + str(e))
            return False
        success = False

        logger.info(f"Possible tasks: {[task['task'] for task in output_dict]}")

        for task in output_dict:
            logger.info(task)

            for col in task["input columns"]:
                if col not in dataset_info.columns:
                    logger.error(
                        f"Input column '{col}' not found in dataset columns: {dataset_info.columns}"
                    )
                    continue
            for col in task["output columns"]:
                if col not in dataset_info.columns:
                    logger.error(
                        f"Output column '{col}' not found in dataset columns: {dataset_info.columns}"
                    )
                    continue
            try:
                # step 2: generate prompts
                prompt_input = prompt_for_template_generation.replace(
                    "{{task_name}}", task["task"]
                )
                prompt_input = prompt_input.replace(
                    "{{input_modality}}", ", ".join(task["input columns"])
                )
                prompt_input = prompt_input.replace(
                    "{{output_modality}}", ", ".join(task["output columns"])
                )
                prompts = gpt4.chat(prompt_input)
                prompts_list = extract_json(prompts)
                # step 3: generate extraction code
                if prompts_list["need_template"]:
                    extract_input = prompt_for_extract_code_generation_template.replace(
                        "{{dataset_name}}", dataset_info.name
                    )
                    extract_input = extract_input.replace("{{columns}}", columns)
                    extract_input = extract_input.replace("{{sample_data}}", examples)
                    extract_input = extract_input.replace(
                        "{{prompt_list}}", str(prompts_list["templates"])
                    )
                    extract_input = extract_input.replace(
                        "{{task_name}}", str(task["task"])
                    )
                    extract_input = extract_input.replace(
                        "{{input_modality}}", ", ".join(task["input columns"])
                    )
                    extract_input = extract_input.replace(
                        "{{output_modality}}", ", ".join(task["output columns"])
                    )
                else:
                    extract_input = prompt_for_extract_code_generation.replace(
                        "{{dataset_name}}", dataset_info.name
                    )
                    extract_input = extract_input.replace("{{columns}}", columns)
                    extract_input = extract_input.replace("{{sample_data}}", examples)
                    extract_input = extract_input.replace(
                        "{{task_name}}", str(task["task"])
                    )
                    extract_input = extract_input.replace(
                        "{{input_modality}}", ", ".join(task["input columns"])
                    )
                    extract_input = extract_input.replace(
                        "{{output_modality}}", ", ".join(task["output columns"])
                    )
            except Exception as e:
                logger.error(f"Error in generating prompts: {e}")
                continue
            max_retries = 3
            chat_history = [{"role": "system", "content": extract_input}]
            process_data: ProcessDataFn | None = None
            code_str = None  # Store the raw code for output
            numbered_code = None  # For debugging
            validated = False
            for attempt in range(max_retries):
                process_data_attempt: ProcessDataFn | None = None
                process_data_attempt, chat_history, new_code_str, new_numbered_code = (
                    self._generate_process_data_function(
                            gpt4, chat_history, return_raw_code=True
                    )
                )
                if new_code_str is not None:
                    code_str = new_code_str
                if new_numbered_code is not None:
                    numbered_code = new_numbered_code
                if process_data_attempt is not None:
                    process_data = process_data_attempt
                if process_data is None:
                    logger.error(
                            f"Failed to generate process_data function on attempt {attempt + 1}. Retrying..."
                    )
                    continue
                    # parse the example data to check the code
                example_data = []
                for example in ds:
                    example_data.append(example)
                    if len(example_data) >= 5:
                        break
                try:
                    if prompts_list["need_template"]:
                        data = process_data(
                            example_data, prompts_list["templates"]
                        )
                    else:
                        data = process_data(example_data)
                except Exception as e:
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
                        # Show the failing line and a few lines of context
                        context = code_lines[
                            max(0, failing_lineno - 3) : failing_lineno + 2
                        ]
                        context_str = "\n".join(context)
                        failing_line_info = f"\n[process_data failed at generated code line: {failing_lineno}]\nContext:\n{context_str}"
                    elif failing_lineno:
                        failing_line_info = f"\n[process_data failed at generated code line: {failing_lineno}]"
                    error_msg = (
                        f"Error processing item:\n"
                        f"Exception: {str(e)}\n"
                        f"Traceback:\n{tb}\n"
                        f"Please refine the process_data function to handle this case.{failing_line_info}"
                    )
                    logger.error(error_msg)
                    chat_history = chat_history + [
                        {"role": "user", "content": error_msg}
                    ]
                    continue
                validation_results = self._validate_process_data(gpt4, data)
                if not validation_results.is_correct:  
                    logger.error(
                        f"Validation failed for process_data function on attempt {attempt + 1}. Retrying..."
                    )
                    chat_history = chat_history + [
                        {
                            "role": "user",
                            "content": f"The data is not valid, {validation_results.thinking}, for example, {validation_results.data}, regenerate the process_data function.",
                        }
                    ]
                else:
                    logger.info(
                        f"Process data function validated successfully on attempt {attempt + 1}."
                    )
                    validated = True
                    break
            if not validated:
                logger.error(
                    f"Failed to validate process_data function after {max_retries} attempts."
                )
                continue
            else:
                try:
                    # prepare output directories
                    if not os.path.exists(self.config.output_dir):
                        os.makedirs(self.config.output_dir)
                    ds_name = dataset_info.name.replace("/", "_").replace(" ", "_")
                    task_name = task["task"].replace("/", "_").replace(" ", "_")
                    base_dir = os.path.join(self.config.output_dir, ds_name, task_name)
                    os.makedirs(os.path.join(base_dir, "raw"), exist_ok=True)
                    with open(os.path.join(base_dir, "raw", "task.json"), "w") as f:
                        json.dump(task, f, indent=4)
                    with open(os.path.join(base_dir, "raw", "prompts.json"), "w") as f:
                        json.dump(prompts_list, f, indent=4)
                    # run data extraction
                    if prompts_list["need_template"]:
                        data = process_data(ds, prompts_list["templates"])  # noqa: F821
                    else:
                        data = process_data(ds)  # noqa: F821
                    with open(os.path.join(base_dir, f"{split}.json"), "w") as f:
                        json.dump(data, f, indent=4)
                    logger.info(f"Advanced processing task '{task['task']}' done")
                    success = True
                except Exception as e:
                    logger.error(f"Error in advanced processing task '{task['task']}': {e}")
                    continue
        return success

    def  _generate_process_data_function(
        self, gpt4, chat_history, return_raw_code=False
    ):
        response = gpt4.chat(chat_history)
        if not response:
            logger.error("No response from OpenAI for code generation.")
            return None, chat_history, None, None
        chat_history = chat_history + [{"role": "assistant", "content": response}]
        code_str = extract_code(response)
        numbered_code = None
        if code_str:
            numbered_code = "\n".join(
                f"{i+1:4}: {line}" for i, line in enumerate(code_str.splitlines())
            )
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
    def _validate_process_data(self, gpt4, data):
        data_dicts = []
        for item in data:
            if not isinstance(item, str):
                continue
            item = item.replace("\\t", "\t")
            if item.count("\t") == 1:
                question, answer = item.split("\t")
                #if answer.lower() in question.lower():
                #    continue
                data_dicts.append(json.dumps({"question": question, "answer": answer}))
        if len(data_dicts) == 0:
            logger.error("No valid data samples found for validation.")
            return ValidationResult(
                is_correct=False,
                thinking="The process_data function did not return any valid data samples with '\t'.",
                data=""
            )
        data_samples = "\n".join(data_dicts)
        for prompt_for_quality_check in tqdm(prompts_for_quality_check):
            prompt_quality = prompt_for_quality_check.replace(
                "{{input}}", data_samples
            )
            if "{{topic}}" in prompt_quality:
                prompt_quality = prompt_quality.replace(
                    "{{topic}}", self.config.prompt
                )
            try:
                quality = gpt4.chat(prompt_quality)
                quality_dict = extract_json(quality)
            except Exception as e:
                logger.error(f"Error in quality check: {e}")
                print(quality)
                continue
            try:
                if isinstance(quality_dict, list):
                    quality_dict = quality_dict[0]
                if quality_dict["answer"] == False:
                    print(f"Quality check failed: {quality_dict}")
                    print(data_samples)
                    return ValidationResult(
                        is_correct=False,
                        thinking=quality_dict.get("reason", "No specific thinking provided"),
                        data=data_samples,
                    )
            except Exception as e:
                logger.error(f"Error processing quality check response: {e}")
                print(quality_dict)
        return ValidationResult(
            is_correct=True,
            thinking="the process_data function passed the quality checks.",
            data=data_samples,
        )
    
if __name__ == "__main__":
    config = InstAgentConfig("medical")
    processor = AdvancedProcessor(
        config=config,
    )
    dataset_info = DatasetInfo(
        name="huzaifa525/Medical_Intelligence_Dataset_40k_Rows_of_Disease_Info_Treatments_and_Medical_QA",
        #name="krvhrv/Healix-2.8B-Token-Medical-Shot",
        subset="default",
        number_of_rows=40000,
        columns=["input", "output"],
        description="Medical Intelligence Dataset 40k Rows of Disease Info, Treatments and Medical QA",
        hf_tags=["medical", "qa", "disease", "treatment"],
        modalities=["text"],
        license="cc-by-nc-4.0",
        n_likes=0,
        n_downloads_last_month=0,
    )
    processor.process(dataset_info)
