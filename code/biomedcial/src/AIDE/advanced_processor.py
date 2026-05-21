import json
import os

from datasets import load_dataset
from loguru import logger

from .configs import InstAgentConfig
from .dataset_info import DatasetInfo
from .prompts import (
    prompt_for_extract_code_generation,
    prompt_for_extract_code_generation_template,
    prompt_for_template_generation,
    prompt_for_usage_brain_storming,
)
from .utils import OpenAIClient, extract_code, extract_json


class AdvancedProcessor:
    def __init__(self, config: InstAgentConfig):
        self.config = config

    def process(self, dataset_info: DatasetInfo) -> bool:
        logger.info(
            f"Advanced processing: dataset {dataset_info.name} - {dataset_info.subset}"
        )
        # Step 1: Brain storming the data usage
        gpt4 = OpenAIClient(
            system_message=self.config.system_message,
        )
        split = "train"
        ds = load_dataset(
            path=dataset_info.name,
            name=dataset_info.subset,
            split=split,
            cache_dir=self.config.cache_dir,
            trust_remote_code=False,
            streaming=True,
        )
        try:
            example_data = []

            for example in ds:
                example_data.append(json.dumps(example, ensure_ascii=False))
                if len(example_data) >= 5:
                    break
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
                    extract_output = gpt4.chat(extract_input)
                    extract_output = extract_code(extract_output)
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
                    extract_output = gpt4.chat(extract_input)
                    extract_output = extract_code(extract_output)
                # execute extraction code
                exec(extract_output, globals())
                if "process_data" in globals():
                    # prepare output directories
                    if not os.path.exists(self.config.output_dir):
                        os.makedirs(self.config.output_dir)
                    ds_name = dataset_info.name.replace("/", "_").replace(" ", "_")
                    task_name = task["task"].replace("/", "_").replace(" ", "_")
                    base_dir = os.path.join(self.config.output_dir, ds_name, task_name)
                    os.makedirs(os.path.join(base_dir, "raw"), exist_ok=True)
                    with open(
                        os.path.join(base_dir, "raw", "task.json"),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        json.dump(task, f, indent=2, ensure_ascii=False)
                    with open(
                        os.path.join(base_dir, "raw", "prompts.json"),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        json.dump(prompts_list, f, indent=2, ensure_ascii=False)
                    with open(
                        os.path.join(base_dir, "raw", "extract_code.py"), "w"
                    ) as f:
                        f.write(extract_output)
                    # run data extraction
                    if prompts_list["need_template"]:
                        data = process_data(ds, prompts_list["templates"])  # noqa: F821
                    else:
                        data = process_data(ds)  # noqa: F821
                    with open(
                        os.path.join(base_dir, f"{split}.json"), "w", encoding="utf-8"
                    ) as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                logger.info(f"Advanced processing task '{task['task']}' done")
                success = True
            except Exception as e:
                logger.error(f"Error in advanced processing task '{task['task']}': {e}")
                continue
        return success


if __name__ == "__main__":
    config = InstAgentConfig("medical")
    processor = AdvancedProcessor(
        config=config,
    )
    dataset_info = DatasetInfo(
        # name="huzaifa525/Medical_Intelligence_Dataset_40k_Rows_of_Disease_Info_Treatments_and_Medical_QA",
        name="krvhrv/Healix-2.8B-Token-Medical-Shot",
        subset="default",
        number_of_rows=40000,
        columns=["text"],
        description="Medical Intelligence Dataset 40k Rows of Disease Info, Treatments and Medical QA",
        hf_tags=["medical", "qa", "disease", "treatment"],
        modalities=["text"],
        license="cc-by-nc-4.0",
        n_likes=0,
        n_downloads_last_month=0,
    )
    processor.process(dataset_info)
