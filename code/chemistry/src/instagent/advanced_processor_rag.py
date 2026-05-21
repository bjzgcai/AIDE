import json
import os
import traceback
from typing import Callable, TypeAlias

from datasets import load_dataset
from loguru import logger
from tqdm import tqdm

from .configs import InstAgentConfig
from .dataset_info import DatasetInfo
from .prompts import prompt_for_rag_process
from .utils import OpenAIClient, extract_code

ProcessDataFn: TypeAlias = Callable[[str], list[tuple[str, str]]]


class AdvancedProcessorRAG:
    def __init__(self, config: InstAgentConfig):
        self.config = config

    def process(self, dataset_info: DatasetInfo) -> bool:
        logger.info(
            f"RAG processing: dataset {dataset_info.name} - {dataset_info.subset}"
        )
        gpt4 = OpenAIClient(
            resource_name=self.config.resource_name,
            api_version=self.config.api_version,
            system_message=self.config.system_message,
        )
        ds = self._load_dataset(dataset_info)
        sample_str = self._sample_examples(ds)
        prompt = prompt_for_rag_process.replace("{{sample_data}}", sample_str)
        max_retries = 3
        chat_history = [{"role": "user", "content": prompt}]
        process_data: ProcessDataFn | None = None
        for attempt in range(max_retries):
            all_chunks = []
            process_data_attempt: ProcessDataFn | None = None
            process_data_attempt, chat_history = self._generate_process_data_function(
                gpt4, chat_history
            )
            if process_data_attempt is not None:
                process_data = process_data_attempt
            if process_data is None:
                logger.error(
                    f"Failed to generate process_data function on attempt {attempt + 1}. Retrying..."
                )
                continue
            for item in tqdm(ds):
                item_str = json.dumps(item)
                try:
                    pairs = process_data(item_str)
                    for idx, val in pairs:
                        all_chunks.append((idx, val))
                except Exception as e:
                    tb = traceback.format_exc()
                    error_msg = (
                        f"Error processing item for RAG:\n"
                        f"Item: {item_str}\n"
                        f"Exception: {str(e)}\n"
                        f"Traceback:\n{tb}\n"
                        f"Please refine the process_data function to handle this case."
                    )
                    logger.error(error_msg)
                    chat_history = chat_history + [
                        {"role": "user", "content": error_msg}
                    ]
                    continue

            logger.info(
                f"Found {len(all_chunks)} chunks from dataset {dataset_info.name} - {dataset_info.subset}"
            )

            self._write_output(dataset_info, all_chunks)
            return True
        logger.error("All retries failed for RAG processing.")
        # Try best-effort processing with last process_data if available
        if process_data is not None:
            logger.info(
                "Attempting best-effort processing with last generated process_data, ignoring failed items."
            )
            all_chunks = []
            skipped_count = 0
            for item in tqdm(ds):
                item_str = json.dumps(item)
                try:
                    pairs = process_data(item_str)
                    if pairs is not None and hasattr(pairs, "__iter__"):
                        for idx, val in pairs:
                            all_chunks.append((idx, val))
                    else:
                        skipped_count += 1
                        continue
                except Exception:
                    skipped_count += 1
                    continue
            logger.info(
                f"Best-effort: Found {len(all_chunks)} chunks from dataset {dataset_info.name} - {dataset_info.subset}. Skipped {skipped_count} items."
            )
            self._write_output(dataset_info, all_chunks)
            return True
        return False

    def _write_output(self, dataset_info: DatasetInfo, all_chunks):
        if not os.path.exists(self.config.output_dir):
            os.makedirs(self.config.output_dir, exist_ok=True)
        ds_name = dataset_info.name.replace("/", "_").replace(" ", "_")
        out_file = os.path.join(
            self.config.output_dir,
            f"{ds_name}_{dataset_info.subset}_rag.json",
        )
        with open(out_file, "w") as f:
            json.dump(all_chunks, f, ensure_ascii=True, indent=2)
        logger.info(f"RAG processing saved to {out_file}")

    def _generate_process_data_function(self, gpt4, chat_history):
        response = gpt4.chat(chat_history)
        if not response:
            logger.error("No response from OpenAI for RAG processing.")
            return None, chat_history
        chat_history = chat_history + [{"role": "assistant", "content": response}]
        code_str = extract_code(response)
        try:
            exec(code_str, globals())
        except Exception as exec_e:
            logger.error(f"Error defining process_data: {exec_e}")
            refine_msg = (
                f"Error occurred when executing process_data:\n"
                f"{code_str}\n"
                f"Exception:\n"
                f"{exec_e}\n"
                f"Please refine the process_data function to fix the error. "
                f"Ensure your output ends with a python code block containing only the function."
            )
            chat_history = chat_history + [{"role": "user", "content": refine_msg}]
            return None, chat_history
        if "process_data" not in globals():
            logger.error("No process function generated for RAG processing.")
            refine_msg = (
                "No process_data function found in the response. "
                "Please refine the process_data function to fix the error. "
                "Ensure your output ends with a python code block containing only the function."
            )
            chat_history = chat_history + [{"role": "user", "content": refine_msg}]
            return None, chat_history
        return globals()["process_data"], chat_history

    def _sample_examples(self, ds, n=5):
        sample_list = []
        for example in ds:
            sample_list.append(json.dumps(example))
            if len(sample_list) >= n:
                break
        sample_str = ""
        for idx, sample in enumerate(sample_list):
            sample_str += f"Sample {idx + 1}:\n{sample}\n"
        return sample_str

    def _load_dataset(self, dataset_info: DatasetInfo):
        split = "train"
        return load_dataset(
            path=dataset_info.name,
            name=dataset_info.subset,
            split=split,
            cache_dir=self.config.cache_dir,
            trust_remote_code=False,
            streaming=True,
        )


if __name__ == "__main__":
    config = InstAgentConfig("medical")
    processor = AdvancedProcessorRAG(
        config=config,
    )
    dataset_info = DatasetInfo(
        # name="huzaifa525/Medical_Intelligence_Dataset_40k_Rows_of_Disease_Info_Treatments_and_Medical_QA",
        # name="ruslanmv/ai-medical-chatbot",
        name="FreedomIntelligence/medical-o1-reasoning-SFT",
        subset="en",
        number_of_rows=40000,
        columns=["Question", "Complex_CoT", "Response"],
        description="Medical Intelligence Dataset 40k Rows of Disease Info, Treatments and Medical QA",
        hf_tags=["medical", "qa", "disease", "treatment"],
        modalities=["text"],
        license="cc-by-nc-4.0",
        n_likes=0,
        n_downloads_last_month=0,
    )
    processor.process(dataset_info)
