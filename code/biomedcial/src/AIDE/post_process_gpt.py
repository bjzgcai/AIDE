import argparse
import json
import os
import random
import time

from tqdm import tqdm

from .prompts import prompt_for_data_post_processing, prompt_for_quality_check
from .utils import OpenAIClient


def post_process(input_dir: str, output_dir: str, client: OpenAIClient):
    """
    Post process the generated data.
    """
    datasets = os.listdir(input_dir)
    prompt = input_dir.split("/")[-1]
    cnt = 0
    for dataset in tqdm(datasets):
        tasks = os.listdir(f"{input_dir}/{dataset}")
        for task in tqdm(tasks):
            files = os.listdir(f"{input_dir}/{dataset}/{task}")
            for file in files:
                if file == "raw":
                    continue
                with open(f"{input_dir}/{dataset}/{task}/{file}", "rb") as f:
                    data = json.loads(f.read().decode("utf-8"))

                data_samples = random.sample(data, min(5, len(data)))
                data_samples = "\n".join(data_samples)
                prompt_quality = prompt_for_quality_check.replace(
                    "{{input}}", data_samples
                )
                try:
                    quality = client.chat(prompt_quality)
                    cnt += 1
                    if cnt % 3 == 0:
                        time.sleep(1)
                    if "no" in quality.lower():
                        f = open("quality_check.txt", "a", encoding="utf-8")
                        f.write(
                            json.dumps(
                                {
                                    "Quality check failed for": f"{dataset}/{task}/{file}"
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        f.close()
                        continue
                except Exception:
                    pass
                batch_data = []
                for item in tqdm(data):
                    input = prompt_for_data_post_processing.replace("{{input}}", item)
                    if len(input) > 10000:
                        continue
                    batch_data.append(input)
                input_file = (
                    f"{output_dir}/batch_input/{prompt}_{dataset}_{task}_{file}.jsonl"
                )
                os.makedirs(f"{output_dir}/batch_input", exist_ok=True)
                try:
                    client.batch_chat(batch_data, input_file)
                except Exception:
                    continue
                # print(input_file, batch_obj.id)
                """
                converted_data = []
                for item in tqdm(data):
                    input = prompt_for_data_post_processing.replace(
                        "{{input}}", item
                    )
                    output = client.chat(input)
                    converted_data.append(output)
                os.makedirs(
                    f"{output_dir}", exist_ok=True
                )
                with open(
                    f"{output_dir}/{prompt}_{dataset}_{task}_{file}.tsv", "w"
                ) as f:
                    f.write("\n".join(converted_data))
                """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default=os.path.expanduser("~/AIDE/processed_data/chemistry"),
        help="Input directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.expanduser("~/AIDE/post_processed_data_gpt"),
        help="Output directory",
    )
    args = parser.parse_args()
    client = OpenAIClient(
        system_message="You are a helpful AI assistant.",
    )
    post_process(args.input_dir, args.output_dir, client)
