from datasketch import MinHash, MinHashLSH
from tqdm import tqdm
import json
import os
import random
from typing import List, Optional, Sequence


def build_minhash(index_sentence):
    index, sentence = index_sentence
    m = MinHash(num_perm=128)
    for word in sentence.split():
        m.update(word.encode("utf8"))
    return (f"s{index}", m, sentence)


def dedup(sentences: Sequence[str], threshold: float = 0.7) -> List[str]:
    minhashes = [build_minhash((i, s)) for i, s in tqdm(enumerate(sentences), total=len(sentences))]

    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    for name, m, _ in minhashes:
        lsh.insert(name, m)

    unique_sentences = []
    seen = set()
    for name, m, sentence in minhashes:
        if name in seen:
            continue
        duplicates = lsh.query(m)
        for dup in duplicates:
            seen.add(dup)
        unique_sentences.append(sentence)

    return unique_sentences


def dedup_datasets(dir_name: str, summary_path: str, threshold: float = 0.7) -> None:
    datasets = os.listdir(dir_name)
    total = 0
    total_before = 0

    with open(summary_path, "w", encoding="utf-8") as fw:
        for dataset in tqdm(datasets):
            dedup_file = os.path.join(dir_name, dataset, "dedup.json")
            if os.path.exists(dedup_file):
                continue

            print(dataset)
            folder = os.path.join(dir_name, dataset)
            tasks = os.listdir(folder)
            sentences = []
            for task in tqdm(tasks):
                file_path = os.path.join(folder, task, "train.json")
                if not os.path.exists(file_path):
                    continue
                with open(file_path, "r", encoding="utf-8") as f:
                    sentences.extend(json.load(f))

            before_count = len(sentences)
            total_before += before_count
            if before_count == 0:
                fw.write(f"{dataset}\t{before_count}\t0\n")
                continue

            try:
                sentences = dedup(sentences, threshold=threshold)
            except Exception as e:
                print(f"Error in deduplication for {dataset}: {e}")
                print(sentences[0] if sentences else "No sentences available")
                continue

            fw.write(f"{dataset}\t{before_count}\t{len(sentences)}\n")
            with open(dedup_file, "w", encoding="utf-8") as f:
                json.dump(sentences, f, ensure_ascii=False, indent=4)
            total += len(sentences)

        fw.write(f"Total before deduplication: {total_before}\n")
        fw.write(f"Total: {total}\n")


def collect_instruction_pairs(dir_name: str) -> List[str]:
    datasets = os.listdir(dir_name)
    sentences = []
    for dataset in tqdm(datasets):
        folder = os.path.join(dir_name, dataset)
        dedup_file = os.path.join(folder, "dedup.json")
        if not os.path.exists(dedup_file):
            continue

        with open(dedup_file, "r", encoding="utf-8") as f:
            records = json.load(f)

        for record in records:
            normalized = record.replace("\\t", "\t")
            if len(normalized.splitlines()) == 1 and len(normalized.split("\t")) == 2:
                sentences.append(normalized)

    return sentences


def write_instruction_jsonl(
    sentences: Sequence[str],
    train_output_path: str,
    val_output_path: str,
    val_ratio: float = 0.05,
    seed: Optional[int] = None,
    apply_dedup: bool = False,
    threshold: float = 0.7,
) -> None:
    before_count = len(sentences)
    print(f"Total sentences before deduplication: {before_count}")

    processed_sentences = list(sentences)
    if apply_dedup and processed_sentences:
        processed_sentences = dedup(processed_sentences, threshold=threshold)

    after_count = len(processed_sentences)
    print(f"Total sentences after deduplication: {after_count}")

    sentences_jsonl = []
    for line in processed_sentences:
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        instruction, output = parts
        sentences_jsonl.append({"instruction": instruction, "output": output})

    rng = random.Random(seed) if seed is not None else random
    rng.shuffle(sentences_jsonl)

    val_size = int(len(sentences_jsonl) * val_ratio)
    val = sentences_jsonl[:val_size]
    train = sentences_jsonl[val_size:]

    with open(train_output_path, "w", encoding="utf-8") as f:
        for item in train:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(val_output_path, "w", encoding="utf-8") as f:
        for item in val:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Train samples: {len(train)}, Val samples: {len(val)}")


if __name__ == "__main__":
    dir_name = "/tos-mlp-zgci/liuzequn/instagent/processed_data/chemistry_text_for_instruction_tuning"
    summary_path = "sample_count.txt"
    train_output_path = "/tos-mlp-zgci/liuzequn/instagent/train_chemistry.jsonl"
    val_output_path = "/tos-mlp-zgci/liuzequn/instagent/valid_chemistry.jsonl"

    dedup_datasets(dir_name, summary_path=summary_path, threshold=0.7)
    sentences = collect_instruction_pairs(dir_name)
    write_instruction_jsonl(
        sentences,
        train_output_path=train_output_path,
        val_output_path=val_output_path,
        val_ratio=0.05,
        seed=None,
        apply_dedup=False,
        threshold=0.7,
    )