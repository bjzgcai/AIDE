import argparse
import json
import os
import random
from collections.abc import Sequence

from datasketch import MinHash, MinHashLSH
from tqdm import tqdm


def build_minhash(index_sentence):
    index, sentence = index_sentence
    m = MinHash(num_perm=128)
    for word in sentence.split():
        m.update(word.encode("utf8"))
    return (f"s{index}", m, sentence)


def dedup(sentences: Sequence[str], threshold: float = 0.7) -> list[str]:
    minhashes = [
        build_minhash((i, s))
        for i, s in tqdm(enumerate(sentences), total=len(sentences))
    ]

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


def dedup_datasets(
    dir_name: str, summary_path: str, threshold: float = 0.7, force: bool = False
) -> None:
    datasets = os.listdir(dir_name)
    total = 0
    total_before = 0

    with open(summary_path, "w", encoding="utf-8") as fw:
        for dataset in tqdm(datasets):
            folder = os.path.join(dir_name, dataset)
            if not os.path.isdir(folder):
                continue

            dedup_file = os.path.join(folder, "dedup.json")
            if os.path.exists(dedup_file) and not force:
                continue

            train_files = []
            for task in os.listdir(folder):
                task_dir = os.path.join(folder, task)
                if not os.path.isdir(task_dir):
                    continue
                train_file = os.path.join(task_dir, "train.json")
                if os.path.exists(train_file):
                    train_files.append(train_file)

            if not train_files:
                continue

            print(dataset)
            sentences = []
            for train_file in tqdm(train_files):
                with open(train_file, encoding="utf-8") as f:
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


def collect_instruction_pairs(dir_name: str) -> list[str]:
    datasets = os.listdir(dir_name)
    sentences = []
    for dataset in tqdm(datasets):
        folder = os.path.join(dir_name, dataset)
        if not os.path.isdir(folder):
            continue
        dedup_file = os.path.join(folder, "dedup.json")
        if not os.path.exists(dedup_file):
            continue

        with open(dedup_file, encoding="utf-8") as f:
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
    seed: int | None = None,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect and optionally deduplicate organized instruction data."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--val-output", required=True)
    parser.add_argument("--summary-path", default="dedup_summary.txt")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-dataset-dedup", action="store_true")
    parser.add_argument("--dedup-final-jsonl", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild existing per-dataset dedup.json files before exporting JSONL.",
    )
    args = parser.parse_args()

    if not args.skip_dataset_dedup:
        dedup_datasets(
            args.input_dir,
            summary_path=args.summary_path,
            threshold=args.threshold,
            force=args.force,
        )
    sentences = collect_instruction_pairs(args.input_dir)
    write_instruction_jsonl(
        sentences,
        train_output_path=args.train_output,
        val_output_path=args.val_output,
        val_ratio=args.val_ratio,
        seed=args.seed,
        apply_dedup=args.dedup_final_jsonl,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
