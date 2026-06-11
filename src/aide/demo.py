"""Offline smoke-test demo for the AIDE release.

This module intentionally avoids network and LLM dependencies. It writes a tiny
curation run that mirrors the real pipeline stages so users can verify the
repository immediately after installation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TOY_DATASETS: list[dict[str, Any]] = [
    {
        "name": "toy/chemistry-qa",
        "subset": "default",
        "score": 8,
        "records": [
            {
                "id": "chem-1",
                "question": "What is the molecular formula of water?",
                "answer": "H2O",
                "context": "Water is a molecule composed of two hydrogen atoms and one oxygen atom.",
            },
            {
                "id": "chem-2",
                "question": "Which element has atomic number 6?",
                "answer": "Carbon",
                "context": "Carbon is the chemical element with symbol C and atomic number 6.",
            },
        ],
    },
    {
        "name": "toy/noisy-metadata",
        "subset": "default",
        "score": 2,
        "records": [
            {
                "id": "noise-1",
                "question": "",
                "answer": "",
                "context": "Incomplete record used to demonstrate the selection gate.",
            }
        ],
    },
]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _rag_items(dataset: dict[str, Any]) -> list[dict[str, str]]:
    rows = []
    for record in dataset["records"]:
        rows.append(
            {
                "id": f"{dataset['name']}:{record['id']}",
                "contents": f"{record['context']} Question: {record['question']} Answer: {record['answer']}",
            }
        )
    return rows


def _instruction_items(dataset: dict[str, Any]) -> list[dict[str, str]]:
    rows = []
    for record in dataset["records"]:
        if not record["question"] or not record["answer"]:
            continue
        rows.append(
            {
                "instruction": record["question"],
                "output": record["answer"],
                "source": f"{dataset['name']}::{dataset['subset']}::{record['id']}",
            }
        )
    return rows


def run_demo(output_dir: str | Path = "outputs/demo") -> dict[str, Any]:
    output_path = Path(output_dir)
    selected = [dataset for dataset in TOY_DATASETS if dataset["score"] >= 5]

    _write_json(output_path / "collection" / "candidates.json", TOY_DATASETS)
    _write_json(output_path / "selection" / "selected_datasets.json", selected)

    rag_rows: list[dict[str, str]] = []
    instruction_rows: list[dict[str, str]] = []
    for dataset in selected:
        rag_rows.extend(_rag_items(dataset))
        instruction_rows.extend(_instruction_items(dataset))

    _write_jsonl(output_path / "organization" / "rag_corpus.jsonl", rag_rows)
    _write_jsonl(
        output_path / "organization" / "instruction_data.jsonl", instruction_rows
    )

    summary = {
        "output_dir": str(output_path),
        "collection_candidates": len(TOY_DATASETS),
        "selected_datasets": len(selected),
        "rag_items": len(rag_rows),
        "instruction_items": len(instruction_rows),
    }
    _write_json(output_path / "run_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the offline AIDE demo.")
    parser.add_argument("--output-dir", default="outputs/demo")
    args = parser.parse_args(argv)
    print(json.dumps(run_demo(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
