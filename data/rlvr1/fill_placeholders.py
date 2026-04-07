#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "datasets>=2.19.0",
# ]
# ///

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Fills placeholders in super-v3 RL jsonl files with data fetched from Hugging Face datasets.

The super-v3 blend contains placeholders for 2 open source math datasets:

- BytedTsinghua-SIA/DAPO-Math-17k
- Skywork/Skywork-OR1-RL-Data

This script processes every .jsonl file in the input directory, downloads the data from
Hugging Face, and replaces placeholders with the actual content.

Usage:
    chmod +x fill_placeholders.py
    ./fill_placeholders.py --input-dir /path/to/placeholders/ --output-dir /path/to/output/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from datasets import load_dataset


TARGET_DATASETS: Dict[str, Dict[str, str]] = {
    "super_v3_lcsft_step1000_dapo17k": {
        "hf_dataset": "BytedTsinghua-SIA/DAPO-Math-17k",
        "split": "train",
        "question_path": ["prompt", 0, "content"],
        "answer_path": ["reward_model", "ground_truth"],
    },
    "super_v3_lcsft_step1000_skyworks": {
        "hf_dataset": "Skywork/Skywork-OR1-RL-Data",
        "split": "math",
        "question_path": ["prompt", 0, "content"],
        "answer_path": ["reward_model", "ground_truth"],
    },
}


def strip_dapo_prompt(text: str) -> str:
    """
    DAPO wraps the math question inside a fixed prompt. Extract the inner question.
    """
    prefix = (
        "Solve the following math problem step by step. "
        "The last line of your response should be of the form "
        "Answer: $Answer (without quotes) where $Answer is the answer to the problem."
    )
    suffix = 'Remember to put your answer on its own line after "Answer:".'

    start = text.index(prefix) + len(prefix)
    end = text.rfind(suffix)
    return text[start:end]


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_hf_dataset(hf_dataset: str, split: str):
    return load_dataset(hf_dataset, split=split)


def extract_path(obj: Any, path: List[Any]) -> Any:
    cur = obj
    for key in path:
        if isinstance(key, int):
            if not isinstance(cur, list):
                raise KeyError(f"Expected list before index {key}, got {type(cur)}")
            cur = cur[key]
        else:
            if not isinstance(cur, dict):
                raise KeyError(f"Expected dict before key {key}, got {type(cur)}")
            cur = cur.get(key)
    return cur


def get_answer(raw: Any) -> Any:
    if isinstance(raw, str):
        s = raw.strip()
        if (s.startswith("[") and s.endswith("]")) or (
            s.startswith("{") and s.endswith("}")
        ):
            loaded = json.loads(s)
            return loaded[0]
        else:
            return s


def restore_dapo_template(text: str, template):
    if template["prefix"]:
        # When the prompt is a prefix template for dapo samples,
        # we remove the trailing newlines from the question.
        return f"{template['prefix']}{text}".removesuffix("\n\n")
    elif template["suffix"]:
        return f"{text}{template['suffix']}"
    else:
        raise ValueError(f"Unknown template: {template}")


def restore_skywork_template(text: str, template):
    return template["template"].replace("{question}", text)


def restore_record(
    record: Dict, hf_row: Dict, question_path: List[Any], answer_path: List[Any]
) -> Dict:
    question = extract_path(hf_row, question_path)
    if record["dataset"] == "super_v3_lcsft_step1000_dapo17k":
        question_stripped = strip_dapo_prompt(question)
        question_template = record["_hf_placeholder"]["question_template"]
        full_question = restore_dapo_template(
            question_stripped, question_template
        )
    elif record["dataset"] == "super_v3_lcsft_step1000_skyworks":
        question_template = record["_hf_placeholder"]["question_template"]
        full_question = restore_skywork_template(question, question_template)
    else:
        raise NotImplementedError(f"Unknown dataset: {record['dataset']}")

    answer = get_answer(extract_path(hf_row, answer_path))

    restored = dict(record)
    restored.pop("_hf_placeholder")
    if record["dataset"] == "super_v3_lcsft_step1000_dapo17k":
        restored["question"] = full_question
    elif record["dataset"] == "super_v3_lcsft_step1000_skyworks":
        restored["question"] = question
    else:
        raise NotImplementedError(f"Unknown dataset: {record['dataset']}")

    restored["expected_answer"] = answer
    restored["responses_create_params"] = {
        "input": [{"role": "user", "content": full_question}]
    }
    return restored


def process_file(input_path: Path, output_path: Path, hf_cache: Dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fout:
        for record in iter_jsonl(input_path):
            placeholder = record.get("_hf_placeholder")
            dataset_name = record.get("dataset")
            if not placeholder or dataset_name not in TARGET_DATASETS:
                fout.write(json.dumps(record) + "\n")
                continue

            cfg = TARGET_DATASETS[dataset_name]
            dataset = hf_cache[dataset_name]
            row_idx = int(placeholder["row"])
            hf_row = dataset[row_idx]
            restored = restore_record(
                record, hf_row, cfg["question_path"], cfg["answer_path"]
            )
            fout.write(json.dumps(restored) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct super-v3 jsonl by replacing placeholders for dapo and skywork datasets."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing jsonl files with placeholders.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Destination directory for restored jsonl files.",
    )
    args = parser.parse_args()

    jsonl_files = sorted(args.input_dir.glob("*.jsonl"))
    if not jsonl_files:
        parser.error(f"No .jsonl files found in {args.input_dir}")

    hf_cache = {}
    for dataset_name, cfg in TARGET_DATASETS.items():
        ds = load_hf_dataset(cfg["hf_dataset"], cfg["split"])
        hf_cache[dataset_name] = ds

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for input_path in jsonl_files:
        output_path = args.output_dir / input_path.name
        print(f"Processing {input_path.name} ...")
        process_file(input_path, output_path, hf_cache)
        print(f"  -> {output_path}")


if __name__ == "__main__":
    main()
