# Copyright 2025 Project-Numina and/or its affiliates
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

import argparse
from pathlib import Path

from datasets import load_dataset

PROMPT = """Think about and solve the following problems step by step in Lean 4.

# Problem:
{informal_problem}

# Formal Statement:
```lean4
{formal_statement}
```
"""

SYSTEM_PROMPT = "You are an expert in mathematics and proving theorems in Lean 4."


def check_is_already_downloaded(path: Path) -> bool:
    return (path / "train.parquet").exists() and (path / "test.parquet").exists()


def format_for_verl(sample: dict) -> str:
    user_prompt = PROMPT.format(
        informal_problem=sample["informal_problem"],
        formal_statement=sample["formal_statement"],
    )

    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    sample["prompt"] = prompt
    sample["extra_info"] = {
        "formal_statement": sample["formal_statement"],
        "index": sample["statement_id"],
    }
    sample["reward_model"] = {
        "style": "rule",
        "ground_truth": "Should be solved in Lean 4.",
    }

    return sample


def prepare_test_split(hf_identifier: str):
    dataset = load_dataset(hf_identifier, split="train")

    dataset = dataset.add_column("data_source", [hf_identifier] * len(dataset))
    dataset = dataset.rename_column("informal_prefix", "informal_problem")
    dataset = dataset.rename_column("name", "statement_id")
    dataset = dataset.map(format_for_verl)

    return dataset


def prepare_train_split(hf_identifier: str):
    dataset = load_dataset(hf_identifier, split="train")

    dataset = dataset.rename_column("source", "data_source")
    dataset = dataset.rename_column("natural_language", "informal_problem")
    dataset = dataset.map(format_for_verl)

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the prompt set.")
    parser.add_argument(
        "--train-dataset",
        type=str,
        help="Huggingface identifier of the training dataset to download.",
        required=True,
    )
    parser.add_argument(
        "--test-dataset",
        type=str,
        help="Huggingface identifier of the test dataset to download.",
        required=True,
    )
    parser.add_argument("--path", type=str, help="Path for storing the prompt set.")
    parser.add_argument(
        "--sample-first-n",
        type=int,
        default=None,
        help="Sample first n examples from both datasets and ignore the rest of the dataset.",
        required=False,
    )

    args = parser.parse_args()

    dataset_name = (
        args.train_dataset if args.sample_first_n is None else f"{args.train_dataset}-first-{args.sample_first_n}"
    )

    if args.path:
        path = Path(args.path) / "prompt_sets" / dataset_name
    else:
        path = Path(__file__).parent.parent / "prompt_sets" / dataset_name

    if check_is_already_downloaded(path):
        print(f"Prompt set already downloaded at {path}.")
    else:
        print(f"Downloading prompt set {args.train_dataset} to {path}...")
        train_split = prepare_train_split(args.train_dataset)
        if args.sample_first_n is not None:
            train_split = train_split.select(range(args.sample_first_n))
        train_split.to_parquet(path / "train.parquet")
        print(f"Prompt set {args.train_dataset} downloaded successfully to {path}.")

        print(f"Downloading test split {args.test_dataset} to {path}...")
        test_split = prepare_test_split(args.test_dataset)
        if args.sample_first_n is not None:
            test_split = test_split.select(range(args.sample_first_n))
        test_split.to_parquet(path / "test.parquet")
        print(f"Test split {args.test_dataset} downloaded successfully to {path}.")
