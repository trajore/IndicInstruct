#!/usr/bin/env python
# coding=utf-8
"""
This script is used to reformat the IndicInstruct dataset into the format that can be used by the model.
Here we use jsonl for the converted data. Each line in the jsonl file is a json object formatted as follows:
{
    "dataset": "dataset_name",
    "id": "unique_id",
    "messages": [
        {"role": "system", "content": "message_text"}, # optional
        {"role": "user", "content": "message_text"},
        {"role": "assistant", "content": "message_text"},
        {"role": "user", "content": "message_text"},
        {"role": "assistant", "content": "message_text"},
        ...
    ],
}
"""
import os
import random
import argparse
from tqdm import tqdm
from indic_num_map import INDIC_NUM_MAP
from datasets import load_dataset, Dataset
from instruction_encode_templates import encode_instruction_example, encode_translation_example


def normalize_indic_numerals(line):
    return "".join([INDIC_NUM_MAP.get(c, c) for c in line])


def reformat_dolly_data(dataset, splits=["en", "hi"]):
    processed_dataset = []
    for split in splits:
        for example in tqdm(dataset[split], desc="Processing data"):
            encoded_example = encode_instruction_example(
                instruction=normalize_indic_numerals(example["instruction"]),
                input=normalize_indic_numerals(example["context"]),
                output=normalize_indic_numerals(example["response"]),
                random_template=True,
                eos_token=None,
            )
            processed_dataset.append(
                {
                    "id": example["id"],
                    "dataset": "dolly",
                    "language": split,
                    "messages": [
                        {"role": "user", "content": encoded_example["prompt"]},
                        {"role": "assistant", "content": encoded_example["completion"]},
                    ],
                }
            )

    return processed_dataset


def reformat_flan_v2_data(dataset, splits=["en", "hi"]):
    processed_dataset = []
    for split in splits:
        for example in tqdm(dataset[split], desc="Processing data"):
            prompt = example["inputs"]
            if not prompt.endswith("\n") and not prompt.rstrip().endswith(":"):
                prompt += "\n"
            completion = example["targets"]
            processed_dataset.append(
                {
                    "id": example["id"],
                    "dataset": "flan_v2",
                    "language": split,
                    "messages": [
                        {"role": "user", "content": normalize_indic_numerals(prompt)},
                        {"role": "assistant", "content": normalize_indic_numerals(completion)},
                    ],
                }
            )

    return processed_dataset


def reformat_hh_rlhf_data(dataset, splits=["en", "hi"]):
    processed_dataset = []
    for split in splits:
        for example in tqdm(dataset[split], desc="Processing data"):
            messages = example["messages"]
            for i in range(len(messages)):
                messages[i]["content"] = normalize_indic_numerals(messages[i]["content"])
            processed_dataset.append(
                {"id": example["id"], "dataset": "hh-rlhf", "language": split, "messages": messages}
            )

    return processed_dataset


def reformat_oasst1_data(dataset, splits=["en", "hi"]):
    processed_dataset = []
    for split in splits:
        for example in tqdm(dataset[split], desc="Processing data"):
            messages = []
            for message in example["messages"]:
                messages.append(
                    {
                        "role": message["role"],
                        "content": normalize_indic_numerals(message["content"]),
                    }
                )
            processed_dataset.append(
                {"id": example["id"], "dataset": "oasst1", "language": split, "messages": messages}
            )

    return processed_dataset


def reformat_lm_sys_data(dataset, splits=["en", "hi"]):
    processed_dataset = []
    for split in splits:
        for example in tqdm(dataset[split], desc="Processing data"):
            messages = []
            for message in example["messages"]:
                messages.append(
                    {
                        "role": message["role"],
                        "content": normalize_indic_numerals(message["content"]),
                    }
                )
            processed_dataset.append(
                {"id": example["id"], "dataset": "lm_sys", "language": split, "messages": messages}
            )

    return processed_dataset


def reformat_anudesh_data(dataset, splits=["en", "hi"]):
    processed_dataset = []
    for split in splits:
        for example in tqdm(dataset[split], desc="Processing data"):
            messages = []
            for message in example["messages"]:
                messages.append(
                    {
                        "role": message["role"],
                        "content": normalize_indic_numerals(message["content"]),
                    }
                )
            processed_dataset.append(
                {"id": example["id"], "dataset": "anudesh", "language": split, "messages": messages}
            )

    return processed_dataset


def reformat_nmt_seed_data(dataset, splits=["hi"]):
    processed_dataset = []
    for split in splits:
        for example in tqdm(dataset[split], desc="Processing data"):
            encoded_example = encode_translation_example(
                src_lang=example["input_language"],
                tgt_lang=example["output_language"],
                src_text=example["input_text"],
                tgt_text=example["output_text"],
            )
            processed_dataset.append(
                {
                    "id": example["id"],
                    "dataset": "nmt",
                    "language": split,
                    "messages": [
                        {
                            "role": "user",
                            "content": normalize_indic_numerals(encoded_example["prompt"]),
                        },
                        {
                            "role": "assistant",
                            "content": normalize_indic_numerals(encoded_example["completion"]),
                        },
                    ],
                }
            )

    return processed_dataset


def reformat_wikihow_data(dataset, splits=["en", "hi"]):
    processed_dataset = []
    for split in splits:
        for example in tqdm(dataset[split], desc="Processing data"):
            messages = []
            for message in example["messages"]:
                messages.append(
                    {
                        "role": message["role"],
                        "content": normalize_indic_numerals(message["content"]),
                    }
                )
            processed_dataset.append(
                {"id": example["id"], "dataset": "anudesh", "language": split, "messages": messages}
            )

    return processed_dataset


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--output_dir", type=str, default="data/processed")
    arg_parser.add_argument("--seed", type=int, default=42)
    args = arg_parser.parse_args()
    random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    processed_datasets = []
    subsets = ["dolly", "flan_v2", "anudesh", "oasst1", "hh-rlhf", "nmt-seed", "wikihow", "lm_sys"]
    for subset in subsets:
        print(f"Processing {subset} data...")
        dataset = load_dataset("ai4bharat/indic-instruct-data-v0.1", subset)
        processed_dataset = globals()[f"reformat_{subset.replace('-', '_')}_data"](dataset)
        processed_datasets.extend(processed_dataset)

    random.shuffle(processed_dataset)
    processed_dataset = Dataset.from_list(processed_datasets)
    processed_dataset.to_json(os.path.join(args.output_dir, "indic_instruct_data.jsonl"))
