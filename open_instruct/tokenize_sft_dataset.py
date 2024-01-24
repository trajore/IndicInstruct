import argparse
import json
import numpy as np
import random
import tqdm.auto as tqdm
import sys
import torch
import datasets
import transformers
from functools import partial
from datasets import load_dataset

from transformers import LlamaTokenizer, LlamaTokenizerFast


def read_jsonl(path):
    # Manually open because .splitlines is different from iterating over lines
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def encode_with_prompt_completion_format(example, tokenizer, max_seq_length):
    """
    Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated
    and it doesn't make sense to follow directly with the completion.
    """
    # if prompt doesn't end with space and completion doesn't start with space, add space
    if not example["prompt"].endswith((" ", "\n", "\t")) and not example["completion"].startswith(
        (" ", "\n", "\t")
    ):
        example_text = example["prompt"] + " " + example["completion"]
    else:
        example_text = example["prompt"] + example["completion"]
    example_text = example_text + tokenizer.eos_token
    # print(example_text)
    tokenized_example = tokenizer(
        example_text, return_tensors="pt", max_length=max_seq_length, truncation=True
    )
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    tokenized_prompt = tokenizer(
        example["prompt"], return_tensors="pt", max_length=max_seq_length, truncation=True
    )
    # print(tokenized_prompt.input_ids)
    # mask the prompt part for avoiding loss
    labels[:, : tokenized_prompt.input_ids.shape[1]] = -100
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


def encode_with_messages_format(example, tokenizer, max_seq_length):
    """
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    """
    messages = example["messages"]
    if len(messages) == 0:
        raise ValueError("messages field is empty.")

    def _concat_messages(messages):
        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += "<|system|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "user":
                message_text += "<|user|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "assistant":
                message_text += (
                    "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
                )
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text

    example_text = _concat_messages(messages).strip()
    tokenized_example = tokenizer(
        example_text, return_tensors="pt", max_length=max_seq_length, truncation=True
    )
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]),
                    return_tensors="pt",
                    max_length=max_seq_length,
                    truncation=True,
                ).input_ids.shape[1]
            if message_idx < len(messages) - 1 and messages[message_idx + 1]["role"] == "assistant":
                # here we also ignore the role of the assistant
                messages_so_far = _concat_messages(messages[: message_idx + 1]) + "<|assistant|>\n"
            else:
                messages_so_far = _concat_messages(messages[: message_idx + 1])
            message_end_idx = tokenizer(
                messages_so_far, return_tensors="pt", max_length=max_seq_length, truncation=True
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100

            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--preprocessing_num_workers", type=int, default=64)
    parser.add_argument("--use_fast", action="store_true")
    args = parser.parse_args()
    print("Loading Tokenizer")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.tokenizer_path, use_fast=args.use_fast
    )
    print("Tokenizer info", tokenizer)
    print("BOS token and id", tokenizer.bos_token, tokenizer.bos_token_id)
    print("EOS token and id", tokenizer.eos_token, tokenizer.eos_token_id)
    print("PAD token and id", tokenizer.pad_token, tokenizer.pad_token_id)
    print("UNK token and id", tokenizer.unk_token, tokenizer.unk_token_id)
    print("Loading Dataset")
    dataset_args = {}

    raw_datasets = load_dataset(
        "json",
        data_files=args.data_path,
    )

    # print(raw_datasets)

    if (
        "prompt" in raw_datasets["train"].column_names
        and "completion" in raw_datasets["train"].column_names
    ):
        encode_function = partial(
            encode_with_prompt_completion_format,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
        )
    elif "messages" in raw_datasets["train"].column_names:
        encode_function = partial(
            encode_with_messages_format,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
        )

    print("Encoding")

    lm_datasets = raw_datasets["train"].map(
        encode_function,
        batched=False,
        num_proc=args.preprocessing_num_workers,
        remove_columns=[
            name
            for name in raw_datasets["train"].column_names
            if name not in ["input_ids", "labels", "attention_mask"]
        ],
        desc="Tokenizing and reformatting instruction data",
    )
    lm_datasets.set_format(type="pt")
    lm_datasets = lm_datasets.filter(lambda example: (example["labels"] != -100).any())
    # print(lm_datasets)

    print("Saving")
    lm_datasets.save_to_disk(args.save_path)


if __name__ == "__main__":
    main()
