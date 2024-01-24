import argparse
import json
import numpy as np
import random
import tqdm.auto as tqdm
import sys
import torch
import datasets
import transformers


def read_jsonl(path):
    # Manually open because .splitlines is different from iterating over lines
    with open(path, "r") as f:
        for line in f:
            yield json.loads(line)


def read_lm_dataformat(path):
    import lm_dataformat

    reader = lm_dataformat.Reader(path)
    yield from reader.stream_data()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--data_format", type=str, default="jsonl")
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_examples", type=int, default=100000000)
    parser.add_argument("--max_tokens", type=int, default=1000000000)
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
    all_tokenized = []
    if args.data_format == "jsonl":
        reader = read_jsonl(args.data_path)
    elif args.data_format == "lm_dataformat":
        reader = read_lm_dataformat(args.data_path)
    elif args.data_format == "parquet":
        reader = datasets.load_dataset("parquet", data_files=[args.data_path])["train"]
    elif args.data_format == "txt":
        reader = open(args.data_path, "r")
    else:
        raise KeyError(args.data_format)

    print("Encoding")
    ctr = 0
    total_len = 0
    total_words = 0
    for elem in reader:
        if ctr == args.max_examples:
            break
        text = (
            elem["text"]
            if args.data_format == "jsonl"
            or args.data_format == "lm_dataformat"
            or args.data_format == "parquet"
            else elem.strip()
        )
        total_words += len(text.split(" "))
        if len(text.split(" ")) > 32768:  # Split into smaller chunks
            # print("Super long document with presplit length", len(text.split(" ")))
            # print("Splitting")
            text = text.split(" ")
            docsplit = [tokenizer.bos_token_id]
            for i in range(0, len(text), 32768):
                # print("Current chunk length", len(text[i:i+32768]))
                docsplit.extend(
                    tokenizer.encode(" ".join(text[i : i + 32768]), add_special_tokens=False)
                )
            docsplit.append(tokenizer.eos_token_id)
            all_tokenized.append(docsplit)
        else:
            # print("Normal document with length", len(text.split(" ")))
            all_tokenized.append(
                [tokenizer.bos_token_id]
                + tokenizer.encode(text, add_special_tokens=False)
                + [tokenizer.eos_token_id]
            )
        len_of_doc = len(all_tokenized[-1])
        total_len += len_of_doc
        if total_len > args.max_tokens:
            print("Total tokens exceeded", total_len)
            break
        # print("Tokenized length of document", len_of_doc)
        ctr += 1
        if ctr % 10000 == 0:
            print("Processed", ctr, "documents")
            print("Total tokens so far", total_len)

    random.shuffle(all_tokenized)
    ## Concatenate all the tokenized documents
    all_tokens = []
    print("Concatenating")
    for doc in tqdm.tqdm(all_tokenized):
        all_tokens.extend(doc)

    ## Truncate the tokens to the max_seq_length
    print("Truncating")
    print("Total words", total_words)
    print("Pretruncation subwords length", len(all_tokens))
    truncated_tokens = all_tokens[: (len(all_tokens) // args.max_seq_length) * args.max_seq_length]
    print("Posttruncation subwords length", len(truncated_tokens))
    ## Now we take each chunk of max_seq_length tokens and make it into pytorch tensors
    print("Creating dataset")

    def gen():
        for i in range(0, len(truncated_tokens), args.max_seq_length):
            input_ids = torch.LongTensor(truncated_tokens[i : i + args.max_seq_length])
            labels = input_ids.clone()
            attention_mask = torch.ones_like(input_ids)
            dict_to_save = {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            }
            yield dict_to_save

    ds = datasets.Dataset.from_generator(gen)

    print("Saving")
    ds.save_to_disk(args.save_path)


if __name__ == "__main__":
    main()
