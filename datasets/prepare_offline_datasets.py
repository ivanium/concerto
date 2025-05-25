import json
import random
import os
import argparse
import tqdm

import pandas as pd
import numpy as np
from transformers import PreTrainedTokenizerBase, AutoTokenizer


def read_dataset(
    dataset_path: str,
    tokenizer: PreTrainedTokenizerBase,
    name: str,
    args: argparse.Namespace,
):
    """
    read_dataset: Read the given dataset and return a list of Request.
    """
    if name.lower() == "sharegpt":
        # Load the dataset.
        with open(dataset_path) as f:
            dataset = json.load(f)

        results = []
        for data in tqdm.tqdm(dataset):
            num_conversations = len(data["conversations"])

            # Filter out the conversations with less than args.sharegpt_min_turns turns.
            if num_conversations < args.sharegpt_min_turns or \
                num_conversations < args.sharegpt_min_prompt_turns + 1:
                continue

            num_prompt_turns = random.randint(
                args.sharegpt_min_prompt_turns,
                min(num_conversations - 1, args.sharegpt_max_prompt_turns))

            prompt = "\n".join([
                data["conversations"][i]["value"]
                for i in range(num_prompt_turns)
            ])
            completion = data["conversations"][num_prompt_turns]["value"]
            prompt_token_ids = tokenizer(prompt).input_ids
            completion_token_ids = tokenizer(completion).input_ids

            prompt_len = len(prompt_token_ids)
            output_len = len(completion_token_ids)
            if prompt_len < 4 and output_len < 4:
                # Prune too short sequences.
                continue
            if prompt_len + output_len >= 2048:
                # Prune too long sequences. (It exceeded max_positional_embedding)
                continue

            results.append((prompt_len, output_len))

        return "sharegpt", results

    elif name.lower() == "alpaca":
        with open(dataset_path) as f:
            dataset = json.load(f)

        # extract the input and output
        dataset = [(data["instruction"] + data["input"], data["output"])
                   for data in dataset]

        prompts = [prompt for prompt, _ in dataset]
        prompt_token_ids = tokenizer(prompts).input_ids
        completions = [completion for _, completion in dataset]
        completion_token_ids = tokenizer(completions).input_ids
        tokenized_dataset = []
        for i in range(len(dataset)):
            output_len = len(completion_token_ids[i])
            tokenized_dataset.append(
                (prompts[i], prompt_token_ids[i], output_len))

        # Filter out too long sequences.
        filtered_results = []
        for prompt, prompt_token_ids, output_len in tokenized_dataset:
            prompt_len = len(prompt_token_ids)
            if prompt_len < 4 and output_len < 4:
                # Prune too short sequences.
                continue
            if prompt_len > 1024 or prompt_len + output_len > 2048:
                # Prune too long sequences.
                continue
            filtered_results.append((prompt_len, output_len))

        return "alpaca", filtered_results

    elif name.lower() == "mmlu":
        dataset = []
        choices = ["A", "B", "C", "D"]
        data_path = dataset_path
        subjects = sorted([
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(data_path, "test"))
            if "_test.csv" in f
        ])

        for sub in subjects:
            test_df = pd.read_csv(os.path.join(data_path, "test",
                                               sub + "_test.csv"),
                                  header=None)
            for i in range(test_df.shape[0]):
                prompt = test_df.iloc[i, 0]
                k = test_df.shape[1] - 2
                for j in range(k):
                    prompt += f"\n{choices[j]}. {test_df.iloc[i, j + 1]}"
                prompt += "\nAnswer:"
                output = test_df.iloc[i, k + 1]
                dataset.append((prompt, output))

        print("MMLU dataset size:", len(dataset))

        prompts = [prompt for prompt, _ in dataset]
        prompt_token_ids = tokenizer(prompts).input_ids
        completions = [completion for _, completion in dataset]
        completion_token_ids = tokenizer(completions).input_ids
        tokenized_dataset = []
        for i in range(len(dataset)):
            output_len = len(completion_token_ids[i])
            tokenized_dataset.append(
                (prompts[i], prompt_token_ids[i], output_len))

        # Filter out too long sequences.
        filtered_results = []
        for prompt, prompt_token_ids, output_len in tokenized_dataset:
            prompt_len = len(prompt_token_ids)
            if prompt_len < 4 and output_len < 4:
                # Prune too short sequences.
                continue
            if prompt_len > 1024 or prompt_len + output_len > 2048:
                # Prune too long sequences.
                continue
            filtered_results.append((prompt_len, output_len))

        return "mmlu", filtered_results

    elif name.lower() == "longbench":
        # find all .jsonl files under the dataset_path
        files = []
        for root, dirs, filenames in os.walk(dataset_path):
            filenames = sorted(filenames)
            for filename in filenames:
                if filename.endswith(".jsonl"):
                    files.append(os.path.join(root, filename))
        files = sorted(files)

        filtered_results = []
        for file in tqdm.tqdm(files):
            with open(file) as f:
                for line in f.readlines():
                    if line.strip() == "":
                        continue
                    data = json.loads(line)

                    # truncate to the first 40000 chars to reduce tokenization time
                    context = data["context"][:40000]
                    context_token_ids = tokenizer(context).input_ids
                    answer_token_ids = tokenizer(data["answers"][0]).input_ids
                    context_len = len(context_token_ids)
                    answer_len = len(answer_token_ids)

                    context_len_allowed = min(context_len,
                                              args.longbench_max_prompt_len)
                    context_token_ids = context_token_ids[:context_len_allowed]

                    filtered_results.append(
                        (len(context_token_ids), answer_len))

        return "longbench", filtered_results

    elif name.lower() == "humaneval":
        filtered_results = []
        with open(dataset_path) as f:
            for line in f.readlines():
                if line.strip() == "":
                    continue
                data = json.loads(line)

                context = data["prompt"]
                context_token_ids = tokenizer(context).input_ids
                answer = data["canonical_solution"]
                answer_token_ids = tokenizer(answer).input_ids

                if len(context_token_ids) + len(answer_token_ids) >= 2048:
                    continue

                filtered_results.append(
                    (len(context_token_ids), len(answer_token_ids)))

        # Copy the dataset for 10 times since it's too small.
        filtered_results = filtered_results * 10

        return "humaneval", filtered_results

    else:
        raise ValueError(f"Unsupported dataset name: {name}")


def prepare_dataset(
    dataset_path: str,
    output_path: str,
    tokenizer: PreTrainedTokenizerBase,
    name: str,
    args: argparse.Namespace,
):
    output_path = f"{output_path}.txt"
    if os.path.exists(output_path):
        print(f"Output file {output_path} already exists. Loading...")
        return np.loadtxt(output_path, dtype=int, delimiter=",")
    else:
        dataset_name, reqs = read_dataset(dataset_path, tokenizer, name, args)
        assert dataset_name == name
        reqs = np.array(reqs, dtype=int)
        np.savetxt(output_path, reqs, fmt="%d,%d")
        return reqs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="longbench")
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--sharegpt-min-turns", type=int, default=3)
    parser.add_argument("--sharegpt-min-prompt-turns", type=int, default=1)
    parser.add_argument("--sharegpt-max-prompt-turns", type=int, default=1000)

    parser.add_argument("--longbench-min-prompt-len", type=int, default=1900)
    parser.add_argument("--longbench-max-prompt-len", type=int, default=2048)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)
    reqs = prepare_dataset(args.dataset_path, args.output_path, tokenizer,
                           args.dataset, args)
    print(f"Loaded {len(reqs)} requests from dataset {args.dataset_path}")
