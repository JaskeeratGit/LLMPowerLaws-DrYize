#!/usr/bin/env python3
"""
Loads the first user prompt from each conversation in LMSYS-Chat-1M and
ShareGPT, and all prompts from LongBench (context + question).
Requires HF_TOKEN in environment for gated datasets.
"""

import os
from datasets import load_dataset

LONGBENCH_SUBSETS = [
    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa",
    "musique", "gov_report_e", "qmsum", "multi_news_e", "trec", "triviaqa",
    "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p",
]


def _extract_lmsys_prompts(dataset_id: str, token: str | None) -> list[str]:
    ds = load_dataset(dataset_id, token=token, split="train")
    prompts = []
    for row in ds:
        for turn in row["conversation"]:
            if turn["role"] == "user":
                prompts.append(turn["content"])
                break
    return prompts


def _extract_sharegpt_prompts(dataset_id: str, token: str | None) -> list[str]:
    ds = load_dataset(dataset_id, token=token, split="train")
    prompts = []
    for row in ds:
        for turn in row["conversations"]:
            if turn["from"] == "human":
                prompts.append(turn["value"])
                break
    return prompts


def _extract_longbench_prompts(dataset_id: str, subsets: list[str], token: str | None) -> list[str]:
    prompts = []
    for subset in subsets:
        ds = load_dataset(dataset_id, name=subset, token=token, split="test")
        for row in ds:
            context = row.get("context", "").strip()
            question = row.get("input", "").strip()
            prompt = f"{context}\n\n{question}" if context else question
            prompts.append(prompt)
    return prompts


def load_prompts(config: dict) -> list[str]:
    token = os.environ.get("HF_TOKEN")
    prompts = []

    for name in config.get("datasets", []):
        if name == "lmsys":
            dataset_id = config["lmsys_dataset"]
            print(f"Loading {dataset_id}...")
            new = _extract_lmsys_prompts(dataset_id, token)
            prompts.extend(new)
            print(f"  +{len(new)} prompts (total {len(prompts)})")

        elif name == "sharegpt":
            dataset_id = config["sharegpt_dataset"]
            print(f"Loading {dataset_id}...")
            new = _extract_sharegpt_prompts(dataset_id, token)
            prompts.extend(new)
            print(f"  +{len(new)} prompts (total {len(prompts)})")

        elif name == "longbench":
            dataset_id = config["longbench_dataset"]
            subsets = config.get("longbench_subsets", LONGBENCH_SUBSETS)
            print(f"Loading {dataset_id} ({len(subsets)} subsets)...")
            new = _extract_longbench_prompts(dataset_id, subsets, token)
            prompts.extend(new)
            print(f"  +{len(new)} prompts (total {len(prompts)})")

    return prompts
