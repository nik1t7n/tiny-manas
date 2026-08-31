from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch

from .data import load_split, load_tokenizer
from .experiment import generate_text, load_checkpoint, require_mps


DEFAULT_PROMPTS = (
    "Манас",
    "Айкөл Манас",
    "Бакай",
    "— Кана, баатыр",
    "Кылычын сууруп",
)
DEFAULT_SEEDS = (1337, 2026, 3407, 6036)


def _words(text: str) -> list[str]:
    return re.findall(r"[\wӨҮҢөүң'-]+", text, flags=re.UNICODE)


def longest_copied_word_span(generated: str, corpus: str, maximum_words: int = 40) -> dict[str, Any]:
    words = _words(generated)
    if not words:
        return {"words": 0, "text": ""}
    upper = min(maximum_words, len(words))
    corpus_casefolded = corpus.casefold()
    for size in range(upper, 3, -1):
        for start in range(len(words) - size + 1):
            candidate = " ".join(words[start : start + size])
            if candidate.casefold() in corpus_casefolded:
                return {"words": size, "text": candidate}
    return {"words": 0, "text": ""}


def repetition_stats(text: str) -> dict[str, Any]:
    words = _words(text)
    if not words:
        return {"words": 0, "unique_word_ratio": 0.0, "repeated_trigram_ratio": 0.0}
    trigrams = [tuple(words[index : index + 3]) for index in range(max(0, len(words) - 2))]
    repeated = len(trigrams) - len(set(trigrams))
    return {
        "words": len(words),
        "unique_word_ratio": len(set(word.casefold() for word in words)) / len(words),
        "repeated_trigram_ratio": repeated / len(trigrams) if trigrams else 0.0,
    }


def run_generation_audit(
    checkpoint: str | Path,
    output: str | Path,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> dict[str, Any]:
    device = require_mps()
    model, payload = load_checkpoint(checkpoint, device)
    dataset_name = payload["experiment_config"]["data"]["dataset"]
    tokenizer = load_tokenizer()
    corpus = tokenizer.decode(load_split(dataset_name, "train").tolist())

    samples: list[dict[str, Any]] = []
    for prompt in DEFAULT_PROMPTS:
        for seed in DEFAULT_SEEDS:
            sample = generate_text(
                model,
                prompt,
                max_new_tokens,
                temperature,
                top_k,
                seed,
                device,
            )
            continuation_ids = sample["all_token_ids"][len(sample["prompt_token_ids"]) :]
            continuation = tokenizer.decode(continuation_ids)
            sample["continuation"] = continuation
            sample["longest_copied_word_span"] = longest_copied_word_span(continuation, corpus)
            sample["repetition"] = repetition_stats(continuation)
            samples.append(sample)

    copied_lengths = [sample["longest_copied_word_span"]["words"] for sample in samples]
    repeated_trigrams = [sample["repetition"]["repeated_trigram_ratio"] for sample in samples]
    result = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "dataset": dataset_name,
        "protocol": {
            "prompts": list(DEFAULT_PROMPTS),
            "seeds": list(DEFAULT_SEEDS),
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
        },
        "summary": {
            "samples": len(samples),
            "maximum_exact_copied_words": max(copied_lengths, default=0),
            "mean_repeated_trigram_ratio": sum(repeated_trigrams) / len(repeated_trigrams),
        },
        "samples": samples,
    }
    output_path = Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {**result["summary"], "output": str(output_path)}
