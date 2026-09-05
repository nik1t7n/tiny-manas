"""Real-data scoring and audit helpers for the O14–O17 follow-up.

No corpus substitution, model training, or artifact writes occur on import.
"""
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from manas_gpt.audit import (DEFAULT_PROMPTS, DEFAULT_SEEDS, WORD_MATCH_PROTOCOL,
                            audit_summary, longest_copied_word_span, repetition_stats)
from manas_gpt.data import load_split, load_tokenizer, TOKENIZER_SHA256
from manas_gpt.experiment import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "artifacts/tiny-manas-27m-rope-20260904.pt"
BASELINE_SHA = "abc13354d5cb1cc94c966985d95252befdfaf9f25b19c1884701442f4e519d8f"
EVALUATION_PREFIX = 512
EVALUATION_STRIDE = 128


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def precision_context(precision, device):
    if precision == "fp32":
        return nullcontext()
    if precision != "bf16" or device.type not in ("mps", "cpu"):
        raise ValueError("Only explicit FP32 or BF16 on MPS/CPU is supported")
    return torch.autocast(device.type, dtype=torch.bfloat16)


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()


class FixedEvaluation:
    """Score each retained target once, with the same targets at 256 and 512.

    Targets start at index 512 in the frozen original split. The first 512
    tokens are context only. Evaluation is not a new independent document.
    """
    def __init__(self, split="validation"):
        if split not in ("validation", "test"):
            raise ValueError(split)
        baseline = json.loads((ROOT / "docs/experiments/00-baseline-manifest.json").read_text())
        rel = f"data/processed/manas01-full/{split}.bin"
        self.source_sha = sha(ROOT / rel)
        if self.source_sha != baseline["files"][rel]["sha256"]:
            raise RuntimeError("Frozen evaluation split changed")
        self.tokenizer = load_tokenizer()
        self.ids = load_split("manas01-full", split)
        self.lengths = torch.tensor([len(self.tokenizer.id_to_token(i)) for i in range(32768)])
        text = self.tokenizer.decode(self.ids.tolist())
        if "\ufffd" in text or self.tokenizer.encode(text, add_special_tokens=False).ids != self.ids.tolist():
            raise RuntimeError("Evaluation source failed exact round trip")
        if int(self.lengths[self.ids].sum()) != len(text.encode("utf8")):
            raise RuntimeError("Byte-level accounting mismatch")
        self.spans = [(i, min(i + EVALUATION_STRIDE, len(self.ids)))
                      for i in range(EVALUATION_PREFIX, len(self.ids), EVALUATION_STRIDE)]
        self.info = {"dataset": "manas01-full", "split": split,
                     "source_sha256": self.source_sha, "tokenizer_sha256": TOKENIZER_SHA256,
                     "prefix_tokens": EVALUATION_PREFIX, "stride": EVALUATION_STRIDE,
                     "targets": len(self.ids) - EVALUATION_PREFIX,
                     "scored_bytes": int(self.lengths[self.ids[EVALUATION_PREFIX:]].sum()),
                     "windows": len(self.spans), "target_coverage": "once; shared across contexts",
                     "independent_document_holdout": False}

    def batches(self, context=256, batch_size=4):
        if context not in (256, 512):
            raise ValueError("Use the preregistered 256 or 512 context")
        for offset in range(0, len(self.spans), batch_size):
            spans = self.spans[offset:offset + batch_size]
            x = torch.zeros((len(spans), context), dtype=torch.long)
            y = torch.full_like(x, -100)
            target_positions = torch.full_like(x, -1)
            for row, (start, end) in enumerate(spans):
                input_start = max(0, end - context - 1)
                length = end - input_start - 1
                x[row, :length] = self.ids[input_start:end - 1]
                first = start - input_start - 1
                y[row, first:length] = self.ids[start:end]
                target_positions[row, first:length] = torch.arange(start, end)
            yield x, y, target_positions


@torch.inference_mode()
def evaluate(model, data, device, context=256, precision="fp32"):
    model.eval()
    nll = 0.0
    targets = top1 = 0
    start = time.perf_counter()
    for x, y, positions in data.batches(context):
        x, y = x.to(device), y.to(device)
        with precision_context(precision, device):
            logits, _ = model(x)
        active = y != -100
        selected = logits[active].float()
        if not torch.isfinite(selected).all():
            raise RuntimeError("Nonfinite evaluation logits")
        nll += F.cross_entropy(selected, y[active], reduction="sum").item()
        top1 += (selected.argmax(-1) == y[active]).sum().item()
        targets += active.sum().item()
    synchronize(device)
    if targets != data.info["targets"]:
        raise RuntimeError("Missing or repeated evaluation targets")
    return {"loss": nll / targets, "perplexity": math.exp(nll / targets),
            "bits_per_byte": nll / math.log(2) / data.info["scored_bytes"],
            "summed_nll": nll, "targets": targets, "scored_bytes": data.info["scored_bytes"],
            "top1_accuracy": top1 / targets, "context": context, "precision": precision,
            "elapsed_seconds": time.perf_counter() - start, "evaluation": data.info}


def baseline_model(device):
    if sha(BASELINE) != BASELINE_SHA:
        raise RuntimeError("Accepted baseline checkpoint identity mismatch")
    model, payload = load_checkpoint(BASELINE, device)
    model.eval()
    return model, payload


@torch.inference_mode()
def generation_audit(model, device, output, precision="fp32", training_text=None):
    tokenizer = load_tokenizer()
    training = training_text if training_text is not None else tokenizer.decode(load_split("manas01-full", "train").tolist())
    samples = []
    for prompt in DEFAULT_PROMPTS:
        for seed in DEFAULT_SEEDS:
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
            x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            synchronize(device)
            started = time.perf_counter()
            with precision_context(precision, device):
                generated = model.generate(x, 256, .8, 40,
                    torch.Generator(device=device).manual_seed(seed))
            synchronize(device)
            text = tokenizer.decode(generated[0, len(prompt_ids):].tolist())
            samples.append({"prompt": prompt, "seed": seed, "new_tokens": 256,
                "continuation": text, "elapsed_seconds": time.perf_counter() - started,
                "repetition": repetition_stats(text),
                "longest_copied_word_span": longest_copied_word_span(text, training)})
            save_json(output, {"protocol": WORD_MATCH_PROTOCOL, "precision": precision,
                "temperature": .8, "top_k": 40, "summary": audit_summary(samples), "samples": samples})
    return audit_summary(samples)
