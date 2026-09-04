"""Correct the word-span audit on saved real generations, without regenerating."""
import argparse
import hashlib
import json
from pathlib import Path

from manas_gpt.audit import WORD_MATCH_PROTOCOL, _words, audit_summary, longest_copied_word_span, repetition_stats
from manas_gpt.data import load_split, load_tokenizer

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = json.loads(args.input.read_text())
    dataset = payload["dataset"]
    corpus = load_tokenizer().decode(load_split(dataset, "train").tolist())
    # Positive control is an actual source excerpt, not invented model output.
    # The processed epic is one line; bound by characters, not line count.
    excerpt = corpus[:1000].rsplit(" ", 1)[0]
    expected = len(_words(excerpt))
    measured = longest_copied_word_span(excerpt, corpus)["words"]
    if expected < 40 or measured != expected:
        raise AssertionError({"expected_real_excerpt_words": expected, "measured": measured})
    previous_summary = payload["summary"]
    changes = []
    for index, sample in enumerate(payload["samples"]):
        old = sample["longest_copied_word_span"]["words"]
        sample["longest_copied_word_span"] = longest_copied_word_span(sample["continuation"], corpus)
        sample["repetition"] = repetition_stats(sample["continuation"])
        new = sample["longest_copied_word_span"]["words"]
        if old != new:
            changes.append({"sample": index, "old_words": old, "new_words": new})
    payload["protocol"]["word_matching"] = WORD_MATCH_PROTOCOL
    payload["summary"] = audit_summary(payload["samples"])
    payload["reanalysis"] = {
        "original_audit": str(args.input.resolve()),
        "original_audit_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "train_tokens_sha256": hashlib.sha256((ROOT / "data/processed" / dataset / "train.bin").read_bytes()).hexdigest(),
        "audit_source_sha256": hashlib.sha256((ROOT / "src/manas_gpt/audit.py").read_bytes()).hexdigest(),
        "original_summary": previous_summary,
        "changed_samples": changes,
        "real_source_positive_control_words": measured,
        "new_generation": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"summary": payload["summary"], "changes": changes, "positive_control_words": measured, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
