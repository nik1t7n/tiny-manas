"""Freeze document splits, byte-exact tokenization, and train-only span removal."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import torch

from extract_manas_verse import SPECS
from followup_common import ROOT, save_json, sha
from manas_gpt.data import load_split, load_tokenizer, TOKENIZER_SHA256

OUT = ROOT / "runs/data-evaluation-20260905/bundle"
SOURCE = OUT.parent / "source-review"
WORD = re.compile(r"[\w]+", re.UNICODE)


def words(text):
    return [(m.group().casefold(), m.start(), m.end()) for m in WORD.finditer(text)]


def keys(rows):
    for i in range(len(rows) - 31):
        yield hashlib.blake2b(" ".join(w[0] for w in rows[i:i+32]).encode(), digest_size=16).digest(), i


def prune(text, forbidden):
    rows = words(text)
    intervals = []
    for key, i in keys(rows):
        if key in forbidden:
            a, b = rows[i][1], rows[i+31][2]
            if intervals and a <= intervals[-1][1]:
                intervals[-1][1] = max(b, intervals[-1][1])
            else:
                intervals.append([a, b])
    segments, cursor = [], 0
    for a, b in intervals:
        segments.append((cursor, a, text[cursor:a]))
        cursor = b
    segments.append((cursor, len(text), text[cursor:]))
    return segments, intervals


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "manifest.json").exists():
        raise FileExistsError("The evaluation bundle is frozen; do not overwrite it")
    tokenizer = load_tokenizer()
    texts = {name: (SOURCE / f"{name}.verse.txt").read_text() for name in SPECS}
    texts["karalaev-train"] = tokenizer.decode(load_split("manas01-full", "train").tolist())
    holdouts = [texts[n] for n in SPECS if SPECS[n][2] != "train"]
    holdouts.extend(tokenizer.decode(load_split("manas01-full", s).tolist()) for s in ("validation", "test"))
    forbidden = {k for text in holdouts for k, _ in keys(words(text))}
    lengths = [len(tokenizer.id_to_token(i)) for i in range(32768)]
    documents, arrays, audit = [], {}, {}
    for name in ["karalaev-train", *SPECS]:
        text = texts[name]
        split = "train" if name == "karalaev-train" else SPECS[name][2]
        if "\ufffd" in text:
            raise RuntimeError(f"Replacement characters in {name}")
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        if tokenizer.decode(ids) != text or sum(lengths[i] for i in ids) != len(text.encode()):
            raise RuntimeError(f"Byte round trip failed: {name}")
        segments, removed = prune(text, forbidden) if split == "train" else ([(0, len(text), text)], [])
        kept_tokens = 0
        segment_records = []
        for index, (a, b, segment) in enumerate(segments):
            token_ids = tokenizer.encode(segment, add_special_tokens=False).ids
            # Same window eligibility at both training contexts.
            if len(token_ids) < 513:
                continue
            if tokenizer.decode(token_ids) != segment:
                raise RuntimeError("Segment failed exact round trip")
            key = f"{name}-{index}"
            arrays[key] = torch.tensor(token_ids, dtype=torch.int32)
            kept_tokens += len(token_ids)
            segment_records.append({"key": key, "characters": [a, b], "tokens": len(token_ids),
                                    "utf8_bytes": len(segment.encode()),
                                    "text_sha256": hashlib.sha256(segment.encode()).hexdigest()})
        record = {"id": name, "split": split, "tokens_before_pruning": len(ids),
                  "tokens_kept": kept_tokens, "text_bytes": len(text.encode()),
                  "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                  "removed_character_spans": removed, "segments": segment_records,
                  "source_group": "Karalaev-2010" if name.startswith("karalaev") else ("Mamay-2004" if name == "mamay" else "Orozbakov-academic"),
                  "roundtrip": True}
        documents.append(record)
        audit[name] = {"tokens": len(ids), "bytes_per_token": len(text.encode())/len(ids),
                       "kept": kept_tokens, "removed_passages": len(removed), "segments": len(segment_records)}
        if split == "train":
            # Remove long exact repeats from later training volumes too.
            forbidden.update(k for k, _ in keys(words(text)))
    torch.save(arrays, OUT / "tokens.pt")
    save_json(OUT / "manifest.json", {"schema_version": 1, "created": "2026-09-05",
        "tokenizer_sha256": TOKENIZER_SHA256, "tokens_sha256": sha(OUT / "tokens.pt"),
        "extractor_sha256": sha(Path(__file__).with_name("extract_manas_verse.py")),
        "builder_sha256": sha(__file__), "documents": documents,
        "source_splits": {n: {"pdf_pages": list(SPECS[n][:2]), "split": SPECS[n][2],
                              "pdf_sha256": sha(SOURCE / f"{n}.pdf")} for n in SPECS},
        "deduplication": "32 Unicode-word, casefolded exact spans; train only; no stitching across removals; later training books deduplicated against earlier ones",
        "evaluation": "first 512 tokens context only; stride 128; 256 evenly spaced validation windows for selection; whole remaining verse for final reporting",
        "mixture": "50% original-source windows, 50% new-source windows proportional to eligible starts; control original-source only; identical pruned original in both",
        "authorization": "Explicit owner research permission, not independently verified blanket redistribution license",
        "limitations": "Residual scan OCR spelling errors remain; no language-model text repair. Books from one academic edition are related, not independent narrators. Tokenizer was previously trained on some Manas text.",
        "excluded": {"orozbakov-1": "Dense inline apparatus and visible OCR errors; not admitted in this bounded extraction pass",
                     "orozbakov-6-7": "Two-column extraction and systematic glyph errors require separate remediation",
                     "orozbakov-8-9": "Two-column extraction and systematic glyph errors require separate remediation"}})
    save_json(OUT / "tokenization-audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
