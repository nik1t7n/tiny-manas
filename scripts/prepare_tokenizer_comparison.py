"""Prepare immutable real-text variants without moving train/validation boundaries."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from tokenizers import Tokenizer

from manas_gpt.data import TOKENIZER_SHA256, _write_uint16, load_split, load_tokenizer

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_ROOT = Path("/Users/nik1t7n/Projects/learning/kyrgyz-tokenizer/artifacts/tokenizer-v1/models")
HASHES = {
    32768: TOKENIZER_SHA256,
    16384: "32c7b3816d03704d6b6ad26a9234de77a40b19bbaca2e60da84f1bf1f7c05abb",
    8192: "ec195af483856a6dee03bfd00af2b4c766f8f2a417213798dc090997f10f59b2",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    baseline = json.loads((ROOT / "docs/experiments/00-baseline-manifest.json").read_text())
    original = load_tokenizer()
    texts = {}
    inputs = {}
    for split in ("train", "validation"):
        relative = f"data/processed/manas01-full/{split}.bin"
        actual = sha(ROOT / relative)
        if actual != baseline["files"][relative]["sha256"]:
            raise RuntimeError(f"Frozen split hash changed: {relative}")
        texts[split] = original.decode(load_split("manas01-full", split).tolist())
        if "\ufffd" in texts[split]:
            raise RuntimeError(f"Replacement character in {split}")
        inputs[relative] = actual
    # Resolve all source artifacts before creating any output directory.
    sources = {}
    for size, expected in HASHES.items():
        path = TOKENIZER_ROOT / f"bpe-{size}" / "tokenizer.json"
        if sha(path) != expected:
            raise RuntimeError(f"Tokenizer hash changed: {path}")
        sources[size] = path
    out = ROOT / "runs" / ("optimization-05-data-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out.mkdir(exist_ok=False)
    manifest = {"schema_version": 1, "status": "preparing", "source_split_hashes": inputs,
                "preparation_source_sha256": sha(Path(__file__)), "variants": {},
                "test_loaded": False, "new_tokenizer_training": False,
                "protocol": {"train_epochs": 30, "context": 256, "micro_batch": 8,
                             "maximum_microbatches_per_update": 2, "validation_stride": 128,
                             "padding_target": -100, "prefix_rule": "first space at or after character 64"}}
    for split, text in texts.items():
        (out / f"{split}.txt").write_text(text, encoding="utf-8")
    for size, source in sources.items():
        destination = out / str(size)
        destination.mkdir()
        shutil.copy2(source, destination / "tokenizer.json")
        tokenizer = Tokenizer.from_file(str(destination / "tokenizer.json"))
        if tokenizer.get_vocab_size() != size:
            raise RuntimeError(f"Vocabulary size mismatch: {source}")
        lengths = [len(tokenizer.id_to_token(i)) for i in range(size)]
        variant = {"vocab_size": size, "tokenizer_sha256": HASHES[size], "source": str(source), "splits": {}}
        for split, text in texts.items():
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            cut = text.index(" ", 64)
            prefix = tokenizer.encode(text[:cut], add_special_tokens=False).ids
            rest = tokenizer.encode(text[cut:], add_special_tokens=False).ids
            if prefix + rest != ids or tokenizer.decode(ids) != text:
                raise RuntimeError(f"Round trip or shared-prefix mismatch: {size}/{split}")
            byte_lengths = [lengths[i] for i in ids]
            raw = text.encode("utf-8")
            scored = text[cut:].encode("utf-8")
            if sum(byte_lengths) != len(raw) or sum(byte_lengths[len(prefix):]) != len(scored):
                raise RuntimeError(f"ByteLevel accounting mismatch: {size}/{split}")
            _write_uint16(destination / f"{split}.bin", ids)
            _write_uint16(destination / f"{split}.byte-lengths.bin", byte_lengths)
            variant["splits"][split] = {
                "tokens": len(ids), "prefix_tokens": len(prefix), "prefix_characters": cut,
                "prefix_utf8_bytes": len(raw) - len(scored), "scored_tokens": len(rest),
                "utf8_bytes": len(raw), "scored_utf8_bytes": len(scored),
                "text_sha256": hashlib.sha256(raw).hexdigest(),
                "scored_text_sha256": hashlib.sha256(scored).hexdigest(),
            }
        manifest["variants"][str(size)] = variant
    for split in texts:
        if len({v["splits"][split]["scored_text_sha256"] for v in manifest["variants"].values()}) != 1:
            raise RuntimeError(f"Different scored text across vocabularies: {split}")
    manifest["files"] = {str(path.relative_to(out)): sha(path) for path in sorted(out.rglob("*")) if path.is_file()}
    manifest["status"] = "verified_prepared_not_trained"
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for path in out.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    print(json.dumps({"directory": str(out), "status": manifest["status"], "manifest_sha256": sha(out / "manifest.json"),
                      "variants": {size: data["splits"] for size, data in manifest["variants"].items()}}, indent=2))


if __name__ == "__main__":
    main()
