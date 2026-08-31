from __future__ import annotations

import hashlib
import html
import json
import re
import sys
import urllib.request
import zipfile
from array import array
from pathlib import Path
from typing import Any

import tokenizers
import torch
from tokenizers import Tokenizer

from .config import DataConfig
from .paths import PROCESSED_DIR, RAW_DIR, TOKENIZER_DIR


MANAS_URL = "https://fedora.clarin-d.uni-saarland.de/kyrgyz/kyrgyz_2022_10_03.zip"
MANAS_SHA1 = "82fc72b9687175c55b5309822da2a7c44bb303f1"
MANAS_MEMBER = "kyrgyz_2022_10_03.vrt"
MANAS_DOCUMENT_ID = "Manas01"
EPIC_START_HEADING = "Манастын туула элегиндеги бабалары"

TOKENIZER_COMMIT = "594d9e142cca1593963ccf12f344ab7ea4938fa5"
TOKENIZER_URL = (
    "https://raw.githubusercontent.com/nik1t7n/kyrgyz-tokenizer/"
    f"{TOKENIZER_COMMIT}/models/kyrgyz-byte-bpe-v1/tokenizer.json"
)
TOKENIZER_SHA256 = "5047b4f427bb1af1c06cfb9cefbe83790b56df409b137b887988db6eba4b159f"

ATTRIBUTE_RE = re.compile(r'(\w+)="([^"]*)"')


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(url: str, destination: Path, algorithm: str, expected: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        actual = file_digest(destination, algorithm)
        if actual != expected:
            raise RuntimeError(
                f"Existing file failed {algorithm} verification: {destination}\n"
                f"expected {expected}, got {actual}. Remove it explicitly and retry."
            )
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "manas-gpt/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)

    actual = file_digest(partial, algorithm)
    if actual != expected:
        raise RuntimeError(
            f"Downloaded file failed {algorithm} verification: {url}\n"
            f"expected {expected}, got {actual}. Partial file remains at {partial}."
        )
    partial.replace(destination)
    return destination


def _detokenize_vrt_sentence(tokens: list[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.;:!?%…\)\]\}»])", r"\1", text)
    text = re.sub(r"([\(\[\{«])\s+", r"\1", text)
    text = re.sub(r"\s+([’'])\s+", r"\1", text)
    return text.strip()


def extract_manas01(archive_path: Path) -> tuple[str, dict[str, str]]:
    attributes: dict[str, str] = {}
    sentences: list[str] = []
    sentence_tokens: list[str] = []
    inside_target = False

    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(MANAS_MEMBER) as binary:
            for raw_line in binary:
                line = raw_line.decode("utf-8").rstrip("\n")
                if line.startswith("<text "):
                    candidate = {
                        key: html.unescape(value) for key, value in ATTRIBUTE_RE.findall(line)
                    }
                    inside_target = candidate.get("id") == MANAS_DOCUMENT_ID
                    if inside_target:
                        attributes = candidate
                        sentences = []
                elif inside_target and line == "<s>":
                    sentence_tokens = []
                elif inside_target and line == "</s>":
                    sentence = _detokenize_vrt_sentence(sentence_tokens)
                    if sentence:
                        sentences.append(sentence)
                elif inside_target and line == "</text>":
                    text = " ".join(sentences).strip()
                    if not text:
                        raise RuntimeError("Manas01 was found but contained no text")
                    return text, attributes
                elif inside_target and line and not line.startswith("<"):
                    sentence_tokens.append(line.split("\t", 1)[0])

    raise RuntimeError(f"Document {MANAS_DOCUMENT_ID} was not found in {MANAS_MEMBER}")


def _write_uint16(path: Path, values: list[int]) -> None:
    if values and max(values) > 65535:
        raise ValueError("Token IDs exceed uint16 capacity")
    payload = array("H", values)
    if sys.byteorder != "little":
        payload.byteswap()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        payload.tofile(handle)


def _read_uint16(path: Path) -> list[int]:
    payload = array("H")
    with path.open("rb") as handle:
        payload.fromfile(handle, path.stat().st_size // payload.itemsize)
    if sys.byteorder != "little":
        payload.byteswap()
    return payload.tolist()


def dataset_dir(name: str) -> Path:
    if not name or Path(name).name != name:
        raise ValueError(f"Invalid dataset name: {name!r}")
    return PROCESSED_DIR / name


def prepare_dataset(config: DataConfig) -> dict[str, Any]:
    archive_path = download_verified(
        MANAS_URL,
        RAW_DIR / "manas-uds" / "kyrgyz_2022_10_03.zip",
        "sha1",
        MANAS_SHA1,
    )
    tokenizer_path = download_verified(
        TOKENIZER_URL,
        TOKENIZER_DIR / "kyrgyz-byte-bpe-v1.json",
        "sha256",
        TOKENIZER_SHA256,
    )

    document_text, source_metadata = extract_manas01(archive_path)
    heading_count = document_text.count(EPIC_START_HEADING)
    if heading_count != 1:
        raise RuntimeError(
            f"Expected one epic start heading, found {heading_count}: {EPIC_START_HEADING!r}"
        )
    epic_start = document_text.index(EPIC_START_HEADING)
    epic_text = document_text[epic_start:]

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    document_ids = tokenizer.encode(document_text, add_special_tokens=False).ids
    if tokenizer.decode(document_ids) != document_text:
        raise RuntimeError("Pinned tokenizer failed full-document round-trip")
    all_ids = tokenizer.encode(epic_text, add_special_tokens=False).ids
    if tokenizer.decode(all_ids) != epic_text:
        raise RuntimeError("Pinned tokenizer failed epic-text round-trip")

    selected_ids = all_ids if config.max_tokens == 0 else all_ids[: config.max_tokens]
    if config.max_tokens and len(all_ids) < config.max_tokens:
        raise ValueError(f"Requested {config.max_tokens} tokens, only {len(all_ids)} exist")

    train_end = int(len(selected_ids) * config.train_fraction)
    validation_end = train_end + int(len(selected_ids) * config.validation_fraction)
    train_ids = selected_ids[:train_end]
    validation_ids = selected_ids[train_end:validation_end]
    test_ids = selected_ids[validation_end:]
    if min(len(train_ids), len(validation_ids)) < 2:
        raise RuntimeError("Train and validation splits both need at least two tokens")

    output_dir = dataset_dir(config.dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "epic.txt").write_text(epic_text, encoding="utf-8")
    _write_uint16(output_dir / "train.bin", train_ids)
    _write_uint16(output_dir / "validation.bin", validation_ids)
    _write_uint16(output_dir / "test.bin", test_ids)

    split_bytes = {
        "train": len(tokenizer.decode(train_ids).encode("utf-8")),
        "validation": len(tokenizer.decode(validation_ids).encode("utf-8")),
        "test": len(tokenizer.decode(test_ids).encode("utf-8")) if test_ids else 0,
    }
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "dataset": config.dataset,
        "source": {
            "url": MANAS_URL,
            "sha1": MANAS_SHA1,
            "member": MANAS_MEMBER,
            "document_id": MANAS_DOCUMENT_ID,
            "metadata": source_metadata,
        },
        "tokenizer": {
            "url": TOKENIZER_URL,
            "commit": TOKENIZER_COMMIT,
            "sha256": TOKENIZER_SHA256,
            "vocab_size": tokenizer.get_vocab_size(),
            "tokenizers_version": tokenizers.__version__,
        },
        "full_document": {
            "characters": len(document_text),
            "utf8_bytes": len(document_text.encode("utf-8")),
            "tokens": len(document_ids),
        },
        "epic": {
            "start_heading": EPIC_START_HEADING,
            "excluded_leading_characters": epic_start,
            "characters": len(epic_text),
            "utf8_bytes": len(epic_text.encode("utf-8")),
            "tokens": len(all_ids),
            "distinct_token_ids": len(set(all_ids)),
        },
        "selection": {
            "max_tokens": config.max_tokens,
            "selected_tokens": len(selected_ids),
            "train_fraction": config.train_fraction,
            "validation_fraction": config.validation_fraction,
            "test_fraction": config.test_fraction,
        },
        "splits": {
            "train": {"tokens": len(train_ids), "decoded_utf8_bytes": split_bytes["train"]},
            "validation": {
                "tokens": len(validation_ids),
                "decoded_utf8_bytes": split_bytes["validation"],
            },
            "test": {"tokens": len(test_ids), "decoded_utf8_bytes": split_bytes["test"]},
        },
        "storage": {"dtype": "uint16", "byte_order": "little"},
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def load_metadata(name: str) -> dict[str, Any]:
    path = dataset_dir(name) / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Prepared dataset is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_split(name: str, split: str) -> torch.Tensor:
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown split: {split}")
    path = dataset_dir(name) / f"{split}.bin"
    if not path.exists():
        raise FileNotFoundError(f"Prepared split is missing: {path}")
    return torch.tensor(_read_uint16(path), dtype=torch.long)


def load_tokenizer(path: str | Path | None = None) -> Tokenizer:
    path = Path(path) if path is not None else TOKENIZER_DIR / "kyrgyz-byte-bpe-v1.json"
    if not path.exists():
        raise FileNotFoundError("Tokenizer is missing. Run `manas-gpt prepare` first.")
    actual = file_digest(path, "sha256")
    if actual != TOKENIZER_SHA256:
        raise RuntimeError(f"Tokenizer hash mismatch: expected {TOKENIZER_SHA256}, got {actual}")
    return Tokenizer.from_file(str(path))


class RandomWindowSampler:
    def __init__(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        block_size: int,
        seed: int,
        fixed: bool = False,
    ) -> None:
        if tokens.ndim != 1:
            raise ValueError("Token split must be one-dimensional")
        if len(tokens) <= block_size:
            raise ValueError(
                f"Split has {len(tokens)} tokens but block_size is {block_size}; need more tokens"
            )
        self.tokens = tokens
        self.batch_size = batch_size
        self.block_size = block_size
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.fixed = fixed
        self._fixed_batch: tuple[torch.Tensor, torch.Tensor] | None = None

    def next(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self.fixed and self._fixed_batch is not None:
            x, y = self._fixed_batch
        else:
            starts = torch.randint(
                0,
                len(self.tokens) - self.block_size,
                (self.batch_size,),
                generator=self.generator,
            )
            x = torch.stack([self.tokens[i : i + self.block_size] for i in starts])
            y = torch.stack([self.tokens[i + 1 : i + self.block_size + 1] for i in starts])
            if self.fixed:
                self._fixed_batch = (x, y)
        return x.to(device, non_blocking=False), y.to(device, non_blocking=False)
