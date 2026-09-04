"""Exact-coverage real-text windows for the preregistered vocabulary experiment."""
import hashlib
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from manas_gpt.data import _read_uint16


class ComparisonData:
    def __init__(self, directory, vocabulary, manifest_sha256):
        self.directory = Path(directory).resolve(strict=True)
        manifest_path = self.directory / "manifest.json"
        if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != manifest_sha256:
            raise RuntimeError("Comparison manifest hash mismatch")
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest["status"] != "verified_prepared_not_trained":
            raise RuntimeError("Comparison data is not verified")
        self.variant = self.manifest["variants"][str(vocabulary)]
        variant_dir = self.directory / str(vocabulary)
        # Verify exactly the files this experiment consumes; never load test.
        for name in ("tokenizer.json", "train.bin", "validation.bin", "train.byte-lengths.bin", "validation.byte-lengths.bin"):
            path = variant_dir / name
            expected = self.manifest["files"][str(path.relative_to(self.directory))]
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Comparison artifact hash mismatch: {path}")
        self.tokenizer = Tokenizer.from_file(str(variant_dir / "tokenizer.json"))
        if self.tokenizer.get_vocab_size() != vocabulary:
            raise RuntimeError("Vocabulary size mismatch")
        self.ids = {}
        self.byte_lengths = {}
        for split in ("train", "validation"):
            self.ids[split] = torch.tensor(_read_uint16(variant_dir / f"{split}.bin"), dtype=torch.long)
            self.byte_lengths[split] = torch.tensor(_read_uint16(variant_dir / f"{split}.byte-lengths.bin"), dtype=torch.long)
            info = self.variant["splits"][split]
            if len(self.ids[split]) != info["tokens"] or len(self.byte_lengths[split]) != info["tokens"]:
                raise RuntimeError(f"Split length mismatch: {split}")
            if not 1 <= info["prefix_tokens"] < 256:
                raise ValueError("Scoring prefix must fit inside the initial context")
            if int(self.byte_lengths[split][info["prefix_tokens"]:].sum()) != info["scored_utf8_bytes"]:
                raise RuntimeError(f"Scored byte count mismatch: {split}")

    def _batch(self, split, spans):
        """A span is (input_start, scored_target_start, scored_target_end)."""
        if not 1 <= len(spans) <= 8:
            raise ValueError("Expected one to eight real windows")
        # Explicit right padding; ignored targets provide no training signal.
        x = torch.zeros((8, 256), dtype=torch.long)
        y = torch.full((8, 256), -100, dtype=torch.long)
        positions = torch.full((8, 256), -1, dtype=torch.long)
        targets = byte_count = 0
        for row, (start, scored_start, end) in enumerate(spans):
            length = end - 1 - start
            if not (0 <= start < scored_start < end <= len(self.ids[split]) and 1 <= length <= 256):
                raise ValueError(f"Invalid {split} span: {(start, scored_start, end)}")
            x[row, :length] = self.ids[split][start:end - 1]
            masked = scored_start - start - 1
            y[row, masked:length] = self.ids[split][scored_start:end]
            positions[row, masked:length] = torch.arange(scored_start, end)
            targets += end - scored_start
            byte_count += int(self.byte_lengths[split][scored_start:end].sum())
        return {"x": x, "y": y, "positions": positions, "targets": targets, "bytes": byte_count}

    def training_updates(self, epoch, seed=1338):
        info = self.variant["splits"]["train"]
        spans = [(start, max(start + 1, info["prefix_tokens"]), min(start + 257, info["tokens"]))
                 for start in range(0, info["tokens"] - 1, 256)]
        generator = torch.Generator(device="cpu").manual_seed(seed + epoch)
        order = torch.randperm(len(spans), generator=generator).tolist()
        for offset in range(0, len(order), 16):
            selected = [spans[index] for index in order[offset:offset + 16]]
            batches = [self._batch("train", selected[start:start + 8]) for start in range(0, len(selected), 8)]
            yield {"batches": batches, "targets": sum(b["targets"] for b in batches),
                   "bytes": sum(b["bytes"] for b in batches)}

    def validation_batches(self):
        info = self.variant["splits"]["validation"]
        spans = []
        for scored_start in range(info["prefix_tokens"], info["tokens"], 128):
            end = min(scored_start + 128, info["tokens"])
            spans.append((max(0, end - 257), scored_start, end))
        for start in range(0, len(spans), 8):
            yield self._batch("validation", spans[start:start + 8])

    def verify_coverage(self):
        """Validate the actual real-data windows, including padding and byte totals."""
        results = {}
        for split in ("train", "validation"):
            info = self.variant["splits"][split]
            seen = torch.zeros(info["tokens"], dtype=torch.long)
            batches = (batch for update in self.training_updates(0) for batch in update["batches"]) if split == "train" else self.validation_batches()
            total_bytes = total_targets = batch_count = 0
            for batch in batches:
                active = batch["y"] != -100
                positions = batch["positions"][active]
                if not torch.equal(batch["y"][active], self.ids[split][positions]):
                    raise AssertionError("Targets differ from the original token sequence")
                if positions.numel() != batch["targets"]:
                    raise AssertionError("Active target count mismatch")
                seen.index_add_(0, positions, torch.ones_like(positions))
                total_bytes += batch["bytes"]
                total_targets += batch["targets"]
                batch_count += 1
            if seen[:info["prefix_tokens"]].any() or not torch.all(seen[info["prefix_tokens"]:] == 1):
                raise AssertionError(f"Duplicate or missing scored positions: {split}")
            if total_bytes != info["scored_utf8_bytes"] or total_targets != info["scored_tokens"]:
                raise AssertionError(f"Wrong byte/target totals: {split}")
            results[split] = {"batches": batch_count, "scored_tokens": total_targets,
                              "scored_utf8_bytes": total_bytes, "all_targets_once": True}
        return results
