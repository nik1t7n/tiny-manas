"""Real-checkpoint A/B experiment; outputs stay local, not a synthetic test suite."""
from pathlib import Path
import hashlib
import json
import statistics
import subprocess
import time

import torch
from torch.nn import functional as F

from manas_gpt.data import load_split, load_tokenizer
from manas_gpt.experiment import environment_info, load_checkpoint, require_mps, seed_everything

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/optimization-01-last-position"
CHECKPOINT = ROOT / "runs/frozen-baseline-20260904/artifacts/tiny-manas-27m.pt"


@torch.no_grad()
def main():
    OUT.mkdir(exist_ok=False)
    device = require_mps()
    seed_everything(1337)
    model, _ = load_checkpoint(CHECKPOINT, device)
    model.eval()
    tokenizer = load_tokenizer()
    tokens = load_split("manas01-full", "validation")
    result = {"environment": environment_info(device), "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
              "working_diff": subprocess.check_output(["git", "diff", "--", "src/manas_gpt/model.py"], cwd=ROOT, text=True),
              "shapes": [], "generations": [], "timing": "synchronized wall time, alternating order, 5 warmups and 30 measured calls per mode"}
    for batch, length in ((1, 1), (1, 64), (1, 256), (8, 64)):
        x = torch.stack([tokens[i * 300:i * 300 + length] for i in range(batch)]).to(device)
        full = model(x)[0]
        last = model(x, last_position_only=True)[0]
        delta = (full[:, -1:, :] - last).abs().max().item()
        torch.testing.assert_close(full[:, -1:, :], last, atol=1e-4, rtol=1e-4)
        entry = {"batch": batch, "length": length, "max_abs_difference": delta,
                 "full_output_bytes": full.numel() * full.element_size(),
                 "last_output_bytes": last.numel() * last.element_size()}
        del full, last
        times = {False: [], True: []}
        for iteration in range(35):
            for flag in ((False, True) if iteration % 2 == 0 else (True, False)):
                torch.mps.synchronize()
                start = time.perf_counter()
                output = model(x, last_position_only=flag)[0]
                torch.mps.synchronize()
                elapsed = time.perf_counter() - start
                if iteration >= 5:
                    times[flag].append(elapsed)
                del output
        for flag, name in ((False, "full"), (True, "last")):
            entry[name + "_seconds"] = times[flag]
            entry[name + "_median_ms"] = 1000 * statistics.median(times[flag])
        entry["speedup"] = entry["full_median_ms"] / entry["last_median_ms"]
        result["shapes"].append(entry)
        print(json.dumps({k: v for k, v in entry.items() if not k.endswith("_seconds")}), flush=True)

    # Identical real prefixes, repeatedly check all logits and greedy continuations.
    for index in range(20):
        start = index * 400
        ids = tokens[start:start + 64][None].to(device)
        prefix = ids.clone()
        for _ in range(24):
            context = ids[:, -model.config.block_size:]
            full = model(context)[0][:, -1:, :]
            last = model(context, last_position_only=True)[0]
            torch.testing.assert_close(full, last, atol=1e-4, rtol=1e-4)
            if not torch.equal(full.argmax(-1), last.argmax(-1)):
                raise AssertionError("Greedy continuation changed")
            ids = torch.cat((ids, last[:, -1].argmax(-1, keepdim=True)), dim=1)
        result["generations"].append({"prompt": tokenizer.decode(prefix[0].tolist()), "continuation": tokenizer.decode(ids[0, 64:].tolist())})

    # Exercise the actual stochastic generate method, including the old crop path.
    prompt = tokens[:250][None].to(device)
    seed_everything(77)
    actual = model.generate(prompt.clone(), 16, 0.8, 40)
    seed_everything(77)
    expected = prompt.clone()
    for _ in range(16):
        logits = model(expected[:, -256:])[0][:, -1] / 0.8
        threshold = logits.topk(40).values[:, [-1]]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
        expected = torch.cat((expected, torch.multinomial(F.softmax(logits, -1), 1)), 1)
    if not torch.equal(actual, expected):
        raise AssertionError("Seeded generation/cropping changed")
    result["seeded_generate_equal"] = True
    # The default loss remains a full-position next-token loss.
    x, y = tokens[:64][None].to(device), tokens[1:65][None].to(device)
    logits, loss = model(x, y)
    torch.testing.assert_close(loss, F.cross_entropy(logits.flatten(0, 1), y.flatten()))
    result["full_position_loss"] = loss.item()
    result["status"] = "passed"
    (OUT / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"PASS: {OUT / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
