"""O09: real accepted-checkpoint cache parity and paired inference measurements."""
import argparse
import json
from pathlib import Path
import statistics
import time
import traceback
from functools import partial

import torch

from architecture_candidates import restore_model
from comparison_windows import ComparisonData
from experiment_tokenizers import BUNDLE, MANIFEST_SHA, ROOT, save_json, sha
from kv_cache_candidate import CacheSession, cached_generate
from manas_gpt.experiment import environment_info, load_checkpoint, require_mps


def load_reference(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "architecture_changes" in payload.get("protocol", {}):
        model = restore_model(payload).to(device)
    else:
        del payload
        model, payload = load_checkpoint(path, device)
    model.eval()
    return model


def timed(function, memory=None):
    torch.mps.synchronize()
    start = time.perf_counter()
    result = function()
    torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    if memory is not None:
        memory["allocated"] = max(memory["allocated"], torch.mps.current_allocated_memory())
        memory["driver"] = max(memory["driver"], torch.mps.driver_allocated_memory())
    return elapsed, result


def require_equal_logits(reference, candidate, label):
    error = (reference - candidate).abs().max().item()
    if not torch.allclose(reference, candidate, atol=1e-4, rtol=1e-4):
        raise AssertionError(f"{label}: cached logit mismatch, max absolute error {error}")
    if not torch.equal(reference.argmax(-1), candidate.argmax(-1)):
        raise AssertionError(f"{label}: greedy token changed")
    return error


@torch.no_grad()
def run(args, out, report):
    device = require_mps()
    torch.mps.set_per_process_memory_fraction(.65)
    if sha(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError("Checkpoint differs from the selected immutable reference")
    data = ComparisonData(BUNDLE, args.vocabulary, MANIFEST_SHA)
    model = load_reference(args.checkpoint, device)
    uncached_generate = partial(model.generate, use_cache=False)
    if model.config.vocab_size != args.vocabulary or model.config.block_size != 256:
        raise ValueError("Checkpoint vocabulary/context differs from the declared experiment")
    tokens = data.ids["validation"]
    report.update(environment=environment_info(device), parameters=model.parameter_count(),
                  tokenizer_sha256=sha(BUNDLE / str(args.vocabulary) / "tokenizer.json"),
                  precision="FP32 parameters, activations, logits and KV cache; dropout disabled")
    memory = {"allocated": 0, "driver": 0}
    for index in range(20):
        length = 32 if index % 2 == 0 else 248
        ids = tokens[index * 400:index * 400 + length][None].to(device)
        prefix = ids.clone()
        session = CacheSession(model)
        maximum_error, rebuilds = 0.0, 0
        for step in range(32):
            reference = model(ids[:, -256:], last_position_only=True)[0]
            if session.layers is None or session.length == 256:
                rebuilds += int(session.layers is not None)
                candidate = session.prefill(ids[:, -256:])
            else:
                candidate = session.decode(ids[:, -1:])
            maximum_error = max(maximum_error, require_equal_logits(reference, candidate, f"prompt {index}, step {step}"))
            ids = torch.cat((ids, reference[:, -1].argmax(-1, keepdim=True)), dim=1)
        report["parity"].append({"prompt_number": index + 1, "prompt_length": length,
                                  "new_tokens": 32, "max_logit_error": maximum_error, "rebuilds": rebuilds,
                                  "prompt": data.tokenizer.decode(prefix[0].tolist()),
                                  "continuation": data.tokenizer.decode(ids[0, length:].tolist())})
        save_json(out / "result.json", report)
    # Independent request state: a second prefill must not change the first cache.
    first, second = CacheSession(model), CacheSession(model)
    prefix = tokens[:64][None].to(device)
    first.prefill(prefix)
    second.prefill(tokens[900:932][None].to(device))
    report["request_isolation_error"] = require_equal_logits(
        model(tokens[:65][None].to(device), last_position_only=True)[0],
        first.decode(tokens[64:65][None].to(device)), "independent requests")
    first.prefill(tokens[:256][None].to(device))
    report["full_cache_storage"] = first.storage()
    expected_bytes = 2 * model.config.n_layer * 256 * model.config.n_embd * 4
    if (report["full_cache_storage"]["logical_bytes"] != expected_bytes
        or report["full_cache_storage"]["storage_bytes"] != expected_bytes):
        raise AssertionError("Full FP32 MHA cache does not own exactly the expected K/V storage")
    del first, second, session, reference, candidate

    # Exercise the actual sampled generation wrapper, also across cropping.
    for length in (32, 248):
        prefix = tokens[:length][None].to(device)
        expected = uncached_generate(prefix, 32, .8, 40, torch.Generator(device=device).manual_seed(6036))
        actual = cached_generate(model, prefix, 32, .8, 40, torch.Generator(device=device).manual_seed(6036))
        if not torch.equal(expected, actual):
            raise AssertionError(f"Actual seeded generation changed for prompt length {length}")
    report["seeded_generation_equal"] = True

    for length in (32, 248):
        prefix = tokens[:length][None].to(device)
        durations = {"uncached": [], "cached": []}
        prefill = {"uncached": [], "cached": []}
        for repetition in range(7):
            order = ("uncached", "cached") if repetition % 2 == 0 else ("cached", "uncached")
            outputs = {}
            for mode in order:
                generator = torch.Generator(device=device).manual_seed(3407)
                function = uncached_generate if mode == "uncached" else lambda *a: cached_generate(model, *a)
                elapsed, outputs[mode] = timed(lambda: function(prefix, 64, .8, 40, generator), memory)
                session = CacheSession(model)
                prefill_function = (lambda: model(prefix, last_position_only=True)[0]) if mode == "uncached" else lambda: session.prefill(prefix)
                first_seconds, logits = timed(prefill_function, memory)
                del logits, session
                if repetition >= 2:
                    durations[mode].append(elapsed)
                    prefill[mode].append(first_seconds)
            if not torch.equal(outputs["uncached"], outputs["cached"]):
                raise AssertionError("Timed complete generations differ")
        medians = {mode: statistics.median(values) for mode, values in durations.items()}
        report["timings"].append({"prompt_length": length, "new_tokens": 64,
                                   "seconds": durations, "median_seconds": medians,
                                   "speedup": medians["uncached"] / medians["cached"],
                                   "prefill_seconds": prefill,
                                   "prefill_median_seconds": {k: statistics.median(v) for k, v in prefill.items()}})
        save_json(out / "result.json", report)
    decode_times, full_times = [], []
    for repetition in range(12):
        session = CacheSession(model)
        session.prefill(tokens[:128][None].to(device))
        calls = {"decode": lambda: session.decode(tokens[128:129][None].to(device)),
                 "full": lambda: model(tokens[:129][None].to(device), last_position_only=True)[0]}
        samples = {}
        for mode in (("decode", "full") if repetition % 2 else ("full", "decode")):
            samples[mode], _ = timed(calls[mode], memory)
        if repetition >= 2:
            decode_times.append(samples["decode"])
            full_times.append(samples["full"])
    report["single_token_128_past"] = {"decode_seconds": decode_times, "uncached_seconds": full_times,
                                        "decode_median_seconds": statistics.median(decode_times),
                                        "uncached_median_seconds": statistics.median(full_times)}
    report["maximum_memory_sampled_after_timed_calls"] = memory
    report["status"] = "passed_not_promoted" if report["timings"][0]["speedup"] >= 1.10 else "speed_gate_failed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--vocabulary", type=int, choices=(32768, 16384, 8192), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "checkpoint": str(args.checkpoint.resolve()),
              "checkpoint_sha256": args.checkpoint_sha256, "vocabulary": args.vocabulary,
              "manifest_sha256": MANIFEST_SHA, "runner_sha256": sha(__file__),
              "cache_source_sha256": sha(Path(__file__).with_name("kv_cache_candidate.py")),
              "model_source_sha256": sha(ROOT / "src/manas_gpt/model.py"),
              "parity": [], "timings": [], "new_training_updates": 0,
              "measurement": "synchronized FP32, alternating paired order; two warmup generation pairs plus five measured"}
    try:
        run(args, out, report)
    except Exception:
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
        save_json(out / "result.json", report)
        raise
    save_json(out / "result.json", report)
    print(json.dumps({key: value for key, value in report.items() if key not in ("parity", "timings")}), flush=True)


if __name__ == "__main__":
    main()
