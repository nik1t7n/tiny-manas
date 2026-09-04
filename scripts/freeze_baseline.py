"""Preserve the pre-optimization checkout and real local research artifacts once."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

from manas_gpt.experiment import environment_info, require_mps

ROOT = Path(__file__).resolve().parents[1]
REVISION = "4ad408ecb327832ea855f392d35ed123638cbd87"
DEST = ROOT / "runs/frozen-baseline-20260904"


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    if DEST.exists():
        raise FileExistsError(f"Refusing to overwrite baseline: {DEST}")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if revision != REVISION:
        raise RuntimeError("Freeze must run at the explicitly selected original revision")
    inputs = [ROOT / "artifacts/tiny-manas-27m.pt"]
    for folder in ("data/processed/manas01-full", "data/tokenizer", "runs/manas01-27m-20260831T160739Z"):
        inputs.extend(p for p in (ROOT / folder).rglob("*") if p.is_file())
    DEST.mkdir()
    archive = DEST / "source.tar"
    subprocess.run(["git", "archive", "--format=tar", f"--output={archive}", REVISION], cwd=ROOT, check=True)
    entries = {}
    for original in sorted(inputs):
        relative = original.relative_to(ROOT)
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        expected = digest(original)
        if digest(target) != expected:
            raise RuntimeError(f"Snapshot verification failed: {relative}")
        target.chmod(0o444)
        entries[str(relative)] = {"sha256": expected, "bytes": target.stat().st_size}
    manifest = {
        "source_revision": REVISION,
        "snapshot": str(DEST),
        "source_archive_sha256": digest(archive),
        "environment": environment_info(require_mps()),
        "files": entries,
        "policy": "Read-only copies; never use this directory as experiment output. Source archive includes original README, configs, docs and implementation. No corpus/checkpoint content committed.",
    }
    payload = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    (DEST / "manifest.json").write_text(payload)
    evidence = ROOT / "docs/experiments/00-baseline-manifest.json"
    if evidence.exists():
        raise FileExistsError(evidence)
    evidence.write_text(payload)
    archive.chmod(0o444)
    (DEST / "manifest.json").chmod(0o444)
    print(json.dumps({"snapshot": str(DEST), "files": len(entries), "revision": REVISION}))


if __name__ == "__main__":
    main()
