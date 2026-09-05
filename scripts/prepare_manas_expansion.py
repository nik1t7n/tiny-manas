"""Fetch and inspect owner-authorized editions before admitting training text."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import quote, urljoin
from urllib.request import urlopen, Request

from followup_common import ROOT, save_json, sha

SLUGS = {
    "orozbakov-1": "epos-manas-akademicheskoe-izdanie-po-variantu-sagymbaya-orozbakova-pervaya-kniga",
    "orozbakov-2": "epos-manas-akademicheskoe-izdanie-po-variantu-sagymbaya-orozbakova-vtoraya-kniga",
    "orozbakov-3": "epos-manas-akademicheskoe-izdanie-po-variantu-sagymbaya-orozbakova-tretya-kniga",
    "orozbakov-4": "epos-manas-akademicheskoe-izdanie-po-variantu-sagymbaya-orozbakova-chetvertaya-kniga",
    "orozbakov-5": "epos-manas-akademicheskoe-izdanie-po-variantu-sagymbaya-orozbakova-pyataya-kniga",
    "orozbakov-6-7": "epos-manas-akademicheskoe-izdanie-po-variantu-sagymbaya-orozbakova-shestaya-sedmaya-kniga",
    "orozbakov-8-9": "epos-manas-akademicheskoe-izdanie-po-variantu-sagymbaya-orozbakova-vosmaya-devyataya-kniga",
    "mamay": "epos-manas-variant-zhusupa-mamaya",
}


def fetch_one(item, directory):
    name, slug = item
    page_url = "https://new.bizdin.kg/kniga/" + slug
    page_path = directory / f"{name}.html"
    if not page_path.exists():
        with urlopen(Request(page_url, headers={"User-Agent": "TinyManasResearch/0.1"}), timeout=45) as r:
            page_path.write_bytes(r.read())
    hrefs = sorted(set(re.findall(r'href="([^"]+\.pdf)"', page_path.read_text())))
    if len(hrefs) != 1:
        raise RuntimeError(f"Ambiguous source PDF for {name}: {hrefs}")
    url = quote(urljoin(page_url, hrefs[0]), safe=":/?=&%")
    path = directory / f"{name}.pdf"
    prior = directory / "orozbakov-1995-book1.pdf"
    if name == "orozbakov-1" and prior.exists() and not path.exists():
        shutil.copyfile(prior, path)
    if not path.exists():
        temp = path.with_suffix(".pdf.part")
        with urlopen(Request(url, headers={"User-Agent": "TinyManasResearch/0.1"}), timeout=90) as r, temp.open("wb") as f:
            while b := r.read(1 << 20):
                f.write(b)
        temp.replace(path)
    if not path.read_bytes()[:5] == b"%PDF-":
        raise RuntimeError(f"Not a PDF: {path}")
    text = directory / f"{name}.txt"
    if not text.exists():
        subprocess.run(["pdftotext", "-layout", str(path), str(text)], check=True, capture_output=True)
    pages = text.read_text().split("\f")
    if not pages[-1].strip():
        pages.pop()
    stats = [{"pdf_page": i + 1, "characters": len(p), "lines": len(p.splitlines()),
              "replacement_characters": p.count("\ufffd"),
              "cyrillic_letters": len(re.findall(r"[А-Яа-яЁёӨөҮүҢң]", p))}
             for i, p in enumerate(pages)]
    record = {"id": name, "page_url": page_url, "pdf_url": url,
              "pdf_sha256": sha(path), "pdf_bytes": path.stat().st_size,
              "text_sha256": sha(text), "pages": len(pages), "page_stats": stats,
              "authorization": "Owner confirmed research access/permission on 2026-09-05; not a verified blanket CC license",
              "status": "downloaded_extracted_pending_manual_boundaries_and_quality"}
    save_json(directory / f"{name}.source.json", record)
    print(f"SOURCE {name} pages={len(pages)} bytes={path.stat().st_size} chars={sum(len(p) for p in pages)}", flush=True)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--directory", type=Path, default=ROOT / "runs/data-evaluation-20260905/source-review")
    args = p.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(lambda item: fetch_one(item, args.directory), SLUGS.items()))
    save_json(args.directory / "sources.json", rows)


if __name__ == "__main__":
    main()
