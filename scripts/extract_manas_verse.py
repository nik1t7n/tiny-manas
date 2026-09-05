"""Coordinate-based verse extraction, with manually frozen edition boundaries.

This does not correct OCR spelling. Source text and review samples stay in runs/.
"""
import argparse
from collections import Counter
from pathlib import Path
import re
import statistics
import subprocess
import xml.etree.ElementTree as ET

from followup_common import ROOT, save_json, sha

SPECS = {
    "orozbakov-2": (17, 380, "train"),
    "orozbakov-3": (10, 299, "train"),
    "orozbakov-4": (16, 332, "validation"),
    "orozbakov-5": (18, 561, "train"),
    "mamay": (29, 1040, "test"),
}
NS = {"p": "http://www.w3.org/1999/xhtml"}
CYR = re.compile(r"[А-Яа-яЁёӨөҮүҢң]")


def box(w, key):
    return float(w.attrib[key])


def extract_page(page, body_height, name, number):
    lines = []
    dropped = Counter()
    for line in page.findall(".//p:line", NS):
        words = []
        for w in line.findall("p:word", NS):
            s = w.text or ""
            h = box(w, "yMax") - box(w, "yMin")
            # Editorial footnotes and superscripts use smaller type than verse.
            if h < body_height * .86:
                dropped["small_type_words"] += 1
                continue
            if "bizdin" in s.lower() or re.fullmatch(r"[\d*•]+", s):
                dropped["marks_counters_site"] += 1
                continue
            words.append(w)
        if not words:
            continue
        if box(line, "yMin") < float(page.attrib["height"]) * .04:
            continue
        parts = []
        previous = None
        for w in words:
            s = w.text or ""
            if previous is not None:
                gap = box(w, "xMin") - box(previous, "xMax")
                if gap > body_height * .25 and not re.match(r"^[,.;:!?»)]", s):
                    parts.append(" ")
            parts.append(s)
            previous = w
        s = "".join(parts).replace("*", "").replace("®", "")
        s = re.sub(r"\d+", "", s)  # edition line counts and residual footnote marks
        s = re.sub(r"\s+", " ", s).strip()
        if not CYR.search(s):
            continue
        if s.upper().replace(" ", "") in {"МАНАС", "ТЕКСТТЕР"}:
            dropped["running_title"] += 1
            continue
        if name == "mamay" and number == 1040 and ("Басмага" in s or "АСАНБАЙ" in s):
            continue
        lines.append((box(line, "yMin"), box(line, "xMin"), s))
    # All admitted editions use a single verse column; multi-column volumes excluded.
    lines.sort()
    return [s for _, _, s in lines], dict(dropped)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--directory", type=Path, default=ROOT / "runs/data-evaluation-20260905/source-review")
    p.add_argument("--book", choices=list(SPECS))
    args = p.parse_args()
    records = []
    for name, (start, stop, split) in SPECS.items():
        if args.book and args.book != name:
            continue
        pdf = args.directory / f"{name}.pdf"
        xml = args.directory / f"{name}.verse.xml"
        if not xml.exists():
            subprocess.run(["pdftotext", "-f", str(start), "-l", str(stop), "-bbox-layout", str(pdf), str(xml)], check=True, capture_output=True)
        pages = ET.parse(xml).getroot().findall(".//p:page", NS)
        heights = Counter(round(box(w, "yMax") - box(w, "yMin"), 1)
                          for page in pages for w in page.findall(".//p:word", NS)
                          if CYR.search(w.text or "") and len(w.text or "") > 2)
        body_height = heights.most_common(1)[0][0]
        output = []
        pagestats = []
        for number, page in enumerate(pages, start):
            lines, dropped = extract_page(page, body_height, name, number)
            output.extend(lines)
            pagestats.append({"page": number, "lines": len(lines), "dropped": dropped,
                              "first": lines[:2], "last": lines[-2:]})
        clean = args.directory / f"{name}.verse.txt"
        clean.write_text("\n".join(output) + "\n")
        record = {"id": name, "split": split, "pdf_pages_inclusive": [start, stop],
                  "pdf_sha256": sha(pdf), "text_sha256": sha(clean),
                  "body_height": body_height, "height_histogram": heights.most_common(8),
                  "lines": len(output), "characters": clean.stat().st_size,
                  "replacement_characters": sum(s.count("\ufffd") for s in output),
                  "single_letter_words": sum(len(re.findall(r"\b[А-Яа-яЁёӨөҮүҢң]\b", s)) for s in output),
                  "page_stats": pagestats}
        save_json(args.directory / f"{name}.extraction.json", record)
        print(name, split, body_height, len(output), heights.most_common(3), flush=True)
        records.append(record)
    if not args.book:
        save_json(args.directory / "extraction.json", records)


if __name__ == "__main__":
    main()
