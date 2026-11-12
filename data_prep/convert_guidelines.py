#!/usr/bin/env python3
"""Convert guideline PDFs to Markdown with MinerU, keeping one sub-directory per source family.

  python data_prep/convert_guidelines.py --input guideline_pdfs --output data/guidelines_md

<input>/<source>/<name>.pdf  ->  <output>/<source>/<name>.md   (MinerU work files under <output>/_mineru/)

The indexes read the edition year from the file name: ADA editions as ada/ada_standards_<year>.md, other
documents as <source>/<name>_<year>.md (last four-digit group). Source families: ada, ada_supplements,
kdigo, esc, acc_aha. The study used `mineru -m ocr -b vlm-transformers` (MinerU 2).
Already converted PDFs are skipped unless --force is given.
"""
import argparse
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def convert(pdf, src_root, out_root, model_source, force):
    rel = pdf.relative_to(src_root).with_suffix(".md")
    target = out_root / rel
    if target.exists() and not force:
        return f"skip  {rel}"
    work = out_root / "_mineru" / rel.with_suffix("")
    work.mkdir(parents=True, exist_ok=True)
    cmd = ["mineru", "-p", str(pdf), "-o", str(work), "-m", "ocr", "-b", "vlm-transformers", "--source", model_source]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=12000)
    if r.returncode:
        return f"FAIL  {rel}: {r.stderr[-500:]}"
    parts = sorted(work.glob("**/*.md"))
    if not parts:
        return f"FAIL  {rel}: no Markdown produced"
    target.parent.mkdir(parents=True, exist_ok=True)
    if len(parts) == 1:
        shutil.copy2(parts[0], target)
    else:
        target.write_text("\n".join(f"<!-- Source: {p.name} -->\n{p.read_text(encoding='utf-8')}\n\n---\n"
                                    for p in parts), encoding="utf-8")
    return f"ok    {rel}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--model_source", default="huggingface", choices=["huggingface", "modelscope", "local"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    pdfs = sorted(a.input.glob("**/*.pdf"))
    with ThreadPoolExecutor(a.workers) as ex:
        for msg in ex.map(lambda p: convert(p, a.input, a.output, a.model_source, a.force), pdfs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
