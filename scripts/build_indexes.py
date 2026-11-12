#!/usr/bin/env python3
"""Build the year-tagged guideline indexes from the Markdown corpus (run from the repository root).

  python scripts/build_indexes.py --guidelines_dir data/guidelines_md --index_dir data/guidelines_indexes

Each source family lives in its own sub-directory (ada, ada_supplements, kdigo, esc, acc_aha).
Every passage keeps its source, edition year and section; retrieval later enforces year <= assigned year.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from diamas_utils.ada_guideline_processor import build_all_indexes  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--guidelines_dir", required=True)
ap.add_argument("--index_dir", required=True)
ap.add_argument("--sources", default="ada,ada_supplements,kdigo,esc,acc_aha")
ap.add_argument("--force", action="store_true")
a = ap.parse_args()
reg = build_all_indexes(Path(a.guidelines_dir), Path(a.index_dir), sources=a.sources.split(","), force=a.force)
print("indexed sources:", reg.active_sources)
