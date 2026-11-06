#!/usr/bin/env python3
"""Return the exact language name and slug for new Wikipedia dump projects."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def metadata(project: str) -> tuple[str, str]:
    path = Path(__file__).resolve().parents[1] / "config" / "additional-languages.json"
    entries = json.loads(path.read_text(encoding="utf-8"))["languages"]
    for entry in entries:
        if entry["project"] == project:
            return entry["name"], entry["slug"]
    # Support any Wikimedia Wikipedia project without maintaining a second
    # language-name registry. Known editions keep their curated names/slugs.
    # Closed projects must never be treated as active Wikipedia dumps.
    if project == "tlhwiki":
        raise ValueError("Closed Wikipedia project: tlhwiki")
    if re.fullmatch(r"[a-z][a-z0-9_-]*wiki", project):
        code = project.removesuffix("wiki")
        return code, code.replace("_", "-")
    raise ValueError(f"Invalid Wikipedia project: {project}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/language_info.py PROJECT")
    print("|".join(metadata(sys.argv[1])))
