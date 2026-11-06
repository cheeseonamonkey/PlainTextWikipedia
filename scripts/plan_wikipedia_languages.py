#!/usr/bin/env python3
"""Plan a manual Wikipedia rebuild matrix from one language catalog."""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOUNDATION = ("simplewiki", "enwiki", "nowiki", "eswiki", "frwiki",
              "dewiki", "itwiki", "ptwiki", "ruwiki", "zhwiki")


def languages(selection: str = "") -> list[str]:
    if selection.strip():
        selected = [entry.strip().lower() for entry in selection.split(",")]
        if any(not entry for entry in selected):
            raise ValueError("Empty language selection")
        projects = [entry if entry.endswith("wiki") else entry + "wiki" for entry in selected]
    else:
        catalog = json.loads((ROOT / "config/additional-languages.json").read_text())
        projects = [*FOUNDATION, *(entry["project"] for entry in catalog["languages"])]
    if any(not re.fullmatch(r"[a-z][a-z0-9_-]*wiki", project) for project in projects):
        raise ValueError("Language selection must contain Wikimedia project codes")
    result = list(dict.fromkeys(projects))
    if not 1 <= len(result) <= 256:
        raise ValueError("GitHub Actions supports at most 256 matrix jobs")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", default="", help="Comma-separated projects; blank = all cataloged")
    args = parser.parse_args()
    result = languages(args.languages)
    output = json.dumps({"include": [{"language": project} for project in result]},
                        separators=(",", ":"))
    if os.getenv("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
            print("matrix=" + output, file=stream)
    print(f"Planned {len(result)} Wikimedia editions: {', '.join(result[:12])}" +
          (" ..." if len(result) > 12 else ""))
    if not os.getenv("GITHUB_OUTPUT"):
        print(output)


if __name__ == "__main__":
    main()
