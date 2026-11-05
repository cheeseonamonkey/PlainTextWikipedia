"""Lightweight article links for the existing Wikipedia dataset catalog.

Links are reconstructed from titles, not canonical page IDs; pages can move.
Neither this module nor the exporter performs network requests per article.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
SIMPLE = "ffatty/plain-text-wikipedia-simpleenglish"


def wiki_code(repo: str) -> str:
    """Return Wikimedia's wiki code, refusing unknown repo-to-project mappings."""
    if repo == SIMPLE:
        return "simple"
    if not re.fullmatch(r"ffatty/plaintext-wikipedia-full-[a-z0-9-]+", repo):
        raise ValueError(f"Unsupported Wikipedia dataset: {repo}")

    for item in json.loads((ROOT / "config/additional-languages.json")
                           .read_text(encoding="utf-8"))["languages"]:
        if repo == f"ffatty/plaintext-wikipedia-full-{item['slug']}":
            project = item["project"]
            if not re.fullmatch(r"[a-z0-9_-]+wiki", project):
                raise ValueError(f"Invalid Wikimedia project: {project}")
            return project.removesuffix("wiki")

    # Original major-edition datasets predate the expanded catalog.
    candidates = []
    for line in (ROOT / "config/datasets.env").read_text(encoding="utf-8").splitlines():
        matched = re.fullmatch(r"FULL_HF_REPO_([A-Z_]+)=(ffatty/[^\s#]+)", line.strip())
        if matched and matched.group(2) == repo:
            candidates.append(matched.group(1).lower())
    if candidates:
        # Norwegian has both NO and NOB compatibility aliases in datasets.env.
        return min(candidates, key=lambda code: (len(code), code))
    raise ValueError(f"No Wikimedia project mapping for {repo}")


def article_url(title: str, code: str) -> str:
    """Construct a best-effort page URL. Titles are not canonical identifiers."""
    if not re.fullmatch(r"[a-z0-9_-]+", code):
        raise ValueError(f"Invalid Wikimedia code: {code}")
    domain = code.replace("_", "-")
    return f"https://{domain}.wikipedia.org/wiki/{quote(title.replace(' ', '_'), safe='/')}"
