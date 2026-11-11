"""Lightweight article links for the existing Wikipedia dataset catalog.

Links are reconstructed from titles, not canonical page IDs; pages can move.
Neither this module nor the exporter performs network requests per article.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
SIMPLE = "ffatty/plain-text-wikipedia-simpleenglish"


@lru_cache(maxsize=1)
def repository_codes():
    mapping = {SIMPLE: "simple"}
    catalog = json.loads((ROOT / "config/additional-languages.json").read_text(encoding="utf-8"))
    for item in catalog["languages"]:
        if not re.fullmatch(r"[a-z0-9_-]+wiki", item["project"]):
            raise ValueError(f"Invalid Wikimedia project: {item['project']}")
        mapping[f"ffatty/plaintext-wikipedia-full-{item['slug']}"] = item["project"].removesuffix("wiki")
    aliases = {}
    for line in (ROOT / "config/datasets.env").read_text(encoding="utf-8").splitlines():
        matched = re.fullmatch(r"FULL_HF_REPO_([A-Z_]+)=(ffatty/[^\s#]+)", line.strip())
        if matched:
            aliases.setdefault(matched[2], []).append(matched[1].lower())
    for repo, codes in aliases.items():
        mapping.setdefault(repo, min(codes, key=lambda code: (len(code), code)))
    return mapping


def wiki_code(repo: str) -> str:
    try:
        return repository_codes()[repo]
    except KeyError:
        raise ValueError(f"No Wikimedia project mapping for {repo}") from None


def project_repo(project: str) -> str:
    if not re.fullmatch(r"[a-z0-9_-]+wiki", project):
        raise ValueError(f"Invalid Wikimedia project: {project}")
    matches = [repo for repo, code in repository_codes().items() if code == project.removesuffix("wiki")]
    if len(matches) != 1:
        raise ValueError(f"No unique existing dataset for {project}")
    return matches[0]


def resolve_language(prefix: str) -> tuple[str, str, str]:
    """Resolve a short language prefix (en, zh, es, simple, enwiki) to
    (repo, project, code). Accepts both bare prefixes and full project codes.
    """
    prefix_lower = prefix.strip().lower()
    if not prefix_lower:
        raise ValueError("Empty language prefix")
    if prefix_lower == "simple":
        return SIMPLE, "simplewiki", "simple"
    if prefix_lower.endswith("wiki"):
        project = prefix_lower
    else:
        project = prefix_lower + "wiki"
    if not re.fullmatch(r"[a-z0-9_-]+wiki", project):
        raise ValueError(f"Invalid Wikimedia project: {project}")
    repo = project_repo(project)
    code = wiki_code(repo)
    return repo, project, code


def article_url(title: str, code: str) -> str:
    """Construct a best-effort page URL. Titles are not canonical identifiers."""
    if not re.fullmatch(r"[a-z0-9_-]+", code):
        raise ValueError(f"Invalid Wikimedia code: {code}")
    domain = code.replace("_", "-")
    return f"https://{domain}.wikipedia.org/wiki/{quote(title.replace(' ', '_'), safe='/')}"
