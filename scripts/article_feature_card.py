"""Keep the canonical enriched Parquet the default HF dataset config."""
from __future__ import annotations

import yaml

CANONICAL_NOTE = "<!-- canonical-article-features-v5 -->"


def update_card(content: str) -> str:
    if not content.startswith("---\n"):
        raise ValueError("Hugging Face README is missing YAML front matter")
    sections = content.split("---\n", 2)
    if len(sections) != 3:
        raise ValueError("Malformed Hugging Face README YAML metadata")
    metadata = yaml.safe_load(sections[1])
    if not isinstance(metadata, dict):
        raise ValueError("Invalid Hugging Face README metadata")
    canonical = [{"config_name": "default", "default": True,
                  "data_files": [{"split": "train", "path": "articles.parquet"}]}]
    # Preserve the original metadata, provenance and article description.
    if metadata.get("configs") == canonical and CANONICAL_NOTE in sections[2]:
        return content
    metadata["configs"] = canonical
    body = sections[2]
    if CANONICAL_NOTE not in body:
        body += ("\n## Canonical article table\n\n"
                 f"{CANONICAL_NOTE}\n"
                 "The default `train` split loads **`articles.parquet`**, one row per "
                 "article with original text, full lexical/semantic features, "
                 "sentiment/readability diagnostics where supported and a 200D "
                 "combined embedding. Source gzip and structured shards remain "
                 "available under `data/` and `structured/`, but are not silently "
                 "mixed into the default analytical dataset. Corpus-level "
                 "vocabulary, descriptive statistics, model and searchable SQLite "
                 "sidecar are under `features/`.\n"
                 "The feature schema and known limitations are documented in "
                 "[the repository](https://github.com/cheeseonamonkey/PlainTextWikipedia/blob/main/docs/article-features.md).\n")
    return "---\n" + yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True) + "---\n" + body
