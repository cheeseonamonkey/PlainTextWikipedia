"""Keep the canonical enriched Parquet the default HF dataset config."""
from __future__ import annotations

import yaml

CANONICAL_NOTE = "<!-- canonical-article-features-v6 -->"


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
        body += ("\n## Article schema\n\n" + CANONICAL_NOTE + "\n"
                 "One row per article. Independent Bekko and fastText embeddings, full word counts, "
                 "TF-IDF and BM25 dictionaries, and compact document metrics. "
                 "Corpus statistics and full word TF/DF/IDF are in `features.zip`; "
                 "the fastText model is in `models.zip`.\n")
    return "---\n" + yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True) + "---\n" + body
