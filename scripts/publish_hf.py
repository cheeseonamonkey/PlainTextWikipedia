#!/usr/bin/env python3
"""Publish the verified v8 bundle to Hugging Face.

One commit adds ``articles.parquet`` + ``corpus-profile.json`` +
``manifest.json``, updates the README card so the default config points at the
parquet, and preserves existing source and legacy export paths. The commit is verified by
re-downloading the uploaded files and comparing SHA256 before success is
claimed.
"""
from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import yaml
from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, hf_hub_download

from scripts.bundle_artifacts import file_metadata, verify_bundle
from scripts.wiki_metadata import resolve_language

CANONICAL_NOTE = "<!-- canonical-article-features-v8 -->"
BUNDLE_FILES = ("articles.parquet", "corpus-profile.json", "manifest.json")


def retry(label, callback, attempts=5):
    for attempt in range(attempts):
        try:
            return callback()
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            retryable = not isinstance(exc, (ValueError, TypeError)) and (
                status is None or status == 429 or status >= 500)
            if not retryable or attempt == attempts - 1:
                raise
            delay = min(120, 10 * 2 ** attempt)
            print(f"Retrying {label} in {delay}s: {type(exc).__name__}", flush=True)
            time.sleep(delay)


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
    if metadata.get("configs") == canonical and CANONICAL_NOTE in sections[2]:
        return content
    metadata["configs"] = canonical
    body = sections[2]
    if CANONICAL_NOTE not in body:
        body += ("\n## Article schema\n\n" + CANONICAL_NOTE + "\n"
                 "One row per article. Native 180-d fastText embedding, "
                 "per-article TF-IDF (stemmed, L2-normalized against the "
                 "language corpus), raw word counts, and grouped JSON "
                 "statistics. Language-level IDF/vocabulary context is in "
                 "`corpus-profile.json`.\n")
    return "---\n" + yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True) + "---\n" + body


def publish_hf(api: HfApi, repo: str, folder: Path, sources_dir: Path | None = None) -> None:
    manifest = verify_bundle(folder)
    api.create_repo(repo_id=repo, repo_type="dataset", exist_ok=True)
    existing = {f.path for f in api.list_repo_tree(repo, repo_type="dataset", recursive=True)}
    operations = [CommitOperationAdd(path_in_repo=name,
                                     path_or_fileobj=str(folder / name))
                  for name in BUNDLE_FILES]

    if sources_dir is not None:
        # Fresh rebuild: the parquet derives from these new shards, so the same
        # commit replaces data/ and drops any old shard the new set does not
        # contain. Without this, a rebuilt bundle would ship with stale sources.
        new_sources = sorted(p for p in sources_dir.iterdir()
                             if p.is_file() and (p.name.endswith(".txt.gz") or p.name == "build-manifest.json"))
        if not new_sources:
            raise ValueError(f"No source archives to upload from {sources_dir}")
        old_sources = {p for p in existing if re.fullmatch(r"data/[^/]+\.txt\.gz", p)}
        keep = {f"data/{p.name}" for p in new_sources if p.name != "build-manifest.json"}
        for path in sorted(old_sources - keep):
            operations.append(CommitOperationDelete(path_in_repo=path))
        for file in new_sources:
            operations.append(CommitOperationAdd(path_in_repo=f"data/{file.name}",
                                                 path_or_fileobj=str(file)))

    # Preserve existing exports; replacing a release must not erase unrelated data.

    if "README.md" in existing:
        readme = Path(retry("download README", lambda: hf_hub_download(
            repo, "README.md", repo_type="dataset", token=api.token)))
        card = update_card(readme.read_text(encoding="utf-8"))
    else:
        card = ("---\nlicense: cc-by-sa-4.0\nconfigs:\n"
                "- config_name: default\n  data_files:\n  - split: train\n"
                "    path: articles.parquet\ndefault: true\n---\n"
                "# Plaintext Wikipedia\n\nArticle-level Parquet dataset "
                "with fastText embeddings and lexical statistics.\n")
    operations.append(CommitOperationAdd(path_in_repo="README.md",
                                         path_or_fileobj=card.encode("utf-8")))

    retry("HF commit", lambda: api.create_commit(
        repo_id=repo, repo_type="dataset", operations=operations,
        commit_message=f"Publish v8 article parquet bundle "
                       f"({manifest['documents']:,} articles)"))

    # Verify the upload landed: re-download and compare digests.
    revision = api.repo_info(repo, repo_type="dataset").sha
    if not revision:
        raise RuntimeError('Cannot verify HF publication without a revision')
    for name in BUNDLE_FILES:
        remote = Path(retry(f"verify {name}", lambda n=name: hf_hub_download(
            repo, n, repo_type="dataset", revision=revision, token=api.token)))
        expected = (file_metadata(folder / name)["sha256"] if name == "manifest.json"
                    else manifest["files"][name]["sha256"])
        if file_metadata(remote)["sha256"] != expected:
            raise ValueError(f"HF upload verification failed: {name}")
    print(f"HF PUBLISHED https://huggingface.co/datasets/{repo} @ {revision[:12]}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", required=True)
    parser.add_argument("--folder", type=Path, default=Path("release"))
    parser.add_argument("--sources-dir", type=Path,
                        help="freshly rebuilt data/*.txt.gz + build-manifest.json to commit with the bundle")
    args = parser.parse_args()
    repo, _project, _code = resolve_language(args.language)
    api = HfApi(token=os.getenv("HF_TOKEN") or None)
    publish_hf(api, repo, args.folder, sources_dir=args.sources_dir)


if __name__ == "__main__":
    main()
