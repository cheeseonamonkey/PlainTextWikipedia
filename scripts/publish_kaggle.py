#!/usr/bin/env python3
"""Publish the verified v8 bundle to Kaggle.

Creates a new version of the existing Kaggle dataset, preserving every file
already published there and adding the v8 bundle (articles.parquet,
corpus-profile.json, manifest.json). Safety rules:

- The dataset must already exist and be READY; a processing dataset is never
  replaced (a prior version may still be inflating).
- Prior files are only kept if they can be downloaded; a failed or empty
  download aborts the version instead of silently dropping old content.
- The staged file count must fit Kaggle's per-version budget.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from functools import lru_cache
from pathlib import Path

from scripts.bundle_artifacts import verify_bundle
from scripts.wiki_metadata import resolve_language

BUNDLE_FILES = ("articles.parquet", "corpus-profile.json", "manifest.json")
KAGGLE_FILE_LIMIT = 50


def run(*args, check=True):
    command = ["kaggle", "datasets", *args]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if check and result.returncode:
        print(result.stdout[-1500:], flush=True)
        print(result.stderr[-2000:], flush=True)
        raise RuntimeError(f"Kaggle command failed: {' '.join(command[2:4])}")
    return result


@lru_cache(maxsize=1)
def _catalog():
    from pathlib import Path as _P
    catalog = _P(__file__).resolve().parents[1] / "config" / "additional-languages.json"
    return json.loads(catalog.read_text(encoding="utf-8"))["languages"]


def language_name(repo: str) -> str:
    slug = repo.rsplit("/", 1)[1].removeprefix("plaintext-wikipedia-full-")
    if slug == "plain-text-wikipedia-simpleenglish":
        return "Simple English"
    return next((item["name"] for item in _catalog() if item["slug"] == slug),
                slug.replace("-", " ").title())


def publish_kaggle(repo: str, folder: Path) -> None:
    # Reject an unverifiable bundle before touching Kaggle.
    verify_bundle(folder)

    dataset = repo
    existing = run("status", dataset, "--format", "json", check=False)
    if existing.returncode:
        raise RuntimeError(f"{dataset}: no accessible existing Kaggle dataset")
    status = json.loads(existing.stdout)
    if str(status.get("status", "")).lower() != "ready":
        raise RuntimeError(f"{dataset}: Kaggle status is "
                           f"{status.get('status')!r}; refusing to replace pending data")

    with tempfile.TemporaryDirectory(prefix="wiki-kaggle-") as tmp:
        stage = Path(tmp)
        # Retrieve every file already on Kaggle; no silent data deletion.
        # A failed download must abort, not create a version that drops files.
        result = run("download", "-d", dataset, "-p", str(stage), "--unzip", check=False)
        if result.returncode != 0:
            raise RuntimeError(f"{dataset}: could not download current Kaggle files: "
                               f"{(result.stderr or result.stdout)[-500:]}")
        prior = [p for p in stage.iterdir() if p.is_file()]
        if not prior or any(p.is_dir() for p in stage.iterdir()):
            raise RuntimeError(f"{dataset}: Kaggle download returned no files or contains directories; "
                               "refusing to build a version that could erase content")
        # Remove the CLI's metadata file if present; we write our own.
        (stage / "dataset-metadata.json").unlink(missing_ok=True)

        for name in BUNDLE_FILES:
            shutil.copyfile(folder / name, stage / name)

        meta = {
            "id": dataset,
            "title": f"Plaintext Wikipedia ({language_name(repo)})",
            "subtitle": "Wikipedia articles · original plaintext + enriched Parquet",
            "description": (
                "Article-level parquet with a native fastText embedding, "
                "per-article TF-IDF (stemmed, L2-normalized against the language "
                "corpus), raw word counts, and grouped JSON statistics. "
                "Language-level IDF/vocabulary context is in `corpus-profile.json`. "
                "Original .txt.gz source archives are retained verbatim.\n\n"
                "Source: Wikimedia contributors (CC BY-SA 4.0); converter: "
                "https://github.com/cheeseonamonkey/PlainTextWikipedia. "
                "Observe Wikimedia attribution and share-alike requirements."
            ),
            "licenses": [{"name": "CC-BY-SA-4.0"}],
            "resources": [],
        }
        (stage / "dataset-metadata.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        count = len([p for p in stage.iterdir() if p.is_file() and p.name != "dataset-metadata.json"])
        if count > KAGGLE_FILE_LIMIT:
            raise RuntimeError(f"{dataset}: {count} files exceeds Kaggle {KAGGLE_FILE_LIMIT}-file budget")

        run("version", "-p", str(stage), "-m",
            "Update enriched Parquet and original Wikipedia text",
            "--dir-mode", "skip", "--keep-tabular")
        # Confirm acceptance: the version must be accepted and eventually READY.
        # A still-processing upload is reported honestly, not retried (a retry
        # would create a duplicate version).
        accepted = False
        for attempt in range(30):
            time.sleep(60)
            probe = run("status", dataset, "--format", "json", check=False)
            if probe.returncode != 0:
                continue
            state = json.loads(probe.stdout)
            status = str(state.get("status", "")).lower()
            if status == "ready":
                accepted = True
                break
            if status not in ("", "processing", "pending"):
                raise RuntimeError(f"{dataset}: Kaggle version status {status!r}; "
                                   "the upload was rejected or failed server-side")
        if accepted:
            print(f"KAGGLE VERSIONED https://www.kaggle.com/datasets/{dataset} (READY)", flush=True)
        else:
            print(f"::warning::{dataset}: Kaggle version accepted but still processing "
                  "after 30 minutes; verify the dataset page before relying on it", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", required=True)
    parser.add_argument("--folder", type=Path, default=Path("release"))
    args = parser.parse_args()
    repo, _project, _code = resolve_language(args.language)
    publish_kaggle(repo, args.folder)


if __name__ == "__main__":
    main()
