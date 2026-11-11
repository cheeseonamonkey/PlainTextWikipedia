#!/usr/bin/env python3
"""Publish the verified v8 bundle to Zenodo.

Uploads the exact local bundle files (articles.parquet, corpus-profile.json,
manifest.json) to a Zenodo deposition and publishes the record. Reuses a
matching published record via a new version when the manifest fingerprint is
unchanged; otherwise creates or updates a draft. Every uploaded file's MD5 is
checked against the local bytes, and the published record's file set is
verified after publication.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from scripts.wiki_metadata import resolve_language

ZENODO_MAX = 50_000_000_000
SOURCE = "https://github.com/cheeseonamonkey/PlainTextWikipedia"
BUNDLE_FILES = ("articles.parquet", "corpus-profile.json", "manifest.json")


def _md5(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "md5").hexdigest()


def list_matching_depositions(requests, url: str, headers: dict, title: str) -> list[dict]:
    """Search owned deposits (not public records); paginate to avoid duplicates."""
    matches = []
    for page in range(1, 102):
        response = requests.get(url, headers=headers,
                                params={"q": "Plaintext Wikipedia", "size": 100,
                                        "page": page, "all_versions": "false"},
                                timeout=30)
        response.raise_for_status()
        batch = response.json()
        if not isinstance(batch, list):
            raise RuntimeError("Unexpected Zenodo deposition listing")
        matches.extend(item for item in batch
                       if item.get("metadata", {}).get("title", item.get("title")) == title)
        if len(batch) < 100:
            break
    else:
        raise RuntimeError("Zenodo deposition search exceeded pagination limit")
    return matches


def verify_published(document, folder: Path) -> None:
    files = document.get("files", [])
    remote = {item["filename"]: item for item in files}
    if document.get("submitted") is not True or len(remote) != len(files) or set(remote) != set(BUNDLE_FILES):
        raise ValueError("Published Zenodo file set differs from the bundle")
    for name in BUNDLE_FILES:
        expected = _md5(folder / name)
        actual = str(remote[name].get("checksum", "")).removeprefix("md5:")
        if int(remote[name].get("filesize", -1)) != (folder / name).stat().st_size \
                or actual.lower() != expected:
            raise ValueError(f"Published Zenodo checksum mismatch: {name}")


def zenodo_publish(folder: Path, language: str, repo: str) -> str:
    import requests

    # Never upload an unverified bundle.
    from scripts.bundle_artifacts import verify_bundle
    verify_bundle(folder)

    token = os.getenv("ZENODO_TOKEN")
    if not token:
        raise RuntimeError("ZENODO_TOKEN environment variable required")
    headers = {"Authorization": f"Bearer {token}"}
    url = "https://zenodo.org/api/deposit/depositions"

    title = f"Plaintext Wikipedia ({language}) — Parquet edition"
    with (folder / "manifest.json").open("rb") as stream:
        fingerprint = hashlib.file_digest(stream, "sha256").hexdigest()
    payload = {
        "metadata": {
            "title": title,
            "notes": f"parquet-manifest-sha256:{fingerprint}",
            "upload_type": "dataset",
            "description": (
                "<p>Article-level Parquet with a native fastText embedding, "
                "per-article TF-IDF (stemmed, L2-normalized against the language "
                "corpus), raw word counts, and grouped JSON statistics. "
                "Language-level IDF/vocabulary context is in corpus-profile.json.</p>"
                f"<p>Maintained at {SOURCE}. Source: Wikimedia contributors (CC BY-SA 4.0). "
                f"Dataset: https://huggingface.co/datasets/{repo}</p>"
            ),
            "creators": [{"name": "cheeseonamonkey"}],
            "license": "cc-by-sa-4.0",
            "keywords": ["wikipedia", "corpus", "parquet", "multilingual", "fasttext"],
        }
    }

    matches = list_matching_depositions(requests, url, headers, title)
    if len(matches) > 1:
        raise RuntimeError("Multiple Zenodo records already match this language; resolve manually")
    prior = matches[0] if matches else None

    if prior and prior.get("submitted") and prior.get("metadata", {}).get("notes") == (
            f"parquet-manifest-sha256:{fingerprint}"):
        verify_published(prior, folder)
        print(f"SKIP Zenodo: published edition is current ({prior['id']})", flush=True)
        return f"https://zenodo.org/records/{prior['id']}"

    if prior and prior.get("submitted"):
        response = requests.post(f"{url}/{prior['id']}/actions/newversion",
                                 headers=headers, timeout=30)
        response.raise_for_status()
        draft_url = response.json()["links"]["latest_draft"]
        if not draft_url.startswith(url + "/"):
            raise RuntimeError("Unexpected Zenodo draft URL")
        response = requests.put(draft_url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        draft = response.json()
    elif prior:
        response = requests.put(f"{url}/{prior['id']}", json=payload,
                                headers=headers, timeout=30)
        response.raise_for_status()
        draft = response.json()
    else:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        draft = response.json()

    bucket = draft["links"]["bucket"]
    publish_url = draft["links"]["publish"]
    if not bucket.startswith('https://zenodo.org/api/files/') or not publish_url.startswith(url + '/'):
        raise RuntimeError('Unexpected Zenodo upload or publish URL')
    total = sum((folder / name).stat().st_size for name in BUNDLE_FILES)
    if total + 1_000_000 > ZENODO_MAX:
        raise RuntimeError("Exceeds Zenodo default 50 GB record limit")
    print(f"Zenodo draft {draft['id']} ready: uploading {total / 1e9:.2f} GB", flush=True)

    old_files = {file["filename"]: file for file in draft.get("files", [])}
    for name, file in old_files.items():
        if name not in BUNDLE_FILES:
            response = requests.delete(f"{url}/{draft['id']}/files/{file['id']}",
                                       headers=headers, timeout=30)
            response.raise_for_status()

    for name in BUNDLE_FILES:
        path = folder / name
        md5 = _md5(path)
        existing = old_files.get(name)
        if existing and str(existing.get("checksum") or "").removeprefix("md5:").lower() == md5 \
                and int(existing.get("filesize", -1)) == path.stat().st_size:
            print(f"REUSED Zenodo draft file: {name}", flush=True)
            continue
        if existing:
            response = requests.delete(f"{url}/{draft['id']}/files/{existing['id']}",
                                       headers=headers, timeout=30)
            response.raise_for_status()
        with path.open("rb") as stream:
            response = requests.put(f"{bucket}/{name}", data=stream,
                                    headers=headers, timeout=(30, 7200))
        response.raise_for_status()
        actual = response.json() if response.content else {}
        if str(actual.get("checksum", "")).removeprefix("md5:").lower() != md5:
            raise ValueError(f"Zenodo upload checksum mismatch for {name}")
        print(f"UPLOADED {name}", flush=True)

    published = requests.post(publish_url, headers=headers, timeout=90)
    published.raise_for_status()
    doc = published.json()
    if "files" not in doc or "submitted" not in doc:
        response = requests.get(f"{url}/{draft['id']}", headers=headers, timeout=30)
        response.raise_for_status()
        doc = response.json()
    verify_published(doc, folder)
    record = doc.get("doi_url") or f"https://zenodo.org/records/{draft['id']}"
    print(f"ZENODO PUBLISHED {record}", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", required=True)
    parser.add_argument("--folder", type=Path, default=Path("release"))
    args = parser.parse_args()
    repo, project, code = resolve_language(args.language)
    zenodo_publish(args.folder, project.removesuffix("wiki"), repo)


if __name__ == "__main__":
    main()
