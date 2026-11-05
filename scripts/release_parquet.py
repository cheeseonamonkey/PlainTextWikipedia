#!/usr/bin/env python3
"""Plan and explicitly publish verified Parquet-only archives to Zenodo.

Manual-only: without --publish, performs read-only verification of HF metadata.
No files are uploaded unless --publish is explicitly supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ZENODO_MAX = 50_000_000_000
SOURCE = "https://github.com/cheeseonamonkey/PlainTextWikipedia"
UPSTREAM = "https://github.com/daveshap/PlainTextWikipedia"


@dataclass(frozen=True)
class Asset:
    path: str
    sha256: str
    size: int


def dataset_slug(repo: str) -> str:
    if not re.fullmatch(
        r"ffatty/(?:plain-text-wikipedia-simpleenglish|plaintext-wikipedia-full-[a-z0-9-]+)",
        repo,
    ):
        raise ValueError("Only existing ffatty Wikipedia repositories are supported")
    return ("simpleenglish" if repo.endswith("/plain-text-wikipedia-simpleenglish")
            else repo.split("plaintext-wikipedia-full-", 1)[1])


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def lfs_sha(info) -> str | None:
    lfs = getattr(info, "lfs", None)
    return lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)


def inventory(api: HfApi, repo: str) -> tuple[list[Asset], dict]:
    slug = dataset_slug(repo)
    files = {file.path: file for file in api.list_repo_tree(
        repo, repo_type="dataset", recursive=True)}
    raw = sorted(f for f in files if re.fullmatch(r"data/[^/]+\.txt\.gz", f))
    if not raw:
        raise ValueError("Dataset has no published plaintext source archives")
    assets = []
    details = []
    for source in raw:
        name = Path(source).name.removesuffix(".txt.gz")
        manifest_path = f"structured/manifests/{name}.parquet.json"
        if manifest_path not in files:
            raise ValueError(f"Missing Parquet manifest: {manifest_path}")
        manifest = json.loads(Path(hf_hub_download(
            repo_id=repo, filename=manifest_path,
            repo_type="dataset", token=api.token)).read_text(encoding="utf-8"))
        artifact = manifest["artifacts"]["parquet"]
        artifact_path = f"structured/{artifact['path']}"
        if artifact_path not in files:
            raise ValueError(f"Manifest references absent file: {artifact_path}")
        if not artifact_path.endswith(".parquet") or not manifest.get("articles"):
            raise ValueError(f"Invalid structured manifest: {manifest_path}")
        source_lfs = lfs_sha(files[source])
        if source_lfs and source_lfs != manifest["source_sha256"]:
            raise ValueError(f"Stale Parquet export for {source}")
        actual_info = files[artifact_path]
        if int(actual_info.size) != int(artifact["size_bytes"]):
            raise ValueError(f"Parquet artifact size mismatch for {artifact_path}")
        parquet_lfs = lfs_sha(actual_info)
        if parquet_lfs and parquet_lfs != artifact["sha256"]:
            raise ValueError(f"Parquet artifact SHA256 mismatch for {artifact_path}")
        assets.append(Asset(artifact_path, artifact["sha256"], int(artifact["size_bytes"])))
        details.append({
            "source_file": source,
            "source_sha256": manifest["source_sha256"],
            "articles": int(manifest["articles"]),
            "parquet": artifact,
        })
    if len({Path(a.path).name for a in assets}) != len(assets):
        raise ValueError("Duplicate Parquet basenames would collide on upload")
    metadata = {
        "version": 1,
        "source_repo": repo,
        "source_url": f"https://huggingface.co/datasets/{repo}",
        "project_url": SOURCE,
        "original_converter": UPSTREAM,
        "language": slug,
        "formats": ["parquet"],
        "coverage_warning": "See data/build-manifest.json on HF before claiming complete Wikipedia coverage.",
        "wikipedia_page_ids_available": False,
        "license": "CC BY-SA 3.0 (check Wikimedia source reuse terms)",
        "files": details,
    }
    return assets, metadata


def limits(assets: list[Asset]):
    total = sum(a.size for a in assets)
    if total + 1_000_000 > ZENODO_MAX or len(assets) + 1 > 100:
        raise ValueError("Exceeds Zenodo default 50 GB / 100-file record limits")
    return total


def fetch_verified(api: HfApi, repo: str, asset: Asset, directory: Path) -> Path:
    path = Path(hf_hub_download(repo_id=repo, filename=asset.path,
                                repo_type="dataset", token=api.token,
                                local_dir=directory))
    if path.stat().st_size != asset.size or sha256(path) != asset.sha256:
        raise ValueError(f"SHA256/size mismatch: {asset.path}")
    return path


def zenodo_publish(api: HfApi, repo: str, assets: list[Asset], metadata: dict,
                   record_id: int | None):
    import requests

    token = os.getenv("ZENODO_TOKEN")
    if not token:
        raise RuntimeError("ZENODO_TOKEN GitHub Actions secret required")
    headers = {"Authorization": f"Bearer {token}"}
    url = "https://zenodo.org/api/deposit/depositions"
    title = f"Plaintext Wikipedia ({metadata['language']}) — Parquet edition"
    payload = {
        "metadata": {
            "title": title,
            "upload_type": "dataset",
            "description": (
                "<p>Compressed Parquet article text, with source SHA256 provenance, "
                "derived from Wikimedia Wikipedia dumps. Completeness varies by "
                "source recovery: consult the build-manifest on Hugging Face.</p>"
                f"<p>Maintained at {SOURCE}. Source data: Wikimedia/Wikipedia contributors. "
                f"Canonical format inventory: {metadata['source_url']}. "
                "Source records do not retain Wikipedia page IDs.</p>"
            ),
            "creators": [{"name": "cheeseonamonkey"}],
            "license": "cc-by-sa-3.0",
            "keywords": ["wikipedia", "corpus", "parquet", "multilingual", "plain text"],
        }
    }
    if record_id:
        original = requests.get(f"{url}/{record_id}", headers=headers, timeout=30)
        original.raise_for_status()
        response = requests.post(f"{url}/{record_id}/actions/newversion",
                                 headers=headers, timeout=30)
        response.raise_for_status()
        draft_url = response.json()["links"]["latest_draft"]
        if not draft_url.startswith(url + "/"):
            raise RuntimeError("Unexpected Zenodo draft URL")
        response = requests.get(draft_url, headers=headers, timeout=30)
        response.raise_for_status()
        draft = response.json()
        updated = requests.put(draft_url, json=payload, headers=headers, timeout=30)
        updated.raise_for_status()
        draft = updated.json()
    else:
        response = requests.get(url, params={"size": 100}, headers=headers, timeout=30)
        response.raise_for_status()
        if any(item.get("title") == title for item in response.json()):
            raise ValueError("Existing Zenodo record with this title: supply --zenodo-id to version it")
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        draft = response.json()
    bucket = draft["links"]["bucket"]
    publish_url = draft["links"]["publish"]
    if not bucket.startswith("https://zenodo.org/api/files/") or not publish_url.startswith(url + "/"):
        raise RuntimeError("Unexpected Zenodo upload/publish endpoint")
    print(f"Zenodo draft {draft['id']} created: uploading verified Parquet only", flush=True)
    with tempfile.TemporaryDirectory(prefix="wiki-zenodo-") as tmp:
        root = Path(tmp)
        info = root / "provenance.json"
        info.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        for index, asset in enumerate(assets, 1):
            with tempfile.TemporaryDirectory(dir=root) as folder:
                source = fetch_verified(api, repo, asset, Path(folder))
                with source.open("rb") as stream:
                    response = requests.put(f"{bucket}/{source.name}", data=stream,
                                            headers=headers, timeout=(30, 7200))
                response.raise_for_status()
            print(f"Uploaded {index}/{len(assets)}: {asset.path}", flush=True)
        with info.open("rb") as stream:
            response = requests.put(f"{bucket}/{info.name}", data=stream,
                                    headers=headers, timeout=(30, 300))
        response.raise_for_status()
    published = requests.post(publish_url, headers=headers, timeout=90)
    published.raise_for_status()
    doc = published.json()
    print(f"Zenodo publication: {doc.get('doi_url') or doc.get('record_url')}", flush=True)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--publish", action="store_true",
                        help="EXPLICITLY authorize third-party uploads and publication")
    parser.add_argument("--zenodo-id", type=int,
                        help="Existing published deposition to create a new Zenodo version")
    args = parser.parse_args()
    api = HfApi(token=os.getenv("HF_TOKEN") or None)
    assets, metadata = inventory(api, args.repo)
    total = limits(assets)
    print(f"Validated {len(assets)} verified Parquet artifacts / {total / 1e9:.2f} GB"
          f" in existing {args.repo}; target=zenodo", flush=True)
    if not args.publish:
        print("DRY RUN ONLY — no files uploaded, no records created", flush=True)
        return
    zenodo_publish(api, args.repo, assets, metadata, args.zenodo_id)


if __name__ == "__main__":
    main()
