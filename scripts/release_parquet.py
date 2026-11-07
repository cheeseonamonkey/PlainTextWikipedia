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
    md5: str | None = None


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
    if "articles.parquet" in files and "features/manifest.json" in files:
        manifest = json.loads(Path(hf_hub_download(
            repo_id=repo, filename="features/manifest.json", repo_type="dataset",
            token=api.token)).read_text(encoding="utf-8"))
        declared = {item["path"]: item["sha256"] for item in manifest.get("source_archives", [])}
        if set(declared) != set(raw) or any(not lfs_sha(files[p]) or lfs_sha(files[p]) != declared[p] for p in raw):
            raise ValueError("Feature table does not match current source archives")
        meta = manifest.get("files", {}).get("articles.parquet", {})
        parquet = files["articles.parquet"]
        if (manifest.get("primary_table") != "articles.parquet" or not manifest.get("documents")
                or not re.fullmatch(r"[0-9a-f]{64}", meta.get("sha256", ""))
                or int(parquet.size) != meta.get("bytes")
                or (lfs_sha(parquet) and lfs_sha(parquet) != meta["sha256"])):
            raise ValueError("Invalid canonical feature Parquet")
        return [Asset("articles.parquet", meta["sha256"], int(meta["bytes"]))], {
            "version": 2, "source_repo": repo,
            "source_url": f"https://huggingface.co/datasets/{repo}",
            "kaggle_url": f"https://www.kaggle.com/datasets/{repo}",
            "project_url": SOURCE, "original_converter": UPSTREAM,
            "language": slug, "formats": ["parquet"],
            "documents": manifest["documents"], "files": [meta],
            "license": "CC BY-SA 4.0 (consult Wikimedia reuse requirements)",
        }
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
        if not source_lfs:
            raise ValueError(f"Unverifiable source hash for {source}; refuse an archival DOI")
        if source_lfs != manifest["source_sha256"]:
            raise ValueError(f"Stale Parquet export for {source}")
        actual_info = files[artifact_path]
        if int(actual_info.size) != int(artifact["size_bytes"]):
            raise ValueError(f"Parquet artifact size mismatch for {artifact_path}")
        parquet_lfs = lfs_sha(actual_info)
        if parquet_lfs and parquet_lfs != artifact["sha256"]:
            raise ValueError(f"Parquet artifact SHA256 mismatch for {artifact_path}")
        md5 = artifact.get("md5")
        if md5 is not None and not re.fullmatch(r"[0-9a-f]{32}", md5):
            raise ValueError(f"Invalid Parquet MD5 in {manifest_path}")
        assets.append(Asset(artifact_path, artifact["sha256"],
                            int(artifact["size_bytes"]), md5))
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
        "license": "CC BY-SA 4.0 (consult Wikimedia reuse requirements)",
        "files": details,
    }
    return assets, metadata


def limits(assets: list[Asset]):
    total = sum(a.size for a in assets)
    if total + 1_000_000 > ZENODO_MAX or len(assets) + 1 > 100:
        raise ValueError("Exceeds Zenodo default 50 GB / 100-file record limits")
    return total


def fetch_verified(api: HfApi, repo: str, asset: Asset) -> Path:
    # Use the standard HF cache rather than an isolated temporary local_dir.
    # The preceding Kaggle step may already have downloaded this exact blob.
    path = Path(hf_hub_download(repo_id=repo, filename=asset.path,
                                repo_type="dataset", token=api.token))
    if path.stat().st_size != asset.size or sha256(path) != asset.sha256:
        raise ValueError(f"SHA256/size mismatch: {asset.path}")
    return path


def provenance_fingerprint(metadata: dict) -> str:
    """Version fingerprint from the complete source and Parquet manifest."""
    encoded = json.dumps(metadata, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def list_matching_depositions(requests, url: str, headers: dict, title: str) -> list[dict]:
    """Search owned deposits, not public records; paginate to avoid duplicate DOIs."""
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


def zenodo_publish(api: HfApi, repo: str, assets: list[Asset], metadata: dict,
                   record_id: int | None, *, local_folder: Path | None = None):
    import requests

    token = os.getenv("ZENODO_TOKEN")
    if not token:
        raise RuntimeError("ZENODO_TOKEN GitHub Actions secret required")
    headers = {"Authorization": f"Bearer {token}"}
    url = "https://zenodo.org/api/deposit/depositions"
    title = f"Plaintext Wikipedia ({metadata['language']}) — Parquet edition"
    fingerprint = provenance_fingerprint(metadata)
    payload = {
        "metadata": {
            "title": title,
            "notes": f"wikidata-parquet-manifest-sha256:{fingerprint}",
            "upload_type": "dataset",
            "description": (
                "<p>Compressed Parquet article text, character/byte/word counts, "
                "best-effort (unverified) Wikipedia title links and SHA256 provenance, "
                "derived from Wikimedia Wikipedia dumps. Completeness varies by "
                "source recovery: consult the build-manifest on Hugging Face.</p>"
                f"<p>Maintained at {SOURCE}. Source data: Wikimedia/Wikipedia contributors. "
                f"Canonical format inventory: {metadata['source_url']}. "
                "Source records do not retain Wikipedia page IDs.</p>"
            ),
            "creators": [{"name": "cheeseonamonkey"}],
            "license": "cc-by-sa-4.0",
            "keywords": ["wikipedia", "corpus", "parquet", "multilingual", "plain text"],
        }
    }
    if local_folder is not None:
        payload["metadata"]["description"] = (
            "<p>Complete Wikipedia article Parquet, combined plaintext, features and models. "
            "Identical data bundle on all three publishers. Text: Wikimedia contributors, CC BY-SA 4.0.</p>" +
            "".join(f'<p>{key}: {value}</p>' for key, value in metadata["links"].items()))
    if record_id is None:
        matches = list_matching_depositions(requests, url, headers, title)
        if len(matches) > 1:
            raise RuntimeError("Multiple Zenodo records already match this language; "
                               "supply --zenodo-id to select the intended record")
        if matches:
            prior = matches[0]
            record_id = int(prior["id"])
        else:
            prior = None
    else:
        response = requests.get(f"{url}/{record_id}", headers=headers, timeout=30)
        response.raise_for_status()
        prior = response.json()
        if prior.get("metadata", {}).get("title") != title:
            raise ValueError("Zenodo record ID belongs to another Wikipedia edition")

    if prior and prior.get("submitted") and prior.get("metadata", {}).get("notes") == (
            f"wikidata-parquet-manifest-sha256:{fingerprint}"):
        if local_folder is not None:
            remote = {f["filename"]: f for f in prior.get("files", [])}
            if set(remote) != {Path(a.path).name for a in assets}:
                raise ValueError("Published Zenodo file set differs from the bundle")
            for asset in assets:
                item = remote[Path(asset.path).name]
                with (local_folder / asset.path).open("rb") as stream:
                    digest = hashlib.file_digest(stream, "md5").hexdigest()
                if int(item.get("filesize", -1)) != asset.size or str(item.get("checksum", "")).removeprefix("md5:") != digest:
                    raise ValueError("Published Zenodo checksum mismatch")
        print(f"SKIP Zenodo: published edition is current ({record_id})", flush=True)
        return

    if prior and prior.get("submitted"):
        response = requests.post(f"{url}/{record_id}/actions/newversion",
                                 headers=headers, timeout=30)
        response.raise_for_status()
        draft_url = response.json()["links"]["latest_draft"]
        if not draft_url.startswith(url + "/"):
            raise RuntimeError("Unexpected Zenodo draft URL")
        response = requests.put(draft_url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        draft = response.json()
    elif prior:
        # Resume an unpublished draft from a previous interrupted workflow.
        response = requests.put(f"{url}/{record_id}", json=payload,
                                headers=headers, timeout=30)
        response.raise_for_status()
        draft = response.json()
    else:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        draft = response.json()
    bucket = draft["links"]["bucket"]
    publish_url = draft["links"]["publish"]
    if not bucket.startswith("https://zenodo.org/api/files/") or not publish_url.startswith(url + "/"):
        raise RuntimeError("Unexpected Zenodo upload/publish endpoint")
    print(f"Zenodo draft {draft['id']} ready: uploading verified release files", flush=True)
    # Preserve completed draft uploads across retries. Older published versions
    # can also carry unchanged Parquet files into the new draft.
    old_files = {file["filename"]: file for file in draft.get("files", [])}
    wanted = {Path(asset.path).name for asset in assets}
    if local_folder is None:
        wanted.add("provenance.json")

    def remote_matches(filename: str, size: int, md5: str) -> bool:
        prior_file = old_files.get(filename)
        if not prior_file:
            return False
        try:
            old_size = int(prior_file.get("filesize", -1))
        except (TypeError, ValueError):
            return False
        return (old_size == size and
                str(prior_file.get("checksum") or "").removeprefix("md5:").lower() == md5)

    def remove_old(file):
        response = requests.delete(f"{url}/{draft['id']}/files/{file['id']}",
                                   headers=headers, timeout=30)
        response.raise_for_status()

    for name, file in old_files.items():
        if name not in wanted:
            remove_old(file)

    def upload_verified(file: Path, timeout: int, known_md5: str | None = None):
        md5 = known_md5
        if md5 is None:
            with file.open("rb") as stream:
                md5 = hashlib.file_digest(stream, "md5").hexdigest()
        prior_file = old_files.get(file.name)
        if prior_file:
            if remote_matches(file.name, file.stat().st_size, md5):
                print(f"REUSED Zenodo draft file: {file.name}", flush=True)
                return
            remove_old(prior_file)
        with file.open("rb") as stream:
            response = requests.put(f"{bucket}/{file.name}", data=stream,
                                    headers=headers, timeout=(30, timeout))
        response.raise_for_status()
        actual = response.json() if response.content else {}
        remote_md5 = str(actual.get("checksum", "")).removeprefix("md5:")
        if (local_folder is not None or remote_md5) and remote_md5.lower() != md5:
            raise ValueError(f"Zenodo returned a checksum mismatch for {file.name}")

    with tempfile.TemporaryDirectory(prefix="wiki-zenodo-") as tmp:
        root = Path(tmp)
        info = root / "provenance.json"
        info.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        for index, asset in enumerate(assets, 1):
            if asset.md5 and remote_matches(Path(asset.path).name, asset.size, asset.md5):
                # Existing draft bytes match the independently recorded HF
                # identity. Do not download the Parquet just to hash it again.
                print(f"REUSED Zenodo draft from manifest: {asset.path}", flush=True)
                continue
            source = local_folder / asset.path if local_folder else fetch_verified(api, repo, asset)
            if local_folder and (source.stat().st_size != asset.size or sha256(source) != asset.sha256):
                raise ValueError(f"checksum mismatch before Zenodo upload: {asset.path}")
            upload_verified(source, 7200, asset.md5)
            print(f"Processed {index}/{len(assets)}: {asset.path}", flush=True)
        if local_folder is None:
            upload_verified(info, 300)
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
    raise SystemExit("Retired publishing entrypoint. Use python -m scripts.wikipedia_bundle; build-only is the default.")
