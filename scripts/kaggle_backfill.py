#!/usr/bin/env python3
"""Backfill Kaggle Wikipedia datasets from published Hugging Face data.

No Wikimedia XML downloads, conversion or new source claims. Download full
existing gzip archives, verify Hugging Face LFS SHA256 and gzip integrity,
then publish the same corpus and its existing completeness manifest to Kaggle.

Example (on a machine with authenticated Kaggle CLI):
  python scripts/kaggle_backfill.py swahili hindi --output /root/wiki-kaggle-stage
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

try:
    from scripts.repack_existing_hf import repack_local, transform_manifest
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from repack_existing_hf import repack_local, transform_manifest


HF_PREFIX = "https://huggingface.co"
OWNER = "ffatty"
USER_AGENT = "PlainTextWikipedia/1.0 (https://github.com/cheeseonamonkey/PlainTextWikipedia)"


def get_json(url: str) -> dict | list:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=50) as response:
        return json.load(response)


def manifest_for(slug: str) -> list[dict]:
    repo = f"{OWNER}/plaintext-wikipedia-full-{slug}"
    url = f"{HF_PREFIX}/api/datasets/{repo}/tree/main/data?recursive=false&expand=true"
    entries = get_json(url)
    if not isinstance(entries, list):
        raise ValueError("Invalid Hub file listing")
    result = [entry for entry in entries if entry.get("path", "").endswith(".txt.gz")]
    if not result:
        raise RuntimeError(f"No gzip archives published: {repo}")
    if len({entry["path"] for entry in result}) != len(result):
        raise ValueError("Duplicate paths in Hub listing")
    return result


def verify_file(path: Path, sha: str | None) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as raw:
        while block := raw.read(1024 * 1024):
            digest.update(block)
    if sha and digest.hexdigest() != sha:
        raise ValueError(f"SHA256 mismatch for {path.name}")
    if path.name.endswith(".gz"):
        with gzip.open(path, "rb") as stream:
            while stream.read(8 * 1024 * 1024):
                pass


def download(repo: str, remote_path: str, target: Path, sha: str | None, size: int | None) -> Path:
    if not re.fullmatch(r"data/[A-Za-z0-9._-]+", remote_path):
        raise ValueError("Unsafe Hub dataset file path")
    if target.exists():
        try:
            if size is not None and target.stat().st_size != size:
                raise ValueError("Incorrect size")
            verify_file(target, sha)
            return target
        except (OSError, ValueError, EOFError):
            target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"{HF_PREFIX}/datasets/{repo}/resolve/main/{remote_path}?download=true"
    temp = target.with_name(target.name + ".download")
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=180) as response, temp.open("wb") as out:
                while chunk := response.read(4 * 1024 * 1024):
                    out.write(chunk)
            if size is not None and temp.stat().st_size != size:
                raise IOError(f"Truncated file {target.name}")
            verify_file(temp, sha) if not target.name.endswith(".gz") else verify_file_as_gzip(temp, sha)
            temp.replace(target)
            return target
        except (OSError, ValueError, EOFError):
            temp.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("download retries exhausted")


def verify_file_as_gzip(path: Path, sha: str | None) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as raw:
        while block := raw.read(1024 * 1024):
            digest.update(block)
    if sha and digest.hexdigest() != sha:
        raise ValueError(f"SHA256 mismatch for {path.name}")
    with gzip.open(path, "rb") as stream:
        while stream.read(8 * 1024 * 1024):
            pass


def dataset_metadata(slug: str, name: str, manifest: dict | None) -> dict:
    root = "https://github.com/cheeseonamonkey/PlainTextWikipedia"
    description = (
        f"Plaintext {name} Wikipedia articles, gzip-compressed and sourced "
        f"from official Wikimedia dumps. Each record contains an article "
        f"title, its cleaned body, and a blank line.\\n\\n"
        f"Repackaged from the verified Hugging Face dataset. Original conversion "
        f"and completeness metadata are preserved, not regenerated. "
        f"Converter: {root}. Upstream: https://github.com/daveshap/PlainTextWikipedia."
    )
    if manifest:
        description += (
            f"\\n\\nOriginal build complete: {bool(manifest.get('complete'))}; "
            f"available source roots: {manifest.get('published_shards', '?')}/"
            f"{manifest.get('expected_shards', '?')}. See build-manifest.json."
        )
    return {
        "title": f"Plaintext Wikipedia ({name})",
        "id": f"{OWNER}/plaintext-wikipedia-full-{slug}",
        "subtitle": "Multilingual Wikipedia corpus by PlainTextWikipedia",
        "description": description,
        "licenses": [{"name": "CC-BY-SA-3.0"}],
    }


def kaggle_status(dataset: str) -> dict | None:
    result = subprocess.run(
        ["kaggle", "datasets", "status", dataset, "--format", "json"],
        capture_output=True, text=True, timeout=90)
    if result.returncode:
        return None
    try:
        value = json.loads(result.stdout)
        return value if isinstance(value, dict) else None
    except ValueError:
        return None


def run_kaggle(dataset: str, work: Path, name: str,
               wait_seconds: int = 1800) -> None:
    auth = subprocess.run(["kaggle", "datasets", "list", "--mine"],
                          capture_output=True, text=True, timeout=120)
    if auth.returncode:
        raise RuntimeError("Kaggle CLI not authenticated on this machine")
    previous = kaggle_status(dataset)
    previous_version = int((previous or {}).get("current_version_number") or 0)
    if previous:
        args = ["kaggle", "datasets", "version", "-p", str(work),
                "-m", f"Publish verified Wikipedia ({name}) corpus", "--dir-mode", "skip"]
    else:
        args = ["kaggle", "datasets", "create", "-p", str(work),
                "--public", "--dir-mode", "skip"]
    subprocess.run(args, check=True, timeout=7200)
    # Kaggle create/version returns after upload, not necessarily after the
    # server has finished processing the dataset. Never erase staging until
    # that new version is actually ready.
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        current = kaggle_status(dataset)
        if current:
            status = str(current.get("status", "")).lower()
            number = int(current.get("current_version_number") or 0)
            if number > previous_version and status == "ready":
                print(f"READY (version {number}): https://www.kaggle.com/datasets/{dataset}",
                      flush=True)
                return
            if status in ("error", "failed", "failure"):
                raise RuntimeError(f"Kaggle dataset {dataset} processing failed")
        time.sleep(12)
    raise TimeoutError(f"Kaggle accepted the upload but did not confirm a ready "
                       f"new version within {wait_seconds}s; retaining staging files")

def backfill(slug: str, name: str, outdir: Path, max_mb: int,
             publish: bool = True, keep_staging: bool = False) -> dict:
    if not re.fullmatch(r"[a-z0-9-]+", slug):
        raise ValueError("Invalid Wikipedia dataset slug")
    repo = f"{OWNER}/plaintext-wikipedia-full-{slug}"
    files = manifest_for(slug)
    total = sum(entry.get("size") or 0 for entry in files)
    if total <= 0 or total > max_mb * 1024 * 1024:
        raise RuntimeError(f"Compressed corpus for {slug}: {total // 1048576} MiB; "
                           f"exceeds maximum {max_mb} MiB or has unknown sizes")
    working = outdir / slug
    working.mkdir(parents=True, exist_ok=True)
    source_dir = working / "source-archives"
    source_dir.mkdir(exist_ok=True)
    for entry in files:
        path = entry["path"]
        sha = (entry.get("lfs") or {}).get("oid")
        target = source_dir / Path(path).name
        # Reuse already downloaded archives from an earlier publication rather
        # than fetching from Hugging Face a second time.
        earlier = working / target.name
        if not target.exists() and earlier.is_file():
            try:
                verify_file(earlier, sha)
                os.link(earlier, target)
            except (OSError,ValueError,EOFError):
                pass
        print(f"Downloading/verifying {slug}: {target.name} ({entry.get('size',0)//1048576} MiB)",
              flush=True)
        download(repo, path, target, sha, entry.get("size"))
    # Recreate the publication folder from immutable, checksum-verified source
    # bytes. The --dir-mode skip flag keeps source-archives off Kaggle.
    for path in working.glob("*.txt.gz"):
        path.unlink()
    for entry in files:
        name=Path(entry["path"]).name
        os.link(source_dir / name, working / name)

    old_manifest = None
    try:
        data = get_json(f"{HF_PREFIX}/datasets/{repo}/raw/main/data/build-manifest.json")
        if isinstance(data, dict):
            old_manifest = data
    except (OSError,ValueError):
        pass
    generated = repack_local(working, 16, min_bytes=20 * 1024 * 1024)
    if generated["changed"]:
        print(f"Losslessly split {slug} to {len(generated['new_files'])} archive(s), "
              "each at least 20 MiB compressed", flush=True)
        if old_manifest:
            old_manifest = transform_manifest(
                old_manifest, generated["replacements"], generated["old_files"])
        storage = {
            "verified": "SHA-256 article-record reconstruction",
            "source_archives": generated["old_files"],
            "published_archives": generated["new_files"],
            "replacements": generated["replacements"],
        }
        (working / "storage-repack.json").write_text(
            json.dumps(storage, indent=2, ensure_ascii=False) + "\n")
    else:
        (working / "storage-repack.json").unlink(missing_ok=True)
    if old_manifest:
        (working / "build-manifest.json").write_text(
            json.dumps(old_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
    else:
        (working / "build-manifest.json").unlink(missing_ok=True)
    meta = dataset_metadata(slug, name, old_manifest)
    (working / "dataset-metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    n_archives=len(list(working.glob("*.txt.gz")))
    if publish:
        run_kaggle(meta["id"], working, name)
        if not keep_staging:
            shutil.rmtree(working)
            print(f"Deleted verified/published staging directory: {working}", flush=True)
    return {"dataset": meta["id"], "gzips": n_archives,
            "bytes": total, "published": publish}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("slugs", nargs="+")
    p.add_argument("--output", type=Path, default=Path("/root/wiki-kaggle-stage"))
    p.add_argument("--max-mb", type=int, default=1400)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--keep-staging", action="store_true",
                   help="Keep downloaded source and repacked local files after Kaggle ready")
    args = p.parse_args()
    entries = json.loads(
        (Path(__file__).resolve().parents[1]/"config"/"additional-languages.json").read_text()
    )["languages"]
    names = {entry["slug"]: entry["name"] for entry in entries}
    failed=[]
    for slug in args.slugs:
        if slug not in names:
            print(f"ERROR: Unknown configured language: {slug}", file=sys.stderr)
            failed.append(slug)
            continue
        try:
            result = backfill(slug,names[slug],args.output,args.max_mb,
                              publish=not args.dry_run,keep_staging=args.keep_staging)
            print(json.dumps(result),flush=True)
        except Exception as exc:
            print(f"ERROR: {slug}: {type(exc).__name__}: {exc}", file=sys.stderr,flush=True)
            failed.append(slug)
    if failed:
        print(f"Failed: {', '.join(failed)}",file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
