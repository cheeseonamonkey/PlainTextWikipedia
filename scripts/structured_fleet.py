#!/usr/bin/env python3
"""Incremental, per-language structured Wikipedia exports from existing HF archives.

No Wikimedia XML conversion. Published Parquet preserves raw-source SHA256
provenance; the original plaintext archives remain available. Each language is independent, resumable, and idempotent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

try:
    from scripts.structured_export import export
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from structured_export import export


PREFIX = "ffatty/"
RAW_RE = re.compile(r"data/[^/]+\.txt\.gz$")


def source_files(api: HfApi, repo: str):
    tree = list(api.list_repo_tree(repo, repo_type="dataset", recursive=True))
    return {f.path: f for f in tree if getattr(f, "path", "").startswith("data/")
            and RAW_RE.fullmatch(f.path)}, {f.path: f for f in tree}


def sha256(file: Path) -> str:
    with file.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_hash(info) -> str | None:
    lfs = getattr(info, "lfs", None)
    if isinstance(lfs, dict):
        return lfs.get("sha256")
    return getattr(lfs, "sha256", None)


def retry(label, callback, attempts=4):
    for n in range(attempts):
        try:
            return callback()
        except Exception as exc:
            if n == attempts - 1:
                raise
            print(f"Retrying {label} after transient error ({type(exc).__name__})", flush=True)
            time.sleep(min(80, 8 * 2 ** n))


def current_manifest(api, repo: str, name: str, kind: str,
                     source_digest: str, remote: dict, require_fts: bool = False) -> bool:
    path = f"structured/manifests/{name}.{kind}.json"
    if path not in remote:
        return False
    try:
        found = json.loads(Path(hf_hub_download(
            repo, filename=path, repo_type="dataset", token=api.token)).read_text())
        record = found["artifacts"][kind]
        artifact = remote.get(f"structured/{record['path']}")
        artifact_sha = source_hash(artifact) if artifact is not None else None
        return (found["source_sha256"] == source_digest
                and artifact is not None
                and (artifact_sha is None or artifact_sha == record["sha256"])
                and (not require_fts or record.get("fts5") is True))
    except (OSError, KeyError, ValueError, TypeError):
        return False


def available_repos(api: HfApi):
    repos = sorted(x.id for x in api.list_datasets(author="ffatty", limit=1000)
                   if x.id.startswith("ffatty/plaintext-wikipedia-full-")
                   or x.id == "ffatty/plain-text-wikipedia-simpleenglish")
    if len(repos) > 256:
        raise RuntimeError(f"More than 256 repositories ({len(repos)}); split the rollout into batches")
    if not repos:
        raise RuntimeError("No published Wikipedia datasets found")
    return repos


def plan() -> None:
    api = HfApi(token=os.getenv("HF_TOKEN") or None)
    repos = available_repos(api)
    # Smaller, short names come first; builds remain independent.
    repos.sort(key=lambda x: (x != "ffatty/plain-text-wikipedia-simpleenglish", x))
    matrix = {"include": [{"repo": x} for x in repos]}
    output = json.dumps(matrix, separators=(",", ":"))
    if out := os.getenv("GITHUB_OUTPUT"):
        with open(out, "a", encoding="utf-8") as stream:
            stream.write(f"matrix={output}\n")
    print(f"Discovered {len(repos)} EXISTING Wikipedia datasets; no new slugs created")
    for repo in repos:
        print(repo)


def build(repo: str) -> None:
    if not re.fullmatch(r"ffatty/(?:plain-text-wikipedia-simpleenglish|plaintext-wikipedia-full-[a-z0-9-]+)", repo):
        raise ValueError(f"Unsupported target: {repo}")
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN required for additive publishing")
    api = HfApi(token=token)
    sources, remote = retry("list existing HF files", lambda: source_files(api, repo))
    if not sources:
        print(f"::warning::No source text archives in {repo}; skipping", flush=True)
        return
    failures = []
    updated = skipped = 0
    for path, info in sorted(sources.items()):
        name = Path(path).name.removesuffix(".txt.gz")
        digest = source_hash(info)
        def is_current(known_hash):
            return current_manifest(api, repo, name, "parquet", known_hash, remote)

        if digest and is_current(digest):
            skipped += 1
            print(f"SKIP already-current {repo}/{name}", flush=True)
            continue
        try:
            with tempfile.TemporaryDirectory(prefix="wiki-structured-") as temporary:
                working = Path(temporary)
                source = Path(retry("download source", lambda: hf_hub_download(
                    repo, filename=path, repo_type="dataset", token=token,
                    local_dir=working / "source")))
                digest_actual = sha256(source)
                if digest and digest != digest_actual:
                    raise ValueError(f"Source LFS SHA256 mismatch: {path}")
                if is_current(digest_actual):
                    skipped += 1
                    print(f"SKIP already-current {repo}/{name}", flush=True)
                    continue
                result = export(source, working / "output", "parquet")
                if result["source_sha256"] != digest_actual:
                    raise ValueError("Export provenance differs from downloaded source")

                # Publish data first, manifests last; no source files are modified.
                for kind, artifact in result["artifacts"].items():
                    location = f"structured/{artifact['path']}"
                    file = working / "output" / artifact["path"]
                    retry(f"upload {location}", lambda file=file, location=location:
                          api.upload_file(repo_id=repo, repo_type="dataset",
                                          path_or_fileobj=str(file),
                                          path_in_repo=location,
                                          commit_message=f"Structured {kind}: {name}"))
                for kind in result["artifacts"]:
                    filename = f"{name}.{kind}.json"
                    retry(f"publish manifest {filename}", lambda filename=filename:
                          api.upload_file(
                              repo_id=repo, repo_type="dataset",
                              path_or_fileobj=str(working / "output" / "manifests" / filename),
                              path_in_repo=f"structured/manifests/{filename}",
                              commit_message=f"Verify {kind} source {name}"))
                updated += 1
                print(f"UPDATED {repo}/{name}: {result['articles']} articles, "
                      f"Parquet", flush=True)
        except Exception as exc:
            failures.append(f"{path}: {type(exc).__name__}: {exc}")
            print(f"::error::{repo}/{path}: {type(exc).__name__}: {exc}", flush=True)
    summary = (f"### {repo}\n"
               f"- Source archives: {len(sources)}\n"
               f"- Updated: {updated}\n"
               f"- Already current: {skipped}\n"
               f"- Failed: {len(failures)}\n"
               f"- Hugging Face: https://huggingface.co/datasets/{repo}\n")
    if dest := os.getenv("GITHUB_STEP_SUMMARY"):
        with open(dest, "a", encoding="utf-8") as output:
            output.write(summary)
    print(summary)
    if failures:
        raise RuntimeError(f"{len(failures)} shard(s) failed in {repo}; rerun is incremental")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    b = sub.add_parser("build")
    b.add_argument("--repo", required=True)
    args = p.parse_args()
    if args.command == "plan":
        plan()
    else:
        build(args.repo)


if __name__ == "__main__":
    main()
