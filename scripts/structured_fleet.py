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

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

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


# Start the expensive editions early to avoid a several-hour tail when
# smaller matrix jobs finish. Simple English is a cheap initial smoke check.
LARGE_LANGUAGES = (
    "english", "german", "french", "japanese", "spanish",
    "russian", "chinese", "portuguese", "italian", "arabic",
)
SIMPLE = "ffatty/plain-text-wikipedia-simpleenglish"


def select_repos(repos: list[str], only: str = "") -> list[str]:
    if only.strip():
        requested = {word.strip().lower() for word in only.split(",") if word.strip()}
        names = {repo: repo.rsplit("/", 1)[-1].removeprefix("plaintext-wikipedia-full-")
                 for repo in repos}
        matches = [repo for repo in repos
                   if repo.lower() in requested or names[repo] in requested
                   or (repo == SIMPLE and "simpleenglish" in requested)]
        unknown = requested - {value for repo in matches for value in
                               (repo.lower(), names[repo])} - (
                                   {"simpleenglish"} if SIMPLE in matches else set())
        if unknown:
            raise ValueError(f"Unknown/unpublished Wikipedia datasets: {', '.join(sorted(unknown))}")
        repos = matches
    priority = {name: index for index, name in enumerate(LARGE_LANGUAGES)}
    return sorted(repos, key=lambda repo: (
        -1 if repo == SIMPLE else priority.get(
            repo.rsplit("/", 1)[-1].removeprefix("plaintext-wikipedia-full-"), 100),
        repo,
    ))


def plan(only: str = "") -> None:
    api = HfApi(token=os.getenv("HF_TOKEN") or None)
    repos = select_repos(available_repos(api), only)
    matrix = {"include": [{"repo": repo} for repo in repos]}
    output = json.dumps(matrix, separators=(",", ":"))
    if out := os.getenv("GITHUB_OUTPUT"):
        with open(out, "a", encoding="utf-8") as stream:
            stream.write(f"matrix={output}\ntotal={len(repos)}\n")
    print(f"Queued {len(repos)} EXISTING Wikipedia datasets; no new slugs created")
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
                # The exporter checksums the compressed source already; avoid
                # another full-file scan before decompression.
                if not digest:
                    digest = sha256(source)
                    if is_current(digest):
                        skipped += 1
                        print(f"SKIP already-current {repo}/{name}", flush=True)
                        continue
                result = export(source, working / "output", "parquet")
                if result["source_sha256"] != digest:
                    raise ValueError(f"Source LFS SHA256 mismatch: {path}")

                # One atomic HF commit per shard: no half-published Parquet
                # without the matching checksum manifest, and one fewer API commit.
                artifact = result["artifacts"]["parquet"]
                parquet_file = working / "output" / artifact["path"]
                manifest_file = working / "output" / "manifests" / f"{name}.parquet.json"
                operations = [
                    CommitOperationAdd(
                        path_in_repo=f"structured/{artifact['path']}",
                        path_or_fileobj=str(parquet_file),
                    ),
                    CommitOperationAdd(
                        path_in_repo=f"structured/manifests/{name}.parquet.json",
                        path_or_fileobj=str(manifest_file),
                    ),
                ]
                retry(f"commit Parquet + provenance {name}", lambda:
                      api.create_commit(repo_id=repo, repo_type="dataset",
                                        operations=operations,
                                        commit_message=f"Verified Parquet: {name}"))
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
    selector = sub.add_parser("plan")
    selector.add_argument("--only", default="",
                          help="Comma-separated existing dataset IDs or language slugs")
    b = sub.add_parser("build")
    b.add_argument("--repo", required=True)
    args = p.parse_args()
    if args.command == "plan":
        plan(args.only)
    else:
        build(args.repo)


if __name__ == "__main__":
    main()
