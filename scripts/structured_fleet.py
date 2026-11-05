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
import random
import re
import tempfile
import time
import urllib.error
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


def transient_error(exc: Exception) -> bool:
    """Retry throttling/transport failures, never permissions or bad manifests."""
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status is not None:
        return status in (408, 409, 425, 429) or 500 <= status < 600
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in (408, 409, 425, 429) or 500 <= exc.code < 600
    if isinstance(exc, (TimeoutError, ConnectionError, urllib.error.URLError)):
        return True
    try:
        import httpx
        import requests
        return isinstance(exc, (httpx.TransportError, requests.exceptions.ConnectionError,
                                requests.exceptions.Timeout))
    except ImportError:
        return False


def retry(label, callback, attempts=4):
    for n in range(attempts):
        try:
            return callback()
        except Exception as exc:
            if n == attempts - 1 or not transient_error(exc):
                raise
            delay = min(80, 8 * 2 ** n) + random.random() * 3
            print(f"Retrying transient {label} ({type(exc).__name__}, "
                  f"attempt {n + 2}/{attempts})", flush=True)
            time.sleep(delay)


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


def language_aliases(repos: list[str]) -> dict[str, set[str]]:
    """Resolve Wikimedia project IDs and legacy dataset slugs to published repos.

    Aliases come from the same source catalog and environment configuration
    used by the dataset publishers, rather than a separate hand-maintained list.
    An alias never creates or selects an unpublished dataset.
    """
    config = Path(__file__).resolve().parents[1] / "config"
    aliases: dict[str, set[str]] = {}

    def add(key: str, repo: str) -> None:
        if repo in repos:
            aliases.setdefault(key.lower(), set()).add(repo)

    for repo in repos:
        slug = repo.rsplit("/", 1)[-1].removeprefix("plaintext-wikipedia-full-")
        add(repo, repo)
        add(slug, repo)

    for language in json.loads(
        (config / "additional-languages.json").read_text(encoding="utf-8")
    )["languages"]:
        project = language["project"].lower()
        if not project.endswith("wiki"):
            raise ValueError(f"Invalid Wikimedia project: {project}")
        repo = f"{PREFIX}plaintext-wikipedia-full-{language['slug']}"
        add(project, repo)
        add(project.removesuffix("wiki"), repo)

    # Main language repos use these existing publication settings, rather
    # than entries in additional-languages.json.
    for line in (config / "datasets.env").read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"FULL_HF_REPO_([A-Z_]+)=(ffatty/[^\s#]+)", line.strip())
        if match:
            code, repo = match.groups()
            add(code.lower(), repo)
            add(f"{code.lower()}wiki", repo)

    # Simple English has a separate existing dataset and Wikimedia project.
    for alias in ("simple", "simplewiki", "simpleenglish"):
        add(alias, SIMPLE)
    return aliases


def select_repos(repos: list[str], only: str = "") -> list[str]:
    if only.strip():
        aliases = language_aliases(repos)
        selected: set[str] = set()
        for token in (part.strip().lower() for part in only.split(",")):
            if not token:
                continue
            matches = aliases.get(token, set())
            if not matches:
                raise ValueError(f"Unknown/unpublished Wikipedia language: {token}")
            if len(matches) != 1:
                raise ValueError(f"Ambiguous Wikipedia language: {token} -> "
                                 f"{', '.join(sorted(matches))}")
            selected.update(matches)
        if not selected:
            raise ValueError("No Wikipedia languages selected")
        repos = list(selected)
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
        raise RuntimeError(f"No source text archives in {repo}; nothing can be published")
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
                          help="Comma-separated Wikimedia codes (en, simple, shwiki), slugs or dataset IDs")
    b = sub.add_parser("build")
    b.add_argument("--repo", required=True)
    args = p.parse_args()
    if args.command == "plan":
        plan(args.only)
    else:
        build(args.repo)


if __name__ == "__main__":
    main()
