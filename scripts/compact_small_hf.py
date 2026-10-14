#!/usr/bin/env python3
"""Consolidate undersized, already-published HF Wikipedia gzip archives.

Only works on tracked leaves belonging to the SAME original build root.
Never reconverts Wikipedia or assumes repacking repairs incomplete coverage.
All new file writes, manifest changes, and superseded-file deletions use one
Hugging Face commit per dataset. Invalid or missing manifests fail closed.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path


MIN_BYTES = 20 * 1024 * 1024
MAX_ARCHIVES = 64


def copy_records(sources: list[Path], output: Path) -> tuple[int, str]:
    """Merge complete title/body/blank records, exactly and in deterministic order."""
    try:
        from scripts.repack_shards import records
    except ModuleNotFoundError as exc:
        if exc.name != "scripts":
            raise
        from repack_shards import records

    digest = hashlib.sha256()
    count = 0
    seen_titles = set()
    with gzip.open(output, "wb", compresslevel=6) as writer:
        for source in sources:
            for record in records(source):
                title = record.split(b"\n", 1)[0]
                if title in seen_titles:
                    raise ValueError("Duplicate Wikipedia title across source archives")
                seen_titles.add(title)
                digest.update(record)
                writer.write(record)
                count += 1

    copied = hashlib.sha256()
    with gzip.open(output, "rb") as check:
        while block := check.read(1024 * 1024):
            copied.update(block)
    if copied.digest() != digest.digest():
        raise ValueError("Consolidation lost or modified article bytes")
    return count, digest.hexdigest()


def rewrite_root(root: dict, old: list[str], new: str | None,
                 count: int, size: int, sha256: str | None) -> None:
    oldset = set(old)
    leaves = root.get("successful_leaves", [])
    if not all(sum(leaf.get("file") == x for leaf in leaves) == 1 for x in old):
        raise ValueError("Recovery manifest doesn't track all source archives")
    surviving = [leaf for leaf in leaves if leaf.get("file") not in oldset]
    originals = [leaf for leaf in leaves if leaf.get("file") in oldset]
    if new is not None:
        new_leaf = {
            "file": new, "articles": count, "bytes": size,
            "segment": "storage-consolidated",
            "sources": [source for leaf in originals for source in leaf.get("sources", [])],
            "storage": {"merged_from": old, "uncompressed_sha256": sha256},
        }
        surviving.append(new_leaf)
    else:
        root["complete"] = False
        root.setdefault("failed_fragments", []).append({
            "segment": "empty-storage-cleanup",
            "error": "Verified empty gzip archives removed; no article records",
            "removed_files": old,
        })
    root["successful_leaves"] = surviving
    if "files" in root:
        root["files"] = [
            f for f in root["files"] if f not in oldset
        ] + ([new] if new is not None else [])


def refresh_coverage(manifest: dict) -> None:
    roots = manifest.get("roots", [])
    present = [int(r["root_shard"]) for r in roots if r.get("files")]
    incomplete = [int(r["root_shard"]) for r in roots if r.get("files") and not r.get("complete")]
    expected = int(manifest["expected_shards"])
    if len({int(r["root_shard"]) for r in roots}) != len(roots):
        raise ValueError("Duplicate root identifier")
    manifest["published_archives"] = sum(len(r.get("files", [])) for r in roots)
    manifest["present_shards"] = sorted(present)
    manifest["published_shards"] = len(present)
    manifest["missing_shards"] = sorted(set(range(1, expected + 1)) - set(present))
    manifest["incomplete_shards"] = sorted(incomplete)
    manifest["complete"] = (
        bool(roots) and not manifest["missing_shards"] and not incomplete
        and all(r.get("complete", False) for r in roots)
    )


def consolidate(repo_id: str, apply: bool = False,
                max_total_mb: int = 24, min_mb: int = 20) -> dict:
    if not re.fullmatch(r"ffatty/plaintext-wikipedia-full-[a-z0-9-]+", repo_id):
        raise ValueError("Dataset repo is not an expected Wikipedia dataset")
    from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd, CommitOperationDelete

    api = HfApi(token=os.getenv("HF_TOKEN") or None)
    try:
        entries = list(api.list_repo_tree(repo_id=repo_id, repo_type="dataset",
                                          path_in_repo="data", recursive=False))
    except Exception as exc:
        return {"repo": repo_id, "status": "skip", "reason": f"Data listing unavailable: {type(exc).__name__}"}
    sizes = {e.path: getattr(e, "size", None) for e in entries if e.path.endswith(".txt.gz")}
    total = sum(v or 0 for v in sizes.values())
    if not sizes or any(v is None for v in sizes.values()):
        return {"repo": repo_id, "status": "skip", "reason": "No archives or unknown sizes"}
    if total > max_total_mb * 1024 * 1024:
        return {"repo": repo_id, "status": "skip", "reason": "Corpus above small-corpus cap",
                "bytes": total}
    remote = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
    if "data/build-manifest.json" not in remote:
        return {"repo": repo_id, "status": "skip", "reason": "No coverage manifest"}

    with tempfile.TemporaryDirectory(prefix="hf-wiki-compact-") as tmp:
        work = Path(tmp)
        original = json.loads(Path(hf_hub_download(
            repo_id=repo_id, repo_type="dataset", filename="data/build-manifest.json"
        )).read_text(encoding="utf-8"))
        result = json.loads(json.dumps(original))
        roots = result.get("roots", [])
        if not roots:
            return {"repo": repo_id, "status": "skip", "reason": "Manifest has no root records"}
        accounted = [f"data/{leaf['file']}" for root in roots for leaf in root.get("successful_leaves", [])]
        if len(set(accounted)) != len(accounted) or set(accounted) != set(sizes):
            return {"repo": repo_id, "status": "skip", "reason": "Archive/manifest mismatch"}
        ops = []
        changes = []
        project = original["language"]
        total_roots = int(original["expected_shards"])
        for root in roots:
            leaves = root.get("successful_leaves", [])
            old = sorted(leaf["file"] for leaf in leaves)
            if not old:
                continue
            group_size = sum(sizes["data/" + name] for name in old)
            # Do not rewrite any full-size archive or break distinct source roots.
            if len(old) < 2 or group_size > min_mb * 1024 * 1024:
                continue
            if len(old) > MAX_ARCHIVES:
                continue
            files = [Path(hf_hub_download(repo_id=repo_id, repo_type="dataset",
                                          filename="data/" + name)) for name in old]
            name = f"{project}_{int(root['root_shard']):02d}_of_{total_roots:02d}.txt.gz"
            if "data/" + name in sizes or name in old:
                name = f"{project}_{int(root['root_shard']):02d}_of_{total_roots:02d}_compact.txt.gz"
            if "data/" + name in remote:
                raise ValueError("Would overwrite an existing remote gzip archive")
            dest = work / name
            count, digest = copy_records(files, dest)
            if count:
                ops.append(CommitOperationAdd(path_in_repo="data/" + name, path_or_fileobj=str(dest)))
                rewrite_root(root, old, name, count, dest.stat().st_size, digest)
            else:
                rewrite_root(root, old, None, 0, 0, None)
            for old_name in old:
                ops.append(CommitOperationDelete(path_in_repo="data/" + old_name))
            changes.append({"root": root["root_shard"], "old_files": len(old),
                            "new_files": int(bool(count)), "articles": count,
                            "uncompressed_sha256": digest if count else None})

        if not changes:
            return {"repo": repo_id, "status": "skip",
                    "reason": "No same-root groups below compressed minimum"}
        refresh_coverage(result)
        result["storage_consolidation"] = {
            "minimum_split_bytes": min_mb * 1024 * 1024,
            "verified": "Full article-record SHA-256 reconstruction",
            "groups": changes,
        }
        output = work / "build-manifest.json"
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        ops.append(CommitOperationAdd(path_in_repo="data/build-manifest.json",
                                      path_or_fileobj=str(output)))
        if not apply:
            return {"repo": repo_id, "status": "ready", "changes": changes,
                    "published_archives_after": result["published_archives"]}
        if not os.getenv("HF_TOKEN"):
            raise RuntimeError("HF_TOKEN required for Hub changes")
        api.create_commit(repo_id=repo_id, repo_type="dataset", operations=ops,
                          commit_message="Consolidate tiny Wikipedia archives and remove verified empty data")
        return {"repo": repo_id, "status": "published", "changes": changes,
                "published_archives_after": result["published_archives"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--max-total-mb", type=int, default=24)
    parser.add_argument("--min-mb", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(consolidate(args.repo, args.apply, args.max_total_mb,
                                 args.min_mb), indent=2))


if __name__ == "__main__":
    main()
