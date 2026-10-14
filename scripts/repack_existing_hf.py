#!/usr/bin/env python3
"""Safely migrate already published, one-file Wikipedia datasets on Hugging Face.

This is a STORAGE migration, not an XML rebuild. Download old gzips once,
split with SHA-256-verified round-robin reconstruction, upload ALL replacements,
update coverage references where present, and only then delete old archives.
Keep the existing factual completeness status untouched.

Run:
  HF_TOKEN=... python scripts/repack_existing_hf.py \
    ffatty/plaintext-wikipedia-full-atikamekw --apply
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

try:
    from scripts.repack_shards import repack
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from repack_shards import repack


def migrate_manifest(manifest: dict, replacements: dict[str, list[dict]]) -> dict:
    """Retain all source coverage/failure claims; change only storage references."""
    result = json.loads(json.dumps(manifest))
    for root in result.get("roots", []):
        new_leaves = []
        for leaf in root.get("successful_leaves", []):
            source = leaf.get("file")
            if source not in replacements:
                new_leaves.append(leaf)
                continue
            for part in replacements[source]:
                new_leaves.append({**leaf, **part})
        if "successful_leaves" in root:
            root["successful_leaves"] = new_leaves
        if "files" in root:
            root["files"] = [
                item["file"] for source in root["files"]
                for item in replacements.get(source, [{"file": source}])
            ]
    if "published_archives" in result:
        result["published_archives"] = sum(
            len(replacements.get(source, [])) or 1
            for source in result.get("_original_storage_files", [])
        ) if "_original_storage_files" in result else result["published_archives"]
    result.pop("_original_storage_files", None)
    result["storage_repacked"] = True
    result["storage_repack_note"] = (
        "Original gzip article records were SHA-256 verified before and after "
        "repacking. Source completeness and missing-range claims are unchanged."
    )
    return result


def transform_manifest(manifest: dict, replacements: dict[str, list[dict]],
                       old_filenames: list[str]) -> dict:
    updated = migrate_manifest(manifest, replacements)
    updated["published_archives"] = sum(len(replacements.get(x, [])) or 1 for x in old_filenames)
    return updated


def repack_local(directory: Path, target: int = 16) -> dict:
    """Pure local portion. Inputs are already downloaded data/*.txt.gz files."""
    initial = sorted(p for p in directory.glob("*.txt.gz") if p.is_file())
    if not 1 <= len(initial) < target:
        return {"changed": False, "old_files": [], "new_files": [], "replacements": {}}
    for i, source in enumerate(initial, 1):
        # Temporary manifests exist only to reuse the same validated repacker
        # as the main publication pipeline. They are NEVER published.
        report = {
            "language": "storage-only", "root_shard": i,
            "planned_shards": len(initial), "complete": True,
            "successful_leaves": [{"file": source.name}],
        }
        (directory / f"storage-{i:03d}.manifest.json").write_text(json.dumps(report))
    result = repack(directory, target)
    replacements = {}
    files = sorted(directory.glob("*.txt.gz"))
    for source in initial:
        successors = []
        for file in files:
            if file.name == source.name:
                successors.append({"file": file.name})
            elif file.name.startswith(source.name.removesuffix(".txt.gz") + "_part"):
                # Read the synthetically generated local leaf metadata.
                successors.append({"file": file.name})
        replacements[source.name] = successors
    for path in directory.glob("storage-*.manifest.json"):
        data = json.loads(path.read_text())
        for leaf in data.get("successful_leaves", []):
            partition = leaf.get("partition", {})
            source = partition.get("source_archive")
            if source in replacements:
                for item in replacements[source]:
                    if item["file"] == leaf["file"]:
                        item.update(leaf)
        path.unlink()
    return {"changed": result["repacked"] > 0,
            "old_files": [p.name for p in initial],
            "new_files": [p.name for p in files],
            "replacements": replacements}


def migrate(repo_id: str, apply: bool = False, target: int = 16,
            max_source_mb: int = 750) -> dict:
    if not re.fullmatch(r"ffatty/[a-z0-9._-]+", repo_id):
        raise ValueError("Only the configured ffatty dataset namespace is allowed")
    if not 2 <= target <= 64:
        raise ValueError("Invalid target shard count")
    from huggingface_hub import HfApi, hf_hub_download
    token = os.environ.get("HF_TOKEN", "")
    if apply and not token:
        raise RuntimeError("HF_TOKEN required for writes")
    api = HfApi(token=token or None)
    from huggingface_hub.errors import RepositoryNotFoundError, HfHubHTTPError
    try:
        remote = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
    except RepositoryNotFoundError:
        return {"repo": repo_id, "status": "skip", "reason": "Dataset not published yet"}
    except HfHubHTTPError as exc:
        if getattr(getattr(exc, "response", None), "status_code", None) == 404:
            return {"repo": repo_id, "status": "skip", "reason": "Dataset not published yet"}
        raise
    originals = sorted(x for x in remote if x.startswith("data/") and x.endswith(".txt.gz"))
    if not originals:
        return {"repo": repo_id, "status": "skip", "reason": "No published gzip archives"}
    if len(originals) >= target or any("_part" in name for name in originals):
        return {"repo": repo_id, "status": "skip",
                "reason": "Already split (or contains existing part archives)"}

    # Refuse unknown massive datasets before downloading any archive.
    tree = list(api.list_repo_tree(repo_id=repo_id, repo_type="dataset", recursive=True))
    sizes = {x.path: getattr(x, "size", None) for x in tree}
    if any(sizes.get(x) is None for x in originals):
        return {"repo": repo_id, "status": "skip", "reason": "Source sizes unavailable"}
    total = sum(int(sizes[x]) for x in originals)
    if total > max_source_mb * 1024 * 1024:
        return {"repo": repo_id, "status": "skip", "reason": f"Source > {max_source_mb}MB",
                "source_bytes": total}

    with tempfile.TemporaryDirectory(prefix="wiki-hf-repack-") as working:
        root = Path(working)
        for path in originals:
            local = Path(hf_hub_download(repo_id=repo_id, filename=path,
                                        repo_type="dataset", token=token or None))
            shutil.copyfile(local, root / Path(path).name)
        generated = repack_local(root, target)
        if not generated["changed"]:
            return {"repo": repo_id, "status": "skip", "reason": "Not enough article records"}
        replaced = generated["replacements"]
        new_names = [name for name in generated["new_files"]
                     if name not in generated["old_files"]]
        if not new_names:
            raise RuntimeError("No new valid split archives produced")

        # Include a separate storage-level mapping even for old releases without
        # an article-coverage manifest. Never invent missing-article coverage.
        storage = {
            "format": "gzip-article-record-repack-v1",
            "source_archives": generated["old_files"],
            "published_archives": generated["new_files"],
            "replacements": replaced,
            "verified": "SHA-256 article-record reconstruction",
        }
        (root / "storage-repack.json").write_text(json.dumps(storage, indent=2) + "\n")
        updated_manifest = None
        if "data/build-manifest.json" in remote:
            original_manifest = Path(hf_hub_download(
                repo_id=repo_id, filename="data/build-manifest.json",
                repo_type="dataset", token=token or None))
            updated_manifest = transform_manifest(
                json.loads(original_manifest.read_text(encoding="utf-8")),
                replaced, generated["old_files"])
            (root / "build-manifest.json").write_text(
                json.dumps(updated_manifest, indent=2, ensure_ascii=False) + "\n")
        summary = {
            "repo": repo_id, "status": "ready" if not apply else "publishing",
            "old_archives": len(originals), "new_archives": len(generated["new_files"]),
            "source_bytes": total, "verified": True,
        }
        if not apply:
            return summary
        # Upload ALL replacements before changing manifests or deleting sources.
        def retry(label, action):
            for n in range(4):
                try:
                    return action()
                except Exception as exc:
                    if n == 3:
                        raise
                    delay = 10 * (2 ** n)
                    print(f"Retrying {label} in {delay}s: {exc}", flush=True)
                    time.sleep(delay)
        for name in new_names:
            retry(name, lambda name=name: api.upload_file(
                repo_id=repo_id, repo_type="dataset",
                path_or_fileobj=str(root / name), path_in_repo=f"data/{name}",
                commit_message=f"Add lossless repacked shard {name}"))
        retry("storage-repack.json", lambda: api.upload_file(
            repo_id=repo_id, repo_type="dataset",
            path_or_fileobj=str(root / "storage-repack.json"),
            path_in_repo="data/storage-repack.json",
            commit_message="Document lossless gzip shard migration"))
        if updated_manifest is not None:
            retry("build-manifest.json", lambda: api.upload_file(
                repo_id=repo_id, repo_type="dataset",
                path_or_fileobj=str(root / "build-manifest.json"),
                path_in_repo="data/build-manifest.json",
                commit_message="Update shard filenames without changing factual coverage"))
        # Delete original archives last. If an upload fails, source remains.
        for old in generated["old_files"]:
            if old not in generated["new_files"]:
                retry(old, lambda old=old: api.delete_file(
                    repo_id=repo_id, repo_type="dataset", path_in_repo=f"data/{old}",
                    commit_message=f"Remove superseded archive {old}"))
        summary["status"] = "published"
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--target", type=int, default=16)
    parser.add_argument("--max-source-mb", type=int, default=750)
    args = parser.parse_args()
    print(json.dumps(migrate(args.repo, args.apply, args.target, args.max_source_mb), indent=2))


if __name__ == "__main__":
    main()
