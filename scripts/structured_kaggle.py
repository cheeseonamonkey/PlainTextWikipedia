#!/usr/bin/env python3
"""Safely version existing Kaggle Wikipedia datasets with verified HF formats.

Never replace Kaggle's current file set with a structured-only upload. Skip
oversized editions rather than dropping raw archives or breaking a published URL.
Run after Kaggle authentication and successful/partial HF structured exports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def run(*args, check=True):
    command = ["kaggle", "datasets", *args]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if check and result.returncode:
        print(result.stdout[-1500:], flush=True)
        print(result.stderr[-2000:], flush=True)
        raise RuntimeError(f"Kaggle command failed: {args[0]} {args[1] if len(args) > 1 else ''}")
    return result


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_hash(item):
    lfs = getattr(item, "lfs", None)
    return (lfs.get("sha256") if isinstance(lfs, dict)
            else getattr(lfs, "sha256", None))


def mirror(repo: str, max_mb: int):
    if not re.fullmatch(r"ffatty/(?:plain-text-wikipedia-simpleenglish|plaintext-wikipedia-full-[a-z0-9-]+)", repo):
        raise ValueError("Unsupported Wikipedia repo")
    if max_mb < 300:
        raise ValueError("max-mb must be >=300")
    api = HfApi(token=os.getenv("HF_TOKEN") or None)
    files = {x.path: x for x in api.list_repo_tree(repo, repo_type="dataset", recursive=True)}
    sources = {p: x for p, x in files.items() if p.startswith("data/") and p.endswith(".txt.gz")}
    if not sources:
        print(f"SKIP no source archives: {repo}")
        return

    dataset = repo
    planned = []
    manifests = []
    for source, info in sorted(sources.items()):
        digest = source_hash(info)
        if not digest:
            print(f"::warning::SKIP {repo}: source {source} has no trusted LFS SHA256")
            return
        planned.append((source, digest, None))
        stem = Path(source).name.removesuffix(".txt.gz")
        for kind in ("parquet", "sqlite"):
            manifest_file = f"structured/manifests/{stem}.{kind}.json"
            if manifest_file not in files:
                print(f"::warning::SKIP {repo}: missing {manifest_file}")
                return
            document = json.loads(Path(hf_hub_download(repo, filename=manifest_file,
                              repo_type="dataset", token=api.token)).read_text())
            if document.get("source_sha256") != digest or not document.get("articles"):
                print(f"::warning::SKIP {repo}: {kind} is stale or empty for {source}")
                return
            artifact = document["artifacts"][kind]
            location = f"structured/{artifact['path']}"
            if location not in files:
                raise RuntimeError(f"Manifest points to missing file {location}")
            planned.append((location, artifact["sha256"], artifact["size_bytes"]))
            manifests.append(document)
    # Kaggle's latest dataset has a 50 top-level-file and 200GB total limit.
    # We flatten 16*3 source/format files plus one compact manifest.
    if len(planned) + 1 > 49:
        print(f"::warning::SKIP {repo}: {len(planned)} input files exceed safe Kaggle file budget")
        return
    total = sum(int(files[name].size) for name, _, _ in planned)
    if total > max_mb * 1024 * 1024:
        print(f"::warning::SKIP {repo}: {total / 1048576:.0f} MiB would exceed "
              f"{max_mb} MiB CI staging budget; HF remains published")
        return

    # Do not create new dataset IDs here: only version existing datasets.
    existing = run("status", dataset, "--format", "json", check=False)
    if existing.returncode:
        print(f"::warning::SKIP {dataset}: not an accessible existing Kaggle dataset")
        return
    try:
        status = json.loads(existing.stdout)
    except ValueError:
        raise RuntimeError(f"Kaggle status output malformed for {dataset}")
    if str(status.get("status", "")).lower() != "ready":
        print(f"::warning::SKIP {dataset}: not READY; refusing to replace pending data")
        return
    with tempfile.TemporaryDirectory(prefix="wiki-kaggle-") as tmp:
        root = Path(tmp)
        stage = root / "stage"
        stage.mkdir()
        # First retrieve EVERY file already on Kaggle; no silent data deletion.
        run("download", "-d", dataset, "-p", str(stage), "--unzip")
        old_files = [p for p in stage.iterdir() if p.is_file()]
        if not old_files:
            raise RuntimeError("Kaggle dataset download returned zero files")
        if any(p.is_dir() for p in stage.iterdir()):
            raise RuntimeError("Dataset contains directories; cannot safely flatten them")
        for location, digest, expected_bytes in planned:
            source = Path(hf_hub_download(repo, filename=location,
                                         repo_type="dataset", token=api.token,
                                         local_dir=root / "hf"))
            if sha256(source) != digest:
                raise RuntimeError(f"SHA256 mismatch for {location}")
            if expected_bytes is not None and source.stat().st_size != int(expected_bytes):
                raise RuntimeError(f"Size mismatch for {location}")
            shutil.copyfile(source, stage / Path(location).name)
        # Compact deterministic provenance document avoids 16+ extra top-level
        # files on Kaggle while preserving per-shard SHA256 and counts.
        (stage / "structured-manifest.json").write_text(
            json.dumps({"version": 1, "source": f"https://huggingface.co/datasets/{repo}",
                        "archives": manifests},
                       ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        meta = json.loads((Path(__file__).resolve().parents[1] /
                          ("kaggle-metadata.json" if "simpleenglish" in dataset
                           else "kaggle-metadata-full.json")).read_text())
        meta["id"] = dataset
        name = dataset.rsplit("-", 1)[-1].replace("-", " ").title()
        if "simpleenglish" not in dataset:
            meta["title"] = f"Plaintext Wikipedia ({name})"
            meta["description"] = meta["description"].replace("English Wikipedia", f"{name} Wikipedia")
        meta["description"] = meta["description"].replace(
            "__REPO_URL__", "https://github.com/cheeseonamonkey/PlainTextWikipedia"
        ).replace("__UPSTREAM_URL__", "https://github.com/daveshap/PlainTextWikipedia")
        meta["description"] += (
            "\n\nVersioned Parquet and compressed SQLite editions with independently "
            "verified source SHA256 hashes; includes the original plaintext archives. "
            "See structured-manifest.json and the maintained GitHub repository."
        )
        (stage / "dataset-metadata.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        count = len([p for p in stage.iterdir() if p.is_file() and p.name != "dataset-metadata.json"])
        if count > 50:
            print(f"::warning::SKIP {repo}: {count} files after preserving existing Kaggle data")
            return
        print(f"Preserving {len(old_files)} old file(s); publishing {len(planned)} source "
              f"and structured files to {dataset} ({total / 1048576:.0f} MiB)", flush=True)
        run("version", "-p", str(stage), "-m",
            "Add verified Parquet and SQLite Wikipedia editions", "--dir-mode", "skip",
            "--keep-tabular")
        print(f"VERSIONED https://www.kaggle.com/datasets/{dataset}")
        if dest := os.getenv("GITHUB_STEP_SUMMARY"):
            with open(dest, "a") as out:
                out.write(f"- Versioned https://www.kaggle.com/datasets/{dataset} "
                          f"({len(planned)} format/source files)\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True)
    p.add_argument("--max-mb", type=int, default=2500,
                   help="Upper bound for compressed source+structured staging")
    args = p.parse_args()
    mirror(args.repo, args.max_mb)


if __name__ == "__main__":
    main()
