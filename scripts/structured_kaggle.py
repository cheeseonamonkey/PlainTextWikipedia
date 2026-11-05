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


def language_name(repo: str) -> str:
    slug = repo.rsplit("/", 1)[1].removeprefix("plaintext-wikipedia-full-")
    if slug == "plain-text-wikipedia-simpleenglish":
        return "Simple English"
    catalog = Path(__file__).resolve().parents[1] / "config" / "additional-languages.json"
    entries = json.loads(catalog.read_text(encoding="utf-8"))["languages"]
    return next((item["name"] for item in entries if item["slug"] == slug),
                slug.replace("-", " ").title())


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
        # A repeated CI run should not re-upload an identical Kaggle version.
        provenance = {"version": 1, "source": f"https://huggingface.co/datasets/{repo}",
                      "archives": manifests}
        old_provenance = stage / "structured-manifest.json"
        if old_provenance.exists():
            try:
                if json.loads(old_provenance.read_text()) == provenance:
                    print(f"SKIP {dataset}: already contains these exact verified exports")
                    return
            except (OSError, ValueError):
                pass

        for location, digest, expected_bytes in planned:
            target = stage / Path(location).name
            if target.is_file() and (expected_bytes is None
                    or target.stat().st_size == int(expected_bytes)) and sha256(target) == digest:
                continue  # Preserve already identical Kaggle files, avoiding HF downloads.
            source = Path(hf_hub_download(repo, filename=location,
                                         repo_type="dataset", token=api.token,
                                         local_dir=root / "hf"))
            if sha256(source) != digest:
                raise RuntimeError(f"SHA256 mismatch for {location}")
            if expected_bytes is not None and source.stat().st_size != int(expected_bytes):
                raise RuntimeError(f"Size mismatch for {location}")
            target.unlink(missing_ok=True)
            try:
                os.link(source, target)  # Same-device hardlink avoids doubling disk I/O.
            except OSError:
                shutil.copyfile(source, target)

        # Compact provenance avoids 16+ extra top-level manifest files.
        old_provenance.write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        meta = json.loads((Path(__file__).resolve().parents[1] /
                          ("kaggle-metadata.json" if "simpleenglish" in dataset
                           else "kaggle-metadata-full.json")).read_text())
        meta["id"] = dataset
        name = language_name(repo)
        meta["title"] = f"Plaintext Wikipedia ({name})"
        if "simpleenglish" not in dataset:
            meta["description"] = meta["description"].replace(
                "English Wikipedia", f"{name} Wikipedia")
        meta["description"] = meta["description"].replace(
            "__REPO_URL__", "https://github.com/cheeseonamonkey/PlainTextWikipedia"
        ).replace("__UPSTREAM_URL__", "https://github.com/daveshap/PlainTextWikipedia")
        meta["licenses"] = [{"name": "CC-BY-SA-4.0"}]
        meta["subtitle"] = "Wikipedia article corpus · Parquet, searchable SQLite, original text"
        meta["description"] += (
            "\n\n## Choose a format\n"
            "- **Parquet (.parquet)**: Zstandard-compressed, ready for pandas, "
            "DuckDB and columnar analytics. Columns: `title`, `text`, "
            "`source_file`, `source_ordinal`.\n"
            "- **SQLite (.sqlite.zst)**: downloadable SQL database; decompress "
            "with `zstd -d FILE.sqlite.zst`. Small shards include FTS5 search; "
            "see their manifests for `fts5`.\n"
            "- **Original (.txt.gz)**: retained verbatim. Each record uses title, "
            "single-line text, blank line.\n\n"
            "## Provenance and limitations\n"
            "All structured files are verified against original source SHA256 "
            "hashes. The `structured-manifest.json` file includes article "
            "counts and checksums. These exports **do not preserve Wikipedia "
            "page IDs**. Coverage can be partial; see the HF source build manifest "
            "before asserting completeness.\n\n"
            f"Source inventory: https://huggingface.co/datasets/{repo}\n"
            "Source articles: Wikimedia contributors (CC BY-SA 4.0); "
            "converter source: https://github.com/cheeseonamonkey/PlainTextWikipedia. "
            "Observe Wikimedia attribution and share-alike requirements."
        )
        def describe(file):
            name = file.name
            if name.endswith(".parquet"):
                return "Columnar UTF-8 Wikipedia title/text, source archive and ordinal; zstd compressed."
            if name.endswith(".sqlite.zst"):
                return "Zstandard-compressed SQLite article database; inspect manifests for FTS5."
            if name.endswith(".txt.gz"):
                return "Original gzip plaintext archive; title, text, blank line per article."
            if name == "structured-manifest.json":
                return "Per-shard article counts, SHA256 digests, and exact source provenance."
            return None
        meta["resources"] = [
            {"path": item.name, "description": desc}
            for item in sorted(stage.iterdir())
            if item.is_file() and (desc := describe(item))
        ]
        (stage / "dataset-metadata.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        count = len([p for p in stage.iterdir() if p.is_file() and p.name != "dataset-metadata.json"])
        if count > 50:
            print(f"::warning::SKIP {repo}: {count} files after preserving existing Kaggle data")
            return
        print(f"Preserving {len(old_files)} old file(s); publishing {len(planned)} source "
              f"and structured files to {dataset} ({total / 1048576:.0f} MiB)", flush=True)
        run("version", "-p", str(stage), "-m",
            "Add verified Parquet and SQLite Wikipedia editions",
            "--dir-mode", "skip", "--keep-tabular")
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
