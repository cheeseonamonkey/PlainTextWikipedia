#!/usr/bin/env python3
"""Version existing Kaggle Wikipedia datasets with raw text and verified Parquet.

Keep the original archives and Parquet in every new dataset version, without
carrying forward old generated SQLite files. Skip oversized editions rather
than dropping raw archives or breaking existing dataset URLs.
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


def mirror(repo: str, max_mb: int, require_publish: bool = False):
    def skip(reason: str):
        if require_publish:
            raise RuntimeError(reason)
        print(f"::warning::{reason}", flush=True)

    if not re.fullmatch(r"ffatty/(?:plain-text-wikipedia-simpleenglish|plaintext-wikipedia-full-[a-z0-9-]+)", repo):
        raise ValueError("Unsupported Wikipedia repo")
    if max_mb < 300:
        raise ValueError("max-mb must be >=300")
    api = HfApi(token=os.getenv("HF_TOKEN") or None)
    files = {x.path: x for x in api.list_repo_tree(repo, repo_type="dataset", recursive=True)}
    sources = {p: x for p, x in files.items() if p.startswith("data/") and p.endswith(".txt.gz")}
    if not sources:
        skip(f"No source archives: {repo}")
        return

    dataset = repo
    planned = []
    manifests = []
    for source, info in sorted(sources.items()):
        digest = source_hash(info)
        if not digest:
            skip(f"{repo}: source {source} has no trusted LFS SHA256")
            return
        planned.append((source, digest, None))
        stem = Path(source).name.removesuffix(".txt.gz")
        manifest_file = f"structured/manifests/{stem}.parquet.json"
        if manifest_file not in files:
            skip(f"{repo}: missing {manifest_file}")
            return
        document = json.loads(Path(hf_hub_download(
            repo, filename=manifest_file, repo_type="dataset", token=api.token)).read_text())
        if document.get("source_sha256") != digest or not document.get("articles"):
            skip(f"{repo}: Parquet is stale or empty for {source}")
            return
        artifact = document["artifacts"]["parquet"]
        location = f"structured/{artifact['path']}"
        if location not in files:
            raise RuntimeError(f"Manifest points to missing file {location}")
        planned.append((location, artifact["sha256"], artifact["size_bytes"]))
        manifests.append(document)
    # Kaggle's latest dataset has a 50 top-level-file and 200GB total limit.
    # Two files per source: original gzip and Parquet, plus one compact manifest.
    if len(planned) + 1 > 49:
        skip(f"{repo}: {len(planned)} input files exceed safe Kaggle file budget")
        return
    total = sum(int(files[name].size) for name, _, _ in planned)
    # Preflight runner capacity before downloading the complete old corpus;
    # account for the HF cache, assembled stage, and transient Kaggle upload.
    available = shutil.disk_usage(tempfile.gettempdir()).free
    if total * 2 + 1_000_000_000 > available:
        skip(f"{repo}: need ~{(total * 2 + 1_000_000_000)/1e9:.1f} GB "
             f"temporary disk; only {available/1e9:.1f} GB free on runner")
        return
    if total > max_mb * 1024 * 1024:
        skip(f"{repo}: {total / 1048576:.0f} MiB exceeds {max_mb} MiB CI staging budget")
        return

    # Do not create new dataset IDs here: only version existing datasets.
    existing = run("status", dataset, "--format", "json", check=False)
    if existing.returncode:
        skip(f"{dataset}: no accessible existing Kaggle dataset")
        return
    try:
        status = json.loads(existing.stdout)
    except ValueError:
        raise RuntimeError(f"Kaggle status output malformed for {dataset}")
    if str(status.get("status", "")).lower() != "ready":
        skip(f"{dataset}: not READY; refusing to replace pending data")
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

        # Keep previous dataset versions intact, but stop carrying obsolete
        # generated SQLite artifacts into the current two-format release.
        obsolete = [p for p in stage.iterdir() if p.is_file() and
                    p.name.endswith((".sqlite", ".sqlite.zst", ".sqlite.json"))]
        for file in obsolete:
            file.unlink()
        if obsolete:
            print(f"Dropped {len(obsolete)} legacy SQLite files from new Kaggle version",
                  flush=True)

        # A repeated CI run should not re-upload an identical Kaggle version.
        provenance = {"version": 1, "source": f"https://huggingface.co/datasets/{repo}",
                      "archives": manifests}
        old_provenance = stage / "structured-manifest.json"
        if old_provenance.exists():
            try:
                if (not obsolete and json.loads(old_provenance.read_text()) == provenance
                        and all((stage / Path(location).name).is_file()
                                and (expected_bytes is None or
                                     (stage / Path(location).name).stat().st_size ==
                                     int(expected_bytes))
                                and sha256(stage / Path(location).name) == digest
                                for location, digest, expected_bytes in planned)):
                    print(f"SKIP {dataset}: verified original + Parquet already current")
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
        meta["subtitle"] = "Wikipedia articles · Original plaintext + Parquet"
        meta["description"] += (
            "\n\n## Choose a format\n"
            "- **Parquet (.parquet)**: Zstandard-compressed, ready for pandas, "
            "DuckDB and columnar analytics. Columns: `title`, `text`, "
            "`source_file`, `source_ordinal`.\n"
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
            skip(f"{repo}: {count} files after preserving existing Kaggle data")
            return
        print(f"Keeping {len(old_files) - len(obsolete)} prior file(s), "
              f"verifying {len(planned)} original/Parquet files "
              f"for {dataset} ({total / 1048576:.0f} MiB)", flush=True)
        run("version", "-p", str(stage), "-m",
            "Update verified Parquet and original Wikipedia text",
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
    p.add_argument("--require-publish", action="store_true",
                   help="Fail rather than silently skip an existing Kaggle dataset")
    args = p.parse_args()
    mirror(args.repo, args.max_mb, require_publish=args.require_publish)


if __name__ == "__main__":
    main()
