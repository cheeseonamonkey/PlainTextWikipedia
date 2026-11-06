#!/usr/bin/env python3
"""Version an existing Kaggle Wikipedia dataset with canonical article features.

Preserve unrelated existing Kaggle files, avoid creating new slugs, and refuse
unsafe staging. Full-text archives remain available in prior Kaggle versions.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import shutil
import time
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

try:
    from scripts.structured_kaggle import language_name, run, sha256, source_hash
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from structured_kaggle import language_name, run, sha256, source_hash


def kaggle_inventory(repo: str) -> dict[str, int]:
    result = run("files", repo, "--csv", "--page-size", "100")
    # CLI may prefix a pagination notice even in --csv mode.
    lines = result.stdout.splitlines()
    start = next((index for index, line in enumerate(lines)
                  if line.lstrip().startswith("name,size,")), None)
    if start is None:
        raise RuntimeError("Unrecognized Kaggle file inventory")
    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    if not reader.fieldnames or not {"name", "size"}.issubset(reader.fieldnames):
        raise RuntimeError("Unrecognized Kaggle file inventory")
    result = {row["name"]: int(row["size"]) for row in reader if row.get("name")}
    if not result:
        raise RuntimeError("Kaggle dataset inventory is empty")
    return result


def feature_inventory(api, repo: str, sqlite_cap: int) -> tuple[dict, list[tuple]]:
    tree = {item.path: item for item in api.list_repo_tree(
        repo, repo_type="dataset", recursive=True)}
    manifest = "features/manifest.json"
    if manifest not in tree:
        raise RuntimeError("Article features are not yet published to Hugging Face")
    source = Path(hf_hub_download(repo_id=repo, filename=manifest,
                                  repo_type="dataset", token=api.token))
    document = json.loads(source.read_text())
    if document.get("primary_table") != "articles.parquet":
        raise RuntimeError("Feature release does not declare the canonical article table")
    files = document.get("files", {})
    required = ("articles.parquet", "corpus-profile.json", "vocabulary.json.gz",
                "fasttext.bin.zst")
    if any(name not in files for name in required):
        raise RuntimeError("Incomplete feature release; no partial Kaggle publication")
    selected = []
    for name in required + ("articles.sqlite.zst",):
        if name not in files:
            continue
        meta = files[name]
        if name == "articles.sqlite.zst" and int(meta["bytes"]) > sqlite_cap:
            print(f"SQLite sidecar omitted from Kaggle: exceeds {sqlite_cap // 1048576} MiB cap", flush=True)
            continue
        remote = name if name == "articles.parquet" else f"features/{name}"
        item = tree.get(remote)
        if item is None or item.size != int(meta["bytes"]):
            raise RuntimeError(f"Missing or wrong-size HF feature artifact: {remote}")
        digest = source_hash(item)
        if digest and digest != meta["sha256"]:
            raise RuntimeError(f"Wrong feature SHA256 in HF: {remote}")
        output = "articles-features.sqlite.zst" if name.endswith(".sqlite.zst") else name
        selected.append((remote, output, meta))
    # Kaggle's dataset is a flat namespace; use a descriptive standalone manifest.
    return document, selected


def same_release(repo: str, manifest: dict, files: dict[str, int],
                 selected: list[tuple], folder: Path) -> bool:
    if "feature-manifest.json" not in files:
        return False
    if any(name not in files or files[name] != int(meta["bytes"])
           for _, name, meta in selected):
        return False
    result = run("download", "-d", repo, "-f", "feature-manifest.json",
                 "-p", str(folder), "--unzip", check=False)
    if result.returncode:
        return False
    try:
        return json.loads((folder / "feature-manifest.json").read_text()) == manifest
    except (OSError, ValueError):
        return False


def mirror(repo: str, max_sqlite_mb: int = 512) -> str:
    if not re.fullmatch(
        r"ffatty/(?:plain-text-wikipedia-simpleenglish|plaintext-wikipedia-full-[a-z0-9-]+)",
        repo,
    ):
        raise ValueError("Existing published Wikipedia dataset required")
    if not 0 <= max_sqlite_mb <= 16384:
        raise ValueError("Invalid SQLite size cap")
    status = run("status", repo, "--format", "json")
    previous_status = json.loads(status.stdout)
    if previous_status.get("status") != "ready":
        raise RuntimeError("Kaggle dataset not READY; refusing concurrent update")
    previous_version = int(previous_status.get("current_version_number") or 0)
    api = HfApi(token=os.getenv("HF_TOKEN") or None)
    manifest, selected = feature_inventory(api, repo, max_sqlite_mb * 1024 * 1024)
    previous = kaggle_inventory(repo)
    with tempfile.TemporaryDirectory(prefix="wiki-feature-kaggle-") as tmp:
        root = Path(tmp)
        if same_release(repo, manifest, previous, selected, root):
            return f"SKIP {repo}: Kaggle features already current"
        # Existing Kaggle content is preserved. Size budget includes Kaggle
        # archive download/extraction plus feature blob caching/staging.
        old = sum(previous.values())
        new = sum(int(meta["bytes"]) for _, _, meta in selected)
        available = shutil.disk_usage(tmp).free
        required = 2 * (old + new) + 1024**3
        if required > available:
            raise RuntimeError(f"Kaggle feature staging requires ~{required / 1e9:.1f} GB; "
                               f"only {available / 1e9:.1f} GB free. "
                               "Use a larger runner, not an incomplete update")
        stage = root / "dataset"
        stage.mkdir()
        run("download", "-d", repo, "-p", str(stage), "--unzip")
        if any(p.is_dir() for p in stage.iterdir()) or not any(stage.iterdir()):
            raise RuntimeError("Downloaded Kaggle dataset has unexpected layout")
        # Remove old generated outputs only from the NEXT dataset version;
        # historical Kaggle versions stay accessible.
        for file in stage.iterdir():
            if re.fullmatch(r"(?:simplewiki|enwiki)_\d+_of_\d+\.parquet", file.name):
                file.unlink()
        selected_names = {target_name for _, target_name, _ in selected}
        if "articles-features.sqlite.zst" not in selected_names:
            (stage / "articles-features.sqlite.zst").unlink(missing_ok=True)
        for remote, target_name, meta in selected:
            source = Path(hf_hub_download(repo_id=repo, filename=remote,
                                          repo_type="dataset", token=api.token))
            if source.stat().st_size != int(meta["bytes"]) or sha256(source) != meta["sha256"]:
                raise RuntimeError(f"HF artifact checksum mismatch: {remote}")
            target = stage / target_name
            target.unlink(missing_ok=True)
            try:
                os.link(source, target)
            except OSError:
                shutil.copyfile(source, target)
        (stage / "feature-manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        metadata = Path(__file__).resolve().parents[1] / (
            "kaggle-metadata.json" if "simpleenglish" in repo else "kaggle-metadata-full.json")
        card = json.loads(metadata.read_text())
        card.update(id=repo, title=f"Plaintext Wikipedia ({language_name(repo)})",
                    subtitle="Original Wikipedia articles + enriched Parquet features",
                    licenses=[{"name": "CC-BY-SA-4.0"}])
        card["description"] = (
            f"Original Wikipedia texts and one complete enriched article table. "
            f"`articles.parquet` includes per-article text, embeddings, lexical "
            f"measurements, VADER/readability where applicable, and intermediates. "
            f"`corpus-profile.json` summarizes the collection; `vocabulary.json.gz` "
            f"records corpus-wide TF/DF/IDF; `feature-manifest.json` gives "
            f"source/checksums. Full SQLite sidecar available on HF when too large "
            f"for this Kaggle version. Original files and older Kaggle versions "
            f"remain accessible. Source: https://huggingface.co/datasets/{repo}. "
            f"Wikipedia text CC BY-SA 4.0; contributors retain attribution.")
        (stage / "dataset-metadata.json").write_text(
            json.dumps(card, ensure_ascii=False, indent=2) + "\n")
        if len(list(stage.iterdir())) - 1 > 50:
            raise RuntimeError("Kaggle file limit exceeded; no update attempted")
        run("version", "-p", str(stage), "-m",
            "Add complete article feature Parquet and corpus profile", "--dir-mode", "skip",
            "--keep-tabular")
        deadline = time.monotonic() + 3600
        while time.monotonic() < deadline:
            status = json.loads(run("status", repo, "--format", "json").stdout)
            version = int(status.get("current_version_number") or 0)
            if version > previous_version and status.get("status") == "ready":
                return f"READY Kaggle version {version}: https://www.kaggle.com/datasets/{repo}"
            if status.get("status") in ("error", "failed", "failure"):
                raise RuntimeError("New Kaggle feature version failed processing")
            time.sleep(12)
        raise TimeoutError("Kaggle upload accepted but new feature version not confirmed READY")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--max-sqlite-mb", type=int, default=512)
    args = parser.parse_args()
    print(mirror(args.repo, args.max_sqlite_mb), flush=True)


if __name__ == "__main__":
    main()
