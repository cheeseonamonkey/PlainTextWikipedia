#!/usr/bin/env python3
"""Publish one validated Wikimedia edition to HF and Kaggle independently.

Both publishers reuse one local staging directory and execute concurrently.
Incomplete destinations are reported, never conflated with source completeness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def card(language: str, name: str, repo: str, manifest: dict,
         source_repo: str, upstream_repo: str) -> str:
    template = (ROOT / "config/hf-dataset-card.md").read_text(encoding="utf-8")
    code = "en" if language == "simplewiki" else language.removesuffix("wiki")
    # HF's dataset-card language IDs are at most three characters. Unsupported
    # project variants retain the language in pretty_name, but not language YAML.
    if not re.fullmatch(r"[a-z]{2,3}", code):
        template = template.replace("language:\n- __LANG_CODE__\n", "")
    replacements = {
        "__LANG_PROJECT__": language,
        "__LANG_CODE__": code,
        "__LANG_NAME__": name,
        "__HF_REPO__": repo,
        "__SOURCE_URL__": f"https://dumps.wikimedia.org/{language}/latest",
        "__REPO_URL__": source_repo,
        "__UPSTREAM_URL__": upstream_repo,
    }
    for before, after in replacements.items():
        template = template.replace(before, after)
    status = "COMPLETE" if manifest["complete"] else "PARTIAL"
    warning = (f"**Build status: {status} — {manifest['published_shards']}/"
               f"{manifest['expected_shards']} root shards recovered "
               f"({manifest['published_archives']} leaf archives).** ")
    if not manifest["complete"]:
        warning += (f"Missing roots: {manifest['missing_shards']}; incomplete roots: "
                    f"{manifest['incomplete_shards']}. See data/build-manifest.json. ")
    else:
        warning += "All planned page ranges recovered. "
    return template.replace("## Format", warning + "\n\n## Format", 1)


def metadata(name: str, dataset: str, manifest: dict,
             source_repo: str, upstream_repo: str) -> dict:
    template = json.loads((ROOT / "kaggle-metadata-full.json").read_text())
    template["id"] = dataset
    template["title"] = f"Plaintext Wikipedia ({name})"
    template["description"] = template["description"].replace("English", name)
    template["description"] = (template["description"]
                               .replace("__REPO_URL__", source_repo)
                               .replace("__UPSTREAM_URL__", upstream_repo))
    template["description"] += (
        f"\n\nBuild status: {'complete' if manifest['complete'] else 'PARTIAL'} "
        f"({manifest['published_shards']}/{manifest['expected_shards']} root shards; "
        f"{manifest['published_archives']} leaf files). "
        f"Missing roots: {manifest['missing_shards']}; "
        f"incomplete roots: {manifest['incomplete_shards']}. "
        "See build-manifest.json for exact coverage."
    )
    return template


def retry(description, action, attempts=5):
    for attempt in range(attempts):
        try:
            return action()
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            # Authentication, malformed metadata and other 4xx errors will not
            # become valid by retrying. Rate limits (429) are transient.
            if (isinstance(exc, (ValueError, TypeError)) or
                    status is not None and 400 <= status < 500 and status != 429 or
                    attempt == attempts - 1):
                raise
            delay = min(120, 10 * 2 ** attempt)
            print(f"Retrying {description} in {delay}s: {type(exc).__name__}", flush=True)
            time.sleep(delay)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def publish_hf(directory: Path, language: str, name: str,
               dataset: str, manifest: dict, source_repo: str,
               upstream_repo: str, token: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=dataset, repo_type="dataset", exist_ok=True)
    existing = set(api.list_repo_files(dataset, repo_type="dataset"))
    # LFS checksums let reruns skip files already committed by a partial run.
    hashes = {}
    try:
        for entry in api.list_repo_tree(repo_id=dataset, repo_type="dataset",
                                        path_in_repo="data", expand=True):
            info = getattr(entry, "lfs", None)
            if info and getattr(info, "sha256", None):
                hashes[entry.path] = info.sha256
    except Exception as exc:
        print(f"::warning::HF remote checksum inventory unavailable: {type(exc).__name__}",
              flush=True)

    archives = sorted(directory.glob("*.txt.gz"))
    wanted = {f"data/{path.name}" for path in archives}
    for path in archives:
        remote = f"data/{path.name}"
        if hashes.get(remote) == sha256(path):
            print(f"HF SKIP existing {path.name}", flush=True)
            continue
        retry(path.name, lambda path=path, remote=remote: api.upload_file(
            repo_id=dataset, repo_type="dataset", path_or_fileobj=str(path),
            path_in_repo=remote, commit_message=f"Update {name} {path.name}"))
    retry("build-manifest.json", lambda: api.upload_file(
        repo_id=dataset, repo_type="dataset", path_or_fileobj=str(directory / "build-manifest.json"),
        path_in_repo="data/build-manifest.json", commit_message=f"Update {name} coverage"))
    readme = directory / "README.md"
    readme.write_text(card(language, name, dataset, manifest, source_repo, upstream_repo),
                      encoding="utf-8")
    retry("README.md", lambda: api.upload_file(
        repo_id=dataset, repo_type="dataset", path_or_fileobj=str(readme),
        path_in_repo="README.md", commit_message=f"Update {name} dataset card"))
    # Partial builds must never delete known-good archives from an older run.
    if manifest["complete"]:
        for old in sorted(existing - wanted):
            if old.startswith("data/") and old.endswith(".txt.gz"):
                retry(old, lambda old=old: api.delete_file(
                    repo_id=dataset, repo_type="dataset", path_in_repo=old,
                    commit_message=f"Remove superseded {old}"))
    print(f"HF PUBLISHED {len(archives)} archives: https://huggingface.co/datasets/{dataset}", flush=True)


def publish_kaggle(directory: Path, name: str, dataset: str,
                   manifest: dict, source_repo: str, upstream_repo: str,
                   token: str) -> None:
    from scripts.kaggle_auth import verify_and_configure
    from scripts.kaggle_backfill import run_kaggle

    verify_and_configure(token, dataset.split("/", 1)[0])
    # A Kaggle version replaces the dataset's entire file set. Never erase
    # independent structured exports by uploading only fresh text archives.
    import subprocess
    inventory = subprocess.run(["kaggle", "datasets", "files", dataset, "--csv"],
                               capture_output=True, text=True, timeout=120)
    if inventory.returncode == 0 and re.search(r"\.(?:parquet|sqlite\.zst)(?:[,\s\"]|$)", inventory.stdout, re.I):
        raise RuntimeError("Kaggle has structured files that this source-only upload "
                           "cannot preserve; retry via the structured publisher")
    target = directory / "dataset-metadata.json"
    target.write_text(json.dumps(metadata(name, dataset, manifest, source_repo, upstream_repo),
                                 ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    run_kaggle(dataset, directory, name)


def publish(directory: Path, language: str, name: str,
            hf_repo: str, kaggle_dataset: str,
            source_repo: str, upstream_repo: str,
            hf_token: str, kaggle_token: str) -> dict[str, str]:
    manifest = json.loads((directory / "build-manifest.json").read_text())
    if manifest["language"] != language or not list(directory.glob("*.txt.gz")):
        raise ValueError("Missing or mismatched source archive manifest")
    publishers = {
        "Hugging Face": (hf_token, publish_hf, (directory, language, name, hf_repo,
                                                manifest, source_repo, upstream_repo, hf_token)),
        "Kaggle": (kaggle_token, publish_kaggle, (directory, name, kaggle_dataset,
                                                  manifest, source_repo, upstream_repo, kaggle_token)),
    }
    outcomes = {label: "skipped (no credentials)" for label in publishers}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(fn, *args): label
                   for label, (token, fn, args) in publishers.items() if token}
        for future in as_completed(futures):
            label = futures[future]
            try:
                future.result()
                outcomes[label] = "success"
            except Exception as exc:
                outcomes[label] = f"failed: {type(exc).__name__}: {exc}"
                print(f"::warning::{label} publication failed: {exc}", file=sys.stderr, flush=True)

    print("; ".join(f"{label}={outcome}" for label, outcome in outcomes.items()), flush=True)
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as out:
            print("| Publisher | Result |\n|---|---|", file=out)
            for label, outcome in outcomes.items():
                print(f"| {label} | {outcome.replace('|', '/')} |", file=out)
    if os.getenv("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as out:
            print(f"hf_success={'true' if outcomes['Hugging Face'] == 'success' else 'false'}", file=out)
            print(f"publishers_ok={'true' if all(v == 'success' for v in outcomes.values()) else 'false'}", file=out)
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("publish"))
    args = parser.parse_args()
    e = os.environ
    outcomes = publish(args.directory, e["LANGUAGE"], e["LANG_NAME"],
                       e["FULL_HF_REPO"], e["FULL_KAGGLE_DATASET"],
                       e["SOURCE_REPO_URL"], e["UPSTREAM_REPO_URL"],
                       e.get("HF_TOKEN", ""), e.get("KAGGLE_TOKEN", ""))
    if any(outcome != "success" for outcome in outcomes.values()):
        raise SystemExit("One or more publishers failed or lacked credentials; retain artifacts for retry")


if __name__ == "__main__":
    main()
