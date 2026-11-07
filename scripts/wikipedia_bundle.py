#!/usr/bin/env python3
"""Build one complete immutable language bundle, then publish those exact files."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import quote

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, hf_hub_download

from scripts.article_features import VERSION, build, sha256
from scripts.article_feature_card import update_card
from scripts.release_parquet import Asset, dataset_slug, zenodo_publish, list_matching_depositions
from scripts.structured_fleet import retry, source_hash
from scripts.structured_kaggle import run
from scripts.wiki_metadata import wiki_code

PROJECT = "https://github.com/cheeseonamonkey/PlainTextWikipedia"
PAYLOAD = {"articles.parquet", "articles.txt.xz", "features.zip", "models.zip"}


def links(repo, zenodo=None):
    return dict(github=PROJECT, huggingface=f"https://huggingface.co/datasets/{repo}",
                kaggle=f"https://www.kaggle.com/datasets/{repo}",
                zenodo=zenodo or "https://zenodo.org/search?q=" + quote(f'"{repo}"'))


def complete_sources(folder):
    manifest = json.loads((folder / "build-manifest.json").read_text())
    if manifest.get("complete") is not True:
        raise ValueError("Incomplete source coverage: refusing to build/publish a partial edition")
    expected = {name for root in manifest.get("roots", []) for name in root.get("files", [])}
    actual = {p.name for p in folder.glob("*.txt.gz")}
    if not expected or actual != expected:
        raise ValueError("Source archives differ from the complete build manifest")
    return [folder / name for name in sorted(expected)]


def download_sources(api, repo, folder):
    revision = api.repo_info(repo, repo_type="dataset").sha
    tree = {f.path: f for f in api.list_repo_tree(repo, repo_type="dataset", revision=revision, recursive=True)}
    def download(name):
        return Path(retry(name, lambda: hf_hub_download(repo, name, repo_type="dataset",
                                                       revision=revision, token=api.token)))
    if "manifest.json" in tree and "articles.txt.xz" in tree:
        prior = json.loads(download("manifest.json").read_text())
        meta = prior.get("files", {}).get("articles.txt.xz", {})
        source = download("articles.txt.xz")
        if prior.get("complete") is not True or source.stat().st_size != meta.get("bytes") or sha256(source) != meta.get("sha256"):
            raise ValueError("Invalid canonical source edition")
        return [source], dict(repo=repo, revision=revision, previous_source_revision=prior.get("source", {}).get("revision"),
                             archives=[dict(path="articles.txt.xz", **meta)])
    manifest = download("data/build-manifest.json")
    shutil.copyfile(manifest, folder / "build-manifest.json")
    coverage = json.loads(manifest.read_text())
    names = {name for root in coverage.get("roots", []) for name in root.get("files", [])}
    actual = {Path(name).name for name in tree if name.startswith("data/") and name.endswith(".txt.gz")}
    if coverage.get("complete") is not True or names != actual or not names:
        raise ValueError("Published source manifest is incomplete or does not match its archives")
    provenance = []
    for name in sorted(names):
        if Path(name).name != name:
            raise ValueError("Unsafe source filename")
        path = "data/" + name
        source = download(path)
        digest = sha256(source)
        if not source_hash(tree[path]) or digest != source_hash(tree[path]):
            raise ValueError(f"Source SHA256 mismatch: {path}")
        # Hardlink cached immutable blobs; avoid copying every input archive.
        os.link(source, folder / name)
        provenance.append(dict(path=path, sha256=digest, bytes=source.stat().st_size))
    return complete_sources(folder), dict(repo=repo, revision=revision, archives=provenance)


def archive(destination, files):
    # Existing files are already compressed. ZIP_STORED avoids compressing twice.
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as out:
        for file in files:
            out.write(file, file.name)


def make_bundle(files, output, code, destinations, source):
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output, prefix=".build-") as tmp:
        stage = Path(tmp)
        report = build(files, stage, code)
        if not report["fasttext"]["available"]:
            raise ValueError("Complete fastText model required")
        archive(stage / "features.zip", [stage / name for name in
                ("corpus-profile.json", "vocabulary.json.gz", "words.jsonl.gz")])
        model_config = stage / "config.json"
        model_config.write_text(json.dumps({key: report[key] for key in ("fasttext", "bekko", "lexical")}, indent=2) + "\n")
        archive(stage / "models.zip", [stage / "fasttext.bin.zst", model_config])
        manifest = dict(version=VERSION, language=code, documents=report["documents"],
                        source=source, links=destinations, complete=True,
                        layout={"features.zip": "Extract into features/", "models.zip": "Extract into models/"},
                        files={})
        for name in sorted(PAYLOAD):
            file = stage / name
            with file.open("rb") as stream:
                md5 = hashlib.file_digest(stream, "md5").hexdigest()
            manifest["files"][name] = dict(bytes=file.stat().st_size, sha256=sha256(file), md5=md5)
        (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        for name in PAYLOAD | {"manifest.json"}:
            os.replace(stage / name, output / name)
    return verify_bundle(output)


def verify_bundle(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest.get("complete") is not True or manifest.get("documents", 0) < 1 or set(manifest.get("files", {})) != PAYLOAD:
        raise ValueError("Incomplete release bundle")
    for name, meta in manifest["files"].items():
        file = folder / name
        if not file.is_file() or file.stat().st_size != meta["bytes"] or sha256(file) != meta["sha256"]:
            raise ValueError(f"Release checksum mismatch: {name}")
    import pyarrow.parquet as pq
    if pq.ParquetFile(folder / "articles.parquet").metadata.num_rows != manifest["documents"]:
        raise ValueError("Parquet row count differs from release manifest")
    return manifest


def publish_hf(api, repo, folder, revision=None):
    revision = revision or api.repo_info(repo, repo_type="dataset").sha
    remote = list(api.list_repo_tree(repo, repo_type="dataset", revision=revision, recursive=True))
    manifest = verify_bundle(folder)
    # Replace generated legacy data in the next commit, with an optimistic lock.
    operations = [CommitOperationDelete(path_in_repo=f.path) for f in remote
                  if getattr(f, "size", None) is not None and
                  (f.path.startswith(("data/", "structured/", "features/", "models/"))
                   or f.path.endswith((".parquet", ".sqlite.zst"))) and f.path not in PAYLOAD]
    operations += [CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(folder / name))
                   for name in sorted(PAYLOAD | {"manifest.json"})]
    card = "---\nlicense: cc-by-sa-4.0\n---\n\n# Plaintext Wikipedia\n\n" + description(manifest)
    operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=update_card(card).encode()))
    commit = retry("publish identical HF bundle", lambda: api.create_commit(repo_id=repo, repo_type="dataset",
          parent_commit=revision, operations=operations, commit_message="Publish complete combined Wikipedia bundle"))
    published = {f.path: f for f in api.list_repo_tree(repo, repo_type="dataset", revision=commit.oid, recursive=True)}
    for name, meta in manifest["files"].items():
        item = published.get(name)
        if item is None or item.size != meta["bytes"]:
            raise RuntimeError(f"Missing/wrong-size HF artifact after commit: {name}")
        digest = source_hash(item)
        if not digest:
            digest = sha256(Path(hf_hub_download(repo, name, repo_type="dataset", revision=commit.oid, token=api.token)))
        if digest != meta["sha256"]:
            raise RuntimeError(f"HF checksum mismatch after commit: {name}")



def description(manifest):
    return (f"{manifest['documents']:,} articles. One `articles.parquet` (one row per article), "
            "one `articles.txt.xz` (title, body, blank line), full word statistics and corpus "
            "profiles in `features.zip`, fastText model and configuration in `models.zip`. "
            "Extract the ZIPs into `features/` and `models/`. Every publisher receives identical data files. "
            "Article URLs are reconstructed from titles; Wikipedia page IDs are unavailable. "
            "Text: Wikimedia contributors, CC BY-SA 4.0.\n\n" +
            "\n".join(f"- [{key}]({value})" for key, value in manifest["links"].items()) + "\n")


def publish_kaggle(repo, folder):
    manifest = verify_bundle(folder)
    # Fresh staging replaces obsolete shards rather than carrying them forever.
    metadata = dict(id=repo, title=f"Plaintext Wikipedia ({manifest['language']})",
                    subtitle="Complete article Parquet, plaintext, features and models",
                    licenses=[{"name": "CC-BY-SA-4.0"}], description=description(manifest))
    with tempfile.TemporaryDirectory(prefix="wiki-kaggle-") as tmp:
        stage = Path(tmp)
        for name in PAYLOAD | {"manifest.json"}:
            os.link(folder / name, stage / name)
        (stage / "dataset-metadata.json").write_text(json.dumps(metadata))
        state = json.loads(run("status", repo, "--format", "json").stdout)
        if state.get("status") != "ready":
            raise RuntimeError("Kaggle dataset is not ready for a new version")
        with tempfile.TemporaryDirectory(prefix="wiki-kaggle-manifest-") as probe:
            result = run("download", "-d", repo, "-f", "manifest.json", "-p", probe, "--unzip", check=False)
            previous = Path(probe) / "manifest.json"
            if result.returncode == 0 and previous.is_file() and previous.read_bytes() == (folder / "manifest.json").read_bytes():
                verify_kaggle(repo, folder)
                return
        run("version", "-p", str(stage), "-m", "Identical complete Wikipedia bundle",
            "--dir-mode", "skip", "--keep-tabular")
        # Processing completion is required before marking the release successful.
        import time
        deadline = time.monotonic() + 7200
        while time.monotonic() < deadline:
            current = json.loads(run("status", repo, "--format", "json").stdout)
            if int(current.get("current_version_number") or 0) > int(state.get("current_version_number") or 0) and current.get("status") == "ready":
                verify_kaggle(repo, folder)
                return
            if current.get("status") in {"error", "failed", "failure"}:
                raise RuntimeError("Kaggle rejected the complete bundle")
            time.sleep(15)
        raise TimeoutError("Kaggle version is not confirmed ready")


def verify_kaggle(repo, folder):
    # Download the ready version, preserving uploaded archives and Parquet.
    # Never claim identical content based on file sizes alone.
    with tempfile.TemporaryDirectory(prefix="wiki-kaggle-verify-") as tmp:
        stage = Path(tmp)
        run("download", "-d", repo, "-p", str(stage), "--unzip")
        if (stage / "manifest.json").read_bytes() != (folder / "manifest.json").read_bytes():
            raise ValueError("Kaggle manifest differs from the built bundle")
        verify_bundle(stage)


def reserve_zenodo(repo):
    import requests
    headers = {"Authorization": "Bearer " + os.environ["ZENODO_TOKEN"]}
    url = "https://zenodo.org/api/deposit/depositions"
    title = f"Plaintext Wikipedia ({dataset_slug(repo)}) — Parquet edition"
    matches = list_matching_depositions(requests, url, headers, title)
    if len(matches) > 1:
        raise ValueError("Multiple matching Zenodo records; resolve before publishing")
    if matches:
        prior = matches[0]
        if not prior.get("submitted"):
            return int(prior["id"])
        response = requests.post(f"{url}/{prior['id']}/actions/newversion", headers=headers, timeout=60)
        response.raise_for_status()
        draft_url = response.json()["links"]["latest_draft"]
        if not draft_url.startswith(url + "/"):
            raise ValueError("Unexpected Zenodo draft URL")
        return int(draft_url.rsplit("/", 1)[1])
    response = requests.post(url, json={}, headers=headers, timeout=60)
    response.raise_for_status()
    draft = response.json()
    # Set identifying metadata so an interrupted build can resume this draft.
    response = requests.put(f"{url}/{draft['id']}", headers=headers, timeout=60,
        json={"metadata": {"title": title, "upload_type": "dataset", "description": f"Complete Wikipedia corpus {repo}",
                           "creators": [{"name": "cheeseonamonkey"}], "license": "cc-by-sa-4.0"}})
    response.raise_for_status()
    return int(draft["id"])


def publish_all(api, repo, folder, zenodo_id):
    manifest = verify_bundle(folder)  # Finish and verify EVERYTHING before data uploads.
    dataset_slug(repo)
    if manifest["language"] != wiki_code(repo) or manifest["links"]["huggingface"] != links(repo)["huggingface"]:
        raise ValueError("Bundle belongs to another language/repository")
    for key in ("HF_TOKEN", "KAGGLE_TOKEN", "ZENODO_TOKEN"):
        if not os.getenv(key):
            raise ValueError(f"{key} required for full publishing")
    # Each adapter consumes this same immutable local set. A failed target fails
    # the job; cross-service publication cannot be an atomic transaction.
    source = manifest.get("source", {})
    revision = api.repo_info(repo, repo_type="dataset").sha
    already_hf = False
    if source.get("revision") and revision != source["revision"]:
        remote_manifest = Path(hf_hub_download(repo, "manifest.json", repo_type="dataset", revision=revision, token=api.token))
        if remote_manifest.read_bytes() != (folder / "manifest.json").read_bytes():
            raise ValueError("Source repository changed during build; refusing stale publication")
        already_hf = True  # Resume a failed downstream publisher from the same immutable bundle.
    if not already_hf:
        publish_hf(api, repo, folder, revision)
    publish_kaggle(repo, folder)
    assets = [Asset(name, meta["sha256"], meta["bytes"], meta["md5"])
              for name, meta in manifest["files"].items()]
    assets.append(Asset("manifest.json", sha256(folder / "manifest.json"), (folder / "manifest.json").stat().st_size))
    zenodo_publish(api, repo, assets, dict(language=dataset_slug(repo), source_url=manifest["links"]["huggingface"],
        links=manifest["links"], bundle=manifest), zenodo_id, local_folder=folder)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True)
    p.add_argument("--sources", type=Path)
    p.add_argument("--output", type=Path, default=Path("release"))
    p.add_argument("--publish", action="store_true")
    p.add_argument("--resume", action="store_true", help="Verify and publish an already built local bundle")
    args = p.parse_args()
    code = wiki_code(args.repo)
    api = HfApi(token=os.getenv("HF_TOKEN"))
    zenodo_id = None
    if args.publish:
        for key in ("HF_TOKEN", "KAGGLE_TOKEN", "ZENODO_TOKEN"):
            if not os.getenv(key):
                raise ValueError(f"{key} required for full publishing")
        if args.resume:
            prior = verify_bundle(args.output)
            prefix = "https://zenodo.org/records/"
            target = prior["links"]["zenodo"]
            if not target.startswith(prefix) or not target.removeprefix(prefix).isdigit():
                raise ValueError("Build-only bundle has no reserved Zenodo record; rebuild with --publish")
            zenodo_id = int(target.removeprefix(prefix))
        else:
            zenodo_id = reserve_zenodo(args.repo)
    if not args.resume:
        with tempfile.TemporaryDirectory(prefix="wiki-sources-") as tmp:
            if args.sources:
                files = complete_sources(args.sources)
                provenance = dict(build_manifest=json.loads((args.sources / "build-manifest.json").read_text()),
                                  archives=[dict(path=f.name, sha256=sha256(f), bytes=f.stat().st_size) for f in files])
            else:
                files, provenance = download_sources(api, args.repo, Path(tmp))
            make_bundle(files, args.output, code,
                        links(args.repo, f"https://zenodo.org/records/{zenodo_id}" if zenodo_id else None), provenance)
    verify_bundle(args.output)
    if args.publish:
        publish_all(api, args.repo, args.output, zenodo_id)


if __name__ == "__main__":
    main()
