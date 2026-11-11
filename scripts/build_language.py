#!/usr/bin/env python3
"""Build one complete, verified language parquet bundle.

This is the single build entrypoint for the v8 pipeline. It resolves a language
prefix (en, zh, es, ...) to the existing dataset repository, stages the source
archives (from Hugging Face, or from a fresh-dump conversion directory),
computes the enriched ``articles.parquet`` + ``corpus-profile.json``, and
verifies them. Publishing is handled by separate publisher modules/workflows
that consume the retained bundle artifact.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from scripts.article_features import build
from scripts.bundle_artifacts import link_or_copy, verify_bundle, verify_sources
from scripts.wiki_metadata import resolve_language

RAW_RE = re.compile(r"data/[^/]+\.txt\.gz$")


def source_archives(api: HfApi, repo: str, folder: Path) -> tuple[list[Path], int]:
    """Download the existing repo's data/*.txt.gz shards and its manifest."""
    revision = api.repo_info(repo, repo_type="dataset").sha
    if not revision:
        raise ValueError('Cannot pin source repository revision')
    files = list(api.list_repo_tree(repo, repo_type="dataset", revision=revision, recursive=True))
    sources = {f.path: f for f in files if RAW_RE.fullmatch(getattr(f, "path", ""))}
    if not sources:
        raise ValueError(f"{repo} has no published source archives")
    manifest_remote = "data/build-manifest.json"
    if not any(f.path == manifest_remote for f in files):
        raise ValueError(f"{repo} has no data/build-manifest.json; rebuild from dumps")
    manifest_local = Path(hf_hub_download(repo, filename=manifest_remote,
                                          repo_type="dataset", revision=revision, token=api.token))
    # Stage the source archives (hardlink where possible for speed).
    staged = []
    for path in sorted(sources):
        local = Path(hf_hub_download(repo, filename=path, repo_type="dataset",
                                     revision=revision, token=api.token))
        target = folder / Path(path).name
        link_or_copy(local, target)
        staged.append(target)
    # Manifest must sit in the same folder for verify_sources.
    shutil.copyfile(manifest_local, folder / "build-manifest.json")
    return staged, len(sources)


def make_bundle(files: list[Path], output: Path, code: str, *,
                vocab_size: int = 12000, dim: int = 180, window: int = 8,
                epoch: int = 1, threads: int = 4) -> dict:
    """Build and verify the v8 bundle, returning the manifest."""
    build(files, output, code, vocab_size=vocab_size,
          dim=dim, window=window, epoch=epoch, threads=threads)
    return verify_bundle(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--language", required=True, help="language prefix, e.g. en, zh, es")
    p.add_argument("--sources", type=Path, help="directory of converted source shards (fresh dump)")
    p.add_argument("--output", type=Path, default=Path("release"))
    p.add_argument("--vocab", type=int, default=12000)
    p.add_argument("--dim", type=int, default=180)
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--epoch", type=int, default=1)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()

    repo, _project, code = resolve_language(args.language)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Release directory not empty: {args.output}")

    api = HfApi(token=os.getenv("HF_TOKEN") or None)
    with tempfile.TemporaryDirectory(prefix="wiki-sources-") as tmp:
        sources_dir = Path(tmp)
        if args.sources:
            # inspect() removes invalid inputs and individual reports: never run it
            # against the caller's directory.
            for f in args.sources.iterdir():
                if f.is_file() and (f.name.endswith('.txt.gz') or f.name.endswith('.manifest.json')
                                    or f.name == 'build-manifest.json'):
                    shutil.copy2(f, sources_dir / f.name)
            if not (sources_dir / 'build-manifest.json').exists():
                from scripts.inspect_adaptive import inspect
                reports = [json.loads(p.read_text(encoding='utf-8'))
                           for p in sources_dir.glob('*.manifest.json')]
                if not reports:
                    raise ValueError('No source build manifest or shard conversion manifests found')
                counts = {r['planned_shards'] for r in reports}
                if len(counts) != 1:
                    raise ValueError('Inconsistent planned shard counts')
                result = inspect(code + 'wiki', counts.pop(), sources_dir)
                if not result['complete']:
                    raise ValueError('Incomplete source coverage; nothing will be built')
            files = sorted(p for p in sources_dir.iterdir() if p.name.endswith('.txt.gz'))
        else:
            # Download sources from the existing HF dataset repository.
            files, _ = source_archives(api, repo, sources_dir)
        fmt, expected = verify_sources(sources_dir, code)
        print(f"Source coverage: {fmt} manifest, {expected} expected shards, "
              f"{len(files)} staged archives; manifest claims checked, "
              "content coverage not independently proven", flush=True)
        report = make_bundle(files, args.output, code, vocab_size=args.vocab,
                             dim=args.dim, window=args.window, epoch=args.epoch,
                             threads=args.threads)
        print(f"Bundle verified: {report['documents']:,} articles", flush=True)


if __name__ == "__main__":
    main()

