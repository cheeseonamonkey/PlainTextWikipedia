#!/usr/bin/env python3
"""Build one complete, verified language parquet bundle.

This is the single build entrypoint for the v8 pipeline. It resolves a language
prefix (en, zh, es, ...) to the existing dataset repository, stages the source
archives (from Hugging Face, or from a fresh-dump conversion directory),
computes the enriched ``articles.parquet`` + ``corpus-profile.json``, and
verifies them. Publishing is handled by separate publisher modules/workflows
that consume the retained bundle artifact.

The work is also runnable one stage at a time (``--stage``) against an explicit
intermediate directory, so each stage can get its own process and its own CI
time budget. In order: ``corpus-stats`` reads the sources and writes
``work/corpus-stats``; ``fasttext-training`` reads that and writes
``work/fasttext-training``; ``enrich-parquet`` reads both and writes
``release/``.
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

from scripts.article_features import (
    CORPUS_STATS_DIR,
    ENRICH_DIR,
    STAGES,
    TRAINING_DIR,
    build,
    ensure_capacity,
    stage_corpus_stats,
    stage_enrich_parquet,
    stage_fasttext_training,
)
from scripts.bundle_artifacts import link_or_copy, verify_bundle, verify_sources
from scripts.wiki_metadata import resolve_language

RAW_RE = re.compile(r"data/[^/]+\.txt\.gz$")


def effective_api(api: HfApi, repo: str) -> HfApi:
    """Fall back to anonymous access when the stored token is unusable.

    HF rejects requests carrying an expired token with 401 even for public
    datasets that anonymous callers can read. Publishing still requires a
    valid token; this only keeps public-source builds unblocked.
    """
    try:
        api.repo_info(repo, repo_type="dataset")
        return api
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403) and api.token:
            print("::warning::HF_TOKEN was rejected; retrying anonymous read-only access. "
                  "Refresh the token secret to publish.", flush=True)
            # token=False (not None): None re-reads HF_TOKEN from the environment,
            # which would re-send the same expired credential.
            anonymous = HfApi(token=False)
            anonymous.repo_info(repo, repo_type="dataset")  # must succeed anonymously
            return anonymous
        raise


def source_archives(api: HfApi, repo: str, folder: Path) -> tuple[list[Path], int]:
    """Download the existing repo's data/*.txt.gz shards and its manifest."""
    api = effective_api(api, repo)
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


def collect_sources(api: HfApi, repo: str, code: str, args, folder: Path) -> list[Path]:
    """Fill ``folder`` with source shards and return the .txt.gz archives.

    Accepts either a caller-supplied conversion directory (``--sources``) or a
    download from the existing Hugging Face repository. The directory is never
    mutated in place: inspect() deletes invalid inputs, so it runs on a copy.
    """
    if args.sources:
        for f in args.sources.iterdir():
            if f.is_file() and (f.name.endswith('.txt.gz') or f.name.endswith('.manifest.json')
                                or f.name == 'build-manifest.json'):
                shutil.copy2(f, folder / f.name)
        if not (folder / 'build-manifest.json').exists():
            from scripts.inspect_adaptive import inspect
            reports = [json.loads(p.read_text(encoding='utf-8'))
                       for p in folder.glob('*.manifest.json')]
            if not reports:
                raise ValueError('No source build manifest or shard conversion manifests found')
            counts = {r['planned_shards'] for r in reports}
            if len(counts) != 1:
                raise ValueError('Inconsistent planned shard counts')
            result = inspect(code + 'wiki', counts.pop(), folder)
            if not result['complete']:
                raise ValueError('Incomplete source coverage; nothing will be built')
            # inspect() writes build-manifest.json into the staging dir. Mirror it
            # back into the caller's sources dir so the workflow's sources artifact
            # (and the later --sources-dir HF publish) carries the manifest
            # alongside the rebuilt shards.
            shutil.copy2(folder / 'build-manifest.json', args.sources / 'build-manifest.json')
        files = sorted(p for p in folder.iterdir() if p.name.endswith('.txt.gz'))
    else:
        # Download sources from the existing HF dataset repository.
        files, _ = source_archives(api, repo, folder)
    fmt, expected = verify_sources(folder, code)
    print(f"Source coverage: {fmt} manifest, {expected} expected shards, "
          f"{len(files)} staged archives; manifest claims checked, "
          "content coverage not independently proven", flush=True)
    return files


def make_bundle(files: list[Path], output: Path, code: str, *,
                vocab_size: int = 12000, dim: int = 180, window: int = 8,
                epoch: int = 1, threads: int = 4) -> dict:
    """Build and verify the v8 bundle, returning the manifest."""
    build(files, output, code, vocab_size=vocab_size,
          dim=dim, window=window, epoch=epoch, threads=threads)
    return verify_bundle(output)


def run_stage(stage: str, args, repo: str, code: str) -> dict:
    """Run exactly one pipeline stage into ``--work`` (or ``--release``)."""
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    if stage == CORPUS_STATS_DIR:
        with tempfile.TemporaryDirectory(prefix="wiki-sources-") as tmp:
            files = collect_sources(args.hf_api, repo, code, args, Path(tmp))
            ensure_capacity(files, work)
            return stage_corpus_stats(files, work, code, vocab_size=args.vocab)
    if stage == TRAINING_DIR:
        return stage_fasttext_training(work, dim=args.dim, window=args.window,
                                       epoch=args.epoch, threads=args.threads)
    if stage == ENRICH_DIR:
        stage_enrich_parquet(work, args.release, code)
        return verify_bundle(args.release)
    raise ValueError(f"Unknown stage: {stage}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--language", required=True, help="language prefix, e.g. en, zh, es")
    p.add_argument("--sources", type=Path, help="directory of staged source shards")
    p.add_argument("--stage-only", type=Path, help="download existing HF sources to this empty directory and exit")
    p.add_argument("--stage", choices=STAGES, help="run one stage into --work instead of the whole build")
    p.add_argument("--work", type=Path, help="shared intermediate artifact directory")
    p.add_argument("--release", type=Path, help="release directory written by the enrich-parquet stage")
    p.add_argument("--output", type=Path, default=Path("release"))
    p.add_argument("--vocab", type=int, default=12000)
    p.add_argument("--dim", type=int, default=180)
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--epoch", type=int, default=1)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()

    repo, _project, code = resolve_language(args.language)
    args.hf_api = HfApi(token=os.getenv('HF_TOKEN') or None)

    if args.stage_only:
        if args.sources:
            p.error('--stage-only cannot be combined with --sources')
        args.stage_only.mkdir(parents=True, exist_ok=True)
        if any(args.stage_only.iterdir()):
            raise FileExistsError(f'Source directory not empty: {args.stage_only}')
        source_archives(args.hf_api, repo, args.stage_only)
        verify_sources(args.stage_only, code)
        return
    if args.stage:
        if args.stage == ENRICH_DIR and not args.release:
            p.error('The enrich-parquet stage requires --release')
        result = run_stage(args.stage, args, repo, code)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Release directory not empty: {args.output}")

    with tempfile.TemporaryDirectory(prefix="wiki-sources-") as tmp:
        sources_dir = Path(tmp)
        files = collect_sources(args.hf_api, repo, code, args, sources_dir)
        report = make_bundle(files, args.output, code, vocab_size=args.vocab,
                             dim=args.dim, window=args.window, epoch=args.epoch,
                             threads=args.threads)
        print(f"Bundle verified: {report['documents']:,} articles", flush=True)


if __name__ == "__main__":
    main()
