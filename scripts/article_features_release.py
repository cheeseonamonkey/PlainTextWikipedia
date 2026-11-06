#!/usr/bin/env python3
"""Build/resume one existing language's corpus features on Hugging Face.

No Wikipedia XML rebuild, no source edits. Explicit --publish is required
before any remote mutation; exact source + config fingerprints allow skips.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

try:
    from scripts.article_features import VERSION, FUSION_IMPORTANCE, build
    from scripts.structured_fleet import retry, source_files, source_hash
    from scripts.wiki_metadata import wiki_code
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from article_features import VERSION, FUSION_IMPORTANCE, build
    from structured_fleet import retry, source_files, source_hash
    from wiki_metadata import wiki_code


def inventory(api, repo):
    sources, remote = retry("list source archives", lambda: source_files(api, repo))
    if not sources:
        raise ValueError(f"No published articles found: {repo}")
    files = []
    for path, entry in sorted(sources.items()):
        if not re.fullmatch(r"data/[^/]+\.txt\.gz", path):
            raise ValueError(f"Invalid source path: {path}")
        sha = source_hash(entry)
        if not sha:
            raise ValueError(f"Missing trusted LFS SHA256: {path}")
        files.append({"path": path, "sha256": sha, "bytes": int(entry.size)})
    return files, remote


def config(args, language):
    return {"version": VERSION, "language": language,
            "fusion_importance": FUSION_IMPORTANCE,
            "vocab": args.vocab, "tfidf": args.tfidf,
            "train_token_cap": args.train_token_cap,
            "sample_mod": args.sample_mod,
            "fasttext_timeout": args.fasttext_timeout,
            "threads": args.threads,
            "fts5": not args.no_fts5,
            "allow_lexical_only": args.allow_lexical_only,
            "bekko_model": "hotchpotch/bekko-embedding-v1-a8m",
            "bekko_max_tokens": 256,
            "casefold_all_embeddings": True}


def is_current(api, repo, source_list, settings, remote):
    manifest_name = "features/manifest.json"
    if manifest_name not in remote:
        return False
    file = retry("download feature manifest", lambda: hf_hub_download(
        repo_id=repo, filename=manifest_name, repo_type="dataset",
        token=api.token))
    found = json.loads(Path(file).read_text(encoding="utf-8"))
    if found.get("source_archives") != source_list or found.get("config") != settings:
        return False
    files = found.get("files")
    if not isinstance(files, dict) or not files:
        return False
    for name, meta in files.items():
        target = remote.get(f"features/{name}")
        if target is None or getattr(target, "size", None) != meta.get("bytes"):
            return False
        digest = source_hash(target)
        if digest and digest != meta.get("sha256"):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--publish", action="store_true",
                        help="Commit derived artifacts to the existing HF repo")
    parser.add_argument("--vocab", type=int, default=10_000)
    parser.add_argument("--tfidf", type=int, default=5_000)
    parser.add_argument("--train-token-cap", type=int, default=2_000_000)
    parser.add_argument("--sample-mod", type=int, default=4)
    parser.add_argument("--fasttext-timeout", type=int, default=180)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--no-fts5", action="store_true")
    parser.add_argument("--allow-lexical-only", action="store_true",
                        help="Label unavailable semantic vectors rather than failing")
    args = parser.parse_args()
    if not 1 <= args.tfidf <= args.vocab <= 50_000:
        parser.error("Invalid vocabulary sizes")
    if not re.fullmatch(
        r"ffatty/(?:plain-text-wikipedia-simpleenglish|plaintext-wikipedia-full-[a-z0-9-]+)",
        args.repo,
    ):
        parser.error("Only existing ffatty Wikipedia datasets supported")

    language = wiki_code(args.repo)
    settings = config(args, language)
    api = HfApi(token=os.getenv("HF_TOKEN") or None)
    source_list, remote = inventory(api, args.repo)
    if is_current(api, args.repo, source_list, settings, remote):
        print(f"SKIP current feature edition: {args.repo}", flush=True)
        return
    required = sum(item["bytes"] for item in source_list)
    available = shutil.disk_usage(tempfile.gettempdir()).free
    # SQLite holds decompressed documents and counts; allow conservative
    # overhead before committing to a large language. Overrides need a
    # larger runner, not silently truncated source coverage.
    capacity = required * 6 + 2_000_000_000
    if capacity > available:
        raise RuntimeError(f"Insufficient disk for full corpus: require ~{capacity/1e9:.1f} GB, "
                           f"available {available/1e9:.1f} GB; use a larger runner")
    if not args.publish:
        print(f"DRY RUN: would process {len(source_list)} archives "
              f"({required/1e9:.2f} GB compressed) from {args.repo}", flush=True)
        return
    if not api.token:
        raise ValueError("HF_TOKEN required for publishing")
    with tempfile.TemporaryDirectory(prefix="wiki-features-") as tmp:
        dest = Path(tmp)
        source_paths = [
            Path(retry(f"download {item['path']}", lambda item=item: hf_hub_download(
                repo_id=args.repo, filename=item["path"],
                repo_type="dataset", token=api.token)))
            for item in source_list
        ]
        out = dest / "output"
        report = build(source_paths, out, language, vocab_size=args.vocab,
                       shortlist=args.tfidf, max_train_tokens=args.train_token_cap,
                       sample_mod=args.sample_mod, timeout=args.fasttext_timeout,
                       fts5=not args.no_fts5,
                       allow_lexical_only=args.allow_lexical_only,
                       threads=args.threads)
        report["source_archives"] = source_list
        report["config"] = settings
        (out / "manifest.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        operations = [
            CommitOperationAdd(path_in_repo=f"features/{name}",
                               path_or_fileobj=str(out / name))
            for name in [*report["files"], "manifest.json"]
        ]
        retry("publish feature corpus", lambda: api.create_commit(
            repo_id=args.repo, repo_type="dataset", operations=operations,
            commit_message=f"Corpus features v{VERSION}: {language}"))
        print(f"PUBLISHED {len(operations)} feature files for {args.repo}, "
              f"{report['documents']} articles, fastText={report['fasttext']['available']}",
              flush=True)


if __name__ == "__main__":
    main()
