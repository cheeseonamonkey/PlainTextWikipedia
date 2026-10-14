#!/usr/bin/env python3
"""Validate recursively split shard artifacts and create an honest coverage manifest."""

from __future__ import annotations

import argparse
import gzip
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def inspect(language: str, expected: int, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    leaf_pattern = re.compile(
        rf"^{re.escape(language)}_(\d{{2}})([ab]*)_of_{expected:02d}"
        rf"(?:_part(\d{{2}})_of_(\d{{2}}))?\.txt\.gz$"
    )
    manifest_pattern = re.compile(rf"^{re.escape(language)}_(\d{{2}})_of_{expected:02d}\.manifest\.json$")
    reports = {}
    for file in sorted(directory.glob("*.manifest.json")):
        m = manifest_pattern.fullmatch(file.name)
        if not m:
            file.unlink()
            continue
        root = int(m.group(1))
        if not 1 <= root <= expected:
            file.unlink()
            continue
        try:
            report = json.loads(file.read_text())
            if (report["language"] != language or report["root_shard"] != root
                    or report["planned_shards"] != expected):
                raise ValueError("wrong language/root/count in manifest")
            reports[root] = report
        except (ValueError, KeyError, TypeError) as exc:
            print(f"::warning::Invalid recovery manifest {file.name}: {exc}")
            file.unlink()

    valid = {}
    for file in sorted(directory.glob("*.txt.gz")):
        m = leaf_pattern.fullmatch(file.name)
        if not m or not 1 <= int(m.group(1)) <= expected:
            print(f"::warning::Unexpected archive {file.name}, excluding")
            file.unlink()
            continue
        root = int(m.group(1))
        if m.group(3) is not None:
            part, parts = int(m.group(3)), int(m.group(4))
            if not (1 <= part <= parts <= 64):
                print(f"::warning::Invalid repacked leaf {file.name}")
                file.unlink()
                continue
        report = reports.get(root)
        named = {leaf["file"] for leaf in report.get("successful_leaves", [])} if report else set()
        if file.name not in named:
            print(f"::warning::Untracked leaf archive {file.name}, excluding")
            file.unlink()
            continue
        try:
            with gzip.open(file, "rb") as stream:
                first = stream.read(1)
                if not first:
                    print(f"::warning::Empty gzip article shard {file.name}, excluding")
                    file.unlink()
                    continue
                while stream.read(8 * 1024 * 1024):
                    pass
        except (OSError, EOFError) as exc:
            print(f"::warning::Corrupt leaf archive {file.name}: {exc}")
            file.unlink()
            continue
        valid.setdefault(root, set()).add(file.name)

    present = sorted(valid)
    missing = sorted(set(range(1, expected + 1)) - set(present))
    incomplete = []
    roots = []
    for root in range(1, expected + 1):
        report = reports.get(root)
        files = sorted(valid.get(root, set()))
        if not report:
            if files:
                incomplete.append(root)
            roots.append({"root_shard": root, "complete": False, "files": files,
                          "error": "No build report/artifact was uploaded"})
            continue
        planned = {leaf["file"] for leaf in report.get("successful_leaves", [])}
        is_complete = bool(report["complete"]) and bool(files) and set(files) == planned
        if not is_complete and files:
            incomplete.append(root)
        roots.append({
            "root_shard": root, "complete": is_complete,
            "files": files,
            "successful_leaves": [leaf for leaf in report.get("successful_leaves", [])
                                  if leaf["file"] in files],
            "failed_fragments": report.get("failed_fragments", []),
            "download_failures": report.get("download_failures", []),
        })
        # Upload only the aggregated manifest, not the private per-root build reports.
    for file in directory.glob("*.manifest.json"):
        file.unlink()

    result = {
        "language": language,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "expected_shards": expected,
        "published_shards": len(present),
        "published_archives": sum(len(files) for files in valid.values()),
        "present_shards": present,
        "missing_shards": missing,
        "incomplete_shards": incomplete,
        "complete": len(present) == expected and not incomplete
                    and all(root["complete"] for root in roots),
        "roots": roots,
    }
    (directory / "build-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", required=True)
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--github-summary", type=Path)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    result = inspect(args.language, args.count, args.directory)
    available = result["published_archives"] > 0
    summary = (f"{args.language}: {result['published_shards']}/{args.count} root shards, "
               f"{result['published_archives']} compressed leaf files. "
               f"Missing roots: {result['missing_shards']}; "
               f"incomplete roots: {result['incomplete_shards']}")
    print(summary, flush=True)
    if args.github_output:
        with args.github_output.open("a") as stream:
            print(f"available={str(available).lower()}", file=stream)
            print(f"complete={str(result['complete']).lower()}", file=stream)
    if args.github_summary:
        with args.github_summary.open("a") as stream:
            print(summary, file=stream)
    if not available:
        print("::warning::No successful shard archives available to publish")


if __name__ == "__main__":
    main()
