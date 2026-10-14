#!/usr/bin/env python3
"""Recover failed Wikipedia conversion shards by recursively bisecting work.

A root shard keeps its original zero-based assignment. A failed root 01
becomes 01a and 01b; further failures become 01aa, 01ab, etc.
Every successful leaf covers disjoint source files or inclusive page-ID ranges.
Never publish a partially written .gz file.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Source:
    path: Path
    low: int | None = None
    high: int | None = None


def split_sources(sources: list[Source]) -> tuple[list[Source], list[Source]] | None:
    if len(sources) > 1:
        middle = len(sources) // 2
        return sources[:middle], sources[middle:]
    if not sources:
        return None
    source = sources[0]
    low, high = source.low, source.high
    if low is None or high is None:
        match = re.search(r"(?:\.xml-p|-p)(\d+)p(\d+)(?:\.xml)?\.bz2$", source.path.name)
        if not match:
            return None
        low, high = map(int, match.groups())
    if low >= high:
        return None
    middle = (low + high) // 2
    return [Source(source.path, low, middle)], [Source(source.path, middle + 1, high)]


def validate_gzip(path: Path) -> None:
    with gzip.open(path, "rb") as stream:
        while stream.read(8 * 1024 * 1024):
            pass


def convert(
    language: str,
    shard: int,
    count: int,
    sources: list[Source],
    output: Path,
    *,
    max_depth: int = 3,
    max_seconds: int = 19000,
    timeouts: tuple[int, ...] = (8100, 4500, 2400, 1200),
    failures: list[dict] | None = None,
) -> dict:
    """Return full recovery report, including terminal failures and leaf coverage."""
    output.mkdir(parents=True, exist_ok=True)
    name_prefix = f"{language}_{shard:02d}"
    name_suffix = f"_of_{count:02d}"
    start = time.monotonic()
    deadline = start + max_seconds
    leaves: list[dict] = []
    failed: list[dict] = []
    attempts: list[dict] = []

    def descend(group: list[Source], suffix: str = "") -> bool:
        name = f"{name_prefix}{suffix}{name_suffix}.txt.gz"
        target = output / name
        budget = int(deadline - time.monotonic())
        depth = len(suffix)
        selected = [
            {"file": item.path.name, "min_page_id": item.low, "max_page_id": item.high}
            for item in group
        ]
        if budget <= 60:
            failed.append({"segment": suffix or "root", "sources": selected, "error": "job time budget exhausted"})
            return False
        timeout = min(budget - 30, timeouts[min(depth, len(timeouts) - 1)])
        cmd = [sys.executable, "wiki_to_text.py",
               *(str(item.path) for item in group), str(output), "--output-name", name]
        # A single source file permits page-ID splitting. Multi-file groups are
        # bisected by file first, so all their filters are unbounded.
        if len(group) == 1:
            if group[0].low is not None:
                cmd += ["--min-page-id", str(group[0].low)]
            if group[0].high is not None:
                cmd += ["--max-page-id", str(group[0].high)]
        if suffix:
            cmd.append("--allow-empty")
        attempt_start = time.monotonic()
        error = None
        articles = None
        target.unlink(missing_ok=True)
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, errors="replace", timeout=timeout, check=False)
            logs = proc.stdout[-4500:]
            print(f"{name}: exit={proc.returncode}, elapsed={time.monotonic()-attempt_start:.1f}s", flush=True)
            if proc.returncode != 0:
                error = f"converter exit {proc.returncode}: {logs[-1500:]}"
            elif not target.is_file() or not target.stat().st_size:
                error = "converter produced no archive"
            else:
                validate_gzip(target)
                m = re.search(r"Wrote ([\d,]+) articles", logs)
                articles = int(m.group(1).replace(",", "")) if m else None
        except subprocess.TimeoutExpired:
            error = f"converter exceeded {timeout}s deadline"
        except (OSError, EOFError) as exc:
            error = f"archive validation: {exc}"

        attempts.append({"segment": suffix or "root", "duration_seconds": round(time.monotonic()-attempt_start, 2),
                         "ok": error is None, "error": error})
        if error is None:
            leaves.append({"file": name, "segment": suffix or "root", "sources": selected,
                           "bytes": target.stat().st_size, "articles": articles})
            return True

        target.unlink(missing_ok=True)
        print(f"::warning::{name} failed: {error[-500:]}", flush=True)
        child_groups = split_sources(group) if depth < max_depth else None
        if child_groups is None:
            failed.append({"segment": suffix or "root", "sources": selected, "error": error[-1500:]})
            return False
        left = descend(child_groups[0], suffix + "a")
        right = descend(child_groups[1], suffix + "b")
        return left and right

    recovered = descend(sources) if sources else False
    download_failures = failures or []
    complete = recovered and not download_failures
    report = {
        "language": language, "root_shard": shard, "planned_shards": count,
        "complete": complete, "successful_leaves": leaves,
        "failed_fragments": failed, "download_failures": download_failures,
        "attempts": attempts, "max_split_depth": max_depth,
    }
    (output / f"{name_prefix}{name_suffix}.manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Shard {shard}/{count}: {'complete' if complete else 'PARTIAL'}, "
          f"{len(leaves)} surviving leaf archives, {len(failed)} terminal failed fragments, "
          f"{len(download_failures)} download failures", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", required=True)
    parser.add_argument("--shard", required=True, type=int)
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--ranges-file", required=True, type=Path)
    parser.add_argument("--failures-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-seconds", type=int, default=19000)
    args = parser.parse_args()
    if not 1 <= args.shard <= args.count or not 0 <= args.max_depth <= 8:
        parser.error("invalid shard/count or split depth")
    paths = [Path(x) for x in args.ranges_file.read_text().splitlines() if x.strip()]
    failures = (json.loads(args.failures_file.read_text())
                if args.failures_file and args.failures_file.exists() else [])
    result = convert(args.language, args.shard, args.count,
                     [Source(path) for path in paths], args.output_dir,
                     max_depth=args.max_depth, max_seconds=args.max_seconds,
                     failures=failures)
    if not result["complete"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
