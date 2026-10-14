#!/usr/bin/env python3
"""Parallelize a *single* Wikimedia dump using indexed multistream byte ranges.

Large, single-file editions may still have a separate multistream archive and
index. Each index entry identifies a compressed bzip2 stream boundary, so
individual CI jobs can fetch disjoint HTTP 206 byte ranges, turn complete
<page> fragments into standalone XML, and convert them independently.

For small one-file dumps, retain the existing one-job path: repeatedly
downloading or dividing a tiny wiki into 16 jobs wastes resources.

Usage:
  python scripts/multistream_shards.py plan --base-url URL --language hawwiki --max-shards 16
  python scripts/multistream_shards.py fetch --base-url URL --language hawwiki \
      --shard 0 --count 4 --expected-size N --output-dir ranges
"""

from __future__ import annotations

import argparse
import bisect
import bz2
import json
import math
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

# Works both as python -m scripts.multistream_shards and as an executable script.
try:
    from scripts.download_page_ranges import page_range_catalog, latest_page_ranges, USER_AGENT
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from download_page_ranges import page_range_catalog, latest_page_ranges, USER_AGENT

MIN_INDEXED_SIZE = 128 * 1024 * 1024
TARGET_BYTES_PER_JOB = 128 * 1024 * 1024
PAGE_OPEN = b"<page>"
PAGE_CLOSE = b"</page>"
PAGE_CLOSE_SIZE = len(PAGE_CLOSE)


def urlopen(url: str, method: str = "GET", headers: dict | None = None, timeout: int = 90):
    headers = {"User-Agent": USER_AGENT, **(headers or {})}
    return urllib.request.urlopen(
        urllib.request.Request(url, headers=headers, method=method),
        timeout=timeout,
    )


def paths(base: str, language: str) -> tuple[str, str]:
    base = base.rstrip("/")
    return (f"{base}/{language}-latest-pages-articles-multistream.xml.bz2",
            f"{base}/{language}-latest-pages-articles-multistream-index.txt.bz2")


def byte_range_available(url: str) -> tuple[bool, int]:
    """Refuse servers ignoring Range (200), rather than downloading all bytes."""
    try:
        with urlopen(url, headers={"Range": "bytes=0-0"}, timeout=30) as response:
            if response.status != 206:
                return False, 0
            header = response.headers.get("Content-Range", "")
            match = re.fullmatch(r"bytes 0-0/(\d+)", header)
            if not match:
                return False, 0
            body = response.read(2)
            return body and len(body) == 1, int(match.group(1))
    except (OSError, ValueError):
        return False, 0


def index_streams(url: str) -> list[tuple[int, int]]:
    """Collect distinct (byte offset, first page ID) pairs, not every title."""
    with tempfile.NamedTemporaryFile(suffix=".bz2", delete=True) as tmp:
        for attempt in range(3):
            try:
                with urlopen(url, timeout=120) as response:
                    while block := response.read(1024 * 1024):
                        tmp.write(block)
                break
            except OSError:
                if attempt == 2:
                    raise
                tmp.seek(0)
                tmp.truncate()
                time.sleep(3 * (attempt + 1))
        tmp.flush()
        entries: list[tuple[int, int]] = []
        with bz2.open(tmp.name, "rt", encoding="utf-8", errors="replace") as data:
            last_offset = -1
            for line in data:
                parts = line.split(":", 2)
                if len(parts) != 3:
                    continue
                offset, page_id = int(parts[0]), int(parts[1])
                if offset != last_offset:
                    if offset <= last_offset:
                        raise ValueError("Non-increasing multistream offsets")
                    entries.append((offset, page_id))
                    last_offset = offset
        if not entries:
            raise ValueError("Empty Wikimedia multistream index")
        return entries


def boundaries(streams: list[tuple[int, int]], size: int, count: int) -> list[tuple[int, int, int, int]]:
    """Exact nonoverlapping [start,end) byte ranges at bzip2 stream offsets."""
    if count < 1 or count > len(streams) or not 0 <= streams[0][0] < size:
        raise ValueError("Invalid multistream partition size or count")
    offsets = [offset for offset, _ in streams]
    indexes = [0]
    for shard in range(1, count):
        desired = streams[0][0] + (size - streams[0][0]) * shard // count
        nearest = bisect.bisect_left(offsets, desired)
        floor = indexes[-1] + 1
        ceiling = len(streams) - (count - shard)
        nearest = max(floor, min(ceiling, nearest))
        if nearest > floor and abs(offsets[nearest - 1] - desired) < abs(offsets[nearest] - desired):
            nearest -= 1
        indexes.append(max(floor, nearest))
    indexes.append(len(streams))
    result = []
    for shard in range(count):
        first = indexes[shard]
        last = indexes[shard + 1]
        start = offsets[first]
        end = offsets[last] if last < len(streams) else size
        lo = streams[first][1]
        hi = streams[last][1] - 1 if last < len(streams) else 2147483647
        if start >= end or lo > hi:
            raise ValueError("Invalid/overlapping multistream page range")
        result.append((start, end, lo, hi))
    assert all(a[1] == b[0] for a, b in zip(result, result[1:]))
    return result


def plan(base: str, language: str, max_shards: int) -> dict:
    """Prefer original numbered page-range files, then safe indexed fallback."""
    catalog = page_range_catalog(base, language)
    current = latest_page_ranges(catalog)
    numbered = [x for x in current if re.search(r"\.xml-p\d+p\d+\.bz2$", x[2])]
    if numbered:
        return {"mode": "normal", "count": min(max_shards, len(current)),
                "archive_bytes": 0, "reason": "official numbered page-range dumps"}
    whole, idx = paths(base, language)
    # If Range is unavailable, original complete single-file conversion remains
    # correct and avoids duplicate downloads.
    supported, size = byte_range_available(whole)
    if supported and size >= MIN_INDEXED_SIZE:
        try:
            streams = index_streams(idx)
            count = min(max_shards, len(streams),
                        max(2, math.ceil(size / TARGET_BYTES_PER_JOB)))
            if count > 1:
                boundaries(streams, size, count)
                return {"mode": "indexed", "count": count,
                        "archive_bytes": size,
                        "reason": "indexed multistream HTTP byte-range partition"}
        except (OSError, ValueError, EOFError) as exc:
            print(f"WARNING: Indexed partition unavailable: {exc}", file=sys.stderr)
    if not current:
        raise RuntimeError("Neither numbered, single-file, nor usable indexed dump found")
    return {"mode": "normal", "count": min(max_shards, len(current)),
            "archive_bytes": 0,
            "reason": "single-file archive: indexed sharding not available or unnecessary"}


def fetch(base: str, language: str, shard: int, count: int,
          expected_size: int, output_dir: Path) -> Path:
    whole, idx = paths(base, language)
    supported, actual_size = byte_range_available(whole)
    if not supported or expected_size != actual_size:
        raise RuntimeError("Source changed or stopped supporting HTTP 206; refusing duplicate full-archive download")
    spans = boundaries(index_streams(idx), actual_size, count)
    if not 0 <= shard < len(spans):
        raise ValueError("Invalid shard index")
    start, end, low, high = spans[shard]
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{language}-latest-pages-articles1.xml-p{low}p{high}.bz2"
    target = output_dir / filename
    if target.is_file() and target.stat().st_size > 0:
        return target
    temp = target.with_suffix(".part")
    partial = output_dir / (filename + ".source.part")
    print(f"Reading {language} shard {shard+1}/{count}: bytes {start}–{end-1}, pages {low}–{high}",
          file=sys.stderr, flush=True)
    try:
        for attempt in range(3):
            try:
                with urlopen(whole, headers={"Range": f"bytes={start}-{end-1}"}, timeout=300) as response:
                    wanted = f"bytes {start}-{end-1}/{actual_size}"
                    if response.status != 206 or response.headers.get("Content-Range") != wanted:
                        raise RuntimeError("Server ignored/changed byte Range; refusing full download")
                    written = 0
                    with partial.open("wb") as out:
                        while block := response.read(1024 * 1024):
                            out.write(block)
                            written += len(block)
                    if written != end - start:
                        raise IOError(f"Truncated HTTP Range: got {written} expected {end-start}")
                break
            except (OSError, RuntimeError):
                partial.unlink(missing_ok=True)
                if attempt == 2:
                    raise
                time.sleep(5 * (attempt + 1))

        # Wikimedia's indexed chunks contain complete <page>...</page>
        # elements but generally omit the document-level <mediawiki> wrapper.
        # Wrap only the pages; do not include the original first/last root tags.
        n_pages = 0
        buffer = b""
        with bz2.open(partial, "rb") as inp, bz2.open(temp, "wb", compresslevel=1) as out:
            out.write(b"<mediawiki>\n")
            while data := inp.read(1024 * 1024):
                buffer += data
                while True:
                    opening = buffer.find(PAGE_OPEN)
                    if opening == -1:
                        buffer = buffer[-len(PAGE_OPEN):]
                        break
                    closing = buffer.find(PAGE_CLOSE, opening)
                    if closing == -1:
                        buffer = buffer[opening:]
                        if len(buffer) > 128 * 1024 * 1024:
                            raise RuntimeError("Page XML exceeds the supported streaming buffer")
                        break
                    closing += PAGE_CLOSE_SIZE
                    out.write(buffer[opening:closing] + b"\n")
                    n_pages += 1
                    buffer = buffer[closing:]
            if PAGE_OPEN in buffer:
                raise RuntimeError("Incomplete XML page at indexed stream boundary")
            out.write(b"</mediawiki>\n")
        if n_pages == 0:
            raise RuntimeError("No XML pages recovered from indexed stream")
        temp.replace(target)
        print(f"{target.resolve()} | {n_pages} pages", file=sys.stderr)
        return target
    finally:
        partial.unlink(missing_ok=True)
        temp.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--base-url", required=True)
    p.add_argument("--language", required=True)
    p.add_argument("--max-shards", type=int, default=16)
    f = sub.add_parser("fetch")
    f.add_argument("--base-url", required=True)
    f.add_argument("--language", required=True)
    f.add_argument("--shard", type=int, required=True)
    f.add_argument("--count", type=int, required=True)
    f.add_argument("--expected-size", type=int, required=True)
    f.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args()
    if args.command == "plan":
        if not 1 <= args.max_shards <= 64:
            p.error("max-shards must be 1..64")
        print(json.dumps(plan(args.base_url, args.language, args.max_shards)))
    else:
        target = fetch(args.base_url, args.language, args.shard,
                       args.count, args.expected_size, args.output_dir)
        print(target.resolve())


if __name__ == "__main__":
    main()
