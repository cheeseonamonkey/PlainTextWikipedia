#!/usr/bin/env python3
"""Download non-multistream Wikimedia XML page ranges, partitioned without overlap."""

from __future__ import annotations

import argparse
from datetime import datetime
import re
import sys
import time
import urllib.request
from pathlib import Path

USER_AGENT = "PlainTextWikipedia/1.0 (https://github.com/cheeseonamonkey/PlainTextWikipedia)"


def open_url(url: str, timeout: int = 60):
    return urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": USER_AGENT}),
        timeout=timeout,
    )


def page_range_catalog(
    base_url: str, language: str,
) -> list[tuple[int, int, str, str, int]]:
    """Read the directory once, retaining each file's published date and size.

    Wikimedia's /latest/ indexes can still include page ranges from last month's
    dump alongside the current one. These often return HTTP 404; some contain
    valid but *overlapping* old content and must not enter the new dataset.
    """
    listing = None
    for attempt in range(4):
        try:
            with open_url(f"{base_url.rstrip('/')}/") as response:
                listing = response.read().decode("utf-8", errors="replace")
            break
        except Exception:
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))

    pattern = re.compile(
        rf"({re.escape(language)}-latest-pages-articles\d+\.xml-p(\d+)p(\d+)\.bz2)"
    )
    found = {
        name: (int(start), int(end))
        for name, start, end in pattern.findall(listing)
    }
    dates: dict[str, tuple[str, int]] = {}
    listing_pattern = re.compile(
        rf'<a\s+href="({re.escape(language)}-latest-pages-articles\d+\.xml-p\d+p\d+\.bz2)"'
        r'[^>]*>[^<]*</a>\s+(\d{2}-[A-Za-z]{3}-\d{4})\s+\d{2}:\d{2}\s+(\d+)'
    )
    for name, date, size in listing_pattern.findall(listing):
        dates[name] = (date, int(size))
    return sorted(
        (
            (start, end, name, *dates.get(name, ("", 0)))
            for name, (start, end) in found.items()
        ),
        key=lambda item: (item[0], item[1], item[2]),
    )


def latest_page_ranges(
    catalog: list[tuple[int, int, str, str, int]],
) -> list[tuple[int, int, str, str, int]]:
    dated = [entry for entry in catalog if entry[3] and entry[4] >= 4096]
    if not dated:
        return catalog  # Unannotated listings in tests/other Wikimedia mirrors.
    newest = max(datetime.strptime(entry[3], "%d-%b-%Y") for entry in dated)
    valid = [
        entry for entry in catalog
        if entry[3] and entry[4] >= 4096
        and datetime.strptime(entry[3], "%d-%b-%Y") == newest
    ]
    # Refuse gaps and overlaps in the current dump, rather than silently
    # publishing partial or duplicated article ranges.
    valid_sorted = sorted(valid, key=lambda x: (x[0], x[1]))
    for prev, curr in zip(valid_sorted, valid_sorted[1:]):
        if curr[0] <= prev[1]:
            raise RuntimeError(
                f"Overlapping current dump page ranges: {prev[2]} and {curr[2]}"
            )
    return valid


def page_ranges(base_url: str, language: str) -> list[tuple[int, int, str]]:
    """Legacy listing API, useful for inspecting original shard assignments."""
    return [(start, end, name) for start, end, name, _, _ in page_range_catalog(base_url, language)]


def download(url: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size:
        return
    temporary = destination.with_name(destination.name + ".part")
    for attempt in range(4):
        try:
            with open_url(url, timeout=300) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            if not temporary.stat().st_size:
                raise RuntimeError(f"Empty download: {url}")
            temporary.replace(destination)
            return
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(10 * (attempt + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--shard", type=int, help="zero-based shard number")
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--count-only", action="store_true")
    parser.add_argument(
        "--preserve-layout", action="store_true",
        help="Keep old round-robin shard positions, but exclude outdated dump files",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    catalog = page_range_catalog(args.base_url, args.language)
    current = latest_page_ranges(catalog)
    if not current:
        raise RuntimeError("No current non-multistream page-range files found")
    if len(current) != len(catalog):
        print(
            f"Excluding {len(catalog) - len(current)} obsolete/overlapping dump links "
            f"from {len(catalog)} listed files", file=sys.stderr,
        )
    if args.count_only:
        print(len(current))
        return
    if args.shard is None or args.shard_count is None or args.output_dir is None:
        parser.error("--shard, --shard-count and --output-dir are required")
    if args.shard_count < 1 or not 0 <= args.shard < args.shard_count:
        parser.error("--shard must be within [0, shard-count)")

    allowed = {entry[2] for entry in current}
    # During recovery, preserve existing per-shard assignments to avoid
    # duplicating coverage or invalidating the 60+ successfully built shards.
    # For a new build, partition only today's valid dump files.
    partition = catalog if args.preserve_layout else current
    selected = [
        (start, end, name) for i, (start, end, name, _, _) in enumerate(partition)
        if i % args.shard_count == args.shard and name in allowed
    ]
    if not selected:
        raise RuntimeError("No current dump files assigned to this shard")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for _, _, filename in selected:
        destination = args.output_dir / filename
        download(f"{args.base_url.rstrip('/')}/{filename}", destination)
        print(destination.resolve())
    print(
        f"Selected {len(selected)}/{len(ranges)} unique page-range files for "
        f"shard {args.shard + 1}/{args.shard_count}", file=sys.stderr
    )


if __name__ == "__main__":
    main()
