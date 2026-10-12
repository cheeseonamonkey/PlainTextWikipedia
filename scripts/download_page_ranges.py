#!/usr/bin/env python3
"""Download non-multistream Wikimedia XML page ranges, partitioned without overlap."""

from __future__ import annotations

import argparse
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


def page_ranges(base_url: str, language: str) -> list[tuple[int, int, str]]:
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

    # Non-multistream files are independently parseable XML. Do not mix them
    # with the multistream variant: that would duplicate the whole corpus.
    pattern = re.compile(
        rf"({re.escape(language)}-latest-pages-articles\d+\.xml-p(\d+)p(\d+)\.bz2)"
    )
    found = {
        name: (int(start), int(end))
        for name, start, end in pattern.findall(listing)
    }
    return sorted(
        ((start, end, name) for name, (start, end) in found.items()),
        key=lambda item: (item[0], item[1], item[2]),
    )


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
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    ranges = page_ranges(args.base_url, args.language)
    if not ranges:
        raise RuntimeError("No non-multistream page-range files found")

    if args.count_only:
        print(len(ranges))
        return
    if args.shard is None or args.shard_count is None or args.output_dir is None:
        parser.error("--shard, --shard-count and --output-dir are required")
    if args.shard_count < 1 or not 0 <= args.shard < args.shard_count:
        parser.error("--shard must be within [0, shard-count)")

    # Every range belongs to exactly one shard, with no boundary overlaps.
    selected = [entry for i, entry in enumerate(ranges) if i % args.shard_count == args.shard]
    if not selected:
        raise RuntimeError("No page-range files assigned to this shard")

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
