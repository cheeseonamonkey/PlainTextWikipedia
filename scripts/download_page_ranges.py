#!/usr/bin/env python3
"""Download only the Wikimedia XML page-range files for one logical shard."""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.request
from pathlib import Path


def page_ranges(base_url: str, language: str) -> list[tuple[int, int, str]]:
    listing_url = f"{base_url.rstrip('/')}/"
    with urllib.request.urlopen(listing_url, timeout=60) as response:
        html = response.read().decode("utf-8", errors="replace")

    pattern = re.compile(
        rf"{re.escape(language)}-latest-pages-articles\d+\.xml-p(\d+)p(\d+)\.bz2"
    )
    found: dict[str, tuple[int, int]] = {}
    for start_text, end_text in pattern.findall(html):
        start, end = int(start_text), int(end_text)
        name = f"{language}-latest-pages-articles"
        for candidate in re.findall(
            rf"{re.escape(language)}-latest-pages-articles\d+\.xml-p{start}p{end}\.bz2",
            html,
        ):
            found[candidate] = (start, end)
    return [(start, end, name) for name, (start, end) in found.items()]


def download(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, 5):
        try:
            with urllib.request.urlopen(url, timeout=300) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            temporary.replace(destination)
            return
        except Exception:
            if attempt == 4:
                raise
            time.sleep(attempt * 10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--shard", type=int, required=True, help="zero-based shard number")
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if not 0 <= args.shard < args.shard_count:
        parser.error("--shard must be within [0, shard-count)")
    ranges = page_ranges(args.base_url, args.language)
    if not ranges:
        raise RuntimeError("No page-range files found in the Wikimedia listing")

    maximum_page_id = max(end for _, end, _ in ranges)
    lower = maximum_page_id * args.shard // args.shard_count
    upper = maximum_page_id * (args.shard + 1) // args.shard_count
    selected = sorted(
        (start, end, filename)
        for start, end, filename in ranges
        if end > lower and start <= upper
    )
    if not selected:
        raise RuntimeError("No page-range files overlap the requested shard")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for start, end, filename in selected:
        destination = args.output_dir / filename
        download(f"{args.base_url.rstrip('/')}/{filename}", destination)
        print(destination.resolve())

    print(
        f"Selected {len(selected)} page-range files for shard "
        f"{args.shard + 1}/{args.shard_count} (page IDs {lower + 1}-{upper})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
