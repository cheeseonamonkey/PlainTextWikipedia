"""Command-line entry point for the Wikipedia plaintext converter."""

from __future__ import annotations

import argparse
from pathlib import Path

from dewiki_functions import write_plaintext_shards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dumps",
        nargs="+",
        type=Path,
        help="one or more XML, XML.BZ2, XML.GZ, or XML.XZ dump files",
    )
    parser.add_argument("output", type=Path, help="directory for compressed text shards")
    parser.add_argument("--shards", type=int, default=1, help="number of stable page-ID shards")
    parser.add_argument("--prefix", default="wiki", help="output filename prefix")
    parser.add_argument(
        "--output-name",
        help="exact output filename; valid when --shards is 1",
    )
    parser.add_argument("--min-page-id", type=int, help="Inclusive minimum page ID for recovery splits")
    parser.add_argument("--max-page-id", type=int, help="Inclusive maximum page ID for recovery splits")
    parser.add_argument("--allow-empty", action="store_true", help="Allow an empty page-ID segment")
    args = parser.parse_args()
    if (args.min_page_id is not None and args.max_page_id is not None
            and args.min_page_id > args.max_page_id):
        parser.error("--min-page-id cannot exceed --max-page-id")

    count = write_plaintext_shards(
        args.dumps,
        args.output,
        shard_count=args.shards,
        prefix=args.prefix,
        output_name=args.output_name,
        min_page_id=args.min_page_id,
        max_page_id=args.max_page_id,
        allow_empty=args.allow_empty,
    )
    print(f"Wrote {count:,} articles to {args.output}")


if __name__ == "__main__":
    main()
