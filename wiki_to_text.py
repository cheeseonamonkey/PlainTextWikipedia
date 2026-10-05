"""Command-line entry point for the Wikipedia plaintext converter."""

from __future__ import annotations

import argparse
from pathlib import Path

from dewiki_functions import write_plaintext_shards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", type=Path, help="XML, XML.BZ2, XML.GZ, or XML.XZ dump")
    parser.add_argument("output", type=Path, help="directory for compressed text shards")
    parser.add_argument("--shards", type=int, default=1, help="number of stable page-ID shards")
    parser.add_argument("--prefix", default="wiki", help="output filename prefix")
    args = parser.parse_args()

    count = write_plaintext_shards(
        args.dump,
        args.output,
        shard_count=args.shards,
        prefix=args.prefix,
    )
    print(f"Wrote {count:,} articles to {args.output}")


if __name__ == "__main__":
    main()
