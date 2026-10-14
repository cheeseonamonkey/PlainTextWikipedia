#!/usr/bin/env python3
"""Repack completed conversion leaves into up to 16 named publication shards.

The downloader and converter should not be repeated merely to obtain more files.
Each build worker streams its *already converted* archives once and assigns its
articles to globally unique output bins. The original root count remains in the
recovery manifest for accurate completeness checks.

For 1 input worker -> 16 output bins (01..16).
For 4 input workers -> 4 bins each (01,05,09,13; 02,06,10,14; ...).
Any number of source XML archives can produce 16 independently gzipped outputs.

Incomplete roots remain incomplete; this does not invent or recover missing text.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from contextlib import ExitStack
from pathlib import Path


def iter_articles(path: Path):
    """Yield exactly the original converter's title/body pairs, incrementally."""
    with gzip.open(path, "rt", encoding="utf-8", newline="") as src:
        title = None
        body = None
        for line in src:
            line = line.rstrip("\n\r")
            if title is None:
                if not line:
                    raise ValueError(f"Unexpected empty article title in {path.name}")
                title = line
            elif body is None:
                if not line:
                    raise ValueError(f"Empty article body in {path.name}: {title}")
                body = line
            else:
                if line:
                    raise ValueError(f"Missing article separator in {path.name}: {title}")
                yield title, body
                title, body = None, None
        if title is not None or body is not None:
            raise ValueError(f"Incomplete trailing article in {path.name}")


def output_ids(root: int, roots: int, target: int) -> list[int]:
    if not 1 <= root <= roots <= target <= 64:
        raise ValueError("Require 1 <= root <= root count <= output count <= 64")
    return list(range(root, target + 1, roots))


def repartition(language: str, root: int, roots: int, target: int, output: Path) -> dict | None:
    """Update one recovery manifest atomically after all new gzips validate."""
    report_path = output / f"{language}_{root:02d}_of_{roots:02d}.manifest.json"
    if not report_path.exists():
        return None  # Build may have failed before conversion.
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report["language"], report["root_shard"], report["planned_shards"]) != (language, root, roots):
        raise ValueError("Inconsistent recovery report")
    if report.get("output_shards") == target:
        return report  # Idempotent for workflow retries.
    old = report["successful_leaves"]
    if not old:
        return report
    ids = output_ids(root, roots, target)
    if len(ids) == 1:
        # Normal already-16-shard conversions retain the original leaf layout.
        report["output_shards"] = target
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        return report

    expected = re.compile(rf"^{re.escape(language)}_{root:02d}[ab]*_of_{roots:02d}\.txt\.gz$")
    inputs = [output / item["file"] for item in old]
    for path in inputs:
        if not expected.fullmatch(path.name) or not path.is_file():
            raise ValueError(f"Missing or unexpected source leaf: {path.name}")
    destinations = [output / f"{language}_{i:02d}_of_{target:02d}.txt.gz" for i in ids]
    if len(set(destinations)) != len(destinations) or any(p in inputs for p in destinations):
        raise ValueError("Output file collision")
    temporaries = [path.with_name(path.name + ".pending") for path in destinations]
    totals = [0] * len(ids)
    overall = 0
    try:
        with ExitStack() as stack:
            writers = [
                stack.enter_context(gzip.open(path, "wt", encoding="utf-8", compresslevel=6,
                                              newline=""))
                for path in temporaries
            ]
            for source in inputs:
                for title, body in iter_articles(source):
                    # Stable, language-independent routing; never relies on
                    # randomized Python hash() or English tokenization.
                    digest = hashlib.blake2b(title.encode("utf-8"), digest_size=8).digest()
                    bucket = int.from_bytes(digest, "big") % len(writers)
                    writers[bucket].write(f"{title}\n{body}\n\n")
                    totals[bucket] += 1
                    overall += 1

        known_counts = [x["articles"] for x in old if x.get("articles") is not None]
        if len(known_counts) == len(old) and overall != sum(known_counts):
            raise ValueError(f"Article count changed while repartitioning: "
                             f"{overall} != {sum(known_counts)}")
        if not overall:
            raise ValueError("Refusing to claim completeness with zero articles")
        for temp in temporaries:
            with gzip.open(temp, "rb") as stream:
                while stream.read(1024 * 1024):
                    pass

        # Preserve source provenance and recovery failure metadata.
        originals = old
        new_leaves = [
            {"file": path.name, "segment": f"output-{i:02d}",
             "articles": n, "bytes": tmp.stat().st_size,
             "partition": {"root": root, "worker_count": roots, "bucket": bucket,
                           "buckets_for_worker": len(ids), "method": "blake2b-64(title)%buckets"},
             "sources": [src for leaf in originals for src in leaf.get("sources", [])]}
            for bucket, (i, path, tmp, n) in enumerate(zip(ids, destinations, temporaries, totals))
        ]
        updated = dict(report, output_shards=target,
                       successful_leaves=new_leaves, repartitioned_articles=overall)
        # Write the manifest before deleting original source leaves. If anything
        # fails, all originals still exist and can be recovered from the job.
        new_report = report_path.with_suffix(report_path.suffix + ".pending")
        new_report.write_text(json.dumps(updated, indent=2, ensure_ascii=False) + "\n",
                              encoding="utf-8")
        for temp, path in zip(temporaries, destinations):
            temp.replace(path)
        new_report.replace(report_path)
        for path in inputs:
            path.unlink()
        print(f"Repartitioned {language} root {root}/{roots}: "
              f"{overall:,} articles into {len(ids)} gzips "
              f"({sum(bool(n) for n in totals)} nonempty)", flush=True)
        return updated
    finally:
        for path in temporaries:
            path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", required=True)
    parser.add_argument("--root", type=int, required=True)
    parser.add_argument("--roots", type=int, required=True)
    parser.add_argument("--target-shards", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repartition(args.language, args.root, args.roots, args.target_shards, args.output_dir)


if __name__ == "__main__":
    main()
