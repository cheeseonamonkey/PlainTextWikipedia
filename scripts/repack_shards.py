#!/usr/bin/env python3
"""Repack a completed Wikipedia output into up to N lossless article shards.

Source dump concurrency is separate from the number of files published.
A tiny Wikipedia can convert in one job without re-downloading 16 copies of
the dump, then publish 16 independently gzip-compressed chunks. The existing
recursive a/b recovery identifiers are retained; a '_partNN_of_MM' suffix
denotes post-conversion file partitioning, NOT failed-source recovery.

Each article remains byte-identical. Original files are only deleted AFTER all
new archives and their round-robin reconstruction have been checked.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import Iterator


def records(file: Path) -> Iterator[bytes]:
    """Read the converter's title\\nbody\\n\\n records without changing bytes."""
    with gzip.open(file, "rb") as stream:
        while True:
            title = stream.readline()
            if not title:
                break
            if not title.strip():
                raise ValueError(f"Unexpected blank title in {file.name}")
            body = stream.readline()
            ending = stream.readline()
            if not body or not body.strip() or ending not in (b"\n", b"\r\n"):
                raise ValueError(f"Malformed or truncated article in {file.name}")
            yield title + body + ending


def sha_and_count(file: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    n = 0
    for record in records(file):
        digest.update(record)
        n += 1
    return digest.hexdigest(), n


def split_archive(file: Path, parts: int, staging: Path) -> list[dict]:
    digest, count = sha_and_count(file)
    if parts < 2 or count < 2:
        return []
    parts = min(parts, count)
    stem = file.name[:-len(".txt.gz")]
    result = []
    filenames = [f"{stem}_part{i+1:02d}_of_{parts:02d}.txt.gz" for i in range(parts)]
    writers = []
    try:
        for name in filenames:
            writers.append(gzip.open(staging / name, "wb", compresslevel=6))
        sizes = [0] * parts
        amounts = [0] * parts
        for i, record in enumerate(records(file)):
            slot = i % parts
            writers[slot].write(record)
            sizes[slot] += len(record)
            amounts[slot] += 1
    finally:
        for writer in writers:
            writer.close()
    # Reconstitute the original article order from the independently compressed
    # shards and verify the SHA-256 digest of every original byte.
    check = hashlib.sha256()
    with ExitStack() as stack:
        sources = [stack.enter_context(gzip.open(staging / name, "rb")) for name in filenames]
        for i in range(count):
            source = sources[i % parts]
            title, body, ending = source.readline(), source.readline(), source.readline()
            if not title or not body or ending not in (b"\n", b"\r\n"):
                raise RuntimeError(f"Invalid split record: {filenames[i%parts]}")
            check.update(title + body + ending)
        if any(source.read(1) for source in sources):
            raise RuntimeError("Extra records found in split shards")
    if digest != check.hexdigest() or sum(amounts) != count:
        raise RuntimeError("Repacked Wikipedia shards FAILED lossless SHA-256 validation")
    for i, name in enumerate(filenames):
        result.append({"file":name,"articles":amounts[i],
                       "bytes":(staging/name).stat().st_size,
                       "uncompressed_bytes":sizes[i],
                       "partition":{"strategy":"article_index_modulo",
                                    "source_archive":file.name,
                                    "source_sha256":digest,
                                    "index":i+1,"parts":parts}})
    return result


def repack(directory: Path, target: int = 16) -> dict:
    if target < 2 or target > 64:
        raise ValueError("Target must be in [2,64]")
    reports = []
    for path in sorted(directory.glob("*.manifest.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        reports.append((path, report))
    if not reports:
        return {"repacked":0,"files_before":0,"files_after":0,"reason":"No conversion manifests"}
    eligible = []
    for path, report in reports:
        # An incomplete root can still contain independently validated,
        # successful leaves. Repacking those does not fill missing source
        # coverage, and the 'complete' flag stays false.
        for leaf in report.get("successful_leaves", []):
            source = directory / leaf["file"]
            if source.is_file() and "_part" not in source.name:
                eligible.append((path, report, leaf, source))
    current_count = sum(len(r.get("successful_leaves", [])) for _,r in reports)
    if current_count >= target or not eligible:
        return {"repacked":0,"files_before":current_count,
                "files_after":current_count,"reason":"Already sharded or no valid inputs"}

    # Divide the desired extra outputs among available original leaf archives,
    # preserving source-root recovery semantics. E.g. 1 input => 16 outputs,
    # 2 inputs => 8 each, 3 => 6/5/5.
    requested = [1] * len(eligible)
    for n in range(max(0,target-current_count)):
        requested[n % len(eligible)] += 1

    changed = 0
    added = 0
    for (path, report, leaf, source), desired in zip(eligible, requested):
        if desired < 2:
            continue
        with tempfile.TemporaryDirectory(prefix="wiki-repack-",dir=directory) as staging_name:
            staging = Path(staging_name)
            parts = split_archive(source, desired, staging)
            if not parts:
                continue
            # Success-path transaction: place finished shards first; update
            # report second; remove original only after manifest commit.
            for item in parts:
                (staging/item["file"]).replace(directory/item["file"])
            enriched = [{**leaf,**item} for item in parts]
            copy = dict(report)
            copy["successful_leaves"] = [
                *[x for x in report["successful_leaves"] if x["file"] != source.name],
                *enriched
            ]
            copy["storage_repacked"] = True
            temp_report = path.with_name(path.name + ".tmp")
            temp_report.write_text(json.dumps(copy,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
            temp_report.replace(path)
            source.unlink()
            report.clear()
            report.update(copy)
            changed += 1
            added += len(parts)-1
            print(f"{source.name} -> {len(parts)} losslessly verified archives",flush=True)
    return {"repacked":changed,"files_before":current_count,
            "files_after":current_count+added}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",type=Path,required=True)
    parser.add_argument("--target",type=int,default=16)
    args=parser.parse_args()
    print(json.dumps(repack(args.directory,args.target),indent=2))


if __name__=="__main__":
    main()
