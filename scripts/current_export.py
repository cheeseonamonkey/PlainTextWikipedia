#!/usr/bin/env python3
"""Discover, download, and checksum-verify Wikimedia's new monthly XML exports.

Authoritative source: https://wikitech.wikimedia.org/wiki/MediaWiki_Content_File_Exports
Use ONLY a completed current-revision snapshot (SHA256SUMS exists). The checksum
manifest is the catalog, and every downloaded XML.bz2 must match SHA256 exactly.
No HTML scraping, silent sample loss, or unrelated 'history' dump substitutions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from urllib.parse import quote

USER_AGENT = "PlainTextWikipedia/1.0 (https://github.com/cheeseonamonkey/PlainTextWikipedia)"
ROOT = "https://dumps.wikimedia.org/other/mediawiki_content_current"
FILE_PATTERN = re.compile(r"^(?P<project>[a-z0-9_]+wiki)-(?P<date>\d{4}-\d{2}-\d{2})-p\d+p\d+\.xml\.bz2$")
CHECKSUM_PATTERN = re.compile(r"^([a-fA-F0-9]{64})\s+\*?(\S+)\s*$")


def months_before(today: date | None = None, count: int = 5) -> list[str]:
    today = today or date.today()
    y, m = today.year, today.month
    result = []
    for _ in range(count):
        result.append(f"{y:04d}-{m:02d}-01")
        m -= 1
        if m == 0:
            m, y = 12, y - 1
    return result


def get(url: str, timeout: int = 60):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(request, timeout=timeout)


def parse_manifest(raw: str, language: str, snapshot: str) -> list[dict]:
    files = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        match = CHECKSUM_PATTERN.fullmatch(line)
        if not match:
            raise ValueError(f"Malformed SHA256SUMS entry: {line[:100]!r}")
        digest, name = match.groups()
        name = name.removeprefix("./")
        if "/" in name or "\\" in name or name.startswith("."):
            raise ValueError("Unsafe relative path in export checksum manifest")
        parsed = FILE_PATTERN.fullmatch(name)
        if parsed is None or parsed["project"] != language or parsed["date"] != snapshot:
            raise ValueError(f"Unexpected XML export: {name}")
        low, high = map(int,re.search(r"-p(\d+)p(\d+)\.xml\.bz2$",name).groups())
        if not (0 <= low <= high):
            raise ValueError("Invalid page ID boundaries")
        files.append({"name":name,"sha256":digest.lower(),"low":low,"high":high})
    if not files:
        raise ValueError("Empty XML current content export")
    files.sort(key=lambda x:(x["low"],x["high"],x["name"]))
    names = set()
    last = -1
    for f in files:
        if f["name"] in names or f["low"] <= last:
            raise ValueError("Overlapping or duplicate page ranges in XML export")
        names.add(f["name"])
        last = f["high"]
    return files


def discover(language: str, snapshots: list[str] | None = None) -> dict | None:
    if not re.fullmatch(r"[a-z0-9_]+wiki", language):
        raise ValueError("Invalid Wikimedia project code")
    for snapshot in snapshots or months_before():
        base=f"{ROOT}/{quote(language)}/{snapshot}/xml/bzip2"
        try:
            with get(f"{base}/SHA256SUMS") as response:
                raw=response.read(64 * 1024 * 1024+1)
                if len(raw)>64*1024*1024:
                    raise ValueError("Unexpectedly large SHA256SUMS")
            entries=parse_manifest(raw.decode("utf-8"),language,snapshot)
            return {"source":"mediawiki_content_current","project":language,
                    "snapshot":snapshot,"base":base,"files":entries}
        except urllib.error.HTTPError as e:
            if e.code in (403,404):
                continue
            raise
        except (TimeoutError,urllib.error.URLError) as exc:
            print(f"::warning::Cannot check current export {snapshot} for {language}: {exc}",
                  file=sys.stderr,flush=True)
            continue
    return None


def partition(catalog: dict, shard: int, count: int) -> list[dict]:
    files=catalog["files"]
    if not 0 <= shard < count or count < 1 or count > len(files):
        raise ValueError("Invalid export file shard/count")
    return [x for i,x in enumerate(files) if i%count==shard]


def download(catalog: dict, shard: int, count: int, output: Path,
             allow_partial: bool = True) -> tuple[list[Path],list[dict]]:
    chosen=partition(catalog,shard,count)
    output.mkdir(parents=True,exist_ok=True)
    successes=[]
    failures=[]
    for file in chosen:
        name=file["name"]
        target=output/name
        checksum=file["sha256"]
        url=f'{catalog["base"]}/{quote(name)}'
        verified=False
        error=""
        if target.is_file():
            digest=hashlib.sha256()
            with target.open("rb") as stream:
                while chunk:=stream.read(1024*1024):
                    digest.update(chunk)
            verified=digest.hexdigest()==checksum
            if not verified:
                target.unlink()
        if not verified:
            temp=target.with_name(name+".download")
            for attempt in range(3):
                digest=hashlib.sha256()
                try:
                    with get(url,timeout=300) as response,temp.open("wb") as sink:
                        while chunk:=response.read(1024*1024):
                            sink.write(chunk)
                            digest.update(chunk)
                    if digest.hexdigest()!=checksum:
                        raise ValueError(f"SHA256 mismatch for {name}")
                    temp.replace(target)
                    verified=True
                    break
                except (OSError,ValueError) as exc:
                    error=str(exc)
                    temp.unlink(missing_ok=True)
                    if attempt<2:
                        time.sleep(5*(attempt+1))
        if verified:
            successes.append(target)
        else:
            failure={"file":name,"source":url,"error":error,
                     "start_page_id":file["low"],"end_page_id":file["high"]}
            failures.append(failure)
            print(f"::warning::Failed export file {name}: {error}",file=sys.stderr)
            if not allow_partial:
                raise RuntimeError(failure)
    return successes,failures


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest="action",required=True)
    p=sub.add_parser("plan")
    p.add_argument("--language",required=True)
    p.add_argument("--max-shards",type=int,default=16)
    q=sub.add_parser("fetch")
    q.add_argument("--language",required=True)
    q.add_argument("--shard",type=int,required=True)
    q.add_argument("--count",type=int,required=True)
    q.add_argument("--snapshot",required=True)
    q.add_argument("--output-dir",type=Path,required=True)
    q.add_argument("--failures-file",type=Path,required=True)
    args=ap.parse_args()
    if args.action=="plan":
        catalog=discover(args.language)
        if not catalog:
            raise RuntimeError(f"No complete new-format current export found for {args.language}")
        print(json.dumps({"mode":"current_export","count":min(args.max_shards,len(catalog["files"])),
                          "archive_bytes":0,"snapshot":catalog["snapshot"],
                          "files":len(catalog["files"]),
                          "reason":"checksum-verified monthly current XML files"}))
    else:
        if not re.fullmatch(r"\d{4}-\d{2}-01",args.snapshot):
            ap.error("Invalid monthly snapshot")
        catalog=discover(args.language,[args.snapshot])
        if not catalog:
            raise RuntimeError("Pinned monthly snapshot is not complete")
        good,bad=download(catalog,args.shard,args.count,args.output_dir)
        args.failures_file.write_text(json.dumps(bad,indent=2)+"\n")
        for file in good:
            print(file.resolve())


if __name__=="__main__":
    main()
