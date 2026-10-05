"""Streaming Wikimedia XML to clean article text.

The converter accepts one or more independently parseable XML dump files. This
makes Wikimedia page-range files usable without loading a full language dump.
"""

from __future__ import annotations

import bz2
import gzip
import lzma
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, TextIO

from html2text import html2text
import wikitextparser as wtp


@dataclass(frozen=True)
class Article:
    page_id: int
    title: str
    text: str


def _open_dump(path: Path) -> TextIO:
    suffix = path.name.lower()
    if suffix.endswith(".bz2"):
        return bz2.open(path, "rt", encoding="utf-8")
    if suffix.endswith(".xz") or suffix.endswith(".lzma"):
        return lzma.open(path, "rt", encoding="utf-8")
    if suffix.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def _child_text(element: ET.Element | None, name: str) -> str | None:
    if element is None:
        return None
    child = element.find(f"{{*}}{name}")
    return child.text if child is not None else None


def clean_wikitext(source: str) -> str:
    plain = wtp.parse(source).plain_text()
    plain = html2text(plain, bodywidth=0)
    return re.sub(r"\s+", " ", plain).strip()


def iter_articles(path: Path) -> Iterator[Article]:
    """Yield current main-namespace, non-redirect articles from one XML file."""
    with _open_dump(path) as stream:
        for _, page in ET.iterparse(stream, events=("end",)):
            if page.tag.rsplit("}", 1)[-1] != "page":
                continue
            try:
                if _child_text(page, "ns") != "0":
                    continue
                if page.find("{*}redirect") is not None:
                    continue

                title = (_child_text(page, "title") or "").strip()
                page_id_text = (_child_text(page, "id") or "").strip()
                revisions = page.findall("{*}revision")
                revision = revisions[-1] if revisions else None
                source = _child_text(revision, "text") if revision is not None else None
                if not title or not page_id_text or not source:
                    continue

                text = clean_wikitext(source)
                if text:
                    yield Article(int(page_id_text), title, text)
            finally:
                page.clear()


def write_plaintext_shards(
    dumps: Path | Iterable[Path],
    output_dir: Path,
    shard_count: int = 1,
    prefix: str = "wiki",
    compression_level: int = 9,
    output_name: str | None = None,
) -> int:
    """Write compressed plaintext, optionally combining several page-range files."""
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    if output_name and shard_count != 1:
        raise ValueError("output_name is only valid when shard_count is 1")

    dump_paths = [dumps] if isinstance(dumps, Path) else list(dumps)
    if not dump_paths:
        raise ValueError("at least one dump is required")

    output_dir.mkdir(parents=True, exist_ok=True)
    handles: dict[int, TextIO] = {}
    count = 0

    try:
        for dump in dump_paths:
            for article in iter_articles(dump):
                shard = (article.page_id - 1) % shard_count
                if shard not in handles:
                    filename = output_name or f"{prefix}_{shard + 1:02d}_of_{shard_count:02d}.txt.gz"
                    handles[shard] = gzip.open(
                        output_dir / filename,
                        "wt",
                        encoding="utf-8",
                        compresslevel=compression_level,
                        newline="",
                    )
                handles[shard].write(f"{article.title}\n{article.text}\n\n")
                count += 1
    finally:
        for handle in handles.values():
            handle.close()

    if count == 0:
        raise RuntimeError("No articles were written; refusing to publish an empty dataset.")
    return count
