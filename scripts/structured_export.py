#!/usr/bin/env python3
"""Stream published Wikipedia article archives into Parquet and/or SQLite.

This is a storage conversion, not a Wikimedia XML rebuild. The existing
three-line title/body/blank article records do not preserve Wikipedia page IDs.
A source filename plus one-based ordinal provides stable source provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path

try:
    from scripts.repack_shards import records
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from repack_shards import records

FIELDS = ("title", "text", "source_file", "source_ordinal")
PARQUET_FIELDS = (*FIELDS, "char_count", "byte_count", "word_count", "article_url")
PARQUET_SCHEMA_VERSION = 2


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def artifact_hashes(path: Path) -> dict[str, str]:
    """Compute SHA256 provenance and Zenodo's MD5 identity in one disk pass."""
    sha, md5 = hashlib.sha256(), hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            sha.update(block)
            md5.update(block)
    return {"sha256": sha.hexdigest(), "md5": md5.hexdigest()}


def batches(source: Path, size: int, max_bytes: int = 32 * 1024 * 1024):
    """Bound both rows and decoded article bytes to avoid large-shard OOMs."""
    batch, batch_bytes = [], 0
    for ordinal, record in enumerate(records(source), 1):
        raw_title, raw_text, _ = record.split(b"\n", 2)
        row_bytes = len(raw_title) + len(raw_text)
        if batch and (len(batch) >= size or batch_bytes + row_bytes > max_bytes):
            yield batch
            batch, batch_bytes = [], 0
        batch.append((
            raw_title.removesuffix(b"\r").decode("utf-8"),
            raw_text.removesuffix(b"\r").decode("utf-8"),
            source.name,
            ordinal,
        ))
        batch_bytes += row_bytes
    if batch:
        yield batch


def open_sqlite(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=DELETE")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("PRAGMA cache_size=-32768")
    db.execute("""CREATE TABLE articles (
        id INTEGER PRIMARY KEY,
        title TEXT NOT NULL,
        text TEXT NOT NULL,
        source_file TEXT NOT NULL,
        source_ordinal INTEGER NOT NULL
    )""")
    return db


def finish_sqlite(db: sqlite3.Connection, count: int, fts: bool) -> None:
    db.execute("CREATE UNIQUE INDEX articles_origin ON articles(source_file, source_ordinal)")
    db.execute("CREATE INDEX articles_title ON articles(title)")
    if fts:
        # External-content FTS: don't store article bodies a second time.
        db.execute(
            "CREATE VIRTUAL TABLE articles_fts USING fts5("
            "title, text, content='articles', content_rowid='id', tokenize='unicode61')"
        )
        db.execute("INSERT INTO articles_fts(articles_fts) VALUES('rebuild')")
        db.execute("INSERT INTO articles_fts(articles_fts) VALUES('integrity-check')")
    db.commit()
    actual = db.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    check = db.execute("PRAGMA quick_check").fetchone()[0]
    if actual != count or check != "ok":
        raise RuntimeError(f"SQLite integrity failure: rows={actual}/{count}, check={check}")


def export(source: Path, destination: Path, format: str = "parquet",
           fts: bool = False, batch_size: int = 16384, batch_max_mb: int = 32,
           wiki_code: str | None = None) -> dict:
    if not source.is_file() or not source.name.endswith(".txt.gz"):
        raise ValueError("Source must be an existing *.txt.gz article archive")
    if format not in ("parquet", "sqlite", "both"):
        raise ValueError("Invalid output format")
    if fts and format == "parquet":
        raise ValueError("FTS5 requires SQLite output")
    if not 1 <= batch_size <= 65536:
        raise ValueError("batch_size must be in [1, 65536]")
    if not 1 <= batch_max_mb <= 512:
        raise ValueError("batch_max_mb must be in [1, 512]")
    if wiki_code:
        try:
            from scripts.wiki_metadata import article_url
        except ModuleNotFoundError as exc:
            if exc.name != "scripts":
                raise
            from wiki_metadata import article_url
        # Validate before creating output files, including for empty input.
        article_url("Main Page", wiki_code)

    want_parquet = format in ("parquet", "both")
    want_sqlite = format in ("sqlite", "both")
    stem = source.name.removesuffix(".txt.gz")
    destination.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".structured-", dir=destination) as temp:
        stage = Path(temp)
        writer = None
        db = None
        count = 0
        parquet_path = stage / f"{stem}.parquet"
        sqlite_path = stage / f"{stem}.sqlite"
        try:
            if want_parquet:
                import pyarrow as pa
                import pyarrow.parquet as pq

                schema = pa.schema([
                    pa.field("title", pa.string(), nullable=False),
                    pa.field("text", pa.string(), nullable=False),
                    pa.field("source_file", pa.string(), nullable=False),
                    pa.field("source_ordinal", pa.int64(), nullable=False),
                    pa.field("char_count", pa.int64(), nullable=False),
                    pa.field("byte_count", pa.int64(), nullable=False),
                    pa.field("word_count", pa.int64(), nullable=False),
                    pa.field("article_url", pa.string(), nullable=True),
                ]).with_metadata({b"source_archive": source.name.encode("utf-8")})
                writer = pq.ParquetWriter(
                    parquet_path, schema, compression="zstd", compression_level=3,
                    use_dictionary=["source_file"],
                    write_statistics=["title", "source_ordinal"],
                )
            if want_sqlite:
                db = open_sqlite(sqlite_path)

            for batch in batches(source, batch_size, batch_max_mb * 1024 * 1024):
                if writer is not None:
                    columns = dict(zip(FIELDS, zip(*batch)))
                    texts = [row[1] for row in batch]
                    columns.update({
                        "char_count": [len(text) for text in texts],
                        "byte_count": [len(text.encode("utf-8")) for text in texts],
                        "word_count": [len(text.split()) for text in texts],
                        "article_url": ([article_url(row[0], wiki_code) for row in batch]
                                        if wiki_code else [None] * len(batch)),
                    })
                    table = pa.table(columns, schema=schema)
                    writer.write_table(table, row_group_size=batch_size)
                if db is not None:
                    db.executemany(
                        "INSERT INTO articles(title,text,source_file,source_ordinal) "
                        "VALUES(?,?,?,?)", batch
                    )
                    db.commit()
                count += len(batch)
            if count == 0:
                raise ValueError("Refusing to publish an empty article archive")
            if writer is not None:
                writer.close()
                writer = None
                if pq.ParquetFile(parquet_path).metadata.num_rows != count:
                    raise RuntimeError("Parquet row-count mismatch")
            if db is not None:
                finish_sqlite(db, count, fts)
                db.close()
                db = None
        finally:
            if writer is not None:
                writer.close()
            if db is not None:
                db.close()

        artifacts = {}
        if want_parquet:
            artifacts["parquet"] = {
                "path": f"parquet/{parquet_path.name}",
                "size_bytes": parquet_path.stat().st_size,
                **artifact_hashes(parquet_path),
            }
        if want_sqlite:
            import zstandard as zstd

            packed = stage / f"{stem}.sqlite.zst"
            with sqlite_path.open("rb") as inp, packed.open("wb") as out:
                zstd.ZstdCompressor(level=8).copy_stream(inp, out)
            artifacts["sqlite"] = {
                "path": f"sqlite/{packed.name}",
                "size_bytes": packed.stat().st_size,
                "sha256": sha256(packed),
                "fts5": fts,
            }
            sqlite_path.unlink()

        manifest = {
            "version": 1,
            "source_file": source.name,
            "source_sha256": sha256(source),
            "articles": count,
            "columns": list(PARQUET_FIELDS if want_parquet else FIELDS),
            "schema_version": PARQUET_SCHEMA_VERSION if want_parquet else 1,
            "wikipedia_code": wiki_code if want_parquet else None,
            "article_urls_verified": False,
            "wikipedia_page_ids_available": False,
            "artifacts": artifacts,
        }
        # Independent format manifests prevent a refreshed Parquet export from
        # silently making an older SQLite download appear current (or vice versa).
        metadata = []
        for kind, artifact in artifacts.items():
            item = {**manifest, "artifacts": {kind: artifact}}
            if kind == "sqlite":
                item.update(columns=list(FIELDS), schema_version=1,
                            wikipedia_code=None)
            path = stage / f"{stem}.{kind}.json"
            path.write_text(json.dumps(item, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
            metadata.append(path)

        # The manifests are installed last. All outputs are local until CI's
        # single-commit Hub upload; source plaintext archives are never changed.
        for info in artifacts.values():
            target = destination / info["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage / Path(info["path"]).name, target)
        for path in metadata:
            target = destination / "manifests" / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(path, target)
        return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--format", choices=("parquet", "sqlite", "both"), default="parquet")
    parser.add_argument("--fts", action="store_true", help="Build optional SQLite FTS5 index")
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--batch-max-mb", type=int, default=32,
                        help="Cap decoded article bytes per batch to reduce memory usage")
    parser.add_argument("--wiki-code", help="Wikimedia project language code for URL generation")
    args = parser.parse_args()
    result = export(args.source, args.destination, args.format, args.fts,
                    args.batch_size, args.batch_max_mb, args.wiki_code)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
