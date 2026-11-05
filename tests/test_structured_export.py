"""Structured-format regression tests on small, representative gzip archives."""

import gzip
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.structured_export import export


class StructuredExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "simplewiki_01_of_01.txt.gz"
        self.rows = [
            ("Orbit", "Planets orbit stars."),
            ("Český Krumlov", "A city in the Czech Republic."),
            ("日本", "Japan is an island country."),
        ]
        with gzip.open(self.source, "wt", encoding="utf-8", newline="") as out:
            for title, body in self.rows:
                out.write(f"{title}\n{body}\n\n")

    def test_both_formats_and_fts5(self):
        import pyarrow.parquet as pq
        import zstandard as zstd

        dest = self.root / "output"
        info = export(self.source, dest, format="both", fts=True, batch_size=2)
        self.assertEqual(info["articles"], 3)
        self.assertFalse(info["wikipedia_page_ids_available"])
        self.assertEqual(set(info["artifacts"]), {"parquet", "sqlite"})

        parquet = dest / info["artifacts"]["parquet"]["path"]
        table = pq.read_table(parquet)
        self.assertEqual(table.num_rows, 3)
        self.assertEqual(table.column("title").to_pylist(), [r[0] for r in self.rows])
        self.assertEqual(table.column("source_ordinal").to_pylist(), [1, 2, 3])

        packed = dest / info["artifacts"]["sqlite"]["path"]
        database = self.root / "lookup.sqlite"
        with packed.open("rb") as inp, database.open("wb") as out:
            zstd.ZstdDecompressor().copy_stream(inp, out)
        with sqlite3.connect(database) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM articles").fetchone()[0], 3)
            self.assertEqual(
                db.execute("SELECT title FROM articles_fts WHERE articles_fts MATCH 'planets'").fetchone()[0],
                "Orbit",
            )
            self.assertEqual(
                db.execute("SELECT source_file, source_ordinal FROM articles WHERE title='日本'").fetchone(),
                ("simplewiki_01_of_01.txt.gz", 3),
            )
            self.assertEqual(db.execute("PRAGMA quick_check").fetchone()[0], "ok")

        for kind in ("parquet", "sqlite"):
            meta = json.loads((dest / "manifests" / f"simplewiki_01_of_01.{kind}.json").read_text())
            self.assertEqual(meta["source_sha256"], info["source_sha256"])
            self.assertEqual(meta["artifacts"], {kind: info["artifacts"][kind]})

    def test_sqlite_without_fts(self):
        import zstandard as zstd

        dest = self.root / "lookup"
        result = export(self.source, dest, format="sqlite")
        self.assertEqual(set(result["artifacts"]), {"sqlite"})
        packed = dest / result["artifacts"]["sqlite"]["path"]
        database = self.root / "test.sqlite"
        with packed.open("rb") as inp, database.open("wb") as out:
            zstd.ZstdDecompressor().copy_stream(inp, out)
        with sqlite3.connect(database) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM articles").fetchone()[0], 3)
            self.assertFalse(db.execute("SELECT name FROM sqlite_master WHERE name='articles_fts'").fetchone())

    def test_parquet_rejects_fts(self):
        with self.assertRaisesRegex(ValueError, "requires SQLite"):
            export(self.source, self.root / "out", format="parquet", fts=True)

    def test_invalid_record_does_not_publish_partial_data(self):
        with gzip.open(self.source, "wb") as out:
            out.write(b"Broken title\nbody without an ending")
        dest = self.root / "failed"
        with self.assertRaises(ValueError):
            export(self.source, dest, format="parquet", batch_size=1)
        self.assertFalse(list(dest.rglob("*.parquet")))
        self.assertFalse(list(dest.rglob("*.json")))

    def test_empty_archive_rejected(self):
        with gzip.open(self.source, "wb"):
            pass
        with self.assertRaisesRegex(ValueError, "empty"):
            export(self.source, self.root / "out", format="parquet")


if __name__ == "__main__":
    unittest.main()
