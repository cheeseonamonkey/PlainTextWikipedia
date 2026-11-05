# Structured Wikipedia exports

The existing plaintext datasets remain canonical and unchanged. GitHub Actions can
turn each published `.txt.gz` archive into **Parquet** (recommended for training
and analytics) and optionally **SQLite** (offline lookup and SQL, with optional
FTS5 indexing). This reuses existing published data; it does not re-download
Wikimedia XML or redo the wikitext conversion.

## Organization

Keep **one dataset per Wikipedia language**, not separate datasets for every
format. Each language's Hugging Face repository holds its existing `data/`
archives plus:

```text
structured/
  parquet/<original-shard>.parquet
  sqlite/<original-shard>.sqlite.zst
  manifests/<original-shard>.parquet.json
  manifests/<original-shard>.sqlite.json
```

Each format shards rather than repeats articles *within* that format. Keeping
multiple representations does duplicate storage. Large languages should
prioritize Parquet; SQLite and FTS5 are optional.

The original three-line article format includes the title, cleaned single-line
body, and blank separator. It **does not preserve Wikimedia page IDs**. Both
structured formats expose `title`, `text`, `source_file`, and
`source_ordinal` (1-based within the input archive). Do not interpret the
ordinal or the SQLite auto-generated `id` as a Wikipedia page ID.

Parquet uses Zstandard compression and small, stream-written row groups.
SQLite files are compressed with Zstandard **after** building and verifying
the database. Decompress `.sqlite.zst` before querying.

## GitHub Actions

Use [Structured Wikipedia (Parquet + SQLite)](../.github/workflows/structured-wikipedia.yml):

- **Automatic on merge to main:** Simple English pilot, both formats and FTS5.
- **Manual:** Select an existing `ffatty/...` Hugging Face dataset,
  `parquet` (default), `sqlite`, or `both`; FTS5 is opt-in.
- **Build-only mode:** Disable publishing to obtain short-lived CI artifacts.
- Source shards become independent CI jobs, at most two in parallel.
- Per-format manifests record source archive SHA256, output SHA256,
  article count, and whether SQLite FTS5 was built.
- Existing `.txt.gz` archives and original coverage manifests are preserved.
  A converted file does **not** establish that the underlying Wikipedia
  snapshot had complete source coverage.

Example local conversion:

```bash
python -m pip install pyarrow zstandard
python scripts/structured_export.py enwiki_01_of_16.txt.gz output --format parquet
python scripts/structured_export.py simplewiki_01_of_01.txt.gz output --format both --fts
python -m unittest discover -s tests -v
```

Example SQLite search after decompressing a shard:

```sql
SELECT articles.title, substr(articles.text, 1, 240) AS excerpt
FROM articles_fts JOIN articles ON articles.id = articles_fts.rowid
WHERE articles_fts MATCH 'astronomy'
ORDER BY bm25(articles_fts)
LIMIT 10;
```

SQLite FTS5 uses the built-in `unicode61` tokenizer. Its token boundaries
are not linguistically optimal for languages without whitespace-delimited words.
For Chinese, Japanese, and similar editions, use a language-aware search layer.

## Publishing considerations

**Hugging Face:** uploads are additive under `structured/` and do not
overwrite the existing text shards. A source archive's Parquet and SQLite
manifests are independent so they can be updated without falsifying provenance
for an older alternate-format artifact.

**Kaggle:** an existing dataset's new version needs its full file set staged.
A structured-only directory must **not** be uploaded as a replacement version
or it could omit the original text archives. The Simple English pilot can be
mirrored after verifying its structured exports. Multi-gigabyte editions
should avoid unnecessary repeated uploads.

The project's Wikimedia attribution and CC BY-SA reuse requirements apply
equally to all derived representations.
