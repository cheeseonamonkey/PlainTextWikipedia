# Wikipedia data formats

Keep the **existing original plaintext gzip archives** and add **Zstandard Parquet**.
The Parquet is generated from already-published text; no Wikimedia XML rebuild.

## Files

Each language's existing Hugging Face dataset contains:

```text
data/<shard>.txt.gz                   # unchanged source text
structured/parquet/<shard>.parquet   # derived columnar representation
structured/manifests/<shard>.parquet.json
```

Every Parquet row is one article. The v2 Parquet schema has:

| Column | Meaning |
|---|---|
| `title`, `text` | Article title and cleaned text |
| `source_file`, `source_ordinal` | Original gzip shard and one-based record position |
| `char_count` | Unicode code points in article text (not grapheme clusters) |
| `byte_count` | UTF-8 bytes in article text (not the compressed size) |
| `word_count` | Whitespace-separated tokens; **not a reliable linguistic word count for CJK and other non-space-separated languages** |
| `article_url` | Best-effort Wikipedia title link, URL-encoded and language-aware; nullable when no language mapping is supplied |

`article_url` is **not verified as a current, canonical article URL**.
The original archives do not preserve stable Wikipedia page IDs, redirects,
image references, or image file names. **No per-article API lookups** occur,
and image URLs are not included.

Each file has a manifest with schema version, Wikimedia project code, source
SHA256, output SHA256/MD5 and article count. `source_ordinal` is one-based
inside its input archive, **not a Wikipedia page ID**. Schema-v1 Parquet is
re-exported once for v2; subsequent runs skip unchanged v2 output.

The original text stores title, cleaned text and a blank line per article.
The underlying source can be partial: always check `data/build-manifest.json`
before claiming full edition coverage.

## Publishing

- [Entire language fleet](../.github/workflows/structured-fleet.yml): manually
  dispatches a single controller which queues reusable language jobs;
  **Hugging Face, Kaggle and Zenodo are all enabled by default**.
- [One language](../.github/workflows/structured-wikipedia.yml): **the same
  reusable workflow** is also directly dispatchable. No duplicated CI code.
  Retry individual failures with the controller's `languages` input.
- Hugging Face publishes under the **same dataset ID**, preserving original files.
- Kaggle versions the **same dataset ID** with original text and Parquet.
  Before downloading multi-GB files, it fetches only the tiny remote manifest
  and checks filenames. When unchanged, nothing is staged or uploaded.
  Changed editions download the prior complete file set first to avoid
  silently dropping original data. Oversized editions fail explicitly.
- Zenodo receives **Parquet only**, plus a provenance manifest. Kaggle and
  Zenodo reuse the same on-runner Hugging Face file cache to avoid fetching
  identical assets twice. Newly generated Parquet manifests also carry an MD5
  alongside SHA256, allowing Zenodo to reuse already-uploaded draft files
  **without even downloading them again** when size/MD5 match.
- Parquet batches are capped by both row count and 32 MiB of decoded article
  bytes, reducing memory spikes from unusually large articles.

No workflow creates new SQLite databases or FTS5 indexes. Older SQLite assets remain available on Hugging Face where previously published;
new Kaggle versions omit legacy SQLite while previous Kaggle versions
remain available for download. The low-level
`scripts/structured_export.py` still supports SQLite for historical consumers
and test compatibility; release jobs use Parquet only.

No large intermediate GitHub Actions artifacts are uploaded. Export jobs stream
source shards, atomically publish each Parquet + source-checksum pair on HF, and
remove temporary runner data. The controller preflights credentials once and
queues up to 8 languages concurrently by default (configurable from 1–12).
GitHub limits matrix job fan-out to 256 per workflow run.

## Example

```sh
python -m pip install pyarrow
python scripts/structured_export.py simplewiki_01_of_01.txt.gz output --format parquet --wiki-code simple
python -m unittest discover -s tests -v
```

```python
from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq

repo = "ffatty/plain-text-wikipedia-simpleenglish"
files = HfApi().list_repo_files(repo, repo_type="dataset")
name = next(p for p in files if p.startswith("structured/parquet/") and
            p.endswith(".parquet"))
path = hf_hub_download(repo_id=repo, filename=name, repo_type="dataset")
print(pq.read_table(path, columns=["title", "text"]).slice(0, 5))
```

The Wikimedia CC BY-SA attribution and share-alike requirements still apply.
