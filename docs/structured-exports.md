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

Every Parquet file has `title`, `text`, `source_file`, `source_ordinal`,
and a manifest with source SHA256, output SHA256 and article count.
`source_ordinal` is one-based inside its input archive, **not a Wikipedia page ID**.

The original text stores title, cleaned text and a blank line per article.
The underlying source can be partial: always check `data/build-manifest.json`
before claiming full edition coverage.

## Publishing

- [Entire language fleet](../.github/workflows/structured-fleet.yml): manually
  run; **Hugging Face, Kaggle and Zenodo are all enabled by default**.
- [One language](../.github/workflows/structured-wikipedia.yml): same publish
  helpers and destinations, scoped to a particular existing dataset.
- Hugging Face publishes under the **same dataset ID**, preserving original files.
- Kaggle versions the **same dataset ID** with original text and Parquet; the
  previous file set is downloaded first to avoid silently dropping existing
  public data. Oversized editions fail explicitly instead of discarding files.
- Zenodo receives **Parquet only**, plus a provenance manifest.

No workflow creates new SQLite databases or FTS5 indexes. Older SQLite
downloads are left available where already published. The low-level
`scripts/structured_export.py` still supports SQLite for historical consumers
and test compatibility; release jobs use Parquet only.

No large intermediate GitHub Actions artifacts are uploaded. Export jobs stream
source shards, publish, and remove temporary runner data.

## Example

```sh
python -m pip install pyarrow
python scripts/structured_export.py simplewiki_01_of_01.txt.gz output --format parquet
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
