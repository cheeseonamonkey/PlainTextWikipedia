# Article features (schema v8)

One shared implementation handles every language. The Wikimedia code affects
article URLs, Unicode tokenization and stopword lookup; it does not select a
separate build path. Chinese uses Wikimedia's `zh` code.

Each language build produces **three files**:

| File | Contents |
|---|---|
| `articles.parquet` | One row per article; 13 columns; Zstandard-compressed row groups |
| `corpus-profile.json` | Language-level vocabulary (TF/DF/IDF), field moments/percentiles, fastText centroid |
| `manifest.json` | Schema version, article count, configuration, byte sizes, SHA256 checksums |

No Bekko or BM25 columns, no combined `.txt.xz` re-upload, no ZIP archives, no
published model files. One temporary SQLite index supports exact corpus counts
during the build and is deleted; it stores counts and metadata, not another
plaintext corpus. The embedding pass streams article bodies from the temporary
spill file and checks them against that index.

## Build stages

The build runs as three stages over one explicit intermediate directory, so each
stage can run in its own process and its own CI job with its own time budget:

| Stage | Reads | Writes |
|---|---|---|
| `corpus-stats` | `data/*.txt.gz` source shards | `corpus-stats/index.sqlite`, `corpus-stats/articles-spill.txt.xz`, `corpus-stats/lexical-training.txt.gz`, `corpus-stats/stats.json` |
| `fasttext-training` | `corpus-stats/stats.json`, `corpus-stats/lexical-training.txt.gz` | `fasttext-training/model.bin`, `fasttext-training/training.json` |
| `enrich-parquet` | the corpus index, the spill file, the trained model | `release/articles.parquet`, `release/corpus-profile.json`, `release/manifest.json` |

Each stage reads only its declared inputs and fails with an explicit
"run the earlier stage first" error when one is missing, so a failed run resumes
at the failing stage instead of restarting the whole build. `python -m
scripts.article_features --stage <name> --work <dir>` runs one stage directly;
`python -m scripts.build_language --language <code> --stage <name>` does the same
with language resolution and source staging. Omitting `--stage` runs all three
in one process into a temporary directory under the release folder.

The split reproduces the single-process bundle exactly: same vocabulary, same
fastText model, byte-identical `manifest.json` and `corpus-profile.json`.

## Article columns

`text` is the first column; `fasttext` is a native `list<float32>`; every other
heavy field is a compact JSON object in a single string cell, grouped by kind:

| Column | Type | Contents |
|---|---|---|
| `text` | string | cleaned article body (verbatim from the source shards) |
| `title` | string | article title |
| `article_url` | string | best-effort URL rebuilt from the title; titles are not canonical IDs |
| `char_count` | int64 | body character count (scalar for cheap filtering) |
| `counts` | JSON | `word_count, sentence_count, distinct_terms, byte_count, mean_sentence_tokens, mean_token_chars, reading_time_minutes` |
| `ratios` | JSON | `lexical_diversity, hapax_ratio, stopword_ratio, digit_ratio, punctuation_ratio, repetition_ratio, vocab_coverage, title_overlap` |
| `diversity` | JSON | `entropy, normalized_entropy, simpson_diversity` |
| `tfidf_stats` | JSON | `l2, top5_share, entropy, mean_idf, idf_std` |
| `word_counts` | JSON | raw surface-form counts per article |
| `tfidf` | JSON | stemmed-term TF-IDF, **L2-normalized per article**; corpus IDF is language-wide |
| `keywords` | JSON | top 15 TF-IDF terms with weights |
| `fasttext` | list&lt;float32&gt; | normalized 180-d CBOW embedding (mean of in-vocabulary stemmed terms) |
| `source` | JSON | `{file, ordinal}` — join key back to the published `data/*.txt.gz` shards |

Paragraph boundaries are not recoverable: the plaintext converter collapses all
whitespace, so the source shards contain single-line bodies.

## fastText

CBOW over every article, restricted to the shared top-N stemmed vocabulary
(default 12 000 terms, DF-capped). No subword hashing (`minn=maxn=0`,
`bucket=0`): OOV buckets are unnecessary because only in-vocabulary terms are
embedded. Defaults: `dim=180`, `ws=8`, `epoch=1`, `neg=8`, `lr=0.05`; all
exposed as workflow inputs. Vectors are L2-normalized per article. Article
vectors are the model's sentence vector over the article's stemmed, in-vocabulary
terms, in source order.

## Normalization

NFKC + casefold + quote/dash folding, Snowball stemming via PyStemmer where the
language is supported, `stopwordsiso` stopword sets. Languages without a
Snowball stemmer retain normalized surface forms; no article is ever omitted.
`word_counts` stay raw (unstemmed); `tfidf`/`keywords`/`fasttext` use stemmed
terms. IDF is `ln((N+1)/(df+1))+1`; terms with DF above half the corpus are
excluded from the vocabulary and treated as stopword-like.

## Corpus profile

`corpus-profile.json` contains: `documents`, `fields` (count/mean/stddev/min/max
and reservoir-sampled p05/p50/p95 per numeric feature), `fasttext_centroid`
(unnormalized mean vector), and `vocabulary` — the full selected vocabulary with
per-term `tf`, `df`, `idf`.

## Verification

`manifest.json` records byte size and SHA256 for both artifacts plus the schema
version and article count. A verifier re-checks checksums, row count, and the
exact Parquet schema before any publisher touches the bundle. Source coverage is
validated before the build: modern per-root manifests are checked strictly;
legacy aggregate manifests are accepted only when every expected unsplit shard
is present. Legacy manifests cannot prove per-page-range completeness, and the
build log says so explicitly.

## Limitations

- Article URLs are reconstructed from titles and can break if pages move.
- Page IDs are not preserved in the source shards, so joins to Wikidata are
  title-based, not ID-based.
- `reading_time_minutes` assumes 200 words/minute; meaningless for unspaced
  scripts.
- Legacy-manifest languages (most pre-2026-11 datasets) carry only global
  completeness evidence.
- The `corpus-stats` intermediate (SQLite index + plaintext spill) is the largest
  artifact the build produces; for full English it is unmeasured and is the main
  scaling risk of the staged handoff. Each stage also re-installs dependencies
  and re-downloads its inputs, so the split trades transfer time for per-stage
  time budgets.
