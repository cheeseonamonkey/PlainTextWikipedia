# Complete Wikipedia bundles (schema v6)

One shared implementation handles every language. The Wikimedia code affects
article URLs, Unicode tokenization and stopword lookup; it does not select a
separate build or publication path. Chinese uses Wikimedia's `zh` code.

Each language dataset receives these **same five files on all three services**:

| File | Contents |
|---|---|
| `articles.parquet` | One row per article; 33 columns; Zstandard-compressed row groups |
| `articles.txt.xz` | All records combined: title, single-line body, blank line |
| `features.zip` | Extract into `features/`: corpus profile, vocabulary and full word statistics |
| `models.zip` | Extract into `models/`: compressed trained fastText model and model configuration |
| `manifest.json` | Full source provenance, article count, byte sizes, SHA256/MD5, all publisher and repository links |

ZIP packaging is identical everywhere because Zenodo cannot store native directories.
The contained statistics/model files are already compressed, so ZIP uses stored entries.
No public databases, per-shard Parquets, SQL dumps or fused embeddings are generated.
One temporary SQLite index supports exact corpus counts during the build and is deleted.
It stores counts and article metadata, not another full plaintext corpus. The
embedding pass streams article bodies from the combined `txt.xz` and checks them
against that index. The temporary training text is deleted once fastText is loaded.

## Article columns

Identity/content: `title`, `text`, `article_url`, `char_count`.

Document metrics: `sentence_count`, `distinct_terms`, `mean_sentence_tokens`,
`mean_token_chars`, `lexical_diversity`, `hapax_ratio`, `stopword_ratio`,
`digit_ratio`, `punctuation_ratio`, `repetition_ratio`, `entropy`,
`normalized_entropy`, `simpson_diversity`, `vocab_coverage`, `tfidf_coverage`,
`mean_idf`, `idf_std`, `tfidf_l2`, `tfidf_top5_share`, `tfidf_entropy`, `title_overlap`.

JSON columns: `word_counts` (all terms), `tfidf` (term → L2-normalized weight),
`bm25` (term → score), `keywords` (top 16 term → raw TF-IDF),
`fasttext` (160 floats), `bekko` (128 floats).

Bekko metadata: `bekko_input_tokens`, `bekko_truncated`.

TF-IDF uses a selected vocabulary (default 10K terms, 5K TF-IDF terms), not a
claim of every term's per-article TF-IDF. Full per-article counts and full corpus
TF/DF/IDF remain available, so other weighting schemes can be reconstructed.
FastText trains on **every token of every article**, one CBOW epoch, and its
article vector uses the full tokenized article. No training token cap or
language-size exclusion exists. Bekko uses a bounded title/lead excerpt (256
tokens); the full text is always retained. These embeddings are independent.
VADER, readability, saliency, contradiction and fused-vector features are removed.

## Corpus statistics

`features/corpus-profile.json` contains count, mean, population standard deviation,
min/max and approximate percentiles for numeric article columns, plus independent
Bekko and fastText centroids. Percentiles use a deterministic 2048-row reservoir.
`features/words.jsonl.gz` streams every term's corpus frequency, document frequency
and smoothed IDF without a vocabulary cap. `features/vocabulary.json.gz` records
the selected lexical vocabulary and TF-IDF flags. Corpus counts/embeddings have
clear aggregate definitions; titles and URLs are article-only attributes.

## Build, then publish

`python -m scripts.wikipedia_bundle --repo ffatty/... --output release` builds
and verifies a local bundle. It never uploads data without `--publish`.

The manual one-language/fleet workflows call this same command. Rebuild workflows
convert XML into temporary source shards, require successful source coverage,
then call the same bundle workflow. Every manual `publish` input defaults to false.
All automatic publication, recovery and repair workflows have been removed.

For `--publish`, all three credentials are required. Zenodo draft metadata is
reserved before building so every file can contain its actual record link.
**All data is built and verified before the first data upload**. HF receives one
commit guarded by its parent revision; Kaggle receives a fresh version with
`--keep-tabular`; Zenodo receives the same local files. The job fails if a target
fails. Services do not support an atomic cross-service transaction: a downstream
failure can leave earlier targets updated. Retained bundle artifacts permit retry
with `--resume --publish`; incomplete remote publication is never labeled complete.
The one-language workflow accepts `bundle_run_id` to restore a retained bundle
and resume it on GitHub. A build-only bundle can be published this way without
recomputing features: only the manifest gets its newly reserved Zenodo link.
Existing output directories are never overwritten by another build.

Local verification checks exact file/archive membership, article schema, row
count, sizes, SHA256 and MD5. HF retries recheck every artifact, including the
manifest; Kaggle waits for an in-progress version and downloads that specific
version for verification. ZIP member metadata is fixed, so file timestamps do
not alter the archive identity.

Source downloads are pinned to an immutable HF revision and their SHA256s checked.
A missing/incomplete source coverage manifest fails the build; it requires source
recovery rather than pretending an existing subset is complete. This does not
independently establish that historical converters retained every eligible article.

No file is omitted for size. API quotas, insufficient disk or the GitHub-hosted
job time limit produce explicit failures. Large-language capacity and real Bekko
throughput still need a controlled CI validation before a fleet rollout; small
local fixtures do not establish that English fits a standard runner. Production
publishing has not been exercised by this refactor.
