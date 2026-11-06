# Wikipedia article features (schema v5)

Canonical analytical output: **`articles.parquet`**, one file per language, one
row per article. It contains the original article title and body **plus** all
per-article features and intermediate measurements. Original source archives
(`data/*.txt.gz`) and optional per-shard Parquet exports remain untouched.

The companion `features/` directory stores only corpus-level artifacts:

| File | Purpose |
|---|---|
| `articles.parquet` (dataset root) | Primary article table, full text, counts, metrics, nested 200D embedding, raw standalone vectors and reproducibility data |
| `features/corpus-profile.json` | Entire-corpus numeric means, population deviations, extrema, sampled p05/p50/p95, combined-vector centroid |
| `features/vocabulary.json.gz` | Top 10K terms, TF/DF/IDF, flags for selected 5K TF-IDF terms |
| `features/articles.sqlite.zst` | Optional searchable sidecar: every article, **all word counts**, every term's total TF/DF, vocabulary, metrics, FTS5 |
| `features/fasttext.bin.zst` | Trained per-language CBOW model with 160 dimensions |
| `features/manifest.json` | Source and artifact hashes, model configuration and decoding instructions |

The Parquet is a single file with multiple row groups, not a single massive
memory allocation. Read it directly with DuckDB, Polars, pandas or PyArrow.
The SQLite sidecar is a compressed **actual SQLite database**, more useful
than a SQL text dump (`.sql.gz`) when restored because indexes and FTS5 are
preserved. No uncompressed copies are shipped. Corpus coverage may be partial
if the original published Wikipedia conversion was partial.

## Per-article columns

### Original and intermediate data

- `text`, `title`, `article_url`, `source_file`, `source_ordinal`.
- All original token counts (`word_counts_all_json`), filtered-vocabulary
  counts (`word_counts_json`), normalized sparse TF-IDF and BM15 score arrays,
  top keywords and representative extractive sentence.
- Token, character, byte, sentence, vocabulary, distinct-word, hapax,
  stopword, selected-vocabulary, TF-IDF-token, digit and punctuation **counts**;
  coverage ratios and intermediate TF-IDF/BM15 sums.
- Original and quantized components: `embedding_200` is a native fixed-length
  **list of 200 float16 values**. Compact int8 encodings and float16 scale
  columns for the combined, 128D Bekko and 160D fastText vectors are also
  preserved. Decode with `scripts.article_features.dequantize`.

### Lexical and extractive scores

- Lexical diversity, normalized/Shannon entropy, Simpson diversity, effective
  vocabulary, hapax ratios, repetition, rarity, mean term frequency, IDF mean
  and deviation, frequency/coverage ratios, long-tail and title overlap.
- TF-IDF L2/max/top-five share/entropy, BM15 sum/mean/max/top-five share,
  keyword share, saliency, title relevance, lead relevance, extractive sentence
  position/margin, sentence-length CV and lead keyword overlap.
- Quantization fidelity (`vector_quant_cosine`), activity rate and realized
  embedding subspace energies.

### Heuristic and model-derived diagnostics

- `saliency`, `sentiment_valence_01`, `sentiment_intensity_01`,
  `reading_difficulty_01`, `technicality_proxy_01`, and
  `lexical_cohesion_01` are bounded in **0–1**; they are *different measures*,
  not one probability. Avoid comparing their values as if calibrated.
- VADER: `vader_compound` (−1 to 1), `vader_positive`, `vader_neutral`,
  `vader_negative`; normalized valence is `(compound + 1)/2`, where **0.5 is
  neutral**. Takes first 3,072 and last 1,024 characters (or entire shorter
  article); actual sample length is recorded. **English only**; null otherwise.
- Readability: raw Flesch-Kincaid grade and Flesch Reading Ease, plus an
  explicitly heuristic bounded difficulty `min(1, max(0, grade/20))`.
  Computed on first 2,048 characters; **English only**; null otherwise.
  Readability formulas are approximations, not evaluations of truth.
- `technicality_proxy_01`: transparent weighted heuristic of rare vocabulary,
  word length, IDF and digit density; *not a technical-topic classifier*.
- `lexical_cohesion_01`: average adjacent-sentence token-set Jaccard in the
  first 12,000 characters; approximates local topical continuity, **not
  absence of contradictions**. `contradiction_score` is null until a real NLI
  model evaluates contradictions. Missing measurements remain null in Parquet,
  not fabricated zero-confidence scores.

## 200-dimensional combined representation

The v5 vector includes **all 128D multilingual Bekko**, a deterministic
44D projection of the separate 160D per-language fastText model, a 16D signed
TF-IDF hash and 12 normalized scalar metrics. Importance in cosine similarity:

| Component | Dimensions | Importance |
|---|---:|---:|
| Bekko | 128 | 46% |
| FastText projection | 44 | 43% |
| Sparse TF-IDF signed hash | 16 | 3% |
| Metrics (including VADER) | 12 | 8% |

Each block is L2-normalized and multiplied by the square root of its weight;
then the complete vector is normalized. Metric fields include length, lexical
richness, vocabulary/TF-IDF coverage, mean IDF, saliency, title overlap,
entropy, valence, sentiment intensity, technicality and lexical cohesion.
Missing model scores are centered at 0.5 only **inside the embedding** and
remain null in the article table. Weights are design choices, not validated
against a retrieval benchmark. The original fastText/Bekko vectors and the
realized block-energy ratios remain available for different weighting.

Bekko: `hotchpotch/bekko-embedding-v1-a8m` with CPU/OpenVINO, NFKC-casefolded
title and bounded article lead (max 256 model tokens), 16 articles per batch.
This is *not* a full-text transformer embedding. Entire texts are used for
TF-IDF, corpus counts and fastText pooling; separate fastText trains one epoch
on up to 2M sampled tokens. The Bekko 128D representation is approximately
cross-language compatible. The fastText models and **combined vectors are not
cross-language aligned**. No silent missing-model fallback is permitted in
full releases.

Quantization: signed int8 per 32-vector block, float16 scale, fixed seed 2026
for fastText projection, independent BLAKE2s bits for lexical bucket and sign.
Different fastText training runs can vary slightly due to stochastic CPU work.

## Operational behavior

The manual [`article-features-fleet.yml`](../.github/workflows/article-features-fleet.yml)
reads only existing HF source archives, builds once per language and publishes
in one HF commit. Source hashes, configuration, output sizes and hashes decide
whether to skip a current dataset. The standard 8-shard release and Parquet
publisher do **not** run this expensive feature build automatically.

Large Wikipedia editions may exceed ordinary hosted GitHub runner disk and CPU
limits. The builder refuses insufficient capacity rather than silently
truncating sources. Kaggle/Zenodo feature mirroring is not automatic; their
independent existing source/Parquet releases remain unchanged.

For local use:

```bash
python -m pip install numpy pyarrow zstandard fasttext-wheel regex stopwordsiso==0.7.1 vaderSentiment textstat
python -m pip install 'sentence-transformers[openvino]>=5.0' 'transformers>=4.57,<5'
python scripts/article_features.py data/*.txt.gz --language en --output article-features
```
