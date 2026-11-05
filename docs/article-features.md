# Per-language article features

A separate, optional enrichment release. Original Wikipedia gzip and structured
Parquet outputs remain unchanged. The pipeline derives corpus-wide lexical
statistics and small semantic vectors from **every available article** in one
existing language dataset.

Use the [manual feature fleet](../.github/workflows/article-features-fleet.yml).
Select languages "simple,en" for a smoke test, or leave blank to queue all
existing published Wikipedia editions. Language vocabularies and models are
independent and **not inherently aligned across languages**.

## Published outputs

All files are added to the same Hugging Face dataset under features/:

| Filename | Contents |
|---|---|
| article-metrics.csv.gz | One CSV row per article; full metric set, sparse lexical vectors and a quantized 200D vector |
| vocabulary.json.gz | Up to 10,000 corpus terms: ID, total count, document frequency, smoothed IDF and 5K TF-IDF selection flag |
| articles.sqlite.zst | Compressed SQLite: source article text, full per-article compressed word-count JSON, term totals, vocabulary, article metrics and optional FTS5 index |
| fasttext.bin.zst | Language-specific fastText CBOW model, if training succeeds |
| manifest.json | Source hashes, file hashes, schema/config, model status, vector decoding scheme |

Original dumps may be partial; features only represent already-published articles.

## Text normalization

Unicode NFKC normalization plus case folding and folding of common typographic
quotes, dashes and spacing. No transliteration to ASCII: preserve accents and
non-Latin scripts. Unicode word extraction, with overlapping character
bigrams for unsegmented CJK/Thai runs. This is deliberately lightweight and
not a sophisticated language-specific morphological tokenizer.

Stopwords use Stopwords ISO for supported languages, an English fallback and
automatic exclusion of terms found in more than 80% of documents. Select
up to 10K terms by corpus frequency, then up to 5K for TF-IDF using
term frequency weighted by smoothed IDF.

The SQLite database retains **all unfiltered word counts per article**, not
just the vocabulary words. CSV stores the selected 10K word counts plus
complete 5K sparse TF-IDF and BM15 score arrays as JSON cells.

## Per-article metrics

| Category | Fields |
|---|---|
| Provenance | source_file, source_ordinal, title, article_url |
| Length | token_count, char_count, byte_count, sentence_count, mean_sentence_tokens, mean_token_chars |
| Richness | distinct_terms, lexical_diversity, hapax_ratio, entropy, repetition_ratio |
| Distribution | stopword_ratio, oov_or_filtered_ratio, digit_ratio, punctuation_ratio, vocab_coverage, tfidf_coverage |
| Importance | mean_idf, max_tfidf, tfidf_l2, tfidf_top5_share, tfidf_entropy, bm15_sum, bm15_max |
| Extractive | saliency, title_overlap, lead_relevance, representative_sentence, keywords_json |
| Intermediate | word_counts_json, tfidf_sparse_json, bm15_sparse_json |
| Embedding | vector_q8_b64, vector_scales_b64, semantic_available, fasttext_training_tokens |

Smoothed TF-IDF uses (1 + ln(tf)) × (ln((N+1)/(df+1)) + 1).
BM15 uses a BM25-family formula with k1=1.5 and b=0.75 and average
document length. "Saliency" here is the relative share of the highest
weighted TF-IDF term; it is **not** a causal or deep-learning saliency score.
A representative sentence is selected from the first 80 article sentences
using keyword-weighted lexical relevance; it is extractive, not generated.
TF-IDF token counts are not counts for any particular LLM tokenizer.

## Combined 200D vector

- 160D: TF-IDF-weighted fastText word vectors from a small per-language
  CBOW model; one epoch, at most 2 million sampled tokens, 30K subword
  buckets, 2 CPU threads and 180-second training timeout.
- 32D: deterministic signed-hash projection of the 5K TF-IDF features.
- 8D: normalized article-level scalar metrics.
- Relative weights: 0.8 / 0.18 / 0.02, then L2 normalize.

The vector is approximately semantic, not a Transformer embedding. If
fastText cannot train and allow_lexical_only=true, the semantic portion is
zero-filled and semantic_available=0. Do not treat fallback vectors as
semantic embeddings.

Each 200D vector is signed int8 with a float16 scale for each block of 32
dimensions: 200 bytes plus 14 scale bytes before base64 encoding. This
preserves substantially more precision than global-scale quantization.
Decode with scripts.article_features.dequantize(encoded, scales).

The signed projection and int8 quantization are deterministic. Multi-threaded
fastText optimization is stochastic; the published model is the reproducibility
reference.

## SQLite FTS5 and performance

**FTS5 is a SQLite full-text search index**, not an embedding algorithm.
It is built as an external-content index over the articles table. Query
the unpacked DB with SQL such as:

    SELECT title FROM article_fts WHERE article_fts MATCH 'orbit' LIMIT 5;

FTS5's Unicode tokenizer is not a full CJK segmenter. The FTS index can be
disabled without losing the ordinary article metrics or counts.

The pipeline streams source gzip archives once to a temporary SQLite corpus,
then calculates vocabulary and metrics from stored compressed counts.
fastText uses a capped training sample; there are no per-article API calls.
When source hashes, config and artifact checksums match a published release,
the workflow skips it. New artifacts are published to Hugging Face in one
atomic commit. A conservative disk preflight prevents giant languages
from exhausting hosted runners.

Runtime will vary by language, indexing cost and runner capacity; there is
no measured guarantee that this takes less than half of conversion runtime.
Very large editions may require larger runners. The workflow is manual and
does not run as part of standard Parquet publication.

Local usage:

    python -m pip install numpy zstandard fasttext-wheel stopwordsiso==0.7.1
    python scripts/article_features.py data/*.txt.gz --language en --output article-features --vocab 10000 --tfidf 5000 --allow-lexical-only
