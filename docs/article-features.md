# Per-language article features

A separate enrichment release, with **all stages enabled by default**. Original Wikipedia gzip and structured
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
| article-metrics.csv.gz | One CSV row per article; full metric set, sparse lexical vectors, separate 160D fastText + 128D Bekko vectors and a combined 200D vector, all quantized |
| vocabulary.json.gz | Up to 10,000 corpus terms: ID, total count, document frequency, smoothed IDF and 5K TF-IDF selection flag |
| articles.sqlite.zst | Compressed SQLite: source article text, full per-article compressed word-count JSON, term totals, vocabulary, article metrics and optional FTS5 index |
| fasttext.bin.zst | Language-specific fastText CBOW model (required by default) |
| manifest.json | Source hashes, file hashes, schema/config, model status, vector decoding scheme |

Original dumps may be partial; features only represent already-published articles.

## Text normalization

Unicode NFKC normalization plus Unicode case folding (language-independent)
and folding of common typographic quotes, dashes and spacing. **Every vocabulary
term, TF-IDF feature, fastText training token, and Bekko input uses this folded
form, for all languages.** Original article titles and text remain unchanged. No transliteration to ASCII: preserve accents and
non-Latin scripts. Unicode word extraction including combining marks, with overlapping character
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
| Additional diversity/rarity | effective_vocabulary, normalized_entropy, simpson_diversity, hapax_token_ratio, rare_token_ratio, rare_term_ratio, mean_term_frequency, idf_std |
| Additional topical / BM15 | keyword_token_share, keyword_title_overlap, title_tfidf_share, bm15_mean, bm15_top5_share |
| Sentence diagnostics | sentence_length_cv, extractive_position, extractive_margin, lead_keyword_overlap |
| Fusion diagnostics | vector_quant_cosine, vector_active_fraction, fusion_component_energy_json |
| Intermediate | word_counts_json, tfidf_sparse_json, bm15_sparse_json |
| Embedding | vector_q8_b64, vector_scales_b64; fasttext_q8_b64, fasttext_scales_b64; bekko_q8_b64, bekko_scales_b64; bekko_input_tokens, bekko_truncated, bekko_available, semantic_available, fasttext_training_tokens |

Smoothed TF-IDF uses (1 + ln(tf)) × (ln((N+1)/(df+1)) + 1).
BM15 uses a BM25-family formula with k1=1.5 and b=0.75 and average
document length. "Saliency" here is the relative share of the highest
weighted TF-IDF term; it is **not** a causal or deep-learning saliency score.
A representative sentence is selected from the first 80 article sentences
using keyword-weighted lexical relevance; it is extractive, not generated.
TF-IDF token counts are not counts for any particular LLM tokenizer.

## Per-article Bekko + fastText and combined 200D vector

**Every article** is encoded by both models by default. No sampling for
article-level embedding; fastText training alone uses bounded sampling.

- **160D standalone fastText:** TF-IDF-weighted, normalized word embeddings
  trained with a separate small CBOW model per language. One epoch,
  up to 2M sampled training tokens, 30K subword buckets, two CPU threads,
  180-second training timeout.
- **128D standalone Bekko:** the multilingual
  `hotchpotch/bekko-embedding-v1-a8m` model via CPU OpenVINO.
  Each title and bounded article lead is Unicode-NFKC case-folded and
  encoded with at most 256 model tokens, in 16-article batches.
  **This is a partial-context article embedding**, not a full-document
  Transformer representation. The entire article still contributes to
  TF-IDF, all word counts, metrics and fastText pooling. Record the
  Bekko input token count and whether the excerpt was truncated.
- **200D mixed vector (v3):** preserve **all 128D Bekko**, randomly
  project the 160D fastText vector into 48D, signed-hash the selected
  TF-IDF into 16D, and add eight centered article-level metrics.
- **Importance weights for cosine similarity:** Bekko 55%, fastText 40%,
  TF-IDF 3%, scalar metrics 2%. Normalize each block independently,
  multiply it by the square root of its target contribution, concatenate,
  then L2 normalize. The published vector stores per-article realized
  component energy in `fusion_component_energy_json`, including any
  absent/zero blocks. These weights are design defaults, **not benchmark-tuned**.
- Signed hashing now uses independent bucket/sign bits to avoid a bias
  that previously forced even and odd hash buckets to fixed signs.

Both standalone embeddings and the combined representation are saved.
The 128D Bekko vectors can be compared *across languages* approximately;
per-language fastText and the combined vectors are **not aligned across languages**.

All vectors use symmetric signed-int8 quantization with one float16 scale
per block of 32 dimensions: 160/128/200 signed bytes respectively,
plus 10/8/14 bytes of per-block scales before base64 encoding.
Decode with `scripts.article_features.dequantize(encoded, scales)`.

There is **no silent Bekko fallback**: unavailable dependencies, invalid
model output, or partial encoding fail the job. fastText also fails by
default if training fails; the `allow_lexical_only` switch exists only
for explicit diagnostics and marks missing fastText components as unavailable.

The signed-hash projection (seed 2026) and vector quantization are deterministic.
Multi-threaded fastText training is stochastic; the released model and
manifest are the reproducibility reference. Case folding can discard
acronym/proper-name distinctions and does not guarantee faster Bekko
inference. Its benefit is normalization consistency.

## SQLite FTS5 and performance

**FTS5 is a SQLite full-text search index**, not an embedding algorithm.
It is built as an external-content index over the articles table. Query
the unpacked DB with SQL such as:

    SELECT title FROM article_fts WHERE article_fts MATCH 'orbit' LIMIT 5;

FTS5's Unicode tokenizer is not a full CJK segmenter. The FTS index can be
disabled without losing the ordinary article metrics or counts.

The pipeline streams source gzip archives once to a temporary SQLite corpus,
then calculates vocabulary and metrics from stored compressed counts.
fastText uses a capped training sample; Bekko runs on every article with bounded input.
There are no per-article API calls.
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
    python -m pip install 'sentence-transformers[openvino]>=5.0' 'transformers>=4.57,<5'
    python scripts/article_features.py data/*.txt.gz --language en --output article-features --vocab 10000 --tfidf 5000
