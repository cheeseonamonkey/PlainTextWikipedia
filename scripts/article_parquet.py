"""Streaming, typed article Parquet and bounded corpus-wide feature profiles."""
from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np

# Keys absent from a source are not hallucinated. Preserve nullability explicitly.
TEXT = {"source_file", "title", "article_url", "representative_sentence",
        "word_counts_json", "word_counts_all_json", "tfidf_sparse_json",
        "bm15_sparse_json", "keywords_json", "fusion_component_energy_json",
        "vector_q8_b64", "vector_scales_b64", "fasttext_q8_b64",
        "fasttext_scales_b64", "bekko_q8_b64", "bekko_scales_b64"}
INTEGER = {"source_ordinal", "token_count", "distinct_terms", "vocab_terms",
           "tfidf_terms", "char_count", "byte_count", "sentence_count",
           "bekko_input_tokens", "bekko_truncated", "bekko_available",
           "semantic_available", "fasttext_training_tokens", "hapax_count",
           "stopword_token_count", "known_token_count", "tfidf_token_count",
           "digit_char_count", "punctuation_char_count", "sentiment_sample_chars",
           "reading_sample_chars"}


def schema(fields):
    import pyarrow as pa
    columns = [pa.field("text", pa.large_string(), nullable=False)]
    columns.extend(pa.field(name,
                            pa.large_string() if name in TEXT else
                            pa.int64() if name in INTEGER else pa.float64())
                   for name in fields)
    columns.append(pa.field("embedding_200", pa.list_(pa.float16(), 200)))
    return pa.schema(columns, metadata={b"article_features_schema": b"5"})


class ArticleParquet:
    def __init__(self, path: Path, fields):
        import pyarrow.parquet as pq
        self.schema = schema(fields)
        self.writer = pq.ParquetWriter(path, self.schema, compression="zstd",
                                       compression_level=3, use_dictionary=["source_file"])
        self.rows = []
        self.bytes = 0
        self.count = 0

    def add(self, text: str, metrics: dict, embedding):
        self.rows.append({"text": text, **metrics,
                          "embedding_200": np.asarray(embedding, dtype=np.float16).tolist()})
        self.count += 1
        self.bytes += len(text) + sum(len(v) for v in metrics.values() if isinstance(v, str))
        if len(self.rows) >= 1024 or self.bytes >= 16 * 1024 * 1024:
            self.flush()

    def flush(self):
        if self.rows:
            import pyarrow as pa
            self.writer.write_table(pa.Table.from_pylist(self.rows, schema=self.schema))
            self.rows.clear()
            self.bytes = 0

    def close(self):
        self.flush()
        self.writer.close()


class CorpusProfile:
    """Population moments plus a deterministic bounded reservoir for quantiles."""
    def __init__(self, limit=2048):
        self.count = 0
        self.stats = {}
        self.sample = []
        self.limit = limit
        self.random = random.Random(2026)
        self.centroid = None

    def add(self, row: dict, embedding=None):
        self.count += 1
        values = {key: float(value) for key, value in row.items()
                  if isinstance(value, (int, float)) and not isinstance(value, bool)
                  and math.isfinite(value)}
        for key, value in values.items():
            stat = self.stats.setdefault(key, [0, 0., 0., value, value])
            stat[0] += 1
            delta = value - stat[1]
            stat[1] += delta / stat[0]
            stat[2] += delta * (value - stat[1])
            stat[3] = min(stat[3], value)
            stat[4] = max(stat[4], value)
        if len(self.sample) < self.limit:
            self.sample.append(values)
        else:
            slot = self.random.randrange(self.count)
            if slot < self.limit:
                self.sample[slot] = values
        if embedding is not None:
            vec = np.asarray(embedding, dtype=np.float64)
            if self.centroid is None:
                self.centroid = np.zeros_like(vec)
            self.centroid += vec

    def export(self):
        fields = {}
        for key, (n, mean, m2, minimum, maximum) in self.stats.items():
            sample = [x[key] for x in self.sample if key in x] or [mean]
            fields[key] = {
                "count": n, "mean": mean, "stddev": math.sqrt(max(0., m2 / n)),
                "min": minimum, "max": maximum,
                "p05": float(np.percentile(sample, 5)),
                "p50": float(np.percentile(sample, 50)),
                "p95": float(np.percentile(sample, 95)),
            }
        return {"documents": self.count, "fields": fields,
                "percentiles": "deterministic reservoir sample, up to 2048 rows",
                "vector_centroid": (self.centroid / self.count).tolist()
                                   if self.centroid is not None and self.count else None,
                "vector_centroid_is_normalized": False}
