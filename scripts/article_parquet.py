"""Streaming, typed article Parquet and bounded corpus-wide feature profiles."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np

VERSION = 7
ARTICLE_FIELDS = (
    "title", "article_url", "char_count", "sentence_count", "distinct_terms",
    "mean_sentence_tokens", "mean_token_chars", "lexical_diversity",
    "hapax_ratio", "stopword_ratio", "digit_ratio", "punctuation_ratio",
    "repetition_ratio", "entropy", "normalized_entropy", "simpson_diversity",
    "vocab_coverage", "mean_idf", "idf_std",
    "tfidf_l2", "tfidf_top5_share", "tfidf_entropy", "title_overlap",
    "word_counts", "tfidf", "bm25", "keywords", "fasttext", "bekko",
    "bekko_input_tokens", "bekko_truncated",
)


# Keys absent from a source are not hallucinated. Preserve nullability explicitly.
TEXT = {"title", "article_url", "word_counts", "tfidf", "bm25", "keywords", "fasttext", "bekko"}
INTEGER = {"char_count", "sentence_count", "distinct_terms", "bekko_input_tokens", "bekko_truncated"}


def schema(fields):
    import pyarrow as pa
    columns = [pa.field("text", pa.large_string(), nullable=False)]
    columns.extend(pa.field(name,
                            pa.large_string() if name in TEXT else
                            pa.int64() if name in INTEGER else pa.float64())
                   for name in fields)
    return pa.schema(columns, metadata={b"article_features_schema": str(VERSION).encode()})


class ArticleParquet:
    def __init__(self, path: Path, fields):
        import pyarrow.parquet as pq
        self.schema = schema(fields)
        self.writer = pq.ParquetWriter(path, self.schema, compression="zstd",
                                       compression_level=3, use_dictionary=False)
        self.rows = []
        self.bytes = 0
        self.count = 0

    def add(self, text: str, metrics: dict):
        self.rows.append({"text": text, **metrics})
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
        self.embeddings = {}

    def add(self, row: dict):
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
        for key in ("fasttext", "bekko"):
            if row.get(key):
                vector = np.asarray(json.loads(row[key]), dtype=np.float64)
                if key not in self.embeddings:
                    self.embeddings[key] = np.zeros_like(vector)
                self.embeddings[key] += vector

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
                "embedding_centroids": {k: (v / self.count).tolist() for k, v in self.embeddings.items()},
                "word_statistics": "words.jsonl.gz: full corpus TF/DF/IDF",
                "percentiles": "deterministic reservoir sample, up to 2048 rows",
                "embedding_centroids_are_normalized": False}
