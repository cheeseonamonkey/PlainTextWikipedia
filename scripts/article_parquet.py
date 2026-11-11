"""Streaming, typed article Parquet (v8) and bounded corpus-wide feature profiles.

The article table is intentionally slim: heavy per-article data lives in a few
JSON-dict cells (counts / ratios / diversity / TF-IDF stats / word_counts /
tfidf / keywords), and the fastText embedding is a native ``list<float32>``
column instead of JSON text. Corpus-level context (shared vocabulary, IDF,
moments, centroid) is exported separately as ``corpus-profile.json``.
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np

VERSION = 8

# Top-level scalar/string columns. Everything else is a JSON cell or a native
# float list, which keeps the physical schema small and stable.
ARTICLE_FIELDS = (
    "title", "article_url", "char_count", "counts", "ratios", "diversity",
    "tfidf_stats", "word_counts", "tfidf", "keywords", "fasttext", "source",
)
TEXT = {"title", "article_url", "counts", "ratios", "diversity", "tfidf_stats",
        "word_counts", "tfidf", "keywords", "source"}
INTEGER = {"char_count"}

GROUPED_KEYS = {
    "counts": ("word_count", "sentence_count", "distinct_terms", "byte_count",
               "mean_sentence_tokens", "mean_token_chars", "reading_time_minutes"),
    "ratios": ("lexical_diversity", "hapax_ratio", "stopword_ratio",
               "digit_ratio", "punctuation_ratio", "repetition_ratio",
               "vocab_coverage", "title_overlap"),
    "diversity": ("entropy", "normalized_entropy", "simpson_diversity"),
    "tfidf_stats": ("l2", "top5_share", "entropy", "mean_idf", "idf_std"),
}


def schema(fields):
    import pyarrow as pa
    columns = [pa.field("text", pa.large_string(), nullable=False)]
    for name in fields:
        if name == "fasttext":
            columns.append(pa.field(name, pa.list_(pa.float32()), nullable=False))
        elif name in TEXT:
            columns.append(pa.field(name, pa.large_string(), nullable=False))
        elif name in INTEGER:
            columns.append(pa.field(name, pa.int64(), nullable=False))
        else:
            raise ValueError(f"Unknown column kind: {name}")
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
            rows = []
            for row in self.rows:
                item = dict(row)
                if isinstance(item.get("fasttext"), np.ndarray):
                    item["fasttext"] = item["fasttext"].astype(np.float32).tolist()
                rows.append(item)
            self.writer.write_table(pa.Table.from_pylist(rows, schema=self.schema))
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

    def add(self, flat: dict):
        self.count += 1
        values = {key: float(value) for key, value in flat.items()
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
        vector = flat.get("fasttext")
        if vector is not None:
            arr = np.asarray(vector, dtype=np.float64)
            if self.centroid is None:
                self.centroid = np.zeros_like(arr)
            self.centroid += arr

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
                "fasttext_centroid": (self.centroid / self.count).tolist()
                if self.centroid is not None else None,
                "percentiles": "deterministic reservoir sample, up to 2048 rows"}
