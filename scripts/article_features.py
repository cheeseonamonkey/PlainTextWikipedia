#!/usr/bin/env python3
"""Per-language, corpus-wide lexical + small fastText article features.

Three phases: stream sources into a temporary SQLite corpus, fit a bounded
vocabulary, then derive article metrics and int8 vectors. Never changes the
canonical article archives or Parquet schema.
"""
from __future__ import annotations

import argparse
import base64
import csv
import gzip
import hashlib
import json
import math
import os
import re
import regex
import sqlite3
import struct
import subprocess
import sys
import tempfile
import unicodedata
import zlib
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

try:
    from scripts.repack_shards import records
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from repack_shards import records

VERSION = 2
VOCAB_SIZE = 10_000
TFIDF_SIZE = 5_000
DENSE_DIMS = 200
FASTTEXT_DIMS = 160
FASTTEXT_PROJECTED_DIMS = 96
BEKKO_DIMS = 128
BEKKO_PROJECTED_DIMS = 64
LEXICAL_DIMS = 32
METRIC_DIMS = 8
K1, B = 1.5, 0.75
# Don't transliterate: those letters are necessary for multilingual text.
FOLD = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"',
                      "‐": "-", "‑": "-", "‒": "-", "–": "-",
                      "—": "-", "―": "-", "\u00a0": " ", "…": "..."})
WORD = regex.compile(r"[\p{L}\p{N}\p{M}]+(?:['-][\p{L}\p{N}\p{M}]+)*")
SENTENCES = re.compile(r"(?<=[.!?。！？])\s+|(?<=[。！？])")
EN_STOP = set(("a an and are as at be been being but by can could did do does "
               "for from had has have he her hers him his how i if in into is it "
               "its me my no nor not of on or our ours she so some such than that "
               "the their them there these they this those through to too was we "
               "were what when where which while who whom why will with would you "
               "your yours also about after all any because before between both "
               "each few more most other over own same should then very "
               "via per may many one two three first second").split())
CSV_FIELDS = (
    "source_file", "source_ordinal", "title", "article_url",
    "token_count", "distinct_terms", "vocab_terms", "tfidf_terms",
    "char_count", "byte_count", "sentence_count", "mean_sentence_tokens",
    "mean_token_chars", "lexical_diversity", "hapax_ratio", "stopword_ratio",
    "oov_or_filtered_ratio", "digit_ratio", "punctuation_ratio", "repetition_ratio", "entropy",
    "vocab_coverage", "tfidf_coverage", "mean_idf", "max_tfidf",
    "tfidf_l2", "tfidf_top5_share", "tfidf_entropy",
    "bm15_sum", "bm15_max", "saliency", "title_overlap",
    "lead_relevance", "representative_sentence",
    "word_counts_json", "tfidf_sparse_json", "bm15_sparse_json",
    "keywords_json", "vector_q8_b64", "vector_scales_b64",
    "fasttext_q8_b64", "fasttext_scales_b64",
    "bekko_q8_b64", "bekko_scales_b64", "bekko_input_tokens",
    "bekko_truncated", "bekko_available",
    "semantic_available", "fasttext_training_tokens",
)


def compact(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def fold(text: str) -> str:
    return unicodedata.normalize("NFKC", text.translate(FOLD)).casefold()


def tokens(text: str) -> list[str]:
    """Unicode word segmentation; add overlapping CJK/Thai character bigrams."""
    normalized = fold(text)
    parts = WORD.findall(normalized)
    result = []
    for word in parts:
        # No whitespace: CJK/Thai word runs need subword tokens.
        if len(word) > 1 and any(
            "\u3040" <= ch <= "\u30ff" or "\u3400" <= ch <= "\u9fff"
            or "\u0e00" <= ch <= "\u0e7f" for ch in word
        ):
            result.extend(word[i:i + 2] for i in range(len(word) - 1))
        else:
            result.append(word)
    return result


@lru_cache(maxsize=256)
def language_stopwords(code: str) -> frozenset[str]:
    """Aggressive multilingual lists where available, plus corpus high-df stopwords."""
    words = set(EN_STOP) if code in {"en", "simple"} else set()
    try:
        from stopwordsiso import stopwords
    except ImportError:
        return frozenset(words)
    iso = "en" if code == "simple" else code.split("_", 1)[0].split("-", 1)[0]
    words.update(fold(term) for term in stopwords(iso))
    return frozenset(words)


def idf(docs: int, df: int) -> float:
    return math.log((docs + 1) / (df + 1)) + 1


def bm_idf(docs: int, df: int) -> float:
    return math.log1p((docs - df + 0.5) / (df + 0.5))


def norm(values: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(values)
    return values / length if length > 1e-10 else values


def quantize(vector: np.ndarray, block: int = 32) -> tuple[str, str]:
    """Symmetric per-block int8 with float16 scales; accepts arbitrary dimensions."""
    if not len(vector) or not np.isfinite(vector).all():
        raise ValueError("Expected nonempty finite vector")
    encoded, scales = bytearray(), bytearray()
    for start in range(0, len(vector), block):
        section = vector[start:start + block]
        scale = float(np.max(np.abs(section))) / 127 if np.any(section) else 1.0
        # Store the *rounded* scale used for quantization, so decoding agrees.
        scale = struct.unpack("<e", struct.pack("<e", scale))[0]
        if scale == 0:
            scale = 1.0
        encoded.extend(np.rint(np.clip(section / scale, -127, 127))
                       .astype(np.int8).tobytes())
        scales.extend(struct.pack("<e", scale))
    return (base64.b64encode(encoded).decode("ascii"),
            base64.b64encode(scales).decode("ascii"))


def dequantize(encoded: str, scales: str) -> np.ndarray:
    data = np.frombuffer(base64.b64decode(encoded), dtype=np.int8).astype(np.float32)
    weights = np.frombuffer(base64.b64decode(scales), dtype="<f2").astype(np.float32)
    return data * np.repeat(weights, 32)[:len(data)]


def lexical_hash(scores: list[tuple[int, float]]) -> np.ndarray:
    arr = np.zeros(LEXICAL_DIMS, dtype=np.float32)
    for term_id, score in scores:
        digest = hashlib.blake2s(str(term_id).encode(), digest_size=4).digest()
        slot = int.from_bytes(digest, "little") % LEXICAL_DIMS
        sign = -1 if digest[0] & 1 else 1
        arr[slot] += sign * score
    return norm(arr)


# Fixed signed random projection; no fitted cross-language parameters needed.
FASTTEXT_PROJECTION = (np.random.default_rng(2026).choice(
    [-1., 1.], (FASTTEXT_DIMS, FASTTEXT_PROJECTED_DIMS))
    .astype(np.float32) / math.sqrt(FASTTEXT_PROJECTED_DIMS))


def vectorize(semantic: np.ndarray | None, lexical: np.ndarray,
              metrics: list[float], bekko: np.ndarray) -> tuple[str, str]:
    """160D fastText -> 96D, plus 64D Bekko, 32D TF-IDF, 8D metrics."""
    if len(metrics) != METRIC_DIMS or bekko.shape != (BEKKO_DIMS,):
        raise ValueError("Invalid mixed-vector inputs")
    semantic96 = (norm(semantic.astype(np.float32) @ FASTTEXT_PROJECTION)
                  if semantic is not None else np.zeros(FASTTEXT_PROJECTED_DIMS, np.float32))
    parts = np.concatenate((semantic96 * .68,
                            norm(bekko[:BEKKO_PROJECTED_DIMS].astype(np.float32)) * .22,
                            lexical * .08,
                            np.asarray(metrics, dtype=np.float32) * .02))
    if len(parts) != DENSE_DIMS:
        raise RuntimeError("Invalid combined vector shape")
    return quantize(norm(parts))


def source_articles(files: list[Path]):
    for file in files:
        for ordinal, record in enumerate(records(file), 1):
            raw_title, raw_text, _ = record.split(b"\n", 2)
            yield file.name, ordinal, raw_title.decode("utf-8").rstrip("\r"), raw_text.decode("utf-8").rstrip("\r")


def open_database(path: Path, fts5: bool):
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("PRAGMA temp_store=MEMORY")
    db.execute("PRAGMA cache_size=-32768")
    db.execute("""CREATE TABLE articles(
        id INTEGER PRIMARY KEY, source_file TEXT NOT NULL,
        source_ordinal INTEGER NOT NULL, title TEXT NOT NULL, article_url TEXT,
        text TEXT, counts BLOB NOT NULL, token_count INTEGER NOT NULL,
        char_count INTEGER NOT NULL, byte_count INTEGER NOT NULL,
        sentence_count INTEGER NOT NULL,
        UNIQUE(source_file,source_ordinal))""")
    db.execute("CREATE TABLE terms(term TEXT PRIMARY KEY, tf INTEGER NOT NULL, df INTEGER NOT NULL)")
    db.execute("CREATE TABLE vocabulary(id INTEGER PRIMARY KEY, term TEXT UNIQUE, tf INTEGER, df INTEGER, idf REAL, tfidf_enabled INTEGER)")
    db.execute("CREATE TABLE metrics(article_id INTEGER PRIMARY KEY, metrics_json TEXT NOT NULL)")
    if fts5:
        db.execute("""CREATE VIRTUAL TABLE article_fts USING fts5(
            title, text, content='articles', content_rowid='id', tokenize='unicode61')""")
    return db


def ingest(files: list[Path], db: sqlite3.Connection, training: Path,
           code: str, fts5: bool, max_train_tokens: int, sample_mod: int) -> dict:
    articles = total_tokens = training_tokens = 0
    terms_tf, terms_df = Counter(), Counter()
    buffered = []

    def flush():
        if buffered:
            db.executemany("""INSERT INTO articles(
                source_file,source_ordinal,title,article_url,text,counts,
                token_count,char_count,byte_count,sentence_count)
                VALUES (?,?,?,?,?,?,?,?,?,?)""", buffered)
            buffered.clear()
        if terms_tf:
            db.executemany("""INSERT INTO terms VALUES(?,?,?)
                ON CONFLICT(term) DO UPDATE SET
                tf = tf + excluded.tf, df = df + excluded.df""",
                ((term, tf, terms_df[term]) for term, tf in terms_tf.items()))
            terms_tf.clear()
            terms_df.clear()
        db.commit()
    train_stream = training.open("w", encoding="utf-8")
    try:
        for source, ordinal, title, body in source_articles(files):
            words = tokens(body)
            counts = Counter(words)
            if not counts:
                counts = Counter()
            sentences = [s.strip() for s in SENTENCES.split(body) if s.strip()]
            url = None
            if code:
                try:
                    from scripts.wiki_metadata import article_url
                except ModuleNotFoundError:
                    from wiki_metadata import article_url
                url = article_url(title, code)
            articles += 1
            total_tokens += len(words)
            terms_tf.update(counts)
            terms_df.update(counts.keys())
            packed = zlib.compress(compact(counts).encode("utf-8"), level=1)
            buffered.append((source, ordinal, title, url, body,
                             packed, len(words), len(body), len(body.encode("utf-8")),
                             len(sentences)))
            # Stable deterministic sampling across every shard, bounded training.
            key = hashlib.blake2s(f"{source}:{ordinal}".encode(), digest_size=4).digest()
            if (training_tokens < max_train_tokens and words and
                    int.from_bytes(key, "little") % sample_mod == 0):
                selected = words[:min(256, max_train_tokens - training_tokens)]
                if selected:
                    train_stream.write(" ".join(selected) + "\n")
                    training_tokens += len(selected)
            if len(buffered) >= 512:
                flush()
        flush()
        if articles == 0:
            raise ValueError("No articles found")
        return {"articles": articles, "tokens": total_tokens,
                "train_tokens": training_tokens,
                "raw_vocabulary": db.execute("SELECT count(*) FROM terms").fetchone()[0]}
    finally:
        train_stream.close()


def vocabulary(db, documents: int, code: str, limit: int, shortlist: int):
    # High-df terms function as language-specific stopwords; English also has
    # a small fixed function-word list. Corpus-specific filtering is reproducible.
    blocked = language_stopwords(code)
    ordered = db.execute("SELECT term,tf,df FROM terms ORDER BY tf DESC, term LIMIT ?", (max(limit * 15, 25000),))
    selected = []
    for term, tf, df in ordered:
        if (term in blocked or df > max(1, documents * .80)
                or not term.strip("-'") or len(term) > 64
                or not any(ch.isalpha() for ch in term)):
            continue
        selected.append((term, int(tf), int(df)))
        if len(selected) == limit:
            break
    # Prefer frequent terms that are also distinctive, not just rare proper nouns.
    tfidf_ranks = sorted(selected, key=lambda x: (-(x[1] * idf(documents, x[2])), x[0]))
    enabled = {term for term, _, _ in tfidf_ranks[:shortlist]}
    entries = [(i, term, tf, df, idf(documents, df), int(term in enabled))
               for i, (term, tf, df) in enumerate(selected)]
    db.executemany("INSERT INTO vocabulary VALUES (?,?,?,?,?,?)", entries)
    db.commit()
    return entries


def train_fasttext_worker(inp: Path, model_path: Path, dim: int, threads: int):
    import fasttext
    # CBOW is a cheap context-learning baseline; subword buckets improve OOV.
    model = fasttext.train_unsupervised(str(inp), model="cbow",
                                       dim=dim, epoch=1, lr=.04, ws=5,
                                       minCount=2, minn=2, maxn=4,
                                       neg=4, bucket=30000, thread=threads,
                                       verbose=0)
    model.save_model(str(model_path))


def train_fasttext(training: Path, dest: Path, train_tokens: int, *,
                   threads: int = 2, timeout: int = 180,
                   allow_lexical_only: bool = False):
    if train_tokens < 1000:
        if allow_lexical_only:
            return None, "insufficient_training_tokens"
        raise ValueError("Insufficient fastText training data; use --allow-lexical-only explicitly")
    command = [sys.executable, str(Path(__file__).resolve()), "_train",
               str(training), str(dest), str(FASTTEXT_DIMS), str(threads)]
    try:
        subprocess.run(command, check=True, timeout=timeout, capture_output=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        if not allow_lexical_only:
            raise RuntimeError(f"fastText failed; install fasttext-wheel or use --allow-lexical-only: {exc}") from exc
        return None, type(exc).__name__
    import fasttext
    return fasttext.load_model(str(dest)), None


def metrics_for_article(row, vocabulary_lookup, stopwords: set[str],
                        docs: int, avglen: float,
                        fasttext_vectors: np.ndarray | None, training_tokens: int,
                        bekko_embedding: np.ndarray, bekko_tokens: int):
    (article_id, source_file, ordinal, title, url, body, blob,
     length, chars, bytes_, sentences) = row
    counts = json.loads(zlib.decompress(blob))
    distinct = len(counts)
    total = max(1, length)
    hapax = sum(c == 1 for c in counts.values())
    entropy = -sum((n / total) * math.log(n / total) for n in counts.values()) if length else 0.
    # Only retain vocabulary word counts in the CSV; the SQLite counts BLOB
    # preserves *all* intermediary terms, including terms outside top 10K.
    known = {term: count for term, count in counts.items() if term in vocabulary_lookup}
    stop = sum(counts.get(term, 0) for term in stopwords)
    weighted, bmweighted, vecterms = [], [], []
    mass = bm_sum = 0.0
    title_terms = set(tokens(title))
    for term, count in known.items():
        index, df, value_idf, enabled = vocabulary_lookup[term]
        if enabled:
            score = (1 + math.log(count)) * value_idf
            bm = bm_idf(docs, df) * (count * (K1 + 1)) / (
                count + K1 * (1 - B + B * length / max(1, avglen)))
            weighted.append((index, score, term))
            bmweighted.append((index, bm, term))
            mass += score * score
            bm_sum += bm
            if fasttext_vectors is not None:
                vecterms.append((index, score))
    l2 = math.sqrt(mass)
    weighted.sort(key=lambda item: (-item[1], item[2]))
    bmweighted.sort(key=lambda item: (-item[1], item[2]))
    top = weighted[:16]
    sparse = sorted((idx, round(value / l2, 7)) for idx, value, _ in weighted) if l2 else []
    semantic = None
    if fasttext_vectors is not None and mass:
        indices = np.fromiter((index for index, _ in vecterms),
                              dtype=np.intp, count=len(vecterms))
        weights = np.fromiter((score / l2 for _, score in vecterms),
                              dtype=np.float32, count=len(vecterms))
        semantic = weights @ fasttext_vectors[indices]
    lexical = lexical_hash(sparse)
    vocab_coverage = sum(known.values()) / total
    tfidf_coverage = sum(counts[t] for _, _, t in weighted) / total
    tfidf_sum = sum(v for _, v, _ in weighted)
    top5share = sum(value for _, value, _ in top[:5]) / max(1e-9, tfidf_sum)
    concentration = (sum((v / tfidf_sum) ** 2 for _, v, _ in weighted)
                     if tfidf_sum else 0.0)
    tfidf_entropy = (-sum((v / tfidf_sum) * math.log(v / tfidf_sum)
                          for _, v, _ in weighted) if tfidf_sum else 0.0)
    mean_idf = (sum(vocabulary_lookup[t][2] * n for t, n in known.items()) /
                max(1, sum(known.values())))
    title_overlap = len(title_terms.intersection(known)) / max(1, len(title_terms))
    mean_token_chars = sum(len(term) * n for term, n in counts.items()) / total
    repetition = max(counts.values(), default=0) / total
    scalar = [
        min(1., math.log1p(length) / 14),
        distinct / total,
        vocab_coverage,
        tfidf_coverage,
        min(1., mean_idf / 10),
        min(1., concentration),
        title_overlap,
        min(1., entropy / 10),
    ]
    encoded, scales = vectorize(semantic, lexical, scalar, bekko_embedding)
    fasttext_encoded, fasttext_scales = quantize(
        norm(semantic) if semantic is not None
        else np.zeros(FASTTEXT_DIMS, np.float32))
    bekko_encoded, bekko_scales = quantize(norm(bekko_embedding))
    # Salient extractive sentence: the highest-scoring sentence among a bounded
    # set, not a language-model summary; empty without stored body.
    topweights = {t: v for _, v, t in weighted}
    lead_relevance = 0.
    representative = ""
    if body:
        parts = [piece.strip() for piece in SENTENCES.split(body) if piece.strip()][:80]
        best_score = -1.
        for pos, sentence in enumerate(parts):
            piece_tokens = tokens(sentence[:1200])
            score = sum(topweights.get(word, 0.) for word in piece_tokens[:160])
            score /= math.sqrt(max(1, len(piece_tokens)))
            if pos == 0:
                lead_relevance = score
            if score > best_score:
                best_score = score
                representative = sentence[:320]
    saliency = top[0][1] / max(1e-9, sum(v for _, v, _ in weighted)) if top else 0.
    result = {
        "source_file": source_file, "source_ordinal": ordinal, "title": title,
        "article_url": url, "token_count": length, "distinct_terms": distinct,
        "vocab_terms": len(known), "tfidf_terms": len(weighted),
        "char_count": chars, "byte_count": bytes_,
        "sentence_count": sentences,
        "mean_sentence_tokens": length / max(1, sentences),
        "mean_token_chars": mean_token_chars, "lexical_diversity": distinct / total,
        "hapax_ratio": hapax / max(1, distinct),
        "stopword_ratio": stop / total,
        "oov_or_filtered_ratio": 1 - vocab_coverage,
        "digit_ratio": sum(ch.isdigit() for ch in (body or "")) / max(1, chars),
        "punctuation_ratio": sum(unicodedata.category(ch).startswith("P")
                                 for ch in (body or "")) / max(1, chars),
        "repetition_ratio": repetition, "entropy": entropy,
        "vocab_coverage": vocab_coverage, "tfidf_coverage": tfidf_coverage,
        "mean_idf": mean_idf, "max_tfidf": top[0][1] if top else 0,
        "tfidf_l2": l2, "tfidf_top5_share": top5share,
        "tfidf_entropy": tfidf_entropy,
        "bm15_sum": bm_sum, "bm15_max": bmweighted[0][1] if bmweighted else 0.,
        "saliency": saliency, "title_overlap": title_overlap,
        "lead_relevance": lead_relevance, "representative_sentence": representative,
        "word_counts_json": compact(known),
        "tfidf_sparse_json": compact(sparse),
        "bm15_sparse_json": compact(sorted((idx, round(v, 6)) for idx, v, _ in bmweighted)),
        "keywords_json": compact([[term, round(v, 6)] for _, v, term in top]),
        "vector_q8_b64": encoded, "vector_scales_b64": scales,
        "fasttext_q8_b64": fasttext_encoded, "fasttext_scales_b64": fasttext_scales,
        "bekko_q8_b64": bekko_encoded, "bekko_scales_b64": bekko_scales,
        "bekko_input_tokens": bekko_tokens, "bekko_truncated": int(bekko_tokens > 256),
        "bekko_available": 1,
        "semantic_available": int(fasttext_vectors is not None),
        "fasttext_training_tokens": training_tokens,
    }
    return result


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compress(source: Path, destination: Path):
    try:
        import zstandard as zstd
        with source.open("rb") as inp, destination.open("wb") as out:
            zstd.ZstdCompressor(level=3, threads=2).copy_stream(inp, out)
    except ImportError as exc:
        raise RuntimeError("zstandard is required to package SQLite and fastText models") from exc


def build(files: list[Path], output: Path, code: str, *,
          vocab_size: int = VOCAB_SIZE, shortlist: int = TFIDF_SIZE,
          max_train_tokens: int = 2_000_000, sample_mod: int = 4,
          timeout: int = 180, fts5: bool = True,
          allow_lexical_only: bool = False, threads: int = 2,
          bekko: bool = True) -> dict:
    if not files or any(not p.is_file() or not p.name.endswith(".txt.gz") for p in files):
        raise ValueError("Provide existing .txt.gz archives")
    if not 1 <= shortlist <= vocab_size <= 50_000:
        raise ValueError("Require 1 <= tfidf size <= vocab size <= 50000")
    if not 1 <= sample_mod <= 100 or max_train_tokens < 100:
        raise ValueError("Invalid training sample settings")
    if not re.fullmatch(r"[a-z0-9_-]+", code):
        raise ValueError("Expected a Wikimedia language code")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".features-", dir=output) as work:
        stage = Path(work)
        database = stage / "articles.sqlite"
        training = stage / "fasttext-training.txt"
        model_path = stage / "fasttext.bin"
        db = open_database(database, fts5)
        try:
            totals = ingest(sorted(files), db, training, code, fts5, max_train_tokens, sample_mod)
            entries = vocabulary(db, totals["articles"], code, vocab_size, shortlist)
            lookup = {term: (index, df, value_idf, enabled)
                      for index, term, _, df, value_idf, enabled in entries}
            stopwords = set(language_stopwords(code)).union(
                term for (term,) in db.execute(
                    "SELECT term FROM terms WHERE df > ?", (totals["articles"] * .80,)))
            model, reason = train_fasttext(training, model_path, totals["train_tokens"],
                                           timeout=timeout, threads=threads,
                                           allow_lexical_only=allow_lexical_only)
            vectors = (np.stack([model.get_word_vector(term) for _, term, *_ in entries])
                       .astype(np.float32) if model is not None and entries else None)
            avglen = totals["tokens"] / totals["articles"]
            if not bekko:
                raise ValueError("Bekko is required for complete semantic features")
            try:
                from scripts.bekko_features import BekkoEncoder, embedding_input
            except ModuleNotFoundError as exc:
                if exc.name != "scripts":
                    raise
                from bekko_features import BekkoEncoder, embedding_input
            encoder = BekkoEncoder()  # Load exactly once per language.
            metrics_path = stage / "article-metrics.csv.gz"
            with gzip.open(metrics_path, "wt", encoding="utf-8", newline="", compresslevel=4) as out:
                writer = csv.DictWriter(out, CSV_FIELDS)
                writer.writeheader()
                cur = db.execute("""SELECT id,source_file,source_ordinal,title,
                    article_url,text,counts,token_count,char_count,byte_count,
                    sentence_count FROM articles ORDER BY id""")
                from itertools import islice
                while chunk := list(islice(cur, 16)):
                    inputs = [embedding_input(row[3], row[5]) for row in chunk]
                    embeddings, lengths = encoder.encode(inputs)
                    if len(embeddings) != len(chunk) or len(lengths) != len(chunk):
                        raise RuntimeError("Bekko did not return one vector per article")
                    for row, bekko_vector, input_tokens in zip(chunk, embeddings, lengths):
                        result = metrics_for_article(row, lookup, stopwords,
                                                     totals["articles"], avglen,
                                                     vectors, totals["train_tokens"],
                                                     bekko_vector, input_tokens)
                        writer.writerow(result)
                        db.execute("INSERT INTO metrics VALUES (?,?)",
                                   (row[0], compact({key: val for key, val in result.items()
                                                    if key not in ("word_counts_json",
                                                                   "tfidf_sparse_json",
                                                                   "bm15_sparse_json")})))
            if fts5:
                db.execute("INSERT INTO article_fts(article_fts) VALUES('rebuild')")
                db.execute("INSERT INTO article_fts(article_fts) VALUES('integrity-check')")
            db.commit()
            if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise RuntimeError("SQLite integrity check failed")
            if db.execute("SELECT count(*) FROM metrics").fetchone()[0] != totals["articles"]:
                raise RuntimeError("Feature row-count mismatch")
            db.close()
            words_path = stage / "vocabulary.json.gz"
            with gzip.open(words_path, "wt", encoding="utf-8", compresslevel=6) as out:
                json.dump({"version": VERSION, "language": code, "documents": totals["articles"],
                           "vocab_size": len(entries),
                           "tfidf_size": sum(entry[5] for entry in entries),
                           "average_document_tokens": avglen,
                           "terms": [{"id": idx, "term": term, "tf": tf, "df": df,
                                      "idf": idfval, "tfidf": bool(active)}
                                     for idx, term, tf, df, idfval, active in entries]},
                          out, ensure_ascii=False, separators=(",", ":"))
            packaged = stage / "articles.sqlite.zst"
            compress(database, packaged)
            artifacts = [metrics_path, words_path, packaged]
            if model is not None:
                model = None
                compressed_model = stage / "fasttext.bin.zst"
                compress(model_path, compressed_model)
                artifacts.append(compressed_model)
            report = {
                "version": VERSION, "language": code,
                "documents": totals["articles"], "raw_tokens": totals["tokens"],
                "average_document_tokens": avglen,
                "raw_vocabulary": totals["raw_vocabulary"],
                "vocabulary_size": len(entries), "tfidf_size": sum(entry[5] for entry in entries),
                "fasttext": {"available": vectors is not None, "reason": reason,
                             "dim": FASTTEXT_DIMS, "algorithm": "cbow",
                             "epoch": 1, "subword_bucket": 30000,
                             "training_tokens": totals["train_tokens"],
                             "sample_mod": sample_mod, "train_token_cap": max_train_tokens,
                             "timeout_seconds": timeout},
                "bekko": {"model": "hotchpotch/bekko-embedding-v1-a8m",
                          "backend": "openvino", "device": "cpu", "dimensions": 128,
                          "max_input_tokens": 256, "batch_size": 16,
                          "input": "NFKC+casefold title and bounded lead excerpt",
                          "required": True},
                "vector": {"dimensions": DENSE_DIMS,
                           "components": {"fasttext_projected": 96, "bekko": 64,
                                          "signed_tfidf_hash": 32, "document_metrics": 8},
                           "weighting": [0.68, 0.22, 0.08, 0.02],
                           "quantization": "symmetric_int8_per_32_float16_scale",
                           "byte_order": "little",
                           "column": "vector_q8_b64", "scales": "vector_scales_b64"},
                "lexical": {"normalization": "NFKC+casefold+common_quote_dash_fold",
                            "max_df_ratio": 0.8,
                            "stopwords": "stopwordsiso (if installed) + English defaults + high-df filtering",
                            "idf": "ln((N+1)/(df+1))+1",
                            "bm15": {"k1": K1, "b": B},
                            "fts5": fts5,
                            "word_counts": "full counts stored as zlib JSON per article in SQLite; 10K projected counts in CSV"},
                "files": {},
            }
            for file in artifacts:
                report["files"][file.name] = {"bytes": file.stat().st_size,
                                              "sha256": sha256(file)}
            manifest = stage / "manifest.json"
            manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
            artifacts.append(manifest)
            for file in artifacts:
                os.replace(file, output / file.name)
            return report
        finally:
            db.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sources", nargs="*", type=Path)
    p.add_argument("--output", type=Path, default=Path("article-features"))
    p.add_argument("--language", default="en")
    p.add_argument("--vocab", type=int, default=VOCAB_SIZE)
    p.add_argument("--tfidf", type=int, default=TFIDF_SIZE)
    p.add_argument("--train-token-cap", type=int, default=2_000_000)
    p.add_argument("--sample-mod", type=int, default=4)
    p.add_argument("--fasttext-timeout", type=int, default=180)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--no-fts5", action="store_true")
    p.add_argument("--allow-lexical-only", action="store_true")
    p.add_argument("--no-bekko", action="store_true",
                   help="Disable Bekko (incomplete release; rejected by full builder)")
    args = p.parse_args()
    if args.sources and args.sources[0].name == "_train":
        train_fasttext_worker(Path(args.sources[1]), Path(args.sources[2]),
                              int(str(args.sources[3])), int(str(args.sources[4])))
        return
    if not args.sources:
        p.error("At least one source .txt.gz required")
    report = build(args.sources, args.output, args.language, vocab_size=args.vocab,
                   shortlist=args.tfidf, max_train_tokens=args.train_token_cap,
                   sample_mod=args.sample_mod, timeout=args.fasttext_timeout,
                   fts5=not args.no_fts5, allow_lexical_only=args.allow_lexical_only,
                   threads=args.threads, bekko=not args.no_bekko)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
