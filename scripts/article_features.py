#!/usr/bin/env python3
"""Per-language, corpus-wide lexical + small fastText article features.

Stream each source once into combined text and one temporary corpus index,
then fit lexical features and independent fastText/Bekko embeddings.
No database or combined embedding is published.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import lzma
import math
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import unicodedata
import zlib
from collections import Counter
from itertools import islice, zip_longest
from pathlib import Path

import numpy as np

try:
    from scripts.repack_shards import records
    from scripts.article_parquet import VERSION, ARTICLE_FIELDS
    from scripts.bekko_features import MODEL_ID, MODEL_REVISION, MODEL_DIMS as BEKKO_DIMS, MAX_TOKENS, BATCH_SIZE
    from scripts.lexical_features import fold, tokens, LexicalNormalizer, MAX_DF_RATIO
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from repack_shards import records
    from article_parquet import VERSION, ARTICLE_FIELDS
    from bekko_features import MODEL_ID, MODEL_REVISION, MODEL_DIMS as BEKKO_DIMS, MAX_TOKENS, BATCH_SIZE
    from lexical_features import fold, tokens, LexicalNormalizer, MAX_DF_RATIO

VOCAB_SIZE = 10_000
FASTTEXT_DIMS = 160
K1, B = 1.5, 0.75
SENTENCES = re.compile(r"(?<=[.!?。！？])\s+|(?<=[。！？])")


def compact(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def idf(docs: int, df: int) -> float:
    return math.log((docs + 1) / (df + 1)) + 1


def bm_idf(docs: int, df: int) -> float:
    return math.log1p((docs - df + 0.5) / (df + 0.5))


def norm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("Expected a finite, nonempty embedding vector")
    length = np.linalg.norm(values.astype(np.float64))
    return values / length if length > 1e-10 else values


def source_articles(files: list[Path]):
    for file in files:
        for record in records(file):
            raw_title, raw_text, _ = record.split(b"\n", 2)
            yield raw_title.decode("utf-8").rstrip("\r"), raw_text.decode("utf-8").rstrip("\r")


def open_database(path: Path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("PRAGMA temp_store=FILE")
    db.execute("PRAGMA cache_size=-32768")
    db.execute("""CREATE TABLE articles(
        id INTEGER PRIMARY KEY, title TEXT NOT NULL, article_url TEXT,
        counts BLOB NOT NULL, token_count INTEGER NOT NULL,
        char_count INTEGER NOT NULL, sentence_count INTEGER NOT NULL)""")
    for table in ("terms", "lexical_terms"):
        db.execute(f"CREATE TABLE {table}(term TEXT PRIMARY KEY, tf INTEGER NOT NULL, df INTEGER NOT NULL)")
    return db


def ingest(files: list[Path], db: sqlite3.Connection, training: Path,
           code: str, normalizer,
           text_output: Path | None = None) -> dict:
    articles = total_tokens = 0
    tallies = {table: (Counter(), Counter()) for table in ("terms", "lexical_terms")}
    buffered = []

    def flush():
        if buffered:
            db.executemany("""INSERT INTO articles(
                title,article_url,counts,token_count,char_count,sentence_count)
                VALUES (?,?,?,?,?,?)""", buffered)
            buffered.clear()
        for table, (tf, df) in tallies.items():
            db.executemany(f"""INSERT INTO {table} VALUES(?,?,?)
                ON CONFLICT(term) DO UPDATE SET
                tf = tf + excluded.tf, df = df + excluded.df""",
                ((term, count, df[term]) for term, count in tf.items()))
            tf.clear()
            df.clear()
        db.commit()
    train_stream = gzip.open(training, "wt", encoding="utf-8", compresslevel=1)
    text_stream = lzma.open(text_output, "wt", encoding="utf-8", newline="\n", preset=3) if text_output else None
    try:
        for title, body in source_articles(files):
            if text_stream:
                text_stream.write(f"{title}\n{body}\n\n")
            words = tokens(body)
            counts = Counter(words)
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
            normalized = normalizer.counts(counts)
            for table, values in (("terms", counts), ("lexical_terms", normalized)):
                tallies[table][0].update(values)
                tallies[table][1].update(values.keys())
            packed = zlib.compress(compact(counts).encode("utf-8"), level=1)
            buffered.append((title, url, packed, len(words), len(body), len(sentences)))
            train_stream.write(" ".join(normalizer.words(words)) + "\n")
            if len(buffered) >= 512 or len(tallies["terms"][0]) >= 100_000:
                flush()
        flush()
        if articles == 0:
            raise ValueError("No articles found")
        return {"articles": articles, "tokens": total_tokens,
                "raw_vocabulary": db.execute("SELECT count(*) FROM terms").fetchone()[0]}
    finally:
        train_stream.close()
        if text_stream:
            text_stream.close()


def article_rows(db, text_path):
    """Read bodies from the combined text instead of storing a second plaintext copy."""
    cursor = db.execute("SELECT * FROM articles ORDER BY id")
    for row, record in zip_longest(cursor, source_articles([text_path])):
        if row is None or record is None or row["title"] != record[0] or row["char_count"] != len(record[1]):
            raise RuntimeError("Combined text does not match the article index")
        yield {**dict(row), "text": record[1]}


def vocabulary(db, documents, limit):
    ordered = db.execute("SELECT term,tf,df FROM lexical_terms WHERE df <= ? ORDER BY tf DESC, term LIMIT ?",
                         (max(1, documents * MAX_DF_RATIO), limit))
    return [dict(term=term, tf=tf, df=df, idf=idf(documents, df)) for term, tf, df in ordered]


def prepare_training(candidate, destination, vocabulary):
    count = 0
    with gzip.open(candidate, "rt", encoding="utf-8") as source, destination.open("w", encoding="utf-8") as out:
        for line in source:
            words = [word for word in line.split() if word in vocabulary]
            out.write(" ".join(words) + "\n")
            count += len(words)
    candidate.unlink()
    return count


def train_fasttext_worker(inp: Path, model_path: Path, dim: int, threads: int):
    import fasttext
    # CBOW is a cheap context-learning baseline; subword buckets improve OOV.
    model = fasttext.train_unsupervised(str(inp), model="cbow",
                                       dim=dim, epoch=1, lr=.04, ws=5,
                                       minCount=1, minn=2, maxn=4,
                                       neg=4, bucket=30000, thread=threads,
                                       verbose=0)
    model.save_model(str(model_path))


def train_fasttext(training: Path, dest: Path, train_tokens: int, *,
                   threads: int = 2, timeout: int = 0):
    if train_tokens < 1:
        raise ValueError("No tokenized text is available to train fastText")
    command = [sys.executable, str(Path(__file__).resolve()), "_train",
               str(training), str(dest), str(FASTTEXT_DIMS), str(threads)]
    try:
        subprocess.run(command, check=True, timeout=timeout or None, capture_output=True)
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or b"").decode("utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"fastText training failed: {details}") from exc
    import fasttext
    return fasttext.load_model(str(dest))


def metrics_for_article(row, vocabulary_lookup, stopwords, docs, avglen,
                        model, bekko_embedding, bekko_tokens, normalizer):
    title, url, body = row["title"], row["article_url"], row["text"]
    blob, length, chars, sentences = (row[key] for key in
        ("counts", "token_count", "char_count", "sentence_count"))
    if np.asarray(bekko_embedding).shape != (BEKKO_DIMS,):
        raise ValueError("Invalid Bekko embedding dimensions")
    counts = json.loads(zlib.decompress(blob))
    total = max(1, length)
    distinct = len(counts)
    entropy = -sum(n / total * math.log(n / total) for n in counts.values())
    known = {term: n for term, n in normalizer.counts(counts).items() if term in vocabulary_lookup}
    weighted, bm = {}, {}
    for term, n in known.items():
        meta = vocabulary_lookup[term]
        weighted[term] = (1 + math.log(n)) * meta["idf"]
        bm[term] = bm_idf(docs, meta["df"]) * n * (K1 + 1) / (
            n + K1 * (1 - B + B * length / max(1, avglen)))
    l2 = math.sqrt(sum(v * v for v in weighted.values()))
    mass = sum(weighted.values())
    top = sorted(weighted.items(), key=lambda pair: (-pair[1], pair[0]))[:16]
    known_count = sum(known.values())
    mean_idf = sum(vocabulary_lookup[t]["idf"] * n for t, n in known.items()) / max(1, known_count)
    title_terms = set(normalizer.words(tokens(title)))
    # Use exactly the same reduced vocabulary as fastText training.
    vector = np.zeros(FASTTEXT_DIMS, dtype=np.float32)
    model_words = list(normalizer.words(tokens(body), vocabulary_lookup))
    if model_words:
        vector = norm(np.asarray(model.get_sentence_vector(" ".join(model_words)), dtype=np.float32))
    if vector.shape != (FASTTEXT_DIMS,):
        raise ValueError("Invalid fastText embedding dimensions")
    return dict(
        title=title, article_url=url, char_count=chars, sentence_count=sentences,
        distinct_terms=distinct, mean_sentence_tokens=length / max(1, sentences),
        mean_token_chars=sum(len(t) * n for t, n in counts.items()) / total,
        lexical_diversity=distinct / total,
        hapax_ratio=sum(n == 1 for n in counts.values()) / max(1, distinct),
        stopword_ratio=sum(n for t, n in counts.items() if t in stopwords) / total,
        digit_ratio=sum(c.isdigit() for c in body) / max(1, chars),
        punctuation_ratio=sum(unicodedata.category(c).startswith("P") for c in body) / max(1, chars),
        repetition_ratio=max(counts.values(), default=0) / total, entropy=entropy,
        normalized_entropy=entropy / math.log(distinct) if distinct > 1 else 0.,
        simpson_diversity=1 - sum((n / total) ** 2 for n in counts.values()) if length else 0.,
        vocab_coverage=known_count / total,
        mean_idf=mean_idf,
        idf_std=math.sqrt(sum(n * (vocabulary_lookup[t]["idf"] - mean_idf) ** 2
                               for t, n in known.items()) / max(1, known_count)),
        tfidf_l2=l2, tfidf_top5_share=sum(v for _, v in top[:5]) / mass if mass else 0.,
        tfidf_entropy=-sum(v / mass * math.log(v / mass) for v in weighted.values()) if mass else 0.,
        title_overlap=len(title_terms & known.keys()) / max(1, len(title_terms)),
        word_counts=compact(counts),
        tfidf=compact({t: v / l2 for t, v in weighted.items()}) if l2 else "{}",
        bm25=compact(bm), keywords=compact(dict(top)),
        fasttext=compact(vector.tolist()),
        bekko=compact(norm(bekko_embedding).tolist()),
        bekko_input_tokens=bekko_tokens, bekko_truncated=int(bekko_tokens > MAX_TOKENS))


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compress(source: Path, destination: Path):
    try:
        import zstandard as zstd
        with source.open("rb") as inp, destination.open("wb") as out:
            zstd.ZstdCompressor(level=3, threads=2).copy_stream(inp, out)
    except ImportError as exc:
        raise RuntimeError("zstandard is required to package fastText models") from exc


def build(files: list[Path], output: Path, code: str, *,
          vocab_size: int = VOCAB_SIZE,
          timeout: int = 0, threads: int = 2) -> dict:
    if not files or len({p.resolve() for p in files}) != len(files):
        raise ValueError("Provide distinct source archives")
    if any(not p.is_file() or not p.name.endswith((".txt.gz", ".txt.xz")) for p in files):
        raise ValueError("Provide existing .txt.gz archives")
    if not 1 <= vocab_size <= VOCAB_SIZE:
        raise ValueError("Require 1 <= vocabulary size <= 10000")
    if not re.fullmatch(r"[a-z0-9_-]+", code):
        raise ValueError("Expected a Wikimedia language code")
    if threads < 1 or timeout < 0:
        raise ValueError("Require positive threads and a nonnegative timeout")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".features-", dir=output) as work:
        stage = Path(work)
        database = stage / "articles.sqlite"
        training = stage / "fasttext-training.txt"
        candidate = stage / "lexical-training.txt.gz"
        normalizer = LexicalNormalizer(code)
        model_path = stage / "fasttext.bin"
        db = open_database(database)
        try:
            text_path = stage / "articles.txt.xz"
            totals = ingest(sorted(files), db, candidate, code, normalizer,
                            text_output=text_path)
            print(f"Indexed {totals['articles']:,} articles; fitting shared vocabulary", flush=True)
            entries = vocabulary(db, totals["articles"], vocab_size)
            lookup = {entry["term"]: entry for entry in entries}
            stopwords = set(normalizer.stopwords).union(
                term for (term,) in db.execute(
                    "SELECT term FROM terms WHERE df > ?", (totals["articles"] * MAX_DF_RATIO,)))
            totals["train_tokens"] = prepare_training(candidate, training, lookup)
            print(f"Training fastText: {len(lookup):,} terms, {totals['train_tokens']:,} tokens", flush=True)
            model = train_fasttext(training, model_path, totals["train_tokens"],
                                           timeout=timeout, threads=threads)
            training.unlink()
            avglen = totals["tokens"] / totals["articles"]
            try:
                from scripts.bekko_features import BekkoEncoder, embedding_input
            except ModuleNotFoundError as exc:
                if exc.name != "scripts":
                    raise
                from bekko_features import BekkoEncoder, embedding_input
            encoder = BekkoEncoder()  # Load exactly once per language.
            try:
                from scripts.article_parquet import ArticleParquet, CorpusProfile
            except ModuleNotFoundError as exc:
                if exc.name != "scripts":
                    raise
                from article_parquet import ArticleParquet, CorpusProfile
            profile = CorpusProfile()
            article_path = stage / "articles.parquet"
            parquet = ArticleParquet(article_path, ARTICLE_FIELDS)
            try:
                cur = article_rows(db, text_path)
                while chunk := list(islice(cur, BATCH_SIZE)):
                    inputs = [embedding_input(row["title"], row["text"]) for row in chunk]
                    embeddings, lengths = encoder.encode(inputs)
                    if len(embeddings) != len(chunk) or len(lengths) != len(chunk):
                        raise RuntimeError("Bekko did not return one vector per article")
                    for row, bekko_vector, input_tokens in zip(chunk, embeddings, lengths):
                        result = metrics_for_article(row, lookup, stopwords,
                                                     totals["articles"], avglen,
                                                     model, bekko_vector, input_tokens, normalizer)
                        if set(result) != set(ARTICLE_FIELDS):
                            raise RuntimeError("Article-feature schema mismatch")
                        parquet.add(row["text"], result)
                        profile.add(result)
                    if parquet.count % 1024 == 0 or parquet.count == totals["articles"]:
                        print(f"Embedded {parquet.count}/{totals['articles']} articles", flush=True)
            finally:
                parquet.close()
            if parquet.count != totals["articles"]:
                raise RuntimeError("Parquet article row-count mismatch")
            profile_path = stage / "corpus-profile.json"
            profile_path.write_text(json.dumps(profile.export(), ensure_ascii=False,
                                               separators=(",", ":")) + "\n", encoding="utf-8")
            # Full corpus TF/DF/IDF, streamed without a vocabulary cap.
            word_files = []
            for table, name in (("terms", "words.jsonl.gz"), ("lexical_terms", "normalized-words.jsonl.gz")):
                path = stage / name
                with gzip.open(path, "wt", encoding="utf-8") as out:
                    for term, tf, df in db.execute(f"SELECT term,tf,df FROM {table} ORDER BY term"):
                        out.write(compact(dict(term=term, tf=tf, df=df, idf=idf(totals["articles"], df))) + "\n")
                word_files.append(path)
            db.close()
            words_path = stage / "vocabulary.json.gz"
            with gzip.open(words_path, "wt", encoding="utf-8", compresslevel=6) as out:
                json.dump({"version": VERSION, "language": code, "documents": totals["articles"],
                           "vocab_size": len(entries),
                           "average_document_tokens": avglen,
                           "normalization": normalizer.config(), "terms": entries},
                          out, ensure_ascii=False, separators=(",", ":"))
            artifacts = [article_path, text_path, profile_path, words_path, *word_files]
            model = None
            compressed_model = stage / "fasttext.bin.zst"
            compress(model_path, compressed_model)
            artifacts.append(compressed_model)
            report = {
                "version": VERSION, "language": code,
                "documents": totals["articles"], "raw_tokens": totals["tokens"],
                "average_document_tokens": avglen,
                "raw_vocabulary": totals["raw_vocabulary"],
                "vocabulary_size": len(entries),
                "fasttext": {"available": model_path.exists(),
                             "dim": FASTTEXT_DIMS, "algorithm": "cbow",
                             "epoch": 1, "subword_bucket": 30000,
                             "training_tokens": totals["train_tokens"],
                             "training": "every article, restricted to the shared top-10000 normalized vocabulary",
                             "timeout_seconds": timeout or None},
                "bekko": {"model": MODEL_ID, "revision": MODEL_REVISION,
                          "backend": "openvino", "device": "cpu", "dimensions": BEKKO_DIMS,
                          "max_input_tokens": MAX_TOKENS, "batch_size": BATCH_SIZE,
                          "input": "original case-preserving title and bounded lead excerpt",
                          "required": True},
                "primary_table": "articles.parquet",
                "corpus_profile": "corpus-profile.json",
                "lexical": {**normalizer.config(), "vocabulary_limit": vocab_size,
                            "idf": "ln((N+1)/(df+1))+1", "bm25": {"k1": K1, "b": B},
                            "word_counts": "raw article counts; raw and normalized corpus TF/DF/IDF exported"},
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
    p.add_argument("--fasttext-timeout", type=int, default=0)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    if args.sources and args.sources[0].name == "_train":
        train_fasttext_worker(Path(args.sources[1]), Path(args.sources[2]),
                              int(str(args.sources[3])), int(str(args.sources[4])))
        return
    if not args.sources:
        p.error("At least one source .txt.gz required")
    report = build(args.sources, args.output, args.language, vocab_size=args.vocab,
                   timeout=args.fasttext_timeout,
                   threads=args.threads)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
