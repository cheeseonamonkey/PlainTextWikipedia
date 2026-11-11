#!/usr/bin/env python3
"""Per-language, corpus-wide lexical + fastText article features (v8).

Stream each source once into a temporary corpus index plus one combined plaintext
spill file, fit a shared stemmed vocabulary, train a lightweight fastText model,
and emit one enriched ``articles.parquet`` plus ``corpus-profile.json``.

Outputs only those two artifacts (plus a small manifest). No SQLite, no zip
archives, no text .xz re-copy, no Bekko, no BM25 are published.
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
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unicodedata
import zlib
from collections import Counter
from itertools import zip_longest
from pathlib import Path

import numpy as np

try:
    from scripts.article_parquet import ARTICLE_FIELDS, GROUPED_KEYS, VERSION
    from scripts.lexical_features import MAX_DF_RATIO, LexicalNormalizer, tokens
    from scripts.repack_shards import records
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from article_parquet import ARTICLE_FIELDS, GROUPED_KEYS, VERSION
    from lexical_features import MAX_DF_RATIO, LexicalNormalizer, tokens
    from repack_shards import records

VOCAB_SIZE = 12_000
FASTTEXT_DIMS = 180
FASTTEXT_WINDOW = 8
FASTTEXT_NEG = 8
FASTTEXT_LR = 0.05
FASTTEXT_EPOCH = 1
KEYWORD_COUNT = 15
SENTENCES = re.compile(r"(?<=[.!?。！？])\s+|(?<=[。！？])")


def compact(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def idf(docs: int, df: int) -> float:
    return math.log((docs + 1) / (df + 1)) + 1


def norm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("Expected a finite, nonempty embedding vector")
    length = np.linalg.norm(values.astype(np.float64))
    return values / length if length > 1e-10 else values


def source_articles(files: list[Path]):
    for file in files:
        for ordinal, record in enumerate(records(file), start=1):
            raw_title, raw_text, _ = record.split(b"\n", 2)
            yield file.name, ordinal, raw_title.decode("utf-8").rstrip("\r"), raw_text.decode("utf-8").rstrip("\r")


def open_database(path: Path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("PRAGMA temp_store=FILE")
    db.execute("PRAGMA cache_size=-32768")
    db.execute("""CREATE TABLE articles(
        id INTEGER PRIMARY KEY, source_file TEXT NOT NULL,
        source_ordinal INTEGER NOT NULL, title TEXT NOT NULL, article_url TEXT,
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
                source_file,source_ordinal,title,article_url,counts,
                token_count,char_count,sentence_count)
                VALUES (?,?,?,?,?,?,?,?)""", buffered)
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
        for source_file, ordinal, title, body in source_articles(files):
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
            buffered.append((source_file, ordinal, title, url, packed,
                             len(words), len(body), len(sentences)))
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
    """Read bodies from the temporary combined text instead of a second copy."""
    cursor = db.execute("SELECT * FROM articles ORDER BY id")
    for row, record in zip_longest(cursor, source_articles([text_path])):
        if row is None or record is None or row["title"] != record[2] or row["char_count"] != len(record[3]):
            raise RuntimeError("Combined text does not match the article index")
        yield {**dict(row), "text": record[3]}


def vocabulary(db, documents, limit, df_cap=MAX_DF_RATIO):
    cap = max(1, documents * df_cap) if df_cap is not None else documents + 1
    ordered = db.execute("SELECT term,tf,df FROM lexical_terms WHERE df <= ? ORDER BY tf DESC, term LIMIT ?",
                         (cap, limit))
    return [{"term": term, "tf": tf, "df": df, "idf": idf(documents, df)}
            for term, tf, df in ordered]


def prepare_training(candidate, destination, vocabulary):
    count = 0
    with gzip.open(candidate, "rt", encoding="utf-8") as source, destination.open("w", encoding="utf-8") as out:
        for line in source:
            words = [word for word in line.split() if word in vocabulary]
            out.write(" ".join(words) + "\n")
            count += len(words)
    candidate.unlink()
    return count


def train_fasttext_worker(inp: Path, model_path: Path, dim: int, window: int,
                          epoch: int, threads: int):
    import fasttext
    # CBOW with no subwords (minn/maxn=0, bucket=0): feeds only the shared
    # stemmed vocabulary, so OOV buckets are unnecessary and training is much
    # faster and smaller than the old subword model.
    model = fasttext.train_unsupervised(
        str(inp), model="cbow", dim=dim, epoch=epoch, lr=FASTTEXT_LR,
        ws=window, minCount=1, minn=0, maxn=0, neg=FASTTEXT_NEG,
        bucket=0, thread=threads, verbose=0)
    model.save_model(str(model_path))


def train_fasttext(training: Path, dest: Path, train_tokens: int, *,
                   dim: int = FASTTEXT_DIMS, window: int = FASTTEXT_WINDOW,
                   epoch: int = FASTTEXT_EPOCH, threads: int = 4,
                   timeout: int = 0):
    if train_tokens < 1:
        raise ValueError("No tokenized text is available to train fastText")
    command = [sys.executable, str(Path(__file__).resolve()), "_train",
               str(training), str(dest), str(dim), str(window), str(epoch),
               str(threads)]
    try:
        subprocess.run(command, check=True, timeout=timeout or None, capture_output=True)
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or b"").decode("utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"fastText training failed: {details}") from exc
    import fasttext
    return fasttext.load_model(str(dest))


def metrics_for_article(row, vocabulary_lookup, stopwords,
                        model, normalizer, dim: int = FASTTEXT_DIMS):
    title, url, body = row["title"], row["article_url"], row["text"]
    blob, length, chars, sentences = (row[key] for key in
        ("counts", "token_count", "char_count", "sentence_count"))
    counts = json.loads(zlib.decompress(blob))
    total = max(1, length)
    distinct = len(counts)
    entropy = -sum(n / total * math.log(n / total) for n in counts.values())
    known = {term: n for term, n in normalizer.counts(counts).items() if term in vocabulary_lookup}
    weighted = {}
    for term, n in known.items():
        weighted[term] = (1 + math.log(n)) * vocabulary_lookup[term]["idf"]
    l2 = math.sqrt(sum(v * v for v in weighted.values()))
    mass = sum(weighted.values())
    top = sorted(weighted.items(), key=lambda pair: (-pair[1], pair[0]))[:KEYWORD_COUNT]
    known_count = sum(known.values())
    mean_idf = sum(vocabulary_lookup[t]["idf"] * n for t, n in known.items()) / max(1, known_count)
    title_terms = set(normalizer.words(tokens(title)))
    model_words = list(normalizer.words(tokens(body), vocabulary_lookup))
    vector = np.zeros(dim, dtype=np.float32)
    if model_words:
        vector = norm(np.asarray(model.get_sentence_vector(" ".join(model_words)), dtype=np.float32))
    if vector.shape != (dim,):
        raise ValueError("Invalid fastText embedding dimensions")
    word_count = length
    sentence_count = sentences
    reading_time = length / 200.0
    counts_cell = {
        "word_count": word_count, "sentence_count": sentence_count,
        "distinct_terms": distinct, "byte_count": len(body.encode("utf-8")),
        "mean_sentence_tokens": length / max(1, sentences),
        "mean_token_chars": sum(len(t) * n for t, n in counts.items()) / total,
        "reading_time_minutes": reading_time,
    }
    ratios_cell = {
        "lexical_diversity": distinct / total,
        "hapax_ratio": sum(n == 1 for n in counts.values()) / max(1, distinct),
        "stopword_ratio": sum(n for t, n in counts.items() if t in stopwords) / total,
        "digit_ratio": sum(c.isdigit() for c in body) / max(1, chars),
        "punctuation_ratio": sum(unicodedata.category(c).startswith("P") for c in body) / max(1, chars),
        "repetition_ratio": max(counts.values(), default=0) / total,
        "vocab_coverage": known_count / total,
        "title_overlap": len(title_terms & known.keys()) / max(1, len(title_terms)),
    }
    diversity_cell = {
        "entropy": entropy,
        "normalized_entropy": entropy / math.log(distinct) if distinct > 1 else 0.,
        "simpson_diversity": 1 - sum((n / total) ** 2 for n in counts.values()) if length else 0.,
    }
    tfidf_stats_cell = {
        "l2": l2,
        "top5_share": sum(v for _, v in top[:5]) / mass if mass else 0.,
        "entropy": -sum(v / mass * math.log(v / mass) for v in weighted.values()) if mass else 0.,
        "mean_idf": mean_idf,
        "idf_std": math.sqrt(sum(n * (vocabulary_lookup[t]["idf"] - mean_idf) ** 2
                                 for t, n in known.items()) / max(1, known_count)),
    }
    return {
        "title": title, "article_url": url, "char_count": chars,
        "counts": compact(counts_cell),
        "ratios": compact(ratios_cell),
        "diversity": compact(diversity_cell),
        "tfidf_stats": compact(tfidf_stats_cell),
        "word_counts": compact(counts),
        "tfidf": compact({t: v / l2 for t, v in weighted.items()}) if l2 else "{}",
        "keywords": compact(dict(top)),
        "fasttext": vector,
        "source": compact({"file": row["source_file"], "ordinal": row["source_ordinal"]}),
    }


def flat_metrics(metrics: dict) -> dict:
    out = {"fasttext": metrics["fasttext"]}
    for group, _keys in GROUPED_KEYS.items():
        values = json.loads(metrics[group])
        out.update({f"{group}.{key}": value for key, value in values.items()})
    out["char_count"] = metrics["char_count"]
    return out


def build(files: list[Path], output: Path, code: str, *,
          vocab_size: int = VOCAB_SIZE,
          dim: int = FASTTEXT_DIMS, window: int = FASTTEXT_WINDOW,
          epoch: int = FASTTEXT_EPOCH,
          timeout: int = 0, threads: int = 4) -> dict:
    if not files or len({p.resolve() for p in files}) != len(files):
        raise ValueError("Provide distinct source archives")
    if any(not p.is_file() or not p.name.endswith((".txt.gz", ".txt.xz")) for p in files):
        raise ValueError("Provide existing .txt.gz archives")
    if not 1 <= vocab_size <= 100_000:
        raise ValueError("Require 1 <= vocabulary size <= 100000")
    if not re.fullmatch(r"[a-z0-9_-]+", code):
        raise ValueError("Expected a Wikimedia language code")
    if threads < 1 or timeout < 0 or dim < 1 or window < 1 or epoch < 1:
        raise ValueError("Require positive dimensions, window, epoch and threads, and a nonnegative timeout")
    output.mkdir(parents=True, exist_ok=True)
    required = sum(p.stat().st_size for p in files) * 1.2
    available = shutil.disk_usage(output).free
    if available < required:
        raise RuntimeError(f"Insufficient disk for the build: need ~{required / 1e9:.1f} GB "
                           f"free at {output}, found {available / 1e9:.1f} GB")
    with tempfile.TemporaryDirectory(prefix=".features-", dir=output) as work:
        stage = Path(work)
        database = stage / "articles.sqlite"
        training = stage / "fasttext-training.txt"
        candidate = stage / "lexical-training.txt.gz"
        normalizer = LexicalNormalizer(code)
        model_path = stage / "fasttext.bin"
        db = open_database(database)
        try:
            text_path = stage / "articles-spill.txt.xz"
            totals = ingest(sorted(files), db, candidate, code, normalizer,
                            text_output=text_path)
            print(f"Indexed {totals['articles']:,} articles; fitting shared vocabulary", flush=True)
            entries = vocabulary(db, totals["articles"], vocab_size)
            if not entries:
                # Tiny corpora (few, similar articles) can place every eligible
                # term above the DF cap. Keep such editions buildable by lifting
                # the cap rather than failing; the vocabulary remains honest.
                print("::warning::Vocabulary empty under the DF cap; refitting without it", flush=True)
                entries = vocabulary(db, totals["articles"], vocab_size, df_cap=None)
            lookup = {entry["term"]: entry for entry in entries}
            stopwords = set(normalizer.stopwords).union(
                term for (term,) in db.execute(
                    "SELECT term FROM terms WHERE df > ?", (totals["articles"] * MAX_DF_RATIO,)))
            totals["train_tokens"] = prepare_training(candidate, training, lookup)
            print(f"Training fastText: {len(lookup):,} terms, {totals['train_tokens']:,} tokens", flush=True)
            model = train_fasttext(training, model_path, totals["train_tokens"],
                                   dim=dim, window=window, epoch=epoch,
                                   timeout=timeout, threads=threads)
            training.unlink()
            avglen = totals["tokens"] / totals["articles"]
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
                for row in cur:
                    result = metrics_for_article(row, lookup, stopwords,
                                                 model, normalizer, dim=dim)
                    if set(result) != set(ARTICLE_FIELDS):
                        raise RuntimeError("Article-feature schema mismatch")
                    parquet.add(row["text"], result)
                    profile.add(flat_metrics(result))
                    if parquet.count % 1024 == 0 or parquet.count == totals["articles"]:
                        print(f"Embedded {parquet.count}/{totals['articles']} articles", flush=True)
            finally:
                parquet.close()
            if parquet.count != totals["articles"]:
                raise RuntimeError("Parquet article row-count mismatch")
            profile_path = stage / "corpus-profile.json"
            profile_data = profile.export()
            profile_data["vocabulary"] = entries
            profile_path.write_text(json.dumps(profile_data, ensure_ascii=False, allow_nan=False,
                                               separators=(",", ":")) + "\n", encoding="utf-8")
            db.close()
            report = {
                "version": VERSION, "language": code,
                "documents": totals["articles"], "raw_tokens": totals["tokens"],
                "average_document_tokens": avglen,
                "raw_vocabulary": totals["raw_vocabulary"],
                "vocabulary_size": len(entries),
                "fasttext": {"dim": dim, "algorithm": "cbow", "epoch": epoch,
                             "window": window, "negative": FASTTEXT_NEG,
                             "lr": FASTTEXT_LR, "subword_bucket": 0,
                             "training_tokens": totals["train_tokens"],
                             "training": "every article, restricted to the shared top-N stemmed vocabulary",
                             "timeout_seconds": timeout or None},
                "primary_table": "articles.parquet",
                "corpus_profile": "corpus-profile.json",
                "lexical": {**normalizer.config(), "vocabulary_limit": vocab_size,
                            "idf": "ln((N+1)/(df+1))+1",
                            "tfidf": "per-article L2-normalized stemmed TF-IDF",
                            "keywords": KEYWORD_COUNT,
                            "word_counts": "raw article counts"},
                "files": {},
            }
            for name, file in (("articles.parquet", article_path),
                               ("corpus-profile.json", profile_path)):
                report["files"][name] = {"bytes": file.stat().st_size,
                                         "sha256": sha256(file)}
            manifest = stage / "manifest.json"
            manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
            publish(stage, output)
            return report
        finally:
            db.close()


def publish(stage: Path, output: Path):
    """Roll back replacements on failure; manifest is committed last.

    Readers should verify manifest checksums: individual file replacements do not
    provide an atomic snapshot to concurrent readers or across process crashes.
    """
    names = ("articles.parquet", "corpus-profile.json", "manifest.json")
    backups = stage / "backups"
    backups.mkdir()
    import shutil
    for name in names:
        target = output / name
        if target.exists():
            shutil.copy2(target, backups / name)
    replaced = []
    try:
        for name in names:
            os.replace(stage / name, output / name)
            replaced.append(name)
    except BaseException:
        for name in reversed(replaced):
            backup = backups / name
            if backup.exists():
                os.replace(backup, output / name)
            else:
                (output / name).unlink(missing_ok=True)
        raise


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sources", nargs="*", type=Path)
    p.add_argument("--output", type=Path, default=Path("article-features"))
    p.add_argument("--language", default="en")
    p.add_argument("--vocab", type=int, default=VOCAB_SIZE)
    p.add_argument("--dim", type=int, default=FASTTEXT_DIMS)
    p.add_argument("--window", type=int, default=FASTTEXT_WINDOW)
    p.add_argument("--epoch", type=int, default=FASTTEXT_EPOCH)
    p.add_argument("--fasttext-timeout", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    if args.sources and args.sources[0].name == "_train":
        train_fasttext_worker(Path(args.sources[1]), Path(args.sources[2]),
                              int(str(args.sources[3])), int(str(args.sources[4])),
                              int(str(args.sources[5])), int(str(args.sources[6])))
        return
    if not args.sources:
        p.error("At least one source .txt.gz required")
    report = build(args.sources, args.output, args.language, vocab_size=args.vocab,
                   dim=args.dim, window=args.window, epoch=args.epoch,
                   timeout=args.fasttext_timeout, threads=args.threads)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
