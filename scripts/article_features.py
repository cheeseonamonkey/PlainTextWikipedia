#!/usr/bin/env python3
"""Per-language, corpus-wide lexical + fastText article features (v8).

Stream each source once into a temporary corpus index plus one combined plaintext
spill file, fit a shared stemmed vocabulary, train a lightweight fastText model,
and emit one enriched ``articles.parquet`` plus ``corpus-profile.json``.

The work is three stages over one explicit intermediate directory, so a stage can
run in its own process (and its own CI job) with its own time budget. In order:
``stage_corpus_stats`` reads the sources and writes ``work/corpus-stats``;
``stage_fasttext_training`` reads that and writes ``work/fasttext-training``;
``stage_enrich_parquet`` reads both and writes the release. ``build()`` runs all
three in one process. Each stage reads only its declared inputs and writes only
its declared outputs, so a failed run resumes at the stage that failed instead
of restarting the whole build.

Outputs only the release artifacts (plus a small manifest). No SQLite, no zip
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
    from scripts.article_parquet import (
        ARTICLE_FIELDS,
        GROUPED_KEYS,
        VERSION,
        ArticleParquet,
        CorpusProfile,
    )
    from scripts.lexical_features import MAX_DF_RATIO, LexicalNormalizer, tokens
    from scripts.repack_shards import records
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from article_parquet import (
        ARTICLE_FIELDS,
        GROUPED_KEYS,
        VERSION,
        ArticleParquet,
        CorpusProfile,
    )
    from lexical_features import MAX_DF_RATIO, LexicalNormalizer, tokens
    from repack_shards import records

# fastText is vendored as a prebuilt extension (fasttext_pybind.cpython-312-*
# .so) plus its thin Python package under vendor/fasttext/, so build/runtime
# never depend on pip resolving fasttext-wheel -> pybind11.
_VENDOR_FASTTEXT = Path(__file__).resolve().parents[1] / "vendor" / "fasttext"
if _VENDOR_FASTTEXT.is_dir() and str(_VENDOR_FASTTEXT) not in sys.path:
    sys.path.insert(0, str(_VENDOR_FASTTEXT))

VOCAB_SIZE = 12_000
FASTTEXT_DIMS = 180
FASTTEXT_WINDOW = 8
FASTTEXT_NEG = 8
FASTTEXT_LR = 0.05
FASTTEXT_EPOCH = 1
KEYWORD_COUNT = 15
SENTENCES = re.compile(r"(?<=[.!?。！？])\s+|(?<=[。！？])")

# --- Split stages ----------------------------------------------------------
# The build runs as three stages over one explicit intermediate directory, so a
# stage can run in its own process (and its own CI job) with its own time
# budget. Each stage reads only its declared inputs and writes only its declared
# outputs:
#
#   corpus-stats/       index.sqlite + articles-spill.txt.xz +
#                       lexical-training.txt.gz + stats.json
#   fasttext-training/  training.txt (removed after training) + model.bin +
#                       training.json
#   enrich-parquet/     articles.parquet + corpus-profile.json + manifest.json
#
# corpus-stats reads the source archives; fasttext-training reads stats.json and
# the training candidate; enrich-parquet reads the index, the spill and the
# model, and publishes the release.
CORPUS_STATS_DIR = "corpus-stats"
TRAINING_DIR = "fasttext-training"
ENRICH_DIR = "enrich-parquet"
INDEX_NAME = "index.sqlite"
SPILL_NAME = "articles-spill.txt.xz"
CANDIDATE_NAME = "lexical-training.txt.gz"
STATS_NAME = "stats.json"
TRAINING_NAME = "training.txt"
MODEL_NAME = "model.bin"
TRAINING_META_NAME = "training.json"
STAGES = (CORPUS_STATS_DIR, TRAINING_DIR, ENRICH_DIR)


def stage_file(work: Path, stage: str, name: str) -> Path:
    return Path(work) / stage / name


def read_stage_json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"Missing stage artifact: {path}; run the earlier stage first")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Malformed stage artifact: {path}")
    return data


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


def connect_database(path: Path):
    """Open an existing index for reading (no schema creation)."""
    if not Path(path).is_file():
        raise ValueError(f"Missing stage artifact: {path}; run the corpus-stats stage first")
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    return db


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


def vocabulary(db, documents, limit, df_cap: float | None = MAX_DF_RATIO):
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
                   timeout: int = 0, load_model: bool = True):
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
    if not load_model:
        # The training stage runs in its own process; the embedding is loaded by
        # the enrich stage instead of being paid for twice.
        return None
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


def validate_request(files: list[Path], code: str, *,
                     vocab_size: int = VOCAB_SIZE,
                     dim: int = FASTTEXT_DIMS, window: int = FASTTEXT_WINDOW,
                     epoch: int = FASTTEXT_EPOCH,
                     timeout: int = 0, threads: int = 4) -> None:
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


def ensure_capacity(files: list[Path], *folders: Path) -> None:
    """Fail before the first byte is written when the disk cannot hold the build."""
    required = sum(p.stat().st_size for p in files) * 1.2
    for folder in folders:
        available = shutil.disk_usage(folder).free
        if available < required:
            raise RuntimeError(f"Insufficient disk for the build: need ~{required / 1e9:.1f} GB "
                               f"free at {folder}, found {available / 1e9:.1f} GB")


def stage_corpus_stats(files: list[Path], work: Path, code: str, *,
                       vocab_size: int = VOCAB_SIZE) -> dict:
    """Stage 1: one streaming pass over the sources.

    Writes the article index, the combined plaintext spill used later to recover
    bodies, the vocabulary-filtered fastText training candidate, and stats.json.
    """
    work = Path(work)
    stage = work / CORPUS_STATS_DIR
    stage.mkdir(parents=True, exist_ok=True)
    database = stage_file(work, CORPUS_STATS_DIR, INDEX_NAME)
    candidate = stage_file(work, CORPUS_STATS_DIR, CANDIDATE_NAME)
    text_path = stage_file(work, CORPUS_STATS_DIR, SPILL_NAME)
    normalizer = LexicalNormalizer(code)
    db = open_database(database)
    try:
        totals = ingest(sorted(files), db, candidate, code, normalizer, text_output=text_path)
        print(f"Indexed {totals['articles']:,} articles; fitting shared vocabulary", flush=True)
        capped = True
        entries = vocabulary(db, totals["articles"], vocab_size)
        if not entries:
            # Tiny corpora (few, similar articles) can place every eligible term
            # above the DF cap. Keep such editions buildable by lifting the cap
            # rather than failing; the vocabulary remains honest.
            print("::warning::Vocabulary empty under the DF cap; refitting without it", flush=True)
            entries = vocabulary(db, totals["articles"], vocab_size, df_cap=None)
            capped = False
        extra_stopwords = sorted(term for (term,) in db.execute(
            "SELECT term FROM terms WHERE df > ?", (totals["articles"] * MAX_DF_RATIO,)))
    finally:
        db.close()
    stats = {
        "version": VERSION, "language": code,
        "documents": totals["articles"], "raw_tokens": totals["tokens"],
        "average_document_tokens": totals["tokens"] / totals["articles"],
        "raw_vocabulary": totals["raw_vocabulary"],
        "vocabulary_size": len(entries), "vocabulary": entries,
        "extra_stopwords": extra_stopwords, "df_cap_applied": capped,
        "lexical": {**normalizer.config(), "vocabulary_limit": vocab_size,
                    "idf": "ln((N+1)/(df+1))+1",
                    "tfidf": "per-article L2-normalized stemmed TF-IDF",
                    "keywords": KEYWORD_COUNT,
                    "word_counts": "raw article counts"},
    }
    stage_file(work, CORPUS_STATS_DIR, STATS_NAME).write_text(
        json.dumps(stats, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n",
        encoding="utf-8")
    return stats


def stage_fasttext_training(work: Path, *,
                            dim: int = FASTTEXT_DIMS, window: int = FASTTEXT_WINDOW,
                            epoch: int = FASTTEXT_EPOCH, threads: int = 4,
                            timeout: int = 0) -> dict:
    """Stage 2: restrict the training text to the shared vocabulary and train."""
    work = Path(work)
    stats = read_stage_json(stage_file(work, CORPUS_STATS_DIR, STATS_NAME))
    lookup = {entry["term"]: entry for entry in stats["vocabulary"]}
    candidate = stage_file(work, CORPUS_STATS_DIR, CANDIDATE_NAME)
    if not candidate.is_file():
        raise ValueError(f"Missing stage artifact: {candidate}; run the corpus-stats stage first")
    stage = work / TRAINING_DIR
    stage.mkdir(parents=True, exist_ok=True)
    training = stage_file(work, TRAINING_DIR, TRAINING_NAME)
    tokens = prepare_training(candidate, training, lookup)
    print(f"Training fastText: {len(lookup):,} terms, {tokens:,} tokens", flush=True)
    model_path = stage_file(work, TRAINING_DIR, MODEL_NAME)
    train_fasttext(training, model_path, tokens, dim=dim, window=window,
                   epoch=epoch, timeout=timeout, threads=threads, load_model=False)
    training.unlink(missing_ok=True)
    meta = {"dim": dim, "algorithm": "cbow", "epoch": epoch, "window": window,
            "negative": FASTTEXT_NEG, "lr": FASTTEXT_LR, "subword_bucket": 0,
            "training_tokens": tokens,
            "training": "every article, restricted to the shared top-N stemmed vocabulary",
            "timeout_seconds": timeout or None}
    stage_file(work, TRAINING_DIR, TRAINING_META_NAME).write_text(
        json.dumps(meta, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    return meta


def stage_enrich_parquet(work: Path, output: Path, code: str) -> dict:
    """Stage 3: embed every article, then write and publish the release bundle."""
    work, output = Path(work), Path(output)
    stats = read_stage_json(stage_file(work, CORPUS_STATS_DIR, STATS_NAME))
    meta = read_stage_json(stage_file(work, TRAINING_DIR, TRAINING_META_NAME))
    database = stage_file(work, CORPUS_STATS_DIR, INDEX_NAME)
    text_path = stage_file(work, CORPUS_STATS_DIR, SPILL_NAME)
    model_path = stage_file(work, TRAINING_DIR, MODEL_NAME)
    if not model_path.is_file():
        raise ValueError(f"Missing stage artifact: {model_path}; run the fasttext-training stage first")
    documents = stats["documents"]
    dim = meta["dim"]
    lookup = {entry["term"]: entry for entry in stats["vocabulary"]}
    normalizer = LexicalNormalizer(code)
    stopwords = set(normalizer.stopwords).union(stats["extra_stopwords"])
    import fasttext
    model = fasttext.load_model(str(model_path))
    stage = work / ENRICH_DIR
    stage.mkdir(parents=True, exist_ok=True)
    article_path = stage / "articles.parquet"
    profile = CorpusProfile()
    db = connect_database(database)
    parquet = ArticleParquet(article_path, ARTICLE_FIELDS)
    try:
        for row in article_rows(db, text_path):
            result = metrics_for_article(row, lookup, stopwords, model, normalizer, dim=dim)
            if set(result) != set(ARTICLE_FIELDS):
                raise RuntimeError("Article-feature schema mismatch")
            parquet.add(row["text"], result)
            profile.add(flat_metrics(result))
            if parquet.count % 1024 == 0 or parquet.count == documents:
                print(f"Embedded {parquet.count}/{documents} articles", flush=True)
    finally:
        parquet.close()
        db.close()
    if parquet.count != documents:
        raise RuntimeError("Parquet article row-count mismatch")
    profile_path = stage / "corpus-profile.json"
    profile_data = profile.export()
    profile_data["vocabulary"] = stats["vocabulary"]
    profile_path.write_text(json.dumps(profile_data, ensure_ascii=False, allow_nan=False,
                                       separators=(",", ":")) + "\n", encoding="utf-8")
    report = {
        "version": VERSION, "language": code,
        "documents": documents, "raw_tokens": stats["raw_tokens"],
        "average_document_tokens": stats["average_document_tokens"],
        "raw_vocabulary": stats["raw_vocabulary"],
        "vocabulary_size": stats["vocabulary_size"],
        "fasttext": meta,
        "primary_table": "articles.parquet",
        "corpus_profile": "corpus-profile.json",
        "lexical": stats["lexical"],
        "files": {},
    }
    for name, file in (("articles.parquet", article_path),
                       ("corpus-profile.json", profile_path)):
        report["files"][name] = {"bytes": file.stat().st_size, "sha256": sha256(file)}
    manifest = stage / "manifest.json"
    manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    publish(stage, output)
    return report


def build(files: list[Path], output: Path, code: str, *,
          vocab_size: int = VOCAB_SIZE,
          dim: int = FASTTEXT_DIMS, window: int = FASTTEXT_WINDOW,
          epoch: int = FASTTEXT_EPOCH,
          timeout: int = 0, threads: int = 4) -> dict:
    """Run all three stages into one explicit work directory (single process)."""
    validate_request(files, code, vocab_size=vocab_size, dim=dim, window=window,
                     epoch=epoch, threads=threads, timeout=timeout)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ensure_capacity(files, output)
    with tempfile.TemporaryDirectory(prefix=".features-", dir=output) as work:
        workdir = Path(work)
        stage_corpus_stats(files, workdir, code, vocab_size=vocab_size)
        stage_fasttext_training(workdir, dim=dim, window=window, epoch=epoch,
                                threads=threads, timeout=timeout)
        return stage_enrich_parquet(workdir, output, code)


def publish(stage: Path, output: Path):
    """Roll back replacements on failure; manifest is committed last.

    Readers should verify manifest checksums: individual file replacements do not
    provide an atomic snapshot to concurrent readers or across process crashes.
    """
    names = ("articles.parquet", "corpus-profile.json", "manifest.json")
    output.mkdir(parents=True, exist_ok=True)
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
    p.add_argument("--stage", choices=STAGES,
                   help="run one stage into --work instead of the whole build")
    p.add_argument("--work", type=Path, help="shared intermediate artifact directory")
    p.add_argument("--release", type=Path, help="release directory written by the enrich stage")
    args = p.parse_args()
    if args.sources and args.sources[0].name == "_train":
        train_fasttext_worker(Path(args.sources[1]), Path(args.sources[2]),
                              int(str(args.sources[3])), int(str(args.sources[4])),
                              int(str(args.sources[5])), int(str(args.sources[6])))
        return
    if args.stage:
        if not args.work:
            p.error("--stage requires --work")
        if args.stage == CORPUS_STATS_DIR:
            if not args.sources:
                p.error("The corpus-stats stage requires at least one source .txt.gz")
            stats = stage_corpus_stats(args.sources, args.work, args.language,
                                       vocab_size=args.vocab)
            print(json.dumps({k: v for k, v in stats.items() if k != "vocabulary"},
                             ensure_ascii=False, indent=2))
        elif args.stage == TRAINING_DIR:
            meta = stage_fasttext_training(args.work, dim=args.dim, window=args.window,
                                           epoch=args.epoch, threads=args.threads,
                                           timeout=args.fasttext_timeout)
            print(json.dumps(meta, ensure_ascii=False, indent=2))
        else:
            if not args.release:
                p.error("The enrich-parquet stage requires --release")
            report = stage_enrich_parquet(args.work, args.release, args.language)
            print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if not args.sources:
        p.error("At least one source .txt.gz required")
    report = build(args.sources, args.output, args.language, vocab_size=args.vocab,
                   dim=args.dim, window=args.window, epoch=args.epoch,
                   timeout=args.fasttext_timeout, threads=args.threads)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
