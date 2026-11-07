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
import regex
import sqlite3
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

VERSION = 6
VOCAB_SIZE = 10_000
TFIDF_SIZE = 5_000
FASTTEXT_DIMS = 160
BEKKO_DIMS = 128
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
    "title", "article_url", "char_count", "sentence_count", "distinct_terms",
    "mean_sentence_tokens", "mean_token_chars", "lexical_diversity",
    "hapax_ratio", "stopword_ratio", "digit_ratio", "punctuation_ratio",
    "repetition_ratio", "entropy", "normalized_entropy", "simpson_diversity",
    "vocab_coverage", "tfidf_coverage", "mean_idf", "idf_std",
    "tfidf_l2", "tfidf_top5_share", "tfidf_entropy", "title_overlap",
    "word_counts", "tfidf", "bm25", "keywords", "fasttext", "bekko",
    "bekko_input_tokens", "bekko_truncated",
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
    words.update(fold(term) for term in (stopwords(iso) or ()))
    return frozenset(words)


def idf(docs: int, df: int) -> float:
    return math.log((docs + 1) / (df + 1)) + 1


def bm_idf(docs: int, df: int) -> float:
    return math.log1p((docs - df + 0.5) / (df + 0.5))


def norm(values: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(values)
    return values / length if length > 1e-10 else values


def source_articles(files: list[Path]):
    for file in files:
        for ordinal, record in enumerate(records(file), 1):
            raw_title, raw_text, _ = record.split(b"\n", 2)
            yield file.name, ordinal, raw_title.decode("utf-8").rstrip("\r"), raw_text.decode("utf-8").rstrip("\r")


def open_database(path: Path):
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
    return db


def ingest(files: list[Path], db: sqlite3.Connection, training: Path,
           code: str,
           text_output: Path | None = None) -> dict:
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
    text_stream = lzma.open(text_output, "wt", encoding="utf-8", newline="\n", preset=3) if text_output else None
    try:
        for source, ordinal, title, body in source_articles(files):
            if text_stream:
                text_stream.write(f"{title}\n{body}\n\n")
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
            if words:
                train_stream.write(" ".join(words) + "\n")
                training_tokens += len(words)
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
        if text_stream:
            text_stream.close()


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
                                       minCount=1, minn=2, maxn=4,
                                       neg=4, bucket=30000, thread=threads,
                                       verbose=0)
    model.save_model(str(model_path))


def train_fasttext(training: Path, dest: Path, train_tokens: int, *,
                   threads: int = 2, timeout: int = 0,
                   allow_lexical_only: bool = False):
    if train_tokens < 1:
        if allow_lexical_only:
            return None, "insufficient_training_tokens"
        raise ValueError("Insufficient fastText training data; use --allow-lexical-only explicitly")
    command = [sys.executable, str(Path(__file__).resolve()), "_train",
               str(training), str(dest), str(FASTTEXT_DIMS), str(threads)]
    try:
        subprocess.run(command, check=True, timeout=timeout or None, capture_output=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        if not allow_lexical_only:
            raise RuntimeError(f"fastText failed; install fasttext-wheel or use --allow-lexical-only: {exc}") from exc
        return None, type(exc).__name__
    import fasttext
    return fasttext.load_model(str(dest)), None


def metrics_for_article(row, vocabulary_lookup, stopwords, docs, avglen,
                        model, bekko_embedding, bekko_tokens):
    _, _, _, title, url, body, blob, length, chars, _, sentences = row
    counts = json.loads(zlib.decompress(blob))
    total = max(1, length)
    distinct = len(counts)
    entropy = -sum(n / total * math.log(n / total) for n in counts.values())
    known = {term: n for term, n in counts.items() if term in vocabulary_lookup}
    weighted, bm = {}, {}
    for term, n in known.items():
        _, df, value_idf, enabled = vocabulary_lookup[term]
        if enabled:
            weighted[term] = (1 + math.log(n)) * value_idf
            bm[term] = bm_idf(docs, df) * n * (K1 + 1) / (
                n + K1 * (1 - B + B * length / max(1, avglen)))
    l2 = math.sqrt(sum(v * v for v in weighted.values()))
    mass = sum(weighted.values())
    top = sorted(weighted.items(), key=lambda pair: (-pair[1], pair[0]))[:16]
    known_count = sum(known.values())
    mean_idf = sum(vocabulary_lookup[t][2] * n for t, n in known.items()) / max(1, known_count)
    title_terms = set(tokens(title))
    # Include all article terms, including terms outside the TF-IDF shortlist.
    vector = np.zeros(FASTTEXT_DIMS, dtype=np.float32)
    if model is not None and length:
        vector = norm(np.asarray(model.get_sentence_vector(" ".join(tokens(body))), dtype=np.float32))
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
        tfidf_coverage=sum(counts[t] for t in weighted) / total,
        mean_idf=mean_idf,
        idf_std=math.sqrt(sum(n * (vocabulary_lookup[t][2] - mean_idf) ** 2
                               for t, n in known.items()) / max(1, known_count)),
        tfidf_l2=l2, tfidf_top5_share=sum(v for _, v in top[:5]) / mass if mass else 0.,
        tfidf_entropy=-sum(v / mass * math.log(v / mass) for v in weighted.values()) if mass else 0.,
        title_overlap=len(title_terms & known.keys()) / max(1, len(title_terms)),
        word_counts=compact(counts),
        tfidf=compact({t: v / l2 for t, v in weighted.items()}) if l2 else "{}",
        bm25=compact(bm), keywords=compact(dict(top)),
        fasttext=compact(vector.tolist()) if model is not None else None,
        bekko=compact(norm(bekko_embedding).tolist()),
        bekko_input_tokens=bekko_tokens, bekko_truncated=int(bekko_tokens > 256))


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
          vocab_size: int = VOCAB_SIZE, shortlist: int = TFIDF_SIZE,
          timeout: int = 0, threads: int = 2,
          bekko: bool = True) -> dict:
    if not files or any(not p.is_file() or not p.name.endswith((".txt.gz", ".txt.xz")) for p in files):
        raise ValueError("Provide existing .txt.gz archives")
    if not 1 <= shortlist <= vocab_size <= 50_000:
        raise ValueError("Require 1 <= tfidf size <= vocab size <= 50000")
    if not re.fullmatch(r"[a-z0-9_-]+", code):
        raise ValueError("Expected a Wikimedia language code")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".features-", dir=output) as work:
        stage = Path(work)
        database = stage / "articles.sqlite"
        training = stage / "fasttext-training.txt"
        model_path = stage / "fasttext.bin"
        db = open_database(database)
        try:
            text_path = stage / "articles.txt.xz"
            totals = ingest(sorted(files), db, training, code,
                            text_output=text_path)
            entries = vocabulary(db, totals["articles"], code, vocab_size, shortlist)
            lookup = {term: (index, df, value_idf, enabled)
                      for index, term, _, df, value_idf, enabled in entries}
            stopwords = set(language_stopwords(code)).union(
                term for (term,) in db.execute(
                    "SELECT term FROM terms WHERE df > ?", (totals["articles"] * .80,)))
            model, reason = train_fasttext(training, model_path, totals["train_tokens"],
                                           timeout=timeout, threads=threads,
                                           allow_lexical_only=False)
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
            try:
                from scripts.article_parquet import ArticleParquet, CorpusProfile
            except ModuleNotFoundError as exc:
                if exc.name != "scripts":
                    raise
                from article_parquet import ArticleParquet, CorpusProfile
            profile = CorpusProfile()
            article_path = stage / "articles.parquet"
            parquet = ArticleParquet(article_path, CSV_FIELDS)
            try:
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
                                                     model, bekko_vector, input_tokens)
                        if set(result) != set(CSV_FIELDS):
                            raise RuntimeError("Article-feature schema mismatch")
                        parquet.add(row[5], result)
                        profile.add(result)
                    print(f"Embedded {parquet.count}/{totals['articles']} articles", flush=True)
            finally:
                parquet.close()
            if parquet.count != totals["articles"]:
                raise RuntimeError("Parquet article row-count mismatch")
            profile_path = stage / "corpus-profile.json"
            profile_path.write_text(json.dumps(profile.export(), ensure_ascii=False,
                                               separators=(",", ":")) + "\n", encoding="utf-8")
            # Full corpus TF/DF/IDF, streamed without a vocabulary cap.
            full_words = stage / "words.jsonl.gz"
            with gzip.open(full_words, "wt", encoding="utf-8") as out:
                for term, tf, df in db.execute("SELECT term,tf,df FROM terms ORDER BY term"):
                    out.write(compact(dict(term=term, tf=tf, df=df,
                                           idf=idf(totals["articles"], df))) + "\n")
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
            artifacts = [article_path, text_path, profile_path, words_path, full_words]
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
                "fasttext": {"available": model_path.exists(), "reason": reason,
                             "dim": FASTTEXT_DIMS, "algorithm": "cbow",
                             "epoch": 1, "subword_bucket": 30000,
                             "training_tokens": totals["train_tokens"],
                             "training": "all tokens in every article",
                             "timeout_seconds": timeout or None},
                "bekko": {"model": "hotchpotch/bekko-embedding-v1-a8m",
                          "backend": "openvino", "device": "cpu", "dimensions": 128,
                          "max_input_tokens": 256, "batch_size": 16,
                          "input": "NFKC+casefold title and bounded lead excerpt",
                          "required": True},
                "primary_table": "articles.parquet",
                "corpus_profile": "corpus-profile.json",
                "lexical": {"normalization": "NFKC+casefold+common_quote_dash_fold",
                            "max_df_ratio": 0.8,
                            "stopwords": "stopwordsiso (if installed) + English defaults + high-df filtering",
                            "idf": "ln((N+1)/(df+1))+1",
                            "bm15": {"k1": K1, "b": B},

                            "word_counts": "full counts per article; full corpus TF/DF/IDF in words.jsonl.gz"},
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
    p.add_argument("--fasttext-timeout", type=int, default=0)
    p.add_argument("--threads", type=int, default=2)
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
                   shortlist=args.tfidf, timeout=args.fasttext_timeout,
                   threads=args.threads, bekko=not args.no_bekko)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
