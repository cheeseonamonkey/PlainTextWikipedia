#!/usr/bin/env python3
"""Lossless indexing of PlainTextWikipedia .txt.gz archives.

Full *cleaned* article text is kept; every chunk stores exact character offsets.
This does not recreate original wiki markup, already removed by the converter.

Usage:
    python research/atlas/build_index.py ingest atlas.sqlite --language frwiki data/*.txt.gz
    python research/atlas/build_index.py search atlas.sqlite "Napoleon" --limit 20
    python research/atlas/build_index.py stats atlas.sqlite
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Iterator


def chunks(text: str, target: int = 1050, overlap: int = 180) -> Iterator[tuple[int, int, str]]:
    """Yield complete, recoverable character spans. Never drop or truncate content."""
    if target < 20 or overlap < 0 or overlap >= target:
        raise ValueError("target must be >= 20 and 0 <= overlap < target")
    start = 0
    while start < len(text):
        end = min(len(text), start + target)
        if end < len(text):
            # Prefer sentence/word boundaries without abandoning the fixed
            # size safety limit; CJK text works without spaces.
            window = text[start:end]
            boundaries = [m.end() for m in re.finditer(r"[.!?。！？؟؛]\s*|\s+", window)]
            choices = [i for i in boundaries if i >= target * 0.72]
            if choices:
                end = start + choices[-1]
        yield start, end, text[start:end]
        if end == len(text):
            break
        start = max(start + 1, end - overlap)


def articles_from_gzip(path: Path) -> Iterator[tuple[str, str]]:
    """Parse converter's title, body, blank-line article format as a stream."""
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        title = None
        body: list[str] = []
        for line in stream:
            line = line.rstrip("\n\r")
            if title is None:
                if line:
                    title = line
            elif not line:
                if body:
                    yield title, "\n".join(body)
                title, body = None, []
            else:
                body.append(line)
        if title is not None and body:
            yield title, "\n".join(body)


SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
  id INTEGER PRIMARY KEY,
  language TEXT NOT NULL,
  title TEXT NOT NULL,
  text TEXT NOT NULL,
  source_archive TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  qid TEXT,
  UNIQUE(language, title)
);
CREATE INDEX IF NOT EXISTS idx_articles_lang ON articles(language);
CREATE INDEX IF NOT EXISTS idx_articles_qid ON articles(qid);
CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY,
  article_id INTEGER NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
  ordinal INTEGER NOT NULL,
  start_char INTEGER NOT NULL,
  end_char INTEGER NOT NULL,
  text TEXT NOT NULL,
  UNIQUE(article_id, ordinal)
);
CREATE INDEX IF NOT EXISTS idx_chunks_article ON chunks(article_id);
"""


def connect(path: Path, trigram: bool = True) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript(SCHEMA)
    db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks_lex USING fts5("
               "text, content='chunks', content_rowid='id', tokenize='unicode61')")
    db.execute("""CREATE TRIGGER IF NOT EXISTS chunks_lex_insert AFTER INSERT ON chunks
                  BEGIN INSERT INTO chunks_lex(rowid,text) VALUES (new.id,new.text); END""")
    db.execute("""CREATE TRIGGER IF NOT EXISTS chunks_lex_delete AFTER DELETE ON chunks
                  BEGIN INSERT INTO chunks_lex(chunks_lex,rowid,text)
                        VALUES ('delete',old.id,old.text); END""")
    if trigram:
        try:
            db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks_tri USING fts5("
                       "text, content='chunks', content_rowid='id', tokenize='trigram')")
            db.execute("""CREATE TRIGGER IF NOT EXISTS chunks_tri_insert AFTER INSERT ON chunks
                          BEGIN INSERT INTO chunks_tri(rowid,text) VALUES (new.id,new.text); END""")
            db.execute("""CREATE TRIGGER IF NOT EXISTS chunks_tri_delete AFTER DELETE ON chunks
                          BEGIN INSERT INTO chunks_tri(chunks_tri,rowid,text)
                                VALUES ('delete',old.id,old.text); END""")
        except sqlite3.OperationalError:
            print("WARNING: SQLite lacks FTS5 trigram; unicode61 index remains available.")
    db.commit()
    return db


def ingest(db: sqlite3.Connection, lang: str, files: list[Path], target: int, overlap: int) -> dict:
    if not re.fullmatch(r"[a-z_]+wiki", lang):
        raise ValueError(f"Invalid Wikimedia project: {lang}")
    stats = Counter()
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
        print(f"Indexing {lang}: {path}", flush=True)
        for title, full_text in articles_from_gzip(path):
            # Hash protects incremental rebuilds from obsolete vectors/chunks.
            checksum = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
            existing = db.execute(
                "SELECT id,sha256 FROM articles WHERE language=? AND title=?",
                (lang,title)).fetchone()
            if existing and existing[1] == checksum:
                stats["unchanged"] += 1
                continue
            if existing:
                article_id = existing[0]
                db.execute("DELETE FROM chunks WHERE article_id=?", (article_id,))
                db.execute(
                    "UPDATE articles SET text=?,source_archive=?,sha256=? WHERE id=?",
                    (full_text,path.name,checksum,article_id))
                stats["updated"] += 1
            else:
                article_id = db.execute(
                    "INSERT INTO articles(language,title,text,source_archive,sha256) "
                    "VALUES(?,?,?,?,?)", (lang,title,full_text,path.name,checksum)
                ).lastrowid
                stats["new"] += 1
            for ordinal,(start,end,chunk) in enumerate(chunks(full_text,target,overlap)):
                assert chunk == full_text[start:end]
                db.execute(
                    "INSERT INTO chunks(article_id,ordinal,start_char,end_char,text) "
                    "VALUES(?,?,?,?,?)", (article_id,ordinal,start,end,chunk))
                stats["chunks"] += 1
            if (stats["new"] + stats["updated"]) % 100 == 0:
                db.commit()
        db.commit()
    return dict(stats)


def search(db: sqlite3.Connection, query: str, limit: int = 20, trigram: bool = False) -> list[dict]:
    if not query.strip():
        return []
    index = "chunks_tri" if trigram else "chunks_lex"
    # Quoting avoids interpreting untrusted text as FTS syntax.
    expression = '"' + query.replace('"','""') + '"'
    try:
        rows = db.execute(
            f"""SELECT a.language,a.title,a.qid,c.id,c.start_char,c.end_char,
                       substr(c.text,1,400),bm25({index}) AS score
                FROM {index} JOIN chunks c ON c.id={index}.rowid
                JOIN articles a ON a.id=c.article_id
                WHERE {index} MATCH ? ORDER BY score LIMIT ?""",
            (expression,limit)).fetchall()
    except sqlite3.OperationalError as exc:
        if trigram:
            raise ValueError("Trigram FTS unavailable, or query too short (<3 characters)") from exc
        raise
    return [dict(zip(("language","title","qid","chunk_id","start","end","snippet","score"),r))
            for r in rows]


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    a=sub.add_parser("ingest",help="Index all input articles without sampling")
    a.add_argument("database",type=Path)
    a.add_argument("--language",required=True)
    a.add_argument("files",nargs="+",type=Path)
    a.add_argument("--target-chars",type=int,default=1050)
    a.add_argument("--overlap-chars",type=int,default=180)
    q=sub.add_parser("search",help="Exact-phrase lexical search across chunk text")
    q.add_argument("database",type=Path)
    q.add_argument("query")
    q.add_argument("--limit",type=int,default=20)
    q.add_argument("--trigram",action="store_true")
    b=sub.add_parser("stats")
    b.add_argument("database",type=Path)
    args=p.parse_args()
    db=connect(args.database)
    try:
        if args.command=="ingest":
            print(ingest(db,args.language,args.files,args.target_chars,args.overlap_chars))
        elif args.command=="search":
            import json
            print(json.dumps(search(db,args.query,args.limit,args.trigram),ensure_ascii=False,indent=2))
        else:
            for lang,n in db.execute("SELECT language,count(*) FROM articles GROUP BY language"):
                print(f"{lang}: {n:,} articles")
            n=db.execute("SELECT count(*) FROM chunks").fetchone()[0]
            print(f"Total indexed chunks: {n:,}")
    finally:
        db.close()


if __name__=="__main__":
    main()
