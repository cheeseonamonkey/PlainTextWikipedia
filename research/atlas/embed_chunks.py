#!/usr/bin/env python3
"""Embed EVERY indexed chunk into its full BGE-M3 vector representation.

Requires: pip install FlagEmbedding numpy torch

Embeddings remain 1024-dimensional and are saved as FP16 shards with stable
chunk IDs. These vectors are never overwritten by the 2-D atlas projection.

Example:
  python research/atlas/embed_chunks.py atlas.sqlite atlas-vectors --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("database",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--model",default="BAAI/bge-m3")
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--shard-chunks",type=int,default=8192)
    p.add_argument("--limit",type=int,default=0,help="0 means no limit (full corpus)")
    p.add_argument("--start-id",type=int,default=0,help="Manually begin after this chunk ID")
    args=p.parse_args()
    if args.batch_size<1 or args.shard_chunks<1 or args.limit<0:
        p.error("batch-size and shard-chunks must be positive; limit must be >=0")
    args.output.mkdir(parents=True,exist_ok=True)
    state_path=args.output/"progress.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else None
    if state and state["model"]!=args.model:
        p.error("Output directory has vectors from a different model")
    if state and state.get("database")!=str(args.database.resolve()):
        p.error("Output directory has a different source database")
    last_id=max(args.start_id,state["last_chunk_id"] if state else 0)
    db=sqlite3.connect(f"file:{args.database.resolve()}?mode=ro",uri=True)
    from FlagEmbedding import BGEM3FlagModel
    model=BGEM3FlagModel(args.model,use_fp16=True)
    processed=0
    pending_ids=[]
    pending_vectors=[]
    output_number=int(state["next_shard"]) if state else 0
    checkpoint={"model":args.model,"dimensions":1024,"database":str(args.database.resolve()),
                "last_chunk_id":last_id,"next_shard":output_number}

    def flush() -> None:
        nonlocal output_number
        if not pending_ids:
            return
        vecs=np.vstack(pending_vectors).astype(np.float16,copy=False)
        if vecs.shape[1]!=1024:
            raise RuntimeError(f"Expected 1024-dimensional BGE-M3 embeddings; got {vecs.shape[1]}")
        path=args.output/f"dense-{output_number:06d}.npz"
        temporary=args.output/f"dense-{output_number:06d}.tmp.npz"
        np.savez_compressed(temporary, ids=np.array(pending_ids,dtype=np.int64),vectors=vecs)
        temporary.replace(path)
        checkpoint.update(last_chunk_id=int(pending_ids[-1]),next_shard=output_number+1)
        state_path.write_text(json.dumps(checkpoint,indent=2)+"\n")
        print(f"Wrote {len(pending_ids)} dense chunk vectors: {path}",flush=True)
        output_number+=1
        pending_ids.clear()
        pending_vectors.clear()

    try:
        while not args.limit or processed<args.limit:
            take=min(args.batch_size, args.limit-processed) if args.limit else args.batch_size
            rows=db.execute("SELECT id,text FROM chunks WHERE id>? ORDER BY id LIMIT ?",
                            (last_id,take)).fetchall()
            if not rows:
                break
            vecs=model.encode(
                [row[1] for row in rows],
                batch_size=len(rows),max_length=2048
            )["dense_vecs"]
            if len(vecs)!=len(rows):
                raise RuntimeError("Embedding count mismatch")
            pending_ids.extend(row[0] for row in rows)
            pending_vectors.append(np.asarray(vecs,dtype=np.float16))
            processed+=len(rows)
            last_id=rows[-1][0]
            if len(pending_ids)>=args.shard_chunks:
                flush()
        flush()
        print(f"Embedded {processed:,} chunks in this invocation.")
    finally:
        db.close()


if __name__=="__main__":
    main()
