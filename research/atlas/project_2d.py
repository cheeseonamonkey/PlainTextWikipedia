#!/usr/bin/env python3
"""Project a stratified SAMPLE of full-dimensional embeddings to 2D.

Source 1024-D vectors are never modified. Only atlas display samples
are reduced. Requires: pip install numpy umap-learn
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("database",type=Path)
    p.add_argument("vectors",type=Path)
    p.add_argument("output_csv",type=Path)
    p.add_argument("--per-language",type=int,default=1000)
    p.add_argument("--seed",type=int,default=20261013)
    args=p.parse_args()
    if args.per_language<1:
        p.error("--per-language must be positive")
    rng=random.Random(args.seed)
    samples:dict[str,list[tuple[int,str,np.ndarray]]]=defaultdict(list)
    seen=Counter()
    db=sqlite3.connect(f"file:{args.database.resolve()}?mode=ro",uri=True)
    vector_files=sorted(args.vectors.glob("dense-*.npz"))
    if not vector_files:
        p.error("No full-dimensional vector shards exist; run embed_chunks.py first")
    try:
        for file in vector_files:
            with np.load(file) as data:
                ids=data["ids"]
                vecs=data["vectors"]
                if vecs.shape[1]!=1024 or len(ids)!=len(vecs):
                    raise ValueError(f"Invalid vector shard: {file}")
                labels={}
                for start in range(0,len(ids),500):
                    subset=[int(x) for x in ids[start:start+500]]
                    marks=",".join("?" for _ in subset)
                    if not subset:
                        continue
                    rows=db.execute(
                        "SELECT c.id,a.language,a.title FROM chunks c "
                        "JOIN articles a ON a.id=c.article_id "
                        f"WHERE c.id IN ({marks})",subset).fetchall()
                    labels.update({i:(lang,title) for i,lang,title in rows})
                for id_value,vec in zip(ids,vecs):
                    i=int(id_value)
                    if i not in labels:
                        continue
                    lang,title=labels[i]
                    seen[lang]+=1
                    slot=samples[lang]
                    item=(i,title,vec.astype(np.float32,copy=True))
                    if len(slot)<args.per_language:
                        slot.append(item)
                    else:
                        replacement=rng.randrange(seen[lang])
                        if replacement<args.per_language:
                            slot[replacement]=item
        items=[(lang,*item) for lang,slot in sorted(samples.items()) for item in slot]
        if len(items)<4:
            raise RuntimeError("At least four embedded chunks are required for 2D projection")
        vectors=np.stack([item[3] for item in items])
        import umap
        projection=umap.UMAP(
            n_components=2,n_neighbors=min(30,len(items)-1),
            min_dist=0.08,metric="cosine",random_state=args.seed
        ).fit_transform(vectors)
        args.output_csv.parent.mkdir(parents=True,exist_ok=True)
        with args.output_csv.open("w",newline="",encoding="utf-8") as output:
            writer=csv.writer(output)
            writer.writerow(["chunk_id","language","title","x","y"])
            for (lang,i,title,_),(x,y) in zip(items,projection):
                writer.writerow([i,lang,title,round(float(x),6),round(float(y),6)])
        metadata={"source_vector_dimensions":1024,"display_dimensions":2,
                  "sampled_chunks":len(items),"sample_limit_per_language":args.per_language,
                  "seed":args.seed,"counts_by_language":dict(sorted(Counter(item[0] for item in items).items())),
                  "warning":"UMAP 2D coordinates are visualization only, not source knowledge or retrieval vectors."}
        args.output_csv.with_suffix(".manifest.json").write_text(
            json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
        print(f"Wrote {len(items):,} 2D display points to {args.output_csv}")
    finally:
        db.close()


if __name__=="__main__":
    main()
