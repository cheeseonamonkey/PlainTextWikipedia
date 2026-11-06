"""Bounded CPU Bekko encoder for every article; no per-document network calls.

The 128D Matryoshka output is retained independently, and all 128 dimensions
contribute to the combined 200D per-language article representation.
"""
from __future__ import annotations

import numpy as np

MODEL_ID = "hotchpotch/bekko-embedding-v1-a8m"
MODEL_DIMS = 128
MAX_TOKENS = 256
BATCH_SIZE = 16


def embedding_input(title: str, body: str, sentence: str = "") -> str:
    """Deterministic bounded excerpt. Exact normalization matches word lists."""
    try:
        from scripts.article_features import fold
    except ModuleNotFoundError as exc:
        if exc.name != "scripts":
            raise
        from article_features import fold
    lead = body[:1100]
    # A separate salient sentence helps topic coverage when it is not the lead.
    relevant = sentence[:300] if sentence not in lead else ""
    return fold(" ".join(part for part in (title, lead, relevant) if part))[:1600]


class BekkoEncoder:
    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            MODEL_ID,
            backend="openvino",
            device="cpu",
            truncate_dim=MODEL_DIMS,
            model_kwargs={"file_name": "openvino/openvino_model.xml", "device": "CPU"},
        )
        self.model.max_seq_length = MAX_TOKENS

    def encode(self, texts: list[str]) -> tuple[np.ndarray, list[int]]:
        if not texts:
            return np.empty((0, MODEL_DIMS), dtype=np.float32), []
        # Count input tokens prior to truncation, preserving visibility
        # into how much of the article's deterministic excerpt was used.
        tokenizer = self.model.tokenizer
        lengths = [len(seq) for seq in tokenizer(
            texts, add_special_tokens=True, truncation=False,
            return_attention_mask=False)["input_ids"]]
        vectors = np.asarray(self.model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ), dtype=np.float32)
        if vectors.shape != (len(texts), MODEL_DIMS) or not np.isfinite(vectors).all():
            raise RuntimeError(f"Invalid Bekko vectors: {vectors.shape}")
        return vectors, lengths
