"""Bounded CPU Bekko encoder for every article; no per-document network calls.

The 128D Matryoshka output is retained independently for each article.
"""
from __future__ import annotations

import numpy as np

MODEL_ID = "hotchpotch/bekko-embedding-v1-a8m"
MODEL_REVISION = "c721113d59a1d91b447450324f51c4b3332c924a"
MODEL_DIMS = 128
MAX_TOKENS = 256
BATCH_SIZE = 16


def embedding_input(title: str, body: str) -> str:
    """Keep readable, case-preserving text for the pretrained model."""
    return f"{title.strip()} {body[:1100].strip()}"[:1600]



class BekkoEncoder:
    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            MODEL_ID,
            revision=MODEL_REVISION,
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
