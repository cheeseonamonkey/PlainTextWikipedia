"""Offline tests for per-language corpus metrics, indexing and int8 vectors."""
import base64
import gzip
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import zstandard as zstd
import pyarrow.parquet as pq

from scripts.article_features import (
    DENSE_DIMS, FUSION_IMPORTANCE, build, combined_vector, dequantize,
    fold, lexical_hash, quantize, tokens,
)
from scripts.bekko_features import embedding_input


class FakeBekko:
    calls = []

    def encode(self, texts):
        self.calls.extend(texts)
        embeddings = np.zeros((len(texts), 128), dtype=np.float32)
        embeddings[:, 0] = .8
        embeddings[:, 1] = .6
        return embeddings, [min(50, len(text.split()) + 2) for text in texts]


class ArticleFeaturesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sources = [self.root / "simplewiki_01.txt.gz",
                        self.root / "simplewiki_02.txt.gz"]
        entries = [
            [("Orbit", "Planets orbit stars. Stars orbit galaxies."),
             ("Quoted", "It’s a “quote”—not just ASCII. Isn’t it neat?")],
            [("Forests", "Trees grow in forests. Many trees shelter wildlife."),
             ("日本", "日本語の文章です。星と惑星について書きます。")],
        ]
        for source, docs in zip(self.sources, entries):
            with gzip.open(source, "wt", encoding="utf-8", newline="") as f:
                for title, body in docs:
                    f.write(f"{title}\n{body}\n\n")

    def read_result(self, dest):
        with patch("scripts.bekko_features.BekkoEncoder", FakeBekko):
            report = build(self.sources, dest, "simple",
                           vocab_size=100, shortlist=20,
                           max_train_tokens=100, sample_mod=100,
                           allow_lexical_only=True)
        rows = pq.read_table(dest / "articles.parquet").to_pylist()
        with gzip.open(dest / "vocabulary.json.gz", "rt", encoding="utf-8") as f:
            vocab = json.load(f)
        original = self.root / "db.sqlite"
        with (dest / "articles.sqlite.zst").open("rb") as src, original.open("wb") as out:
            zstd.ZstdDecompressor().copy_stream(src, out)
        return report, rows, vocab, original

    def test_corpus_metrics_and_fts5(self):
        dest = self.root / "output"
        report, rows, vocab, dbpath = self.read_result(dest)
        self.assertEqual(report["documents"], 4)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["text"], "Planets orbit stars. Stars orbit galaxies.")
        self.assertEqual(len(rows[0]["embedding_200"]), 200)
        self.assertIn("planets", json.loads(rows[0]["word_counts_all_json"]))
        self.assertEqual(rows[0]["hapax_count"], 2)
        self.assertIsNone(rows[0]["contradiction_score"])
        self.assertIsNotNone(rows[0]["reading_grade"])
        self.assertIsNotNone(rows[0]["vader_compound"])
        self.assertTrue(0 <= rows[0]["sentiment_valence_01"] <= 1)
        self.assertTrue(0 <= rows[0]["technicality_proxy_01"] <= 1)
        with (dest / "corpus-profile.json").open() as stream:
            profile = json.load(stream)
        self.assertEqual(profile["documents"], 4)
        self.assertEqual(len(profile["vector_centroid"]), 200)
        self.assertIn("saliency", profile["fields"])
        self.assertEqual(report["vector"]["dimensions"], 200)
        self.assertEqual(report["bekko"]["dimensions"], 128)
        self.assertTrue(report["bekko"]["required"])
        self.assertTrue(report["lexical"]["fts5"])
        self.assertEqual(vocab["documents"], 4)
        self.assertEqual(report["vocabulary_size"], len(vocab["terms"]))
        self.assertLessEqual(report["tfidf_size"], 20)
        self.assertFalse(report["fasttext"]["available"])
        self.assertEqual([int(r["source_ordinal"]) for r in rows], [1, 2, 1, 2])
        self.assertEqual(rows[0]["article_url"],
                         "https://simple.wikipedia.org/wiki/Orbit")
        self.assertEqual(len(base64.b64decode(rows[0]["vector_q8_b64"])), 200)
        for row in rows:
            self.assertEqual(len(dequantize(row["vector_q8_b64"],
                                            row["vector_scales_b64"])), 200)
            self.assertEqual(len(dequantize(row["fasttext_q8_b64"],
                                            row["fasttext_scales_b64"])), 160)
            self.assertEqual(len(dequantize(row["bekko_q8_b64"],
                                            row["bekko_scales_b64"])), 128)
            self.assertEqual(row["bekko_available"], 1)
            self.assertIsInstance(json.loads(row["word_counts_json"]), dict)
            self.assertIsInstance(json.loads(row["tfidf_sparse_json"]), list)
            self.assertIsInstance(json.loads(row["bm15_sparse_json"]), list)
            self.assertIsInstance(json.loads(row["keywords_json"]), list)
            self.assertGreaterEqual(float(row["entropy"]), 0)
            self.assertGreaterEqual(float(row["saliency"]), 0)
            self.assertGreaterEqual(float(row["vocab_coverage"]), 0)
            self.assertLessEqual(float(row["vocab_coverage"]), 1)
            for field in ("normalized_entropy", "simpson_diversity",
                          "hapax_token_ratio", "rare_token_ratio",
                          "rare_term_ratio", "keyword_token_share",
                          "keyword_title_overlap", "title_tfidf_share",
                          "bm15_top5_share", "extractive_position",
                          "extractive_margin", "lead_keyword_overlap",
                          "vector_active_fraction"):
                self.assertGreaterEqual(float(row[field]), 0, field)
                self.assertLessEqual(float(row[field]), 1.00000001, field)
            self.assertGreaterEqual(float(row["sentence_length_cv"]), 0)
            self.assertGreaterEqual(float(row["idf_std"]), 0)
            self.assertGreaterEqual(float(row["bm15_mean"]), 0)
            self.assertGreaterEqual(float(row["effective_vocabulary"]), 1)
            self.assertLessEqual(float(row["effective_vocabulary"]),
                                 int(row["distinct_terms"]) + .00001)
            self.assertGreater(float(row["vector_quant_cosine"]), .98)
            self.assertEqual(set(json.loads(row["fusion_component_energy_json"])),
                             set(FUSION_IMPORTANCE))
        with sqlite3.connect(dbpath) as conn:
            self.assertEqual(conn.execute("PRAGMA quick_check").fetchone()[0], "ok")
            self.assertEqual(conn.execute("SELECT count(*) FROM articles").fetchone()[0], 4)
            self.assertEqual(conn.execute("SELECT count(*) FROM metrics").fetchone()[0], 4)
            self.assertGreater(conn.execute("SELECT count(*) FROM terms").fetchone()[0], 0)
            self.assertEqual(conn.execute(
                "SELECT title FROM article_fts WHERE article_fts MATCH 'planets'"
            ).fetchone()[0], "Orbit")
            self.assertIn("planets", json.loads(zlib_decompress(conn.execute(
                "SELECT counts FROM articles WHERE title='Orbit'"
            ).fetchone()[0])))
        self.assertEqual(set(report["files"]), {
            "articles.parquet", "corpus-profile.json",
            "vocabulary.json.gz", "articles.sqlite.zst",
        })

    def test_fasttext_semantic_path_with_stub(self):
        class FakeFastText:
            def get_word_vector(self, term):
                vector = np.zeros(160, dtype=np.float32)
                vector[sum(map(ord, term)) % 160] = 1.
                return vector
        def train_stub(training, dest, *_args, **_kwargs):
            dest.write_bytes(b"fake-fasttext-model")
            return FakeFastText(), None
        with (patch("scripts.article_features.train_fasttext", side_effect=train_stub),
              patch("scripts.bekko_features.BekkoEncoder", FakeBekko)):
            dest = self.root / "semantic"
            result = build(self.sources, dest, "simple", vocab_size=100, shortlist=20)
            rows = pq.read_table(dest / "articles.parquet").to_pylist()
        self.assertTrue(result["fasttext"]["available"])
        self.assertTrue((dest / "fasttext.bin.zst").exists())
        self.assertTrue(all(row["semantic_available"] == 1 for row in rows))
        combined = dequantize(rows[0]["vector_q8_b64"], rows[0]["vector_scales_b64"])
        self.assertTrue(np.any(combined[:128]))  # Full Bekko Matryoshka vector.
        self.assertTrue(np.any(combined[128:172]))  # Projected fastText.
        self.assertTrue(np.any(combined[172:188]))  # Signed TF-IDF hash.
        self.assertTrue(np.any(combined[188:]))  # Centered scalar metrics.
        actual = json.loads(rows[0]["fusion_component_energy_json"])
        for component, expected in FUSION_IMPORTANCE.items():
            self.assertAlmostEqual(actual[component], expected, places=5)

    def test_fusion_weights_match_current_policy(self):
        self.assertEqual(FUSION_IMPORTANCE, {
            "bekko": 0.46, "fasttext": 0.43, "lexical": 0.03, "metrics": 0.08,
        })

    def test_cosine_importance_is_the_requested_share(self):
        semantic = np.linspace(-1, 1, 160, dtype=np.float32)
        bekko = np.linspace(-1, 1, 128, dtype=np.float32)
        lexical = np.linspace(-1, 1, 16, dtype=np.float32)
        metrics = [.1, .2, .3, .4, .6, .7, .8, .9, .1, .2, .8, .6]
        mixed, shares = combined_vector(semantic, lexical, metrics, bekko)
        self.assertEqual(mixed.shape, (200,))
        self.assertAlmostEqual(float(np.linalg.norm(mixed)), 1., places=6)
        self.assertEqual(sum((128, 44, 16, 12)), 200)
        for name, importance in FUSION_IMPORTANCE.items():
            self.assertAlmostEqual(shares[name], importance, places=6)
        self.assertAlmostEqual(sum(shares.values()), 1., places=6)
        inverted, _ = combined_vector(semantic, lexical, metrics, -bekko)
        self.assertAlmostEqual(float(np.dot(mixed, inverted)),
                               1 - 2 * FUSION_IMPORTANCE["bekko"], places=6)
        self.assertTrue(np.array_equal(
            mixed, combined_vector(semantic, lexical, metrics, bekko)[0]))
        _, missing = combined_vector(None, lexical, metrics, bekko)
        self.assertAlmostEqual(missing["fasttext"], 0., places=6)
        self.assertAlmostEqual(sum(missing.values()), 1., places=6)

    def test_signed_lexical_hash_does_not_conflate_sign_and_bucket(self):
        from collections import defaultdict
        signs = defaultdict(set)
        for index in range(2048):
            vector = lexical_hash([(index, 1.)])
            slot = int(np.argmax(np.abs(vector)))
            signs[slot].add(int(np.sign(vector[slot])))
        self.assertEqual(len(signs), 16)
        self.assertTrue(all(values == {-1, 1} for values in signs.values()))

    def test_fasttext_worker_can_train_and_reload(self):
        import fasttext
        import subprocess
        import sys

        corpus = self.root / "tiny-train.txt"
        corpus.write_text(
            ("planets orbit stars and moons orbit planets in space\n" * 256),
            encoding="utf-8")
        output = self.root / "small-model.bin"
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parents[1] /
                                  "scripts/article_features.py"),
             "_train", str(corpus), str(output), "16", "1"],
            check=False, timeout=55, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        model = fasttext.load_model(str(output))
        self.assertEqual(model.get_dimension(), 16)
        self.assertEqual(len(model.get_word_vector("orbit")), 16)

    def test_unicode_normalization_and_int8_round_trip(self):
        self.assertEqual(fold("It’s “Nice”—WOW"), "it's \"nice\"-wow")
        self.assertIn("日本", tokens("日本語"))
        self.assertEqual(tokens("İstanbul"), ["i\u0307stanbul"])
        self.assertEqual(tokens("ΣΊΓΜΑ"), ["σίγμα"])
        self.assertEqual(tokens("ПРИВЕТ"), ["привет"])
        self.assertEqual(tokens("CAFÉ"), ["café"])
        from scripts.article_features import language_stopwords
        self.assertIsInstance(language_stopwords("zz_unknown"), frozenset)
        self.assertEqual(fold("ΣΊΓΜΑ"), "σίγμα")
        self.assertEqual(fold("СТАТЬЯ"), "статья")
        self.assertEqual(fold("İSTANBUL"), "i\u0307stanbul")
        self.assertIn("月", fold("月"))
        self.assertEqual(embedding_input("ΣΊΓΜΑ", "“HELLO” İstanbul WORLD"),
                         'σίγμα "hello" i\u0307stanbul world')
        self.assertEqual(fold("Ｆｏｏ"), "foo")
        vec = np.linspace(-1., 1., DENSE_DIMS, dtype=np.float32)
        encoded, scales = quantize(vec)
        again = dequantize(encoded, scales)
        self.assertEqual(len(again), DENSE_DIMS)
        self.assertLess(float(np.max(np.abs(vec - again))), .02)

    def test_every_article_is_embedded_not_sampled(self):
        FakeBekko.calls = []
        self.read_result(self.root / "once")
        self.assertEqual(len(FakeBekko.calls), 4)
        self.assertEqual(FakeBekko.calls[0], "orbit planets orbit stars. stars orbit galaxies.")
        self.assertTrue(all(text == fold(text) for text in FakeBekko.calls))

    def test_bad_inputs_rejected(self):
        with self.assertRaisesRegex(ValueError, "tfidf size"):
            build(self.sources, self.root / "bad", "en",
                  vocab_size=4, shortlist=5, allow_lexical_only=True)
        with self.assertRaisesRegex(ValueError, "Wikimedia"):
            build(self.sources, self.root / "bad", "../x",
                  allow_lexical_only=True)


def zlib_decompress(blob):
    import zlib
    return zlib.decompress(blob).decode("utf-8")


if __name__ == "__main__":
    unittest.main()
