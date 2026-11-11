import gzip
import json
import math
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq
from scripts.article_features import build
from scripts.bundle_artifacts import verify_bundle


class LeanBuildTests(unittest.TestCase):
    def test_build_preserves_articles_and_tfidf(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / 'enwiki_01_of_01.txt.gz'
            documents = [('Stars', 'Stars shine brightly across the sky.'),
                         ('Oceans', 'Oceans cover most of the planet.'),
                         ('Trees', 'Trees grow leaves in forests.')]
            with gzip.open(src, 'wt', encoding='utf-8') as stream:
                for title, text in documents:
                    stream.write(f'{title}\n{text}\n\n')
            out = root / 'release'
            build([src], out, 'en', dim=16, threads=1)
            self.assertEqual(verify_bundle(out)['documents'], len(documents))
            rows = pq.read_table(out / 'articles.parquet').to_pylist()
            self.assertEqual([(r['title'], r['text']) for r in rows], documents)
            for row in rows:
                self.assertEqual(len(row['fasttext']), 16)
                weights = json.loads(row['tfidf'])
                if weights:
                    self.assertAlmostEqual(sum(v*v for v in weights.values()), 1.0)
                self.assertLessEqual(len(json.loads(row['keywords'])), 15)
                self.assertTrue(all(math.isfinite(x) for x in row['fasttext']))
            profile = json.loads((out / 'corpus-profile.json').read_text())
            self.assertTrue(profile['vocabulary'])
