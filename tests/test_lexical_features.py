"""Real multilingual normalization and consistent corpus vocabulary/training."""
import gzip
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from scripts.article_features import ingest, open_database, vocabulary, prepare_training
from scripts.lexical_features import LexicalNormalizer, tokens


class LexicalFeaturesTests(unittest.TestCase):
    def test_dependency_stemming_and_stopwords(self):
        normalizer = LexicalNormalizer('en')
        self.assertEqual(list(normalizer.words(tokens('THE Connected connections RUNNING runs'))),
                         ['connect', 'connect'])
        self.assertEqual(normalizer.stem('running'), normalizer.stem('runs'))
        self.assertEqual(normalizer.counts(Counter({'connected':2,'connections':3})), {'connect':5})
        self.assertEqual(list(LexicalNormalizer('simple').words(tokens('CONNECTED connections'))), ['connect','connect'])
        self.assertIsNotNone(LexicalNormalizer('fr').stemmer)
        self.assertIsNotNone(LexicalNormalizer('de').stemmer)

    def test_unsupported_stemming_is_explicit_and_preserves_unicode(self):
        normalizer = LexicalNormalizer('zz_unknown')
        self.assertIsNone(normalizer.stemmer)
        self.assertEqual(list(normalizer.words(tokens('CAFÉ ΣΊΓΜΑ'))), ['café','σίγμα'])
        self.assertIn('unavailable', normalizer.config()['morphology'])

    def test_stemmed_document_frequency_and_shared_training_vocabulary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'source.txt.gz'
            with gzip.open(source,'wt') as out:
                out.write('One\nConnected connections and running runs.\n\n'
                          'Two\nOrbits galaxies.\n\n'
                          'Three\nForests woodland.\n\n')
            db = open_database(root/'index.sqlite')
            self.addCleanup(db.close)
            candidate, training = root/'candidate.gz', root/'training.txt'
            ingest([source], db, candidate, 'en', LexicalNormalizer('en'), root/'articles.txt.xz')
            self.assertEqual(tuple(db.execute("SELECT tf,df FROM lexical_terms WHERE term='connect'").fetchone()), (2,1))
            entries = vocabulary(db, 3, 1)
            self.assertEqual(entries[0]['term'], 'connect')
            count = prepare_training(candidate, training, {'connect'})
            self.assertEqual(count, 2)
            self.assertEqual(training.read_text().splitlines(), ['connect connect','',''])
            self.assertFalse(candidate.exists())
            self.assertEqual(db.execute('SELECT count(*) FROM articles').fetchone()[0],3)

    def test_selected_vocabulary_is_bounded_and_frequency_ranked(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = open_database(Path(tmp)/'index.sqlite')
            self.addCleanup(db.close)
            db.executemany('INSERT INTO lexical_terms VALUES(?,?,?)',
                           ((f'lexeme{i:05}', i + 1, 1) for i in range(10025)))
            selected = vocabulary(db, 100, 10000)
            self.assertEqual(len(selected), 10000)
            self.assertEqual(selected[0]['term'], 'lexeme10024')
            self.assertEqual(selected[-1]['term'], 'lexeme00025')

