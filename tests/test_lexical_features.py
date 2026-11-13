"""Real multilingual normalization and consistent corpus vocabulary/training."""
import gzip
import json
import tempfile
import unittest
import zlib
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

    def test_all_text_is_unicode_normalized_before_indexing(self):
        normalizer = LexicalNormalizer('en')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source.txt.gz'
            title = 'Caf\u00e9\u2014Strau\u00dfe'
            body = 'The caf\u00e9\u2019s na\u00efve façade uses a non-breaking\u00a0space.'
            with gzip.open(source, 'wt', encoding='utf-8') as out:
                out.write(f'{title}\n{body}\n\n')
            db = open_database(root / 'index.sqlite')
            self.addCleanup(db.close)
            candidate, training = root / 'candidate.gz', root / 'training.txt'
            ingest([source], db, candidate, 'en', normalizer, root / 'articles.txt.xz')
            row = db.execute('SELECT title,counts,token_count,char_count FROM articles').fetchone()
            # Titles are stored verbatim: NFKC keeps 'ß' intact (it does not expand
            # it to 'ss'; that is casefold, which is only applied to tokens).
            self.assertEqual(row['title'], 'Café—Strauße')
            counts = json.loads(zlib.decompress(row['counts']))
            # The tokenizer keeps intra-word apostrophes, so "café's" is one token.
            self.assertIn("café's", counts)
            self.assertIn('naïve', counts)
            self.assertIn('façade', counts)
            self.assertEqual(row['char_count'], len('The café’s naïve façade uses a non-breaking space.'))
            # The token pipeline is where 'ß' folds to 'ss' (casefold); stem() itself
            # never folds, it only reduces an already-folded token.
            self.assertEqual(tokens('straße'), ['strasse'])
            self.assertEqual(normalizer.stem(tokens('straße')[0]), 'strass')

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

