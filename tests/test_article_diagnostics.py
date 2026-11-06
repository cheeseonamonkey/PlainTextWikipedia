"""Metrics are explicit about language coverage and the limits of proxies."""
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

from scripts.article_diagnostics import Diagnostics
from scripts.article_parquet import ArticleParquet, CorpusProfile
from scripts.article_features import CSV_FIELDS, SENTENCES, tokens


class DiagnosticTests(unittest.TestCase):
    def kwargs(self):
        return dict(rare_ratio=.1, digit_ratio=.02, mean_token_chars=5.,
                    mean_idf=2.4, tokenizer=tokens, splitter=SENTENCES)

    def test_english_scores_and_noncontradiction_flag(self):
        analyzer = Diagnostics('simple')
        positive = analyzer.calculate('Wonderful lovely good news. Good news for everybody!', **self.kwargs())
        negative = analyzer.calculate('Terrible awful bad news. This is a horrible disaster.', **self.kwargs())
        self.assertGreater(positive['sentiment_valence_01'], negative['sentiment_valence_01'])
        self.assertIsNone(positive['contradiction_score'])
        self.assertIsNotNone(positive['reading_grade'])
        for key in ('sentiment_valence_01', 'sentiment_intensity_01', 'reading_difficulty_01',
                    'lexical_cohesion_01', 'technicality_proxy_01'):
            self.assertLessEqual(0, positive[key], key)
            self.assertLessEqual(positive[key], 1, key)

    def test_non_english_does_not_fabricate_vader_or_readability(self):
        result = Diagnostics('ja').calculate('星と惑星。星と銀河。', **self.kwargs())
        self.assertIsNone(result['vader_compound'])
        self.assertIsNone(result['reading_grade'])
        self.assertIsNone(result['contradiction_score'])
        self.assertEqual(result['sentiment_sample_chars'], 0)

    def test_parquet_is_one_file_even_with_many_row_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / 'articles.parquet'
            writer = ArticleParquet(file, CSV_FIELDS)
            record = {field: ('example' if field in {
                'title', 'source_file', 'representative_sentence', 'article_url',
                'word_counts_json', 'word_counts_all_json', 'tfidf_sparse_json',
                'bm15_sparse_json', 'keywords_json', 'fusion_component_energy_json',
                'vector_q8_b64', 'vector_scales_b64', 'fasttext_q8_b64',
                'fasttext_scales_b64', 'bekko_q8_b64', 'bekko_scales_b64'}
                else 1 if field in {'token_count', 'source_ordinal'} else None)
                      for field in CSV_FIELDS}
            for n in range(1030):
                writer.add(f'article {n}', record, [0.] * 200)
            writer.close()
            dataset = pq.ParquetFile(file)
            self.assertEqual(dataset.metadata.num_rows, 1030)
            self.assertEqual(dataset.metadata.num_row_groups, 2)
            self.assertEqual(len(dataset.schema_arrow), len(CSV_FIELDS) + 2)

    def test_profile_streaming_statistics(self):
        profile = CorpusProfile(limit=10)
        for n in (1., 2., 3.):
            profile.add({'saliency': n / 4, 'reading_grade': None}, [n] * 200)
        data = profile.export()
        self.assertEqual(data['documents'], 3)
        self.assertAlmostEqual(data['fields']['saliency']['mean'], .5)
        self.assertEqual(data['vector_centroid'], [2.] * 200)
        self.assertNotIn('reading_grade', data['fields'])


if __name__ == '__main__':
    unittest.main()
