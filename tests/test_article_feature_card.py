import unittest
import yaml

from scripts.article_feature_card import update_card


class FeatureCardTests(unittest.TestCase):
    def test_default_loader_uses_only_canonical_parquet_and_preserves_description(self):
        initial = '---\nlanguage:\n- en\nlicense: cc-by-sa-4.0\n---\n\n# Original prose\n'
        result = update_card(initial)
        meta = yaml.safe_load(result.split('---\n')[1])
        self.assertEqual(meta['configs'][0]['data_files'][0]['path'], 'articles.parquet')
        self.assertEqual(meta['license'], 'cc-by-sa-4.0')
        self.assertIn('# Original prose', result)
        self.assertEqual(update_card(result), result)

    def test_refuses_malformed_card(self):
        with self.assertRaises(ValueError):
            update_card('Plain text without front matter')


if __name__ == '__main__':
    unittest.main()
