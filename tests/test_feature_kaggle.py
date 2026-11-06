"""Kaggle feature sync must preserve provenance and refuse partial releases."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.feature_kaggle import feature_inventory, kaggle_inventory, mirror, same_release


class KaggleFeatureTests(unittest.TestCase):
    def test_inventory_ignores_cli_pagination_notice(self):
        result = SimpleNamespace(stdout=("Next Page Token = sample\n"
                                         "name,size,creationDate\n"
                                         "foo.txt,10,2026-11-06\n"))
        with patch('scripts.feature_kaggle.run', return_value=result):
            self.assertEqual(kaggle_inventory('ffatty/plaintext-wikipedia-full-english'),
                             {'foo.txt': 10})

    def test_requires_complete_current_feature_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'manifest.json'
            expected = ('articles.parquet', 'corpus-profile.json',
                        'vocabulary.json.gz', 'fasttext.bin.zst', 'articles.sqlite.zst')
            p.write_text(json.dumps({'primary_table': 'articles.parquet',
                                     'files': {name: {'bytes': 10, 'sha256': 'a' * 64}
                                               for name in expected}}))
            entries = [SimpleNamespace(path=name if name == 'articles.parquet'
                                        else f'features/{name}', size=10, lfs={'sha256': 'a' * 64})
                       for name in expected]
            entries.append(SimpleNamespace(path='features/manifest.json', size=99))
            api = SimpleNamespace(token='placeholder', list_repo_tree=lambda *a, **k: entries)
            with patch('scripts.feature_kaggle.hf_hub_download', return_value=str(p)):
                manifest, selected = feature_inventory(api, 'ffatty/plain-text-wikipedia-simpleenglish', 9)
                self.assertEqual(manifest['primary_table'], 'articles.parquet')
                self.assertEqual(len(selected), 4)  # SQLite cap is honored.
                self.assertEqual(selected[0][1], 'articles.parquet')
                self.assertNotIn('articles-features.sqlite.zst', [x[1] for x in selected])
                _, all_selected = feature_inventory(api, 'ffatty/plain-text-wikipedia-simpleenglish', 20)
                self.assertEqual(len(all_selected), 5)

    def test_reject_new_dataset_slug_before_network(self):
        with self.assertRaisesRegex(ValueError, 'Existing'):
            mirror('bad-account/new-repo')

    def test_skip_only_identical_manifest_and_all_required_files(self):
        document = {'files': {'articles.parquet': {'sha256': 'abc'}}}
        selected = [('articles.parquet', 'articles.parquet', {'bytes': 10})]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            def run(*args, **kwargs):
                (directory / 'feature-manifest.json').write_text(json.dumps(document))
                return SimpleNamespace(returncode=0)
            with patch('scripts.feature_kaggle.run', side_effect=run):
                self.assertTrue(same_release('ffatty/plain-text-wikipedia-simpleenglish',
                                             document, {'feature-manifest.json': 4,
                                                        'articles.parquet': 10},
                                             selected, directory))
                self.assertFalse(same_release('ffatty/plain-text-wikipedia-simpleenglish',
                                              document, {'feature-manifest.json': 4,
                                                         'articles.parquet': 7},
                                              selected, directory))


if __name__ == '__main__':
    unittest.main()
