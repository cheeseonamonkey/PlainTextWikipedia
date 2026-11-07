"""Regressions for incomplete sources, interrupted publishing and invalid bundles."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import wikipedia_bundle as bundle
import test_article_features as fixtures

REPO = 'ffatty/plaintext-wikipedia-full-french'


class BundleSafetyTests(unittest.TestCase):
    setUp = fixtures.BundleTests.setUp

    def make_release(self, zenodo_url=None):
        output = self.root / 'release'
        with patch('scripts.bekko_features.BekkoEncoder', fixtures.FakeBekko):
            bundle.make_bundle(self.files, output, 'fr', bundle.links(REPO, zenodo_url), {'revision': 'source'})
        return output

    def test_top_level_complete_cannot_hide_failed_root(self):
        manifest = dict(complete=True, expected_shards=1, language='frwiki',
                        roots=[dict(root_shard=1, complete=False,
                                    files=[p.name for p in self.files])])
        (self.root / 'build-manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'coverage'):
            bundle.complete_sources(self.root)

    def test_archive_identity_does_not_depend_on_local_mtime(self):
        source = self.root / 'stats.json'
        source.write_text('{"count":3}')
        bundle.archive(self.root / 'first.zip', [source])
        os.utime(source, (1700000000, 1700000000))
        bundle.archive(self.root / 'second.zip', [source])
        self.assertEqual((self.root / 'first.zip').read_bytes(), (self.root / 'second.zip').read_bytes())

    def test_existing_bundle_is_not_overwritten_by_another_build(self):
        output = self.make_release()
        before = (output / 'manifest.json').read_bytes()
        with patch('scripts.wikipedia_bundle.build') as build:
            with self.assertRaises(FileExistsError):
                bundle.make_bundle(self.files, output, 'fr', bundle.links(REPO), {})
            build.assert_not_called()
        self.assertEqual((output / 'manifest.json').read_bytes(), before)

    def test_resume_rechecks_hf_artifacts_even_when_manifest_matches(self):
        output = self.make_release("https://zenodo.org/records/123")
        api = SimpleNamespace(token=None, repo_info=lambda *a, **k: SimpleNamespace(sha='published'),
                              list_repo_tree=lambda *a, **k: [])
        with patch.dict(os.environ, {'HF_TOKEN':'x','KAGGLE_TOKEN':'x','ZENODO_TOKEN':'x'}), \
             patch('scripts.wikipedia_bundle.hf_hub_download', return_value=str(output / 'manifest.json')), \
             patch('scripts.wikipedia_bundle.publish_kaggle') as kaggle, \
             patch('scripts.wikipedia_bundle.zenodo_publish') as zenodo:
            with self.assertRaises(ExceptionGroup) as failure:
                bundle.publish_all(api, REPO, output, 123)
            self.assertIn('HF artifact', str(failure.exception.exceptions[0]))
            kaggle.assert_called_once()
            zenodo.assert_called_once()

    def test_kaggle_processing_upload_can_be_resumed_without_another_version(self):
        output = self.make_release()
        states = iter([{'status':'processing', 'current_version_number':2},
                       {'status':'ready', 'current_version_number':3}])
        calls = []
        def run(*args, **kwargs):
            calls.append(args)
            if args[0] == 'status':
                return SimpleNamespace(stdout=json.dumps(next(states)), returncode=0)
            if args[0] == 'download':
                probe = Path(args[args.index('-p') + 1])
                (probe / 'manifest.json').write_bytes((output / 'manifest.json').read_bytes())
                return SimpleNamespace(returncode=0)
            self.fail(f'Unexpected Kaggle mutation: {args}')
        with patch('scripts.wikipedia_bundle.run', side_effect=run), \
             patch('time.sleep'), \
             patch('scripts.wikipedia_bundle.verify_kaggle') as verify:
            bundle.publish_kaggle(REPO, output)
            verify.assert_called_once()
        self.assertNotIn('version', [call[0] for call in calls])


    def test_build_only_bundle_can_publish_without_recalculating_features(self):
        import sys
        output = self.make_release()
        before = {name: (output / name).read_bytes() for name in bundle.PAYLOAD}
        with patch.dict(os.environ, {'HF_TOKEN':'x','KAGGLE_TOKEN':'x','ZENODO_TOKEN':'x'}), \
             patch.object(sys, 'argv', ['bundle', '--repo', REPO, '--output', str(output), '--resume', '--publish']), \
             patch('scripts.wikipedia_bundle.reserve_zenodo', return_value=123), \
             patch('scripts.wikipedia_bundle.publish_all') as publish, \
             patch('scripts.wikipedia_bundle.build') as build:
            bundle.main()
            build.assert_not_called()
            publish.assert_called_once()
        after = json.loads((output / 'manifest.json').read_text())
        self.assertEqual(after['links']['zenodo'], 'https://zenodo.org/records/123')
        self.assertEqual(before, {name: (output / name).read_bytes() for name in bundle.PAYLOAD})

    def test_rehashed_parquet_with_wrong_schema_is_rejected(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from scripts.bundle_artifacts import file_metadata
        output = self.make_release()
        pq.write_table(pa.table({'unrelated': [1, 2, 3]}), output / 'articles.parquet')
        manifest = json.loads((output / 'manifest.json').read_text())
        manifest['files']['articles.parquet'] = file_metadata(output / 'articles.parquet')
        (output / 'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'schema'):
            bundle.verify_bundle(output)

    def test_failed_build_leaves_no_partially_committed_bundle(self):
        with patch('scripts.wikipedia_bundle.build', side_effect=RuntimeError('inference failed')):
            with self.assertRaisesRegex(RuntimeError, 'inference failed'):
                bundle.make_bundle(self.files, self.root / 'failed', 'fr', bundle.links(REPO), {})
        self.assertFalse((self.root / 'failed').exists())
        self.assertFalse(list(self.root.glob('.failed-*')))

    def test_invalid_bekko_vector_fails_without_a_release(self):
        import numpy as np
        class BrokenBekko:
            def encode(self, texts):
                return np.full((len(texts), 128), np.nan), [12] * len(texts)
        with patch('scripts.bekko_features.BekkoEncoder', BrokenBekko):
            with self.assertRaisesRegex(ValueError, 'finite'):
                bundle.make_bundle(self.files, self.root / 'bad-vector', 'fr', bundle.links(REPO), {})
        self.assertFalse((self.root / 'bad-vector').exists())

    def test_cross_filesystem_staging_copies_but_other_errors_propagate(self):
        import errno
        from scripts.bundle_artifacts import link_or_copy
        source = self.files[0]
        destination = self.root / 'copy.txt.gz'
        with patch('scripts.bundle_artifacts.os.link', side_effect=OSError(errno.EXDEV, 'cross-device')):
            link_or_copy(source, destination)
        self.assertEqual(destination.read_bytes(), source.read_bytes())
        with patch('scripts.bundle_artifacts.os.link', side_effect=OSError(errno.ENOSPC, 'full')):
            with self.assertRaises(OSError):
                link_or_copy(source, self.root / 'full.txt.gz')

    def test_lost_hf_commit_response_is_observed_without_a_second_commit(self):
        from scripts.bundle_artifacts import file_metadata
        output = self.make_release()
        manifest = bundle.verify_bundle(output)
        files = [SimpleNamespace(path=name, size=(output / name).stat().st_size,
                                 lfs={'sha256':file_metadata(output / name)['sha256']})
                 for name in bundle.RELEASE_FILES]
        api = SimpleNamespace(token=None, repo_info=lambda *a, **k:SimpleNamespace(sha='published'),
                              list_repo_tree=lambda *a, **k:files)
        from unittest.mock import Mock
        api.create_commit = Mock(side_effect=TimeoutError('response lost'))
        bundle.publish_hf(api, REPO, output, 'source', manifest=manifest)
        api.create_commit.assert_called_once()

    def test_combined_text_must_match_the_intermediate_article_index(self):
        import lzma
        import sqlite3
        from scripts.article_features import article_rows
        text = self.root / 'articles.txt.xz'
        with lzma.open(text, 'wt') as stream:
            stream.write('Wrong title\nBody\n\n')
        with sqlite3.connect(':memory:') as db:
            db.row_factory = sqlite3.Row
            db.execute('CREATE TABLE articles(id INTEGER, title TEXT, char_count INTEGER)')
            db.execute('INSERT INTO articles VALUES (1, ?, 4)', ('Correct title',))
            with self.assertRaisesRegex(RuntimeError, 'article index'):
                list(article_rows(db, text))

    def test_all_publishers_are_attempted_and_failures_are_preserved(self):
        output = self.make_release('https://zenodo.org/records/123')
        with patch.dict(os.environ, {'HF_TOKEN':'x','KAGGLE_TOKEN':'x','ZENODO_TOKEN':'x'}), \
             patch('scripts.wikipedia_bundle.publish_hf_current', side_effect=RuntimeError('HF unavailable')) as hf, \
             patch('scripts.wikipedia_bundle.publish_kaggle', side_effect=TimeoutError('Kaggle unavailable')) as kaggle, \
             patch('scripts.wikipedia_bundle.zenodo_publish') as zenodo:
            with self.assertRaises(ExceptionGroup) as failure:
                bundle.publish_all(None, REPO, output, 123)
            self.assertEqual(len(failure.exception.exceptions), 2)
            hf.assert_called_once()
            kaggle.assert_called_once()
            zenodo.assert_called_once()

    def test_small_source_git_blob_identity_is_supported(self):
        import hashlib
        path = self.files[0]
        data = path.read_bytes()
        digest = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
        entry = SimpleNamespace(lfs=None, blob_id=digest)
        bundle.verify_source(path, entry, bundle.sha256(path))
        entry.blob_id = '0' * 40
        with self.assertRaisesRegex(ValueError, 'Git blob identity'):
            bundle.verify_source(path, entry, bundle.sha256(path))
