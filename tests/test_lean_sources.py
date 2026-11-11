import json
import tempfile
import unittest
from pathlib import Path

from scripts.bundle_artifacts import verify_sources


class SourceCoverageTests(unittest.TestCase):
    def test_legacy_requires_every_unsplit_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            manifest = {'language': 'dewiki', 'complete': True, 'expected_shards': 2,
                        'published_shards': 2, 'present_shards': [1, 2], 'missing_shards': []}
            (folder / 'build-manifest.json').write_text(json.dumps(manifest))
            for index in (1, 2):
                (folder / f'dewiki_{index:02d}_of_02.txt.gz').touch()
            self.assertEqual(verify_sources(folder, 'de'), ('legacy', 2))
            (folder / 'dewiki_02_of_02.txt.gz').rename(folder / 'dewiki_02a_of_02.txt.gz')
            with self.assertRaises(ValueError):
                verify_sources(folder, 'de')

    def test_modern_manifest_must_be_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            manifest = {'language': 'ruwiki', 'complete': False, 'expected_shards': 1,
                        'roots': [{'root_shard': 1, 'complete': True,
                                   'files': ['ruwiki_01_of_01.txt.gz']}]}
            (folder / 'build-manifest.json').write_text(json.dumps(manifest))
            (folder / 'ruwiki_01_of_01.txt.gz').touch()
            with self.assertRaises(ValueError):
                verify_sources(folder, 'ru')
