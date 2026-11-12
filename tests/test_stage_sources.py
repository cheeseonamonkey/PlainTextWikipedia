"""Tests for the separate source staging entrypoint."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import build_language


class StageSourcesTests(unittest.TestCase):
    def test_stage_downloads_and_verifies_without_building(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'sources'
            with patch.object(sys, 'argv', ['build_language', '--language', 'en', '--stage-only', str(target)]), \
                 patch.object(build_language, 'resolve_language', return_value=('owner/repo', 'enwiki', 'en')), \
                 patch.object(build_language, 'HfApi'), \
                 patch.object(build_language, 'source_archives') as download, \
                 patch.object(build_language, 'verify_sources') as verify, \
                 patch.object(build_language, 'make_bundle') as build:
                build_language.main()
            download.assert_called_once()
            self.assertEqual(download.call_args.args[1:], ('owner/repo', target))
            verify.assert_called_once_with(target, 'en')
            build.assert_not_called()

    def test_stage_rejects_nonempty_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            (target / 'existing').touch()
            with patch.object(sys, 'argv', ['build_language', '--language', 'en', '--stage-only', str(target)]), \
                 patch.object(build_language, 'resolve_language', return_value=('owner/repo', 'enwiki', 'en')), \
                 patch.object(build_language, 'source_archives') as download:
                with self.assertRaises(FileExistsError):
                    build_language.main()
            download.assert_not_called()


if __name__ == '__main__':
    unittest.main()
