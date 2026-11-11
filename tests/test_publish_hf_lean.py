import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.publish_hf import publish_hf


class PublishHfTests(unittest.TestCase):
    def test_manifest_is_verified_without_self_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for name in ('articles.parquet', 'corpus-profile.json', 'manifest.json'):
                (folder / name).write_text(name)
            from scripts.bundle_artifacts import file_metadata
            manifest = {'documents': 1, 'files': {name: file_metadata(folder / name)
                        for name in ('articles.parquet', 'corpus-profile.json')}}
            class Api:
                token = None
                def create_repo(self, **kwargs): pass
                def list_repo_tree(self, *args, **kwargs): return []
                def create_commit(self, **kwargs): return None
                def repo_info(self, *args, **kwargs):
                    return type('Info', (), {'sha': '1234567890123456'})()
            def download(repo, name, **kwargs): return str(folder / name)
            with patch('scripts.publish_hf.verify_bundle', return_value=manifest), \
                 patch('scripts.publish_hf.hf_hub_download', side_effect=download):
                publish_hf(Api(), 'ffatty/test', folder)
