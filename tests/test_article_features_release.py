import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.article_features_release import is_current


class ReleaseCompletenessTests(unittest.TestCase):
    def test_missing_required_artifact_cannot_be_current(self):
        names = ("articles.parquet", "corpus-profile.json", "vocabulary.json.gz",
                 "articles.sqlite.zst", "fasttext.bin.zst")
        files = {name: {"bytes": 10, "sha256": "a" * 64} for name in names}
        remote = {("features/" + name if name != "articles.parquet" else name):
                  SimpleNamespace(size=10) for name in names}
        remote["features/manifest.json"] = SimpleNamespace(size=1)
        settings = {"allow_lexical_only": False}
        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder) / "manifest.json"
            with patch("scripts.article_features_release.hf_hub_download", return_value=str(manifest)), \
                 patch("scripts.article_features_release.source_hash", return_value=None):
                def check():
                    manifest.write_text(json.dumps({"source_archives": [], "config": settings,
                                                    "files": files}))
                    return is_current(SimpleNamespace(token=None), "owner/repo", [], settings, remote)
                self.assertTrue(check())
                del files["articles.parquet"]
                self.assertFalse(check())


if __name__ == "__main__":
    unittest.main()
