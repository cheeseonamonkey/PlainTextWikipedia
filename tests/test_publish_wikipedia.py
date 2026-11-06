"""Offline contract tests for resumable, independent Wikipedia publication."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

import yaml

from scripts.publish_wikipedia import card, metadata, publish, retry, publish_hf, publish_kaggle

MANIFEST = {"language": "simplewiki", "complete": True, "published_shards": 8,
            "expected_shards": 8, "published_archives": 8,
            "missing_shards": [], "incomplete_shards": []}
URL = "https://github.com/cheeseonamonkey/PlainTextWikipedia"


class PublisherTests(unittest.TestCase):
    def test_simple_english_card_uses_valid_iso_language(self):
        output = card("simplewiki", "Simple English", "ffatty/plain-text-wikipedia-simpleenglish",
                      MANIFEST, URL, URL)
        meta = yaml.safe_load(output.split("---", 2)[1])
        self.assertEqual(meta["language"], ["en"])
        self.assertIn("8/8 root shards", output)
        self.assertNotIn("__LANG_", output)

    def test_extended_wikimedia_code_omits_invalid_hf_language_tag(self):
        output = card("zh_classicalwiki", "Classical Chinese", "ffatty/plaintext-wikipedia-full-classical-chinese",
                      MANIFEST, URL, URL)
        meta = yaml.safe_load(output.split("---", 2)[1])
        self.assertNotIn("language", meta)
        self.assertIn("Classical Chinese", meta["pretty_name"])

    def test_source_only_kaggle_card_preserves_edition_name(self):
        meta = metadata("Simple English", "ffatty/plain-text-wikipedia-simpleenglish",
                        MANIFEST, URL, URL)
        self.assertEqual(meta["id"], "ffatty/plain-text-wikipedia-simpleenglish")
        self.assertIn("Simple English Wikipedia", meta["description"])

    def test_permanent_metadata_error_is_not_retried(self):
        calls = []
        def invalid():
            calls.append(True)
            raise ValueError("Invalid metadata in README.md")
        with patch("scripts.publish_wikipedia.time.sleep") as sleep:
            with self.assertRaisesRegex(ValueError, "Invalid metadata"):
                retry("card", invalid)
        self.assertEqual(len(calls), 1)
        sleep.assert_not_called()

    def test_independent_publishers_run_even_when_one_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "build-manifest.json").write_text(json.dumps(MANIFEST))
            (folder / "simplewiki_01_of_08.txt.gz").write_bytes(b"test")
            called = []
            def fail(*args):
                called.append("hf")
                raise ValueError("invalid HF card")
            def pass_kaggle(*args):
                called.append("kaggle")
            with (patch("scripts.publish_wikipedia.publish_hf", side_effect=fail),
                  patch("scripts.publish_wikipedia.publish_kaggle", side_effect=pass_kaggle)):
                results = publish(folder, "simplewiki", "Simple English", "hf/test", "kaggle/test",
                                  URL, URL, "hf-secret", "kaggle-secret")
            self.assertEqual(set(called), {"hf", "kaggle"})
            self.assertEqual(results["Kaggle"], "success")
            self.assertTrue(results["Hugging Face"].startswith("failed:"))

    def test_source_archives_are_not_deleted_on_partial_publish(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "simplewiki_01_of_08.txt.gz").write_bytes(b"test")
            (folder / "build-manifest.json").write_text(json.dumps(MANIFEST))
            api = MagicMock()
            api.list_repo_files.return_value = ["data/old.txt.gz"]
            api.list_repo_tree.return_value = []
            import sys
            with patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(HfApi=lambda token: api)}):
                publish_hf(folder, "simplewiki", "Simple English", "ffatty/test",
                           {**MANIFEST, "complete": False}, URL, URL, "token")
            api.delete_file.assert_not_called()

    def test_kaggle_refuses_to_replace_structured_files(self):
        with tempfile.TemporaryDirectory() as temp:
            with (patch("scripts.kaggle_auth.verify_and_configure"),
                  patch("subprocess.run", return_value=SimpleNamespace(
                      returncode=0, stdout="file,size\nstructured/parquet/a.parquet,100\n")),
                  patch("scripts.kaggle_backfill.run_kaggle") as uploader):
                with self.assertRaisesRegex(RuntimeError, "structured files"):
                    publish_kaggle(Path(temp), "English", "ffatty/test", MANIFEST,
                                   URL, URL, "token")
                uploader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
