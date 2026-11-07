"""Offline tests for the reusable publishing workflow and preflight handling."""

import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from scripts.publish_preflight import check
from scripts.structured_fleet import retry, transient_error


ROOT = Path(__file__).resolve().parents[1] / ".github" / "workflows"


class WikipediaArticleLinksTests(unittest.TestCase):
    def test_catalog_and_primary_repos_have_wikipedia_domains(self):
        import json
        from scripts.wiki_metadata import ROOT, SIMPLE, wiki_code

        self.assertEqual(wiki_code(SIMPLE), "simple")
        catalog = json.loads((ROOT / "config/additional-languages.json").read_text())
        self.assertGreaterEqual(len(catalog["languages"]), 200)
        for language in catalog["languages"]:
            with self.subTest(slug=language["slug"]):
                repo = f"ffatty/plaintext-wikipedia-full-{language['slug']}"
                self.assertEqual(wiki_code(repo),
                                 language["project"].removesuffix("wiki"))
        for slug, code in (("english", "en"), ("german", "de"),
                           ("spanish", "es"), ("french", "fr"),
                           ("norwegian", "no"), ("chinese", "zh"),
                           ("italian", "it"), ("russian", "ru"),
                           ("portuguese", "pt")):
            with self.subTest(slug=slug):
                self.assertEqual(wiki_code(f"ffatty/plaintext-wikipedia-full-{slug}"),
                                 code)
        with self.assertRaisesRegex(ValueError, "No Wikimedia project mapping"):
            wiki_code("ffatty/plaintext-wikipedia-full-not-published")

class PreflightTests(unittest.TestCase):
    def test_optional_publishers_are_never_touched_when_disabled(self):
        with (patch.dict(os.environ, {"HF_TOKEN": "test"}, clear=True),
              patch("scripts.publish_preflight.HfApi") as hub,
              patch("scripts.publish_preflight.subprocess.run") as command,
              patch("requests.get") as get):
            check(kaggle=False, zenodo=False)
            hub.return_value.whoami.assert_called_once_with()
            command.assert_not_called()
            get.assert_not_called()

    def test_enabled_publishers_are_read_only(self):
        response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: [])
        with (patch.dict(os.environ, {"HF_TOKEN": "test", "KAGGLE_TOKEN": "test",
                                      "ZENODO_TOKEN": "test"}, clear=True),
              patch("scripts.publish_preflight.HfApi"),
              patch("scripts.publish_preflight.subprocess.run") as command,
              patch("requests.get", return_value=response) as get):
            check()
            command.assert_called_once()
            self.assertTrue(command.call_args.kwargs["check"])
            self.assertEqual(get.call_count, 1)

    def test_missing_required_tokens_fail_before_any_upload(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "HF_TOKEN"):
                check()


class SelectiveRetriesTests(unittest.TestCase):
    def test_permanent_urllib_http_failures_never_retry(self):
        def problem(code):
            return urllib.error.HTTPError(
                "https://example.com", code, "error", {}, None)
        self.assertFalse(transient_error(problem(401)))
        self.assertFalse(transient_error(problem(403)))
        self.assertFalse(transient_error(problem(404)))
        self.assertTrue(transient_error(problem(429)))
        self.assertTrue(transient_error(problem(503)))
        calls = []
        with patch("scripts.structured_fleet.time.sleep") as sleep:
            with self.assertRaises(urllib.error.HTTPError):
                retry("permanent 404", lambda: (calls.append(1), (_ for _ in ()).throw(
                    problem(404)))[1])
        self.assertEqual(len(calls), 1)
        sleep.assert_not_called()

    def test_transient_errors_retry_with_bounded_backoff(self):
        attempts = []
        def callback():
            attempts.append(1)
            if len(attempts) < 3:
                raise TimeoutError("transient network")
            return "ok"
        with (patch("scripts.structured_fleet.time.sleep") as sleep,
              patch("scripts.structured_fleet.random.random", return_value=0)):
            self.assertEqual(retry("network", callback), "ok")
        self.assertEqual(len(attempts), 3)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [8, 16])


if __name__ == "__main__":
    unittest.main()
