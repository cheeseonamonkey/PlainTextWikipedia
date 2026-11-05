"""Offline regression checks for safe, resumable dataset publication.

No credentials, external HTTP requests, uploads or Zenodo records are used.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from scripts.release_parquet import Asset, dataset_slug, inventory, limits
from scripts.structured_fleet import current_manifest
from scripts.structured_kaggle import language_name

ROOT = Path(__file__).resolve().parents[1]


class TestAcademicRelease(unittest.TestCase):
    def test_existing_repository_names_only(self):
        self.assertEqual(dataset_slug("ffatty/plain-text-wikipedia-simpleenglish"), "simpleenglish")
        self.assertEqual(dataset_slug("ffatty/plaintext-wikipedia-full-serbo-croatian"),
                         "serbo-croatian")
        for name in ("someone-else/wiki", "ffatty/unrelated", "ffatty/x/../wiki"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                dataset_slug(name)

    def test_reject_zenodo_quota_and_file_count(self):
        self.assertEqual(limits([Asset("a.parquet", "1" * 64, 1_000)]), 1_000)
        with self.assertRaisesRegex(ValueError, "50 GB"):
            limits([Asset("huge.parquet", "1" * 64, 50_000_000_000)])
        with self.assertRaisesRegex(ValueError, "100-file"):
            limits([Asset(f"{i}.parquet", "1" * 64, 1) for i in range(100)])

    def test_remote_inventory_is_checked_against_original_source(self):
        repo = "ffatty/plain-text-wikipedia-simpleenglish"
        source = "data/simplewiki_01_of_01.txt.gz"
        artifact = "structured/parquet/simplewiki_01_of_01.parquet"
        manifest = "structured/manifests/simplewiki_01_of_01.parquet.json"
        entry = {
            "source_sha256": "a" * 64, "articles": 42,
            "artifacts": {"parquet": {
                "path": "parquet/simplewiki_01_of_01.parquet",
                "sha256": "b" * 64, "size_bytes": 123,
            }},
        }
        files = [
            SimpleNamespace(path=source, lfs={"sha256": "a" * 64}, size=321),
            SimpleNamespace(path=artifact, lfs={"sha256": "b" * 64}, size=123),
            SimpleNamespace(path=manifest, lfs=None, size=200),
        ]
        api = SimpleNamespace(token=None, list_repo_tree=lambda *a, **k: files)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps(entry))
            with patch("scripts.release_parquet.hf_hub_download", return_value=str(path)):
                assets, metadata = inventory(api, repo)
                self.assertEqual(assets[0].path, artifact)
                self.assertEqual(metadata["files"][0]["articles"], 42)
                self.assertEqual(metadata["formats"], ["parquet"])
                self.assertFalse(metadata["wikipedia_page_ids_available"])
                files[0].lfs = {"sha256": "c" * 64}
                with self.assertRaisesRegex(ValueError, "Stale"):
                    inventory(api, repo)


class TestIncrementalFleet(unittest.TestCase):
    def test_respects_fts_policy_and_remote_hash(self):
        repo = "ffatty/plain-text-wikipedia-simpleenglish"
        name = "simplewiki_01_of_01"
        result = {"source_sha256": "a" * 64, "artifacts": {
            "sqlite": {"path": f"sqlite/{name}.sqlite.zst", "sha256": "b" * 64,
                       "fts5": False}}}
        remote = {
            f"structured/manifests/{name}.sqlite.json": object(),
            f"structured/sqlite/{name}.sqlite.zst": SimpleNamespace(lfs={"sha256": "b" * 64}),
        }
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "manifest.json"
            file.write_text(json.dumps(result))
            with patch("scripts.structured_fleet.hf_hub_download", return_value=str(file)):
                self.assertTrue(current_manifest(SimpleNamespace(token=None), repo, name,
                                                  "sqlite", "a" * 64, remote))
                self.assertFalse(current_manifest(SimpleNamespace(token=None), repo, name,
                                                   "sqlite", "a" * 64, remote, require_fts=True))
                remote[f"structured/sqlite/{name}.sqlite.zst"].lfs["sha256"] = "c" * 64
                self.assertFalse(current_manifest(SimpleNamespace(token=None), repo, name,
                                                   "sqlite", "a" * 64, remote))
                self.assertFalse(current_manifest(SimpleNamespace(token=None), repo, name,
                                                   "sqlite", "x" * 64, remote))

    def test_language_titles_preserve_hyphenated_names(self):
        self.assertEqual(language_name("ffatty/plain-text-wikipedia-simpleenglish"),
                         "Simple English")
        self.assertEqual(language_name("ffatty/plaintext-wikipedia-full-serbo-croatian"),
                         "Serbo-Croatian")


class TestCIContracts(unittest.TestCase):
    def test_new_release_workflows_are_manual_only(self):
        for name in ("structured-fleet.yml", "zenodo-parquet.yml",
                     "check-publish-credentials.yml"):
            with self.subTest(name=name):
                workflow = yaml.load(
                    (ROOT / ".github" / "workflows" / name).read_text(),
                    Loader=yaml.BaseLoader,
                )
                self.assertEqual(set(workflow["on"]), {"workflow_dispatch"})
                self.assertEqual(workflow["permissions"]["contents"], "read")
        zenodo = (ROOT / ".github/workflows/zenodo-parquet.yml").read_text()
        self.assertIn("default: false", zenodo)
        self.assertIn("--publish", zenodo)

    def test_cli_entrypoints_import_without_scripts_package_errors(self):
        for script in ("scripts/multistream_shards.py", "scripts/current_export.py",
                       "scripts/structured_fleet.py", "scripts/structured_kaggle.py",
                       "scripts/release_parquet.py"):
            with self.subTest(script=script):
                r = subprocess.run(
                    [sys.executable, script, "--help"], cwd=ROOT,
                    capture_output=True, text=True, timeout=30)
                self.assertEqual(r.returncode, 0, r.stderr[-1200:])

    def test_kaggle_cli_supports_exact_version_flags(self):
        r = subprocess.run(
            [sys.executable, "-m", "kaggle", "datasets", "version", "--help"],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr[-1200:])
        self.assertIn("--keep-tabular", r.stdout)
        self.assertIn("--dir-mode", r.stdout)


if __name__ == "__main__":
    unittest.main()
