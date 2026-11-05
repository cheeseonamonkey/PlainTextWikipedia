"""Offline regression checks for safe, resumable dataset publication.

No credentials, external HTTP requests, uploads or Zenodo records are used.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from scripts.release_parquet import (Asset, dataset_slug, inventory, limits,
                                    provenance_fingerprint, list_matching_depositions, zenodo_publish)
from scripts.structured_fleet import current_manifest
from scripts.structured_kaggle import language_name, mirror

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


class TestPublishByDefault(unittest.TestCase):
    def test_zenodo_version_hash_is_deterministic(self):
        data = {"files": [{"source_sha256": "a" * 64}], "source_repo": "ffatty/wiki"}
        self.assertEqual(provenance_fingerprint(data),
                         provenance_fingerprint(dict(reversed(list(data.items())))))
        self.assertNotEqual(provenance_fingerprint(data),
                            provenance_fingerprint({"files": [], "source_repo": "ffatty/wiki"}))

    def test_zenodo_ownership_search_paginates_and_filters_titles(self):
        good = {"id": 123, "metadata": {"title": "target"}}
        irrelevant = {"id": 44, "metadata": {"title": "other"}}
        def query(url, headers, params, timeout):
            page = params["page"]
            items = [irrelevant] * 100 if page == 1 else [good]
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: items)
        matches = list_matching_depositions(SimpleNamespace(get=query),
                                            "https://zenodo.org/api/deposit/depositions",
                                            {"Authorization": "fake"}, "target")
        self.assertEqual(matches, [good])

    def test_zenodo_does_not_republish_identical_parquet_record(self):
        metadata = {"language": "simpleenglish", "source_url": "https://example.org",
                    "files": [{"source_sha256": "a" * 64}]}
        prior = {
            "id": 999, "submitted": True,
            "metadata": {"title": "Plaintext Wikipedia (simpleenglish) — Parquet edition",
                         "notes": "wikidata-parquet-manifest-sha256:" +
                                  provenance_fingerprint(metadata)},
        }
        with (patch.dict(os.environ, {"ZENODO_TOKEN": "test-value"}),
              patch("scripts.release_parquet.list_matching_depositions", return_value=[prior]),
              patch("requests.post") as post):
            zenodo_publish(SimpleNamespace(token=None), "ffatty/plain-text-wikipedia-simpleenglish",
                           [Asset("a.parquet", "b" * 64, 123)], metadata, None)
            post.assert_not_called()

    def test_required_kaggle_cannot_skip_unpublished_dataset(self):
        with patch("scripts.structured_kaggle.HfApi") as api:
            api.return_value.list_repo_tree.return_value = []
            with self.assertRaisesRegex(RuntimeError, "No source archives"):
                mirror("ffatty/plain-text-wikipedia-simpleenglish",
                       max_mb=2500, require_publish=True)

class TestParquetOnlyPublishing(unittest.TestCase):
    def test_kaggle_does_not_require_or_generate_sqlite(self):
        import hashlib
        from scripts.structured_kaggle import mirror

        repo = "ffatty/plain-text-wikipedia-simpleenglish"
        stem = "simplewiki_01_of_01"
        raw_name = f"{stem}.txt.gz"
        parquet_name = f"{stem}.parquet"
        raw_path = f"data/{raw_name}"
        parquet_path = f"structured/parquet/{parquet_name}"
        manifest_path = f"structured/manifests/{stem}.parquet.json"
        with tempfile.TemporaryDirectory() as folder:
            temp = Path(folder)
            raw = temp / raw_name
            parquet = temp / parquet_name
            raw.write_bytes(b"Title\nBody\n\n")
            parquet.write_bytes(b"PAR1testPAR1")
            h = lambda file: hashlib.sha256(file.read_bytes()).hexdigest()
            doc = {
                "source_sha256": h(raw), "articles": 1,
                "artifacts": {"parquet": {"path": f"parquet/{parquet_name}",
                                          "sha256": h(parquet),
                                          "size_bytes": parquet.stat().st_size}},
            }
            manifest = temp / "manifest.json"
            manifest.write_text(json.dumps(doc))
            tree = [
                SimpleNamespace(path=raw_path, size=raw.stat().st_size,
                                lfs={"sha256": h(raw)}),
                SimpleNamespace(path=parquet_path, size=parquet.stat().st_size,
                                lfs={"sha256": h(parquet)}),
                SimpleNamespace(path=manifest_path, size=manifest.stat().st_size, lfs=None),
            ]
            hf_files = {raw_path: raw, parquet_path: parquet, manifest_path: manifest}
            publish = []

            def runner(*args, check=True):
                if args[0] == "status":
                    return SimpleNamespace(returncode=0, stdout='{"status":"ready"}')
                if args[0] == "files":
                    return SimpleNamespace(returncode=1, stdout="")
                if args[0] == "download":
                    stage = Path(args[args.index("-p") + 1])
                    (stage / raw_name).write_bytes(raw.read_bytes())
                    (stage / f"{stem}.sqlite.zst").write_bytes(b"legacy-sqlite")
                    return SimpleNamespace(returncode=0, stdout="")
                if args[0] == "version":
                    stage = Path(args[args.index("-p") + 1])
                    publish.extend(p.name for p in stage.iterdir())
                    meta = json.loads((stage / "dataset-metadata.json").read_text())
                    self.assertNotIn("SQLite", meta["description"])
                    self.assertNotIn("sqlite", json.dumps(meta["resources"]).lower())
                    return SimpleNamespace(returncode=0, stdout="")
                raise AssertionError(args)

            with (patch("scripts.structured_kaggle.HfApi") as mock_api,
                  patch("scripts.structured_kaggle.hf_hub_download",
                        side_effect=lambda repo, filename, **kw: str(hf_files[filename])),
                  patch("scripts.structured_kaggle.run", side_effect=runner)):
                mock_api.return_value.list_repo_tree.return_value = tree
                mirror(repo, 300, require_publish=True)
            self.assertIn(parquet_name, publish)
            self.assertIn(raw_name, publish)
            self.assertIn("structured-manifest.json", publish)
            self.assertFalse(any("sqlite" in name for name in publish))


    def test_fleet_build_only_publishes_parquet(self):
        import gzip
        import hashlib
        from scripts.structured_fleet import build

        repo = "ffatty/plain-text-wikipedia-simpleenglish"
        name = "simplewiki_01_of_01"
        with tempfile.TemporaryDirectory() as folder:
            raw = Path(folder) / f"{name}.txt.gz"
            with gzip.open(raw, "wt", encoding="utf-8") as stream:
                stream.write("Title\nBody\n\n")
            with raw.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            source = SimpleNamespace(path=f"data/{raw.name}",
                                     lfs={"sha256": digest}, size=raw.stat().st_size)
            uploaded = []

            def create_commit(*, operations, **kwargs):
                self.assertEqual(len(operations), 2)
                for operation in operations:
                    self.assertTrue(Path(operation.path_or_fileobj).is_file())
                    uploaded.append(operation.path_in_repo)

            with (patch.dict(os.environ, {"HF_TOKEN": "test-token"}),
                  patch("scripts.structured_fleet.HfApi") as api,
                  patch("scripts.structured_fleet.hf_hub_download", return_value=str(raw))):
                api.return_value.list_repo_tree.return_value = [source]
                api.return_value.create_commit.side_effect = create_commit
                build(repo)
                api.return_value.create_commit.assert_called_once()
                api.return_value.upload_file.assert_not_called()
            self.assertEqual(sorted(uploaded), [
                f"structured/manifests/{name}.parquet.json",
                f"structured/parquet/{name}.parquet",
            ])


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
        root = ROOT / ".github" / "workflows"
        zenodo = yaml.load((root / "zenodo-parquet.yml").read_text(), Loader=yaml.BaseLoader)
        fleet = yaml.load((root / "structured-fleet.yml").read_text(), Loader=yaml.BaseLoader)
        self.assertEqual(zenodo["on"]["workflow_dispatch"]["inputs"]["publish"]["default"],
                         "true")
        inputs = fleet["on"]["workflow_dispatch"]["inputs"]
        self.assertEqual(inputs["publish_zenodo"]["default"], "true")
        self.assertEqual(inputs["mirror_kaggle"]["default"], "true")
        self.assertNotIn("format", inputs)
        for name in ("structured-fleet.yml", "structured-wikipedia.yml"):
            workflow_text = (root / name).read_text()
            self.assertNotIn("sqlite", workflow_text.lower())
            self.assertNotIn("upload-artifact", workflow_text)
            single = yaml.load(workflow_text, Loader=yaml.BaseLoader)
            self.assertEqual(set(single["on"]),
                             {"workflow_dispatch", "workflow_call"} if name == "structured-wikipedia.yml"
                             else {"workflow_dispatch"})
            if name == "structured-wikipedia.yml":
                self.assertEqual(single["on"]["workflow_dispatch"]["inputs"]["mirror_kaggle"]["default"], "true")
                self.assertEqual(single["on"]["workflow_dispatch"]["inputs"]["publish_zenodo"]["default"], "true")
        fleet_job = fleet["jobs"]["publish"]
        self.assertEqual(fleet_job["uses"], "./.github/workflows/structured-wikipedia.yml")
        self.assertEqual(fleet_job["secrets"], "inherit")
        self.assertEqual(fleet_job["strategy"]["fail-fast"], "false")
        self.assertEqual(fleet["on"]["workflow_dispatch"]["inputs"]["parallelism"]["default"], "8")
        self.assertIn("--publish", (root / "zenodo-parquet.yml").read_text())

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
