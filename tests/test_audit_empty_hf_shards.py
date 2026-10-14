import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_empty_hf_shards import classify, datasets, manifest_files


class EmptyPublishedArchiveTests(unittest.TestCase):
    def test_only_tiny_gzip_files_are_candidates(self):
        sizes = {
            "data/a.txt.gz": 0,
            "data/b.txt.gz": 20,
            "data/c.txt.gz": 40_000_000,
            "data/build-manifest.json": 0,
            "README.md": 0,
        }
        plan = classify(sizes, {"b.txt.gz", "c.txt.gz"})
        self.assertEqual(plan["files"], 3)
        self.assertEqual(plan["suspected_empty"],
                         ["data/a.txt.gz", "data/b.txt.gz"])
        self.assertEqual(plan["referenced_candidates"], ["data/b.txt.gz"])
        self.assertEqual(plan["untracked_candidates"], ["data/a.txt.gz"])
        self.assertEqual(plan["valid_size_archives"], 1)

    def test_no_manifest_never_confers_permission_to_delete(self):
        plan = classify({"data/a.txt.gz": 20}, None)
        self.assertEqual(plan["untracked_candidates"], [])
        self.assertEqual(plan["referenced_candidates"], [])

    def test_all_root_references_count_even_in_incomplete_builds(self):
        report = {"complete": False, "roots": [
            {"files": ["a.txt.gz"],
             "successful_leaves": [{"file": "b.txt.gz"}]},
        ]}
        self.assertEqual(manifest_files(report), {"a.txt.gz", "b.txt.gz"})

    def test_historical_projects_exclude_current_active_build_batch(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "langs.json"
            path.write_text(json.dumps({"languages": [
                {"project": "euwiki", "slug": "basque"},
                {"project": "idwiki", "slug": "indonesian",
                 "collection": "2026-october-expansion"},
            ]}))
            repos = datasets(path)
            self.assertIn("ffatty/plaintext-wikipedia-full-basque", repos)
            self.assertNotIn("ffatty/plaintext-wikipedia-full-indonesian", repos)


if __name__ == "__main__":
    unittest.main()
