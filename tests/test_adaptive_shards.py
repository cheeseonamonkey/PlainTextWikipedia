import gzip
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.convert_adaptive import Source, split_sources, convert
from scripts.inspect_adaptive import inspect


class AdaptiveSplitTests(unittest.TestCase):
    def test_file_groups_are_split_without_overlap(self):
        sources = [Source(Path(f"file{i}.bz2")) for i in range(5)]
        left, right = split_sources(sources)
        self.assertEqual(len(left), 2)
        self.assertEqual(len(right), 3)
        self.assertEqual(set(left).intersection(right), set())

    def test_single_range_bisects_page_ids_without_gap(self):
        source = Source(Path("enwiki-latest-pages-articles3.xml-p101p200.bz2"))
        a, b = split_sources([source])
        self.assertEqual((a[0].low, a[0].high), (101, 150))
        self.assertEqual((b[0].low, b[0].high), (151, 200))
        self.assertIsNone(split_sources([Source(source.path, 5, 5)]))

    def test_failed_root_recovers_as_a_and_b(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td)
            def fake_run(cmd, **kwargs):
                path = output / cmd[cmd.index("--output-name") + 1]
                if path.name == "enwiki_01_of_16.txt.gz":
                    return SimpleNamespace(returncode=1, stdout="converter failed")
                with gzip.open(path, "wb") as stream:
                    stream.write(b"Article\\nBody\\n\\n")
                return SimpleNamespace(returncode=0, stdout="Wrote 1 articles to output")

            source = Source(Path("enwiki-latest-pages-articles1.xml-p1p100.bz2"))
            with patch("scripts.convert_adaptive.subprocess.run", side_effect=fake_run):
                result = convert("enwiki", 1, 16, [source], output,
                                 max_depth=2, max_seconds=60 * 60)
            self.assertTrue(result["complete"])
            self.assertEqual([x["segment"] for x in result["successful_leaves"]], ["a", "b"])
            self.assertEqual(sorted(f.name for f in output.glob("*.txt.gz")),
                             ["enwiki_01a_of_16.txt.gz", "enwiki_01b_of_16.txt.gz"])
            self.assertTrue((output / "enwiki_01_of_16.manifest.json").is_file())

    def test_one_failed_child_publishes_other_half_but_marks_partial(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td)
            def fake_run(cmd, **kwargs):
                path = output / cmd[cmd.index("--output-name") + 1]
                if path.name in ("enwiki_01_of_16.txt.gz",
                                 "enwiki_01b_of_16.txt.gz"):
                    return SimpleNamespace(returncode=1, stdout="failed")
                with gzip.open(path, "wb") as stream:
                    stream.write(b"OK")
                return SimpleNamespace(returncode=0, stdout="Wrote 1 articles to output")
            source = Source(Path("enwiki-latest-pages-articles1.xml-p1p8.bz2"))
            with patch("scripts.convert_adaptive.subprocess.run", side_effect=fake_run):
                result = convert("enwiki", 1, 16, [source], output, max_depth=1,
                                 max_seconds=60 * 60)
            self.assertFalse(result["complete"])
            self.assertEqual(len(result["successful_leaves"]), 1)
            self.assertEqual(result["successful_leaves"][0]["segment"], "a")
            self.assertEqual(len(result["failed_fragments"]), 1)


class CoverageManifestTests(unittest.TestCase):
    def test_partial_child_and_missing_root_are_reported(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td)
            name = "enwiki_01a_of_02.txt.gz"
            with gzip.open(output / name, "wb") as stream:
                stream.write(b"OK")
            report = {
                "language": "enwiki", "root_shard": 1, "planned_shards": 2,
                "complete": False, "successful_leaves": [{"file": name}],
                "failed_fragments": [{"segment": "b"}], "download_failures": [],
            }
            (output / "enwiki_01_of_02.manifest.json").write_text(json.dumps(report))
            manifest = inspect("enwiki", 2, output)
            self.assertEqual(manifest["published_archives"], 1)
            self.assertEqual(manifest["present_shards"], [1])
            self.assertEqual(manifest["missing_shards"], [2])
            self.assertEqual(manifest["incomplete_shards"], [1])
            self.assertFalse(manifest["complete"])
            self.assertFalse(list(output.glob("*.manifest.json")))

    def test_complete_split_has_all_leaf_archives(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td)
            names = ["enwiki_01a_of_01.txt.gz", "enwiki_01b_of_01.txt.gz"]
            for name in names:
                with gzip.open(output / name, "wb") as stream:
                    stream.write(b"OK")
            report = {
                "language": "enwiki", "root_shard": 1, "planned_shards": 1,
                "complete": True, "successful_leaves": [{"file": name} for name in names],
                "failed_fragments": [], "download_failures": [],
            }
            (output / "enwiki_01_of_01.manifest.json").write_text(json.dumps(report))
            manifest = inspect("enwiki", 1, output)
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["published_shards"], 1)
            self.assertEqual(manifest["published_archives"], 2)


if __name__ == "__main__":
    unittest.main()
