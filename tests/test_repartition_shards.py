import gzip
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from scripts.repartition_shards import iter_articles, output_ids, repartition
from scripts.inspect_adaptive import inspect


def source_leaf(directory: Path, language: str, root: int, roots: int,
                records: list[tuple[str, str]], complete: bool = True) -> Path:
    name = f"{language}_{root:02d}_of_{roots:02d}.txt.gz"
    file = directory / name
    with gzip.open(file, "wt", encoding="utf-8") as out:
        for title, body in records:
            out.write(f"{title}\n{body}\n\n")
    report = {
        "language": language, "root_shard": root, "planned_shards": roots,
        "complete": complete, "successful_leaves": [
            {"file": name, "segment": "root", "sources": [{"file": "source.xml.bz2"}],
             "articles": len(records)}
        ],
        "failed_fragments": [] if complete else [{"segment": "b", "error": "time budget"}],
        "download_failures": [],
    }
    (directory / f"{language}_{root:02d}_of_{roots:02d}.manifest.json").write_text(
        json.dumps(report))
    return file


class OutputShardTests(unittest.TestCase):
    def test_unique_global_output_numbering(self):
        self.assertEqual(output_ids(1, 1, 16), list(range(1, 17)))
        self.assertEqual(output_ids(1, 4, 16), [1, 5, 9, 13])
        self.assertEqual(output_ids(2, 4, 16), [2, 6, 10, 14])
        all_ids = [i for root in range(1, 4) for i in output_ids(root, 3, 16)]
        self.assertEqual(sorted(all_ids), list(range(1, 17)))
        with self.assertRaises(ValueError):
            output_ids(2, 2, 1)

    def test_single_archive_repacked_into_16_complete_archives(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            originals = [(f"Article {i}", f"Body detail {i}") for i in range(600)]
            original_file = source_leaf(root, "hawwiki", 1, 1, originals)
            result = repartition("hawwiki", 1, 1, 16, root)
            self.assertEqual(result["output_shards"], 16)
            self.assertEqual(result["repartitioned_articles"], 600)
            self.assertFalse(original_file.exists())
            self.assertEqual(len(result["successful_leaves"]), 16)
            actual = [
                pair for leaf in result["successful_leaves"]
                for pair in iter_articles(root / leaf["file"])
            ]
            self.assertEqual(Counter(actual), Counter(originals))
            self.assertEqual(len({x["file"] for x in result["successful_leaves"]}), 16)
            self.assertEqual(repartition("hawwiki", 1, 1, 16, root), result)
            coverage = inspect("hawwiki", 1, root)
            self.assertTrue(coverage["complete"])
            self.assertEqual(coverage["planned_output_shards"], 16)
            self.assertEqual(coverage["published_archives"], 16)

    def test_two_workers_produce_disjoint_global_output_names(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            all_records = []
            for worker in (1, 2):
                records = [(f"Worker {worker} Title {i}", f"Article body {i}")
                           for i in range(200)]
                all_records.extend(records)
                source_leaf(root, "euwiki", worker, 2, records)
                repartition("euwiki", worker, 2, 16, root)
            files = sorted(root.glob("*.txt.gz"))
            self.assertEqual(len(files), 16)
            actual = [pair for file in files for pair in iter_articles(file)]
            self.assertEqual(Counter(actual), Counter(all_records))
            coverage = inspect("euwiki", 2, root)
            self.assertTrue(coverage["complete"])
            self.assertEqual(coverage["published_archives"], 16)

    def test_tiny_wikipedia_does_not_publish_empty_placeholders(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_leaf(root, "tigwiki", 1, 1, [("Only article", "Some detail")])
            result = repartition("tigwiki", 1, 1, 16, root)
            self.assertEqual(len(result["successful_leaves"]), 1)
            self.assertEqual(len(list(root.glob("*.txt.gz"))), 1)
            self.assertTrue(inspect("tigwiki", 1, root)["complete"])

    def test_incomplete_recovery_stays_partial_after_repartition(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_leaf(root, "nvwiki", 1, 1,
                        [(f"Article {i}", "Content") for i in range(25)],
                        complete=False)
            result = repartition("nvwiki", 1, 1, 16, root)
            self.assertFalse(result["complete"])
            self.assertEqual(len(result["failed_fragments"]), 1)
            coverage = inspect("nvwiki", 1, root)
            self.assertFalse(coverage["complete"])
            self.assertEqual(coverage["incomplete_shards"], [1])

    def test_malformed_input_keeps_original_and_recovery_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = source_leaf(root, "hewiki", 1, 1, [("Original", "Body")])
            with gzip.open(path, "wt", encoding="utf-8") as output:
                output.write("Title\nBody without terminating separator\n")
            report_path = root / "hewiki_01_of_01.manifest.json"
            baseline = report_path.read_text()
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                repartition("hewiki", 1, 1, 16, root)
            self.assertTrue(path.exists())
            self.assertEqual(report_path.read_text(), baseline)
            self.assertFalse(list(root.glob("*.pending")))


if __name__ == "__main__":
    unittest.main()
