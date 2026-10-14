import gzip
import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.compact_small_hf import copy_records, refresh_coverage, rewrite_root


class SmallHubArchiveTests(unittest.TestCase):
    def test_consolidation_is_byte_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            items = [
                [b"A\nFirst body\n\n", b"C\nThird body\n\n"],
                [b"B\nSecond body\n\n", b"D\nFourth body\n\n"],
            ]
            files = []
            for n, rows in enumerate(items):
                path = folder / f"part{n}.txt.gz"
                with gzip.open(path, "wb") as out:
                    out.write(b"".join(rows))
                files.append(path)
            dest = folder / "merged.txt.gz"
            count, digest = copy_records(files, dest)
            expected = b"".join(sum(items, []))
            self.assertEqual(count, 4)
            self.assertEqual(digest, hashlib.sha256(expected).hexdigest())
            with gzip.open(dest, "rb") as stream:
                self.assertEqual(stream.read(), expected)
            self.assertTrue(all(path.exists() for path in files))

    def test_empty_input_and_duplicate_titles(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            a, b = folder / "a.gz", folder / "b.gz"
            with gzip.open(a, "wb") as f:
                pass
            with gzip.open(b, "wb") as f:
                f.write(b"Title\nData\n\n")
            count, _ = copy_records([a, b], folder / "good.gz")
            self.assertEqual(count, 1)
            with gzip.open(a, "wb") as f:
                f.write(b"Title\nOther Data\n\n")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                copy_records([a, b], folder / "bad.gz")

    def test_rewrite_preserves_partial_roots_and_recalculates_coverage(self):
        root = {
            "root_shard": 1, "complete": False,
            "successful_leaves": [
                {"file": "a.gz", "articles": 2, "sources": [{"file": "page1.bz2"}]},
                {"file": "b.gz", "articles": 3, "sources": [{"file": "page2.bz2"}]},
            ],
            "files": ["a.gz", "b.gz"],
            "failed_fragments": [{"segment": "c", "error": "missing"}],
        }
        manifest = {"expected_shards": 2, "roots": [root, {
            "root_shard": 2, "complete": True,
            "files": ["other.gz"], "successful_leaves": [{"file": "other.gz"}]}]}
        rewrite_root(root, ["a.gz", "b.gz"], "merged.gz", 5, 100, "digest")
        refresh_coverage(manifest)
        self.assertEqual(root["files"], ["merged.gz"])
        self.assertEqual(root["successful_leaves"][0]["articles"], 5)
        self.assertEqual(manifest["published_archives"], 2)
        self.assertEqual(manifest["incomplete_shards"], [1])
        self.assertFalse(manifest["complete"])
        self.assertEqual(len(root["failed_fragments"]), 1)

    def test_entirely_empty_root_is_marked_missing(self):
        root = {"root_shard": 1, "complete": True,
                "files": ["empty.gz"],
                "successful_leaves": [{"file": "empty.gz", "articles": 0}]}
        manifest = {"expected_shards": 1, "roots": [root]}
        rewrite_root(root, ["empty.gz"], None, 0, 0, None)
        refresh_coverage(manifest)
        self.assertEqual(manifest["missing_shards"], [1])
        self.assertEqual(manifest["published_archives"], 0)
        self.assertFalse(manifest["complete"])
        self.assertFalse(root["complete"])


if __name__ == "__main__":
    unittest.main()
