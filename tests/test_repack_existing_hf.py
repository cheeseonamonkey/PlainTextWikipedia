import gzip
import tempfile
import unittest
from pathlib import Path

from scripts.repack_existing_hf import repack_local, transform_manifest
from scripts.repack_shards import records


class ExistingHubMigrationTests(unittest.TestCase):
    def test_local_repack_preserves_every_article_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            old="atjwiki_01_of_01.txt.gz"
            rows=[f"Title {i}\nBody with content {i}\n\n".encode() for i in range(80)]
            with gzip.open(directory/old,"wb") as out:
                for row in rows:
                    out.write(row)
            result=repack_local(directory,16)
            self.assertTrue(result["changed"])
            self.assertEqual(len(result["new_files"]),16)
            self.assertFalse((directory/old).exists())
            self.assertCountEqual(
                [row for file in directory.glob("*.txt.gz") for row in records(file)],rows)
            self.assertEqual(len(result["replacements"][old]),16)
            self.assertFalse(list(directory.glob("*.manifest.json")))

    def test_partial_manifest_preserves_coverage_claims(self):
        old="iuwiki_01_of_01.txt.gz"
        pieces=[{"file":f"iuwiki_01_of_01_part{i:02d}_of_02.txt.gz"} for i in (1,2)]
        original={
            "language":"iuwiki","complete":False,"expected_shards":1,
            "published_archives":1,"missing_shards":[],
            "incomplete_shards":[1],
            "roots":[{"root_shard":1,"complete":False,"files":[old],
                      "successful_leaves":[{"file":old,"articles":6}],
                      "failed_fragments":[{"segment":"b","error":"timeout"}]}]
        }
        new=transform_manifest(original,{old:pieces},[old])
        self.assertFalse(new["complete"])
        self.assertEqual(new["incomplete_shards"],[1])
        self.assertEqual(len(new["roots"][0]["successful_leaves"]),2)
        self.assertEqual(len(new["roots"][0]["files"]),2)
        self.assertEqual(new["published_archives"],2)
        self.assertEqual(original["published_archives"],1)
        self.assertEqual(new["roots"][0]["failed_fragments"],original["roots"][0]["failed_fragments"])

    def test_malformed_local_article_does_not_delete_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            source=directory/"hawwiki_01_of_01.txt.gz"
            with gzip.open(source,"wb") as f:
                f.write(b"Incomplete\nBody without final separator")
            with self.assertRaises(ValueError):
                repack_local(directory,16)
            self.assertTrue(source.exists())


if __name__ == "__main__":
    unittest.main()
