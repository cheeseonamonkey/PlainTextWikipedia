import gzip
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.current_export import parse_manifest, partition, download, months_before
from scripts.repack_shards import records, repack
from scripts.inspect_adaptive import inspect
from scripts.convert_adaptive import Source, split_sources


class MonthlyContentExportTests(unittest.TestCase):
    def test_snapshot_month_rollover(self):
        from datetime import date
        self.assertEqual(months_before(date(2026, 1, 13), 3),
                         ["2026-01-01", "2025-12-01", "2025-11-01"])

    def test_verified_files_partition_without_duplication(self):
        entries = [
            f"{'a'*64}  xwiki-2026-09-01-p1p100.xml.bz2",
            f"{'b'*64} *xwiki-2026-09-01-p101p200.xml.bz2",
            f"{'c'*64}  xwiki-2026-09-01-p201p300.xml.bz2",
        ]
        files = parse_manifest("\n".join(entries), "xwiki", "2026-09-01")
        catalogue = {"files": files}
        x = partition(catalogue, 0, 2)
        y = partition(catalogue, 1, 2)
        self.assertEqual(len(x), 2)
        self.assertEqual(len(y), 1)
        self.assertEqual(set(f["name"] for f in x + y),
                         set(f["name"] for f in files))

    def test_manifest_rejects_overlap_and_path_traversal(self):
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            parse_manifest(
                f"{'a'*64}  xwiki-2026-09-01-p1p100.xml.bz2\n"
                f"{'b'*64}  xwiki-2026-09-01-p50p150.xml.bz2",
                "xwiki", "2026-09-01")
        with self.assertRaises(ValueError):
            parse_manifest(f"{'c'*64}  ../bad.xml.bz2", "xwiki", "2026-09-01")

    def test_checksum_rejects_corrupt_xml_archives_without_publishing(self):
        data = b"old xml bytes"
        entry = {"name":"xwiki-2026-09-01-p1p10.xml.bz2",
                 "sha256":hashlib.sha256(b"good archive").hexdigest(),
                 "low":1,"high":10}
        catalogue = {"files":[entry],"base":"https://example.test"}
        class Response(io.BytesIO):
            pass
        with tempfile.TemporaryDirectory() as tmp:
            with (patch("scripts.current_export.get",return_value=Response(data)),
                  patch("scripts.current_export.time.sleep")):
                good,failed = download(catalogue,0,1,Path(tmp))
            self.assertFalse(good)
            self.assertEqual(len(failed),1)
            self.assertFalse(list(Path(tmp).glob("*.bz2")))


class MonthlyPageRangeTests(unittest.TestCase):
    def test_modern_xml_filename_can_split_recursively(self):
        source=Source(Path("atjwiki-2026-09-01-p100p199.xml.bz2"))
        left,right=split_sources([source])
        self.assertEqual((left[0].low,left[0].high),(100,149))
        self.assertEqual((right[0].low,right[0].high),(150,199))


class RepackTests(unittest.TestCase):
    def make_file(self,root,count=33,language="atjwiki"):
        filename=f"{language}_01_of_01.txt.gz"
        path=root/filename
        with gzip.open(path,"wb") as f:
            for i in range(count):
                f.write(f"Article {i}\nBody in {{language}} {i}\n\n".encode())
        manifest = {
            "language":language,"root_shard":1,"planned_shards":1,
            "complete":True,"failed_fragments":[],"download_failures":[],
            "successful_leaves":[{"file":filename,"articles":count}]
        }
        (root/f"{language}_01_of_01.manifest.json").write_text(json.dumps(manifest))
        return path

    def test_one_input_file_yields_sixteen_published_archives(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            self.make_file(root,33)
            result=repack(root,16)
            self.assertEqual(result["files_after"],16)
            self.assertEqual(len(list(root.glob("*.txt.gz"))),16)
            self.assertFalse((root/"atjwiki_01_of_01.txt.gz").exists())
            manifest=inspect("atjwiki",1,root)
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["published_archives"],16)
            self.assertEqual(sum(len(list(records(file))) for file in root.glob("*.txt.gz")),33)
            self.assertEqual(repack(root,16)["repacked"],0)

    def test_respect_article_count_and_do_not_create_empty_parts(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            self.make_file(root,5)
            result=repack(root,16)
            self.assertEqual(result["files_after"],5)
            self.assertEqual(sum(len(list(records(f))) for f in root.glob("*.txt.gz")),5)
            self.assertTrue(inspect("atjwiki",1,root)["complete"])

    def test_incomplete_edition_retains_partial_leaf_without_repacking(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            self.make_file(root,20)
            p=root/"atjwiki_01_of_01.manifest.json"
            report=json.loads(p.read_text())
            report["complete"]=False
            report["failed_fragments"]=[{"segment":"b"}]
            p.write_text(json.dumps(report))
            result=repack(root,16)
            self.assertEqual(result["repacked"],0)
            self.assertEqual(len(list(root.glob("*.txt.gz"))),1)

if __name__ == "__main__":
    unittest.main()
