import base64
import gzip
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.kaggle_auth import candidates, verify_and_configure
from scripts.kaggle_backfill import backfill, run_kaggle, FOUNDATION_LANGUAGES
from scripts.repack_shards import repack, records


class KaggleAuthTests(unittest.TestCase):
    def test_legacy_json_key(self):
        result=candidates('{"username":"ffatty","key":"fake-key"}',"ffatty")
        self.assertEqual(result,[("legacy-json",{"KAGGLE_USERNAME":"ffatty",
                                                   "KAGGLE_KEY":"fake-key"})])

    def test_modern_and_legacy_token_fallback(self):
        result=candidates("an-example-token","ffatty")
        self.assertEqual([item[0] for item in result],
                         ["access-token","legacy-key"])
        self.assertEqual(result[1][1]["KAGGLE_USERNAME"],"ffatty")

    def test_reject_invalid_legacy_json(self):
        with self.assertRaises(ValueError):
            candidates('{"key":"only-key"}',"ffatty")

    def test_configuration_only_written_after_successful_authentication(self):
        class Result:
            def __init__(self,code):
                self.returncode=code
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch("scripts.kaggle_auth.subprocess.run",
                       side_effect=[Result(1),Result(0)]) as p:
                mode=verify_and_configure("legacy-key","ffatty",home=root)
            self.assertEqual(mode,"legacy-key")
            self.assertEqual(p.call_count,2)
            config=json.loads((root/".kaggle/kaggle.json").read_text())
            self.assertEqual(config,{"username":"ffatty","key":"legacy-key"})
            self.assertFalse((root/".kaggle/access_token").exists())

    def test_invalid_auth_does_not_persist_token(self):
        class Result:
            returncode=1
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch("scripts.kaggle_auth.subprocess.run",return_value=Result()):
                with self.assertRaises(RuntimeError):
                    verify_and_configure("fake","ffatty",home=root)
            self.assertFalse(list((root/".kaggle").iterdir()))


class FoundationLanguagesTests(unittest.TestCase):
    def test_original_large_edition_destinations_are_supported(self):
        self.assertEqual(FOUNDATION_LANGUAGES["spanish"], "Spanish")
        self.assertEqual(FOUNDATION_LANGUAGES["french"], "French")
        self.assertEqual(FOUNDATION_LANGUAGES["chinese"], "Chinese")
        self.assertEqual(FOUNDATION_LANGUAGES["portuguese"], "Portuguese")
        self.assertEqual(FOUNDATION_LANGUAGES["english"], "English")

class KaggleBackfillTests(unittest.TestCase):
    def test_archive_filename_does_not_replace_language_title(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            item={"path":"data/hiwiki_01_of_01.txt.gz","size":1024}
            def fake_download(repo,remote,target,sha,size):
                target.parent.mkdir(parents=True,exist_ok=True)
                with gzip.open(target,"wb") as out:
                    out.write(b"Title\\nBody\\n\\n")
                return target
            with (patch("scripts.kaggle_backfill.manifest_for",return_value=[item]),
                  patch("scripts.kaggle_backfill.download",side_effect=fake_download),
                  patch("scripts.kaggle_backfill.get_json",return_value={"complete":True})):
                result=backfill("hindi","Hindi",root,1400,publish=False)
            info=json.loads((root/"hindi"/"dataset-metadata.json").read_text())
            self.assertEqual(info["title"],"Plaintext Wikipedia (Hindi)")
            self.assertNotIn("hiwiki_01_of_01",info["title"])
            self.assertEqual(result["gzips"],1)

    def test_existing_dataset_is_versioned_and_waited_until_ready(self):
        class Success:
            returncode=0
        with (patch("scripts.kaggle_backfill.kaggle_status",
                    side_effect=[{"status":"ready","current_version_number":1},
                                 {"status":"processing","current_version_number":2},
                                 {"status":"ready","current_version_number":2}]),
              patch("scripts.kaggle_backfill.subprocess.run",return_value=Success()) as proc,
              patch("scripts.kaggle_backfill.time.sleep")):
            run_kaggle("ffatty/plaintext-wikipedia-full-hindi",
                       Path("/tmp/published-test"),"Hindi",wait_seconds=120)
        self.assertTrue(any("version" in args[0] for args,kwargs in proc.call_args_list))


class MinimumCompressedShardSizeTests(unittest.TestCase):
    def make_source(self,root:Path,records_count:int,body_size:int=1800):
        file=root/"hawwiki_01_of_01.txt.gz"
        with gzip.open(file,"wb") as out:
            for i in range(records_count):
                body=base64.b64encode(os.urandom(body_size)).decode("ascii")
                out.write(f"Article {i}\n{body}\n\n".encode())
        report={"language":"hawwiki","root_shard":1,"planned_shards":1,
                "complete":True,"successful_leaves":[{"file":file.name,"articles":records_count}],
                "failed_fragments":[],"download_failures":[]}
        (root/"hawwiki_01_of_01.manifest.json").write_text(json.dumps(report))
        return file

    def test_minimum_compressed_size_prevents_tiny_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            original=self.make_source(root,12)
            size=original.stat().st_size
            result=repack(root,16,min_bytes=size//2+1)
            self.assertEqual(result["repacked"],0)
            self.assertTrue(original.exists())

    def test_minimum_size_split_and_full_record_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            original=self.make_source(root,72)
            entries=list(records(original))
            result=repack(root,16,min_bytes=12000)
            self.assertTrue(2<=result["files_after"]<=16)
            output=list(root.glob("*.txt.gz"))
            self.assertTrue(all(x.stat().st_size>=12000 for x in output))
            self.assertCountEqual([row for path in output for row in records(path)],entries)


if __name__=="__main__":
    unittest.main()
