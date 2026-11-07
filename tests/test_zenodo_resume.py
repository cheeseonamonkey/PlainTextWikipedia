"""Offline evidence that partially uploaded Zenodo draft files are reused."""

import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.release_parquet import Asset, zenodo_publish


class ZenodoResumeTests(unittest.TestCase):
    def test_matching_draft_parquet_is_not_deleted_or_reuploaded(self):
        with tempfile.TemporaryDirectory() as temporary:
            parquet = Path(temporary) / "foo.parquet"
            parquet.write_bytes(b"PAR1 already verified PAR1")
            checksum = hashlib.md5(parquet.read_bytes()).hexdigest()
            title = "Plaintext Wikipedia (simpleenglish) — Parquet edition"
            metadata = {"language": "simpleenglish", "source_url": "https://example.org",
                        "files": [{"source_sha256": "a" * 64}]}
            asset = Asset("structured/parquet/foo.parquet", "b" * 64,
                          parquet.stat().st_size, md5=checksum)
            draft = {
                "id": 999, "submitted": False, "metadata": {"title": title},
                "links": {
                    "bucket": "https://zenodo.org/api/files/draft-bucket",
                    "publish": "https://zenodo.org/api/deposit/depositions/999/actions/publish",
                },
                "files": [{
                    "id": "file-1", "filename": parquet.name,
                    "filesize": str(parquet.stat().st_size), "checksum": checksum,
                }],
            }
            uploaded = []

            def put(url, **kwargs):
                if "/api/deposit/depositions/" in url:
                    return SimpleNamespace(raise_for_status=lambda: None, json=lambda: draft)
                uploaded.append((url, kwargs["data"].read()))
                return SimpleNamespace(
                    content=b"", raise_for_status=lambda: None, json=lambda: {}
                )

            published = SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"doi_url": "https://doi.org/10.5281/zenodo.999"},
            )
            with (patch.dict(os.environ, {"ZENODO_TOKEN": "fake"}),
                  patch("scripts.release_parquet.list_matching_depositions", return_value=[draft]),
                  patch("scripts.release_parquet.fetch_verified", return_value=parquet) as fetch,
                  patch("requests.put", side_effect=put),
                  patch("requests.delete") as delete,
                  patch("requests.post", return_value=published) as post):
                zenodo_publish(SimpleNamespace(token=None),
                               "ffatty/plain-text-wikipedia-simpleenglish",
                               [asset], metadata, None)

            delete.assert_not_called()
            fetch.assert_not_called()  # Skip both the download and extra SHA256 scan.
            self.assertEqual(len(uploaded), 1)
            self.assertTrue(uploaded[0][0].endswith("/provenance.json"))
            post.assert_called_once()

    def test_verified_parquet_download_uses_shared_hf_cache(self):
        from scripts.release_parquet import fetch_verified

        with tempfile.TemporaryDirectory() as folder:
            parquet = Path(folder) / "test.parquet"
            parquet.write_bytes(b"PAR1 cache hit PAR1")
            digest = hashlib.sha256(parquet.read_bytes()).hexdigest()
            asset = Asset("structured/parquet/test.parquet", digest, parquet.stat().st_size)
            with patch("scripts.release_parquet.hf_hub_download", return_value=str(parquet)) as hub:
                self.assertEqual(fetch_verified(SimpleNamespace(token=None), "ffatty/wiki", asset),
                                 parquet)
                self.assertNotIn("local_dir", hub.call_args.kwargs)
            with patch("scripts.release_parquet.hf_hub_download", return_value=str(parquet)):
                bad = Asset(asset.path, "0" * 64, asset.size)
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    fetch_verified(SimpleNamespace(token=None), "ffatty/wiki", bad)

    def test_rejected_checksum_never_publishes_a_doi(self):
        with tempfile.TemporaryDirectory() as temporary:
            parquet = Path(temporary) / "foo.parquet"
            parquet.write_bytes(b"PAR1 verified PAR1")
            asset = Asset("structured/parquet/foo.parquet", "b" * 64,
                          parquet.stat().st_size,
                          md5=hashlib.md5(parquet.read_bytes(), usedforsecurity=False).hexdigest())
            draft = {
                "id": 111, "submitted": False, "metadata": {
                    "title": "Plaintext Wikipedia (simpleenglish) — Parquet edition"},
                "links": {
                    "bucket": "https://zenodo.org/api/files/a",
                    "publish": "https://zenodo.org/api/deposit/depositions/111/actions/publish",
                },
                "files": [],
            }
            def put(url, **kwargs):
                return SimpleNamespace(
                    content=b'{"checksum":"md5:incorrect"}',
                    raise_for_status=lambda: None,
                    json=lambda: {"checksum": "md5:incorrect"},
                ) if "/api/files/" in url else SimpleNamespace(
                    raise_for_status=lambda: None, json=lambda: draft)
            with (patch.dict(os.environ, {"ZENODO_TOKEN": "fake"}),
                  patch("scripts.release_parquet.list_matching_depositions", return_value=[draft]),
                  patch("scripts.release_parquet.fetch_verified", return_value=parquet),
                  patch("requests.put", side_effect=put),
                  patch("requests.post") as publish):
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    zenodo_publish(SimpleNamespace(token=None),
                                   "ffatty/plain-text-wikipedia-simpleenglish",
                                   [asset], {"language": "simpleenglish",
                                             "source_url": "https://example.org"}, None)
                publish.assert_not_called()


if __name__ == "__main__":
    unittest.main()

class CompleteBundleZenodoTests(unittest.TestCase):
    def test_local_bundle_uploads_only_declared_files_and_requires_checksums(self):
        from scripts.release_parquet import sha256
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            names=['articles.parquet','articles.txt.xz','features.zip','models.zip','manifest.json']
            assets=[]
            for name in names:
                path=folder/name; path.write_bytes(name.encode())
                assets.append(Asset(name,sha256(path),path.stat().st_size))
            title='Plaintext Wikipedia (french) — Parquet edition'
            draft={'id':123,'submitted':False,'metadata':{'title':title},'files':[],
                   'links':{'bucket':'https://zenodo.org/api/files/bundle',
                            'publish':'https://zenodo.org/api/deposit/depositions/123/actions/publish'}}
            uploaded=[]
            def put(url,**kwargs):
                if '/api/files/' not in url:
                    return SimpleNamespace(raise_for_status=lambda:None,json=lambda:draft)
                data=kwargs['data'].read();uploaded.append(url.rsplit('/',1)[1])
                return SimpleNamespace(content=b'ok',raise_for_status=lambda:None,
                                       json=lambda:{'checksum':'md5:'+hashlib.md5(data).hexdigest()})
            with patch.dict(os.environ,{'ZENODO_TOKEN':'fake'}), patch('requests.get',return_value=SimpleNamespace(raise_for_status=lambda:None,json=lambda:draft)), patch('requests.put',side_effect=put), patch('requests.post',return_value=SimpleNamespace(raise_for_status=lambda:None,json=lambda:{})) as publish:
                zenodo_publish(None,'ffatty/plaintext-wikipedia-full-french',assets,
                               {'language':'french','source_url':'https://example.org','links':{}},123,local_folder=folder)
                self.assertEqual(set(uploaded),set(names))
                publish.assert_called_once()
