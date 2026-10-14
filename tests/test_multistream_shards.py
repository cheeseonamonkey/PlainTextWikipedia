import bz2
import io
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

from scripts.multistream_shards import boundaries, fetch, plan


class IndexedDumpTests(unittest.TestCase):
    def test_partition_is_complete_nonoverlapping_and_page_ordered(self):
        streams = [(10 + i * 100, 1 + i * 100) for i in range(30)]
        pieces = boundaries(streams, 3020, 8)
        self.assertEqual(len(pieces), 8)
        self.assertEqual(pieces[0][0], 10)
        self.assertEqual(pieces[-1][1], 3020)
        for a,b in zip(pieces,pieces[1:]):
            self.assertEqual(a[1],b[0])
            self.assertEqual(a[3]+1,b[2])
        self.assertEqual(sum(b-a for a,b,_,_ in pieces),3010)

    def test_invalid_split_count_is_rejected(self):
        with self.assertRaises(ValueError):
            boundaries([(0,1)],100,2)
        with self.assertRaises(ValueError):
            boundaries([(100,1)],100,1)

    def test_large_single_file_is_indexed_but_small_dump_stays_single(self):
        catalog=[(1,2147483647,"hawwiki-latest-pages-articles.xml.bz2",
                  "01-Oct-2026",2_000_000_000)]
        offsets=[(i*128*1024*1024,i*100+1) for i in range(20)]
        with (patch("scripts.multistream_shards.page_range_catalog",return_value=catalog),
              patch("scripts.multistream_shards.byte_range_available",
                    return_value=(True,2_200_000_000)),
              patch("scripts.multistream_shards.index_streams",return_value=offsets)):
            p=plan("https://dumps.wikimedia.org/hawwiki/latest","hawwiki",16)
        self.assertEqual(p["mode"],"indexed")
        self.assertEqual(p["count"],16)
        with (patch("scripts.multistream_shards.page_range_catalog",return_value=catalog),
              patch("scripts.multistream_shards.byte_range_available",
                    return_value=(True,2*1024*1024))):
            p=plan("https://dumps.wikimedia.org/hawwiki/latest","hawwiki",16)
        self.assertEqual(p["mode"],"normal")
        self.assertEqual(p["count"],1)

    def test_numbered_sources_keep_original_partition(self):
        catalog=[(1,100,"hewiki-latest-pages-articles1.xml-p1p100.bz2",
                  "01-Oct-2026",10000),
                 (101,200,"hewiki-latest-pages-articles2.xml-p101p200.bz2",
                  "01-Oct-2026",10000)]
        with patch("scripts.multistream_shards.page_range_catalog",return_value=catalog):
            self.assertEqual(plan("https://dumps.wikimedia.org/hewiki/latest",
                                  "hewiki",16)["count"],2)

    def test_fetch_range_wraps_exact_pages_and_no_duplicates(self):
        def make_page(i):
            return (f"<page><title>Article {i}</title><ns>0</ns><id>{i}</id>"
                    f"<revision><text>Test {i}</text></revision></page>\n").encode()
        streams=[bz2.compress(make_page(i)) for i in range(1,5)]
        joined=b"".join(streams)
        off=0
        index=[]
        for i,part in enumerate(streams,1):
            index.append((off,i))
            off+=len(part)

        class FakeHTTP(io.BytesIO):
            def __init__(self,start,end):
                super().__init__(joined[start:end])
                self.status=206
                self.headers={"Content-Range":f"bytes {start}-{end-1}/{len(joined)}"}

        def fake_open(url,method="GET",headers=None,timeout=90):
            match=headers["Range"].removeprefix("bytes=").split("-")
            return FakeHTTP(int(match[0]),int(match[1])+1)

        with tempfile.TemporaryDirectory() as tmp:
            with (patch("scripts.multistream_shards.byte_range_available",
                        return_value=(True,len(joined))),
                  patch("scripts.multistream_shards.index_streams",
                        return_value=index),
                  patch("scripts.multistream_shards.urlopen",
                        side_effect=fake_open)):
                titles=[]
                for i in range(2):
                    path=fetch("https://example.test/nowiki/latest","nowiki",
                               i,2,len(joined),Path(tmp))
                    with bz2.open(path,"rb") as inp:
                        root=ET.fromstring(inp.read())
                    titles.extend(x.findtext("title") for x in root.findall("page"))
                self.assertEqual(sorted(titles),[f"Article {i}" for i in range(1,5)])
                self.assertEqual(len(list(Path(tmp).glob("*.bz2"))),2)

    def test_http_range_ignored_is_a_hard_error(self):
        with patch("scripts.multistream_shards.byte_range_available",
                   return_value=(False,123)):
            with self.assertRaisesRegex(RuntimeError,"HTTP 206"):
                fetch("https://example.test/a/latest","awiki",0,2,123,Path("."))


if __name__ == "__main__":
    unittest.main()
