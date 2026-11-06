import bz2
import gzip
import tempfile
import unittest
from pathlib import Path

from dewiki_functions import write_plaintext_shards


XML = """<?xml version="1.0" encoding="UTF-8"?>
<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">
  <page><title>One</title><ns>0</ns><id>1</id>
    <revision><id>11</id><text>First '''article'''</text></revision>
  </page>
</mediawiki>
"""


class ConverterTests(unittest.TestCase):
    def test_combines_page_range_files_into_named_gzip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.xml.bz2"
            second = root / "second.xml.bz2"
            for path in (first, second):
                with bz2.open(path, "wt", encoding="utf-8") as stream:
                    stream.write(XML.replace("One", path.stem))
            output = root / "output"

            count = write_plaintext_shards(
                [first, second],
                output,
                output_name="enwiki_01_of_10.txt.gz",
            )

            self.assertEqual(count, 2)
            result = output / "enwiki_01_of_10.txt.gz"
            self.assertTrue(result.exists())
            with gzip.open(result, "rt", encoding="utf-8") as stream:
                text = stream.read()
            self.assertIn("First article", text)
            self.assertEqual(text.count("\n\n"), 2)


    def test_respects_declared_xml_encoding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "latin1.xml"
            source.write_bytes(
                ('<?xml version="1.0" encoding="ISO-8859-1"?>'
                 '<mediawiki><page><title>Café</title><ns>0</ns><id>2</id>'
                 '<revision><text>Résumé</text></revision></page></mediawiki>').encode("iso-8859-1")
            )
            self.assertEqual(write_plaintext_shards(source, root / "output"), 1)
            with gzip.open(root / "output/wiki_01_of_01.txt.gz", "rt", encoding="utf-8") as stream:
                self.assertEqual(stream.read(), "Café\nRésumé\n\n")

    def test_compression_level_override_preserves_article_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sample.xml"
            source.write_text(XML, encoding="utf-8")
            for level in (1, 3, 6, 9):
                output = root / str(level)
                self.assertEqual(write_plaintext_shards(source, output, compression_level=level), 1)
                with gzip.open(output / "wiki_01_of_01.txt.gz", "rt", encoding="utf-8") as stream:
                    self.assertEqual(stream.read(), "One\nFirst article\n\n")


if __name__ == "__main__":
    unittest.main()
