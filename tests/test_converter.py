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


if __name__ == "__main__":
    unittest.main()
