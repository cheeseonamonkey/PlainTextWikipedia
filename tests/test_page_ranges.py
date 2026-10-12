import io
import unittest
from unittest.mock import patch

from scripts.download_page_ranges import page_ranges


class PageRangeTests(unittest.TestCase):
    def test_non_multistream_ranges_only_and_sorted(self):
        html = b"""
        <a href="eswiki-latest-pages-articles2.xml-p101p200.bz2">2</a>
        <a href="eswiki-latest-pages-articles1.xml-p1p100.bz2">1</a>
        <a href="eswiki-latest-pages-articles1.xml-p1p100.bz2">duplicate link</a>
        <a href="eswiki-latest-pages-articles-multistream1.xml-p1p100.bz2">multistream</a>
        """
        stream = io.BytesIO(html)
        with patch("scripts.download_page_ranges.open_url", return_value=stream):
            ranges = page_ranges("https://dumps.wikimedia.org/eswiki/latest", "eswiki")
        self.assertEqual(
            ranges,
            [(1, 100, "eswiki-latest-pages-articles1.xml-p1p100.bz2"),
             (101, 200, "eswiki-latest-pages-articles2.xml-p101p200.bz2")],
        )

    def test_round_robin_has_no_duplicate_ranges(self):
        ranges = [(i, i + 9, f"range-{i}") for i in range(0, 110, 10)]
        all_selected = [
            [entry for i, entry in enumerate(ranges) if i % 5 == shard]
            for shard in range(5)
        ]
        self.assertEqual(len({name for shard in all_selected for _, _, name in shard}), 11)
        self.assertEqual(sum(len(shard) for shard in all_selected), len(ranges))


if __name__ == "__main__":
    unittest.main()
