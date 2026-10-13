import io
import unittest
from unittest.mock import patch

from scripts.download_page_ranges import page_ranges, latest_page_ranges


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

    def test_exclude_obsolete_404_and_overlapping_ranges(self):
        # Wikimedia latest/ can retain September files whose starts overlap
        # the current October partition, including nonempty stale archives.
        catalog = [
            (1, 100, "oct-1.bz2", "01-Oct-2026", 5_000_000),
            (101, 200, "sep-tombstone.bz2", "01-Sep-2026", 70),
            (101, 250, "sep-valid-but-obsolete.bz2", "02-Sep-2026", 190_000_000),
            (101, 270, "oct-2.bz2", "01-Oct-2026", 210_000_000),
        ]
        current = latest_page_ranges(catalog)
        self.assertEqual([entry[2] for entry in current], ["oct-1.bz2", "oct-2.bz2"])
        # The repair workflow must preserve the *original* round-robin
        # positions, only dropping obsolete filenames after assignment.
        shard = 0
        selected = [
            entry[2] for i, entry in enumerate(catalog)
            if i % 2 == shard and entry in current
        ]
        self.assertEqual(selected, ["oct-1.bz2"])

    def test_overlapping_current_ranges_are_not_silently_published(self):
        catalog = [
            (1, 200, "first.bz2", "01-Oct-2026", 100_000),
            (199, 300, "overlap.bz2", "01-Oct-2026", 100_000),
        ]
        with self.assertRaisesRegex(RuntimeError, "Overlapping"):
            latest_page_ranges(catalog)

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
