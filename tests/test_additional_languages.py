import json
import unittest
from pathlib import Path

from scripts.language_info import metadata


class AdditionalLanguageTests(unittest.TestCase):
    def test_registry_has_distinct_projects_and_slugs(self):
        path = Path(__file__).resolve().parents[1] / "config" / "additional-languages.json"
        rows = json.loads(path.read_text(encoding="utf-8"))["languages"]
        self.assertGreaterEqual(len(rows), 115)
        self.assertEqual(len({row["project"] for row in rows}), len(rows))
        self.assertEqual(len({row["slug"] for row in rows}), len(rows))
        for row in rows:
            self.assertTrue(row["project"].endswith("wiki"))
            self.assertEqual(metadata(row["project"]), (row["name"], row["slug"]))

    def test_special_editions(self):
        self.assertEqual(metadata("cebwiki"), ("Cebuano", "cebuano"))
        self.assertEqual(metadata("tokwiki"), ("Toki Pona", "toki-pona"))
        self.assertEqual(metadata("zh_classicalwiki"), ("Classical Chinese", "classical-chinese"))
        self.assertEqual(metadata("arcwiki"), ("Aramaic (Syriac)", "aramaic-syriac"))
        self.assertEqual(metadata("hewiki"), ("Hebrew", "hebrew"))
        self.assertEqual(metadata("hawiki"), ("Hausa", "hausa"))
        self.assertEqual(metadata("hawwiki"), ("Hawaiian", "hawaiian"))
        self.assertEqual(metadata("tigwiki"), ("Tigre", "tigre"))
        self.assertEqual(metadata("tiwiki"), ("Tigrinya", "tigrinya"))
        self.assertEqual(metadata("smnwiki"), ("Inari Sami", "inari-sami"))
        self.assertEqual(metadata("atjwiki"), ("Atikamekw", "atikamekw"))
        with self.assertRaises(ValueError):
            metadata("tlhwiki")  # Closed Klingon Wikipedia


if __name__ == "__main__":
    unittest.main()
