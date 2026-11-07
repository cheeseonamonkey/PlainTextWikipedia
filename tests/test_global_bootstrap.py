"""Static consistency checks for the independent 58-edition bootstrap."""
import json
import re
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


class GlobalBootstrapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.entries=json.loads((ROOT/"config/additional-languages.json").read_text())["languages"]
        cls.full_workflow=(ROOT/".github/workflows/rebuild-full.yml").read_text()

    def test_all_58_projects_are_configured_once(self):
        wave=[x for x in self.entries if x.get("collection")=="global-expansion-2026-10-15"]
        self.assertEqual(len(wave),58)
        projects=[x["project"] for x in self.entries]
        slugs=[x["slug"] for x in self.entries]
        self.assertEqual(len(projects),len(set(projects)))
        self.assertEqual(len(slugs),len(set(slugs)))
        self.assertTrue(all(re.fullmatch(r"[a-z][a-z0-9_-]*wiki",p) for p in projects))
        self.assertTrue(all(re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*",s) for s in slugs))



if __name__=="__main__":
    unittest.main()
