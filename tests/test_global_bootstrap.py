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
        cls.workflow=(ROOT/".github/workflows/bootstrap-global-oct15.yml").read_text()
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

    def test_wave_matches_configuration_without_rebuilding_old_editions(self):
        configured={x["project"] for x in self.entries
                    if x.get("collection")=="global-expansion-2026-10-15"}
        listed=set(re.findall(r"^\s+- ([a-z][a-z0-9_-]*wiki)$",
                               self.workflow,re.MULTILINE))
        self.assertEqual(listed,configured)
        self.assertEqual(len(listed),58)

    def test_safe_runner_limits_and_small_wiki_shard_policy(self):
        self.assertIn("max-parallel: 2",self.workflow)
        self.assertIn("cancel-in-progress: false",self.workflow)
        self.assertIn("secrets: inherit",self.workflow)
        self.assertIn("--min-mb 20",self.full_workflow)
        self.assertIn("kaggle_publish",self.full_workflow)
        self.assertIn("hf_publish",self.full_workflow)


if __name__=="__main__":
    unittest.main()
