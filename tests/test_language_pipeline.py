"""Contract checks for the single reusable Wikimedia edition pipeline."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github/workflows"


def workflow(name):
    return yaml.load((WORKFLOWS / name).read_text(), Loader=yaml.BaseLoader)


class LanguagePipelineTests(unittest.TestCase):
    def test_entrypoints_share_one_rebuild(self):
        for name, language in (
            ("rebuild-simple.yml", "simplewiki"),
            ("refresh-simple.yml", "simplewiki"),
            ("refresh-full-monthly.yml", "enwiki"),
        ):
            with self.subTest(workflow=name):
                data = workflow(name)
                self.assertEqual(len(data["jobs"]), 1)
                job = next(iter(data["jobs"].values()))
                self.assertEqual(job["uses"], "./.github/workflows/rebuild-full.yml")
                self.assertEqual(job["with"]["language"], language)
                self.assertEqual(job["secrets"], "inherit")
        all_languages = workflow("rebuild-languages.yml")
        for name, job in all_languages["jobs"].items():
            with self.subTest(language=name):
                self.assertEqual(job["uses"], "./.github/workflows/rebuild-full.yml")
                self.assertEqual(job["with"]["language"], f"{name}wiki" if name != "simple" else "simplewiki")

    def test_destination_resolution_reuses_existing_datasets(self):
        data = workflow("rebuild-full.yml")
        publish = data["jobs"]["publish"]
        script = next(step["run"] for step in publish["steps"]
                      if step.get("name") == "Resolve language and publishing destinations")
        cases = (
            ("simplewiki", "ffatty/plain-text-wikipedia-simpleenglish"),
            ("enwiki", "ffatty/plaintext-wikipedia-full-english"),
            ("frwiki", "ffatty/plaintext-wikipedia-full-french"),
        )
        for language, expected in cases:
            with self.subTest(language=language), tempfile.TemporaryDirectory() as tmp:
                env = {**os.environ, "LANGUAGE": language, "HF_REPO_INPUT": "",
                       "KAGGLE_DATASET_INPUT": "", "GITHUB_ENV": str(Path(tmp) / "env")}
                result = subprocess.run(["bash", "-e", "-c", script], cwd=ROOT,
                                        env=env, capture_output=True, text=True, check=True)
                settings = dict(line.split("=", 1) for line in Path(env["GITHUB_ENV"]).read_text().splitlines())
                self.assertEqual(settings["FULL_HF_REPO"], expected)
                self.assertEqual(settings["FULL_KAGGLE_DATASET"], expected)
                self.assertIn(expected, result.stdout)


if __name__ == "__main__":
    unittest.main()
