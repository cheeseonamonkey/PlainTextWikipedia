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
        self.assertEqual(all_languages["jobs"]["rebuild"]["uses"],
                         "./.github/workflows/rebuild-full.yml")
        self.assertEqual(all_languages["jobs"]["rebuild"]["with"]["language"],
                         "${{ matrix.language }}")
        self.assertEqual(all_languages["jobs"]["rebuild"]["strategy"]["fail-fast"], "false")
        from scripts.plan_wikipedia_languages import languages
        self.assertEqual(languages("simplewiki,enwiki"), ["simplewiki", "enwiki"])
        self.assertEqual(languages("simple,en"), ["simplewiki", "enwiki"])
        self.assertGreaterEqual(len(languages()), 210)
        self.assertEqual(len(languages()), len(set(languages())))

    def test_uniform_sharding_and_recursive_recovery(self):
        data = workflow("rebuild-full.yml")["jobs"]
        script = next(step["run"] for step in data["plan"]["steps"]
                      if step.get("id") == "plan")
        self.assertIn('max_shards="$WIKIPEDIA_SHARD_TARGET"', script)
        self.assertNotIn("simplewiki", script)
        self.assertEqual(data["build"]["strategy"]["max-parallel"], "8")
        self.assertIn("--max-depth 3", str(data["build"]))
        self.assertIn("overwrite", str(data["build"]))
        self.assertEqual(data["publish"]["uses"], "./.github/workflows/publish-wikipedia.yml")
        self.assertEqual(data["publish"]["secrets"], "inherit")
        publisher = workflow("publish-wikipedia.yml")
        self.assertIn("workflow_dispatch", publisher["on"])
        self.assertIn("workflow_call", publisher["on"])
        script = str(publisher["jobs"]["publish"]["steps"])
        self.assertIn('--target "$WIKIPEDIA_SHARD_TARGET"', script)
        self.assertIn('inputs.source_run_id || github.run_id', script)
        self.assertIn("-m scripts.publish_wikipedia", script)
        self.assertEqual(publisher["jobs"]["zenodo"]["uses"],
                         "./.github/workflows/structured-wikipedia.yml")
        self.assertEqual(publisher["jobs"]["zenodo"]["with"]["mirror_kaggle"], "false")
        self.assertEqual(publisher["jobs"]["zenodo"]["with"]["publish_zenodo"], "true")
        self.assertIn("hf_success == 'true'", publisher["jobs"]["zenodo"]["if"])
        self.assertEqual(publisher["jobs"]["publish"]["outputs"]["publishers_ok"],
                         "${{ steps.publisher.outputs.publishers_ok }}")
        self.assertEqual(publisher["jobs"]["report"]["needs"], ["publish", "zenodo"])
        self.assertNotIn('!= simplewiki', script)

    def test_unlisted_wikipedia_uses_generic_metadata_in_same_pipeline(self):
        from scripts.language_info import metadata
        self.assertEqual(metadata("zz_customwiki"), ("zz_custom", "zz-custom"))
        with self.assertRaises(ValueError):
            metadata("../badwiki")

    def test_destination_resolution_reuses_existing_datasets(self):
        data = workflow("publish-wikipedia.yml")
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
                       "KAGGLE_DATASET_INPUT": "", "GITHUB_ENV": str(Path(tmp) / "env"),
                       "GITHUB_OUTPUT": str(Path(tmp) / "output")}
                result = subprocess.run(["bash", "-e", "-c", script], cwd=ROOT,
                                        env=env, capture_output=True, text=True, check=True)
                settings = dict(line.split("=", 1) for line in Path(env["GITHUB_ENV"]).read_text().splitlines())
                self.assertEqual(settings["FULL_HF_REPO"], expected)
                self.assertEqual(settings["FULL_KAGGLE_DATASET"], expected)
                self.assertIn(expected, result.stdout)


if __name__ == "__main__":
    unittest.main()
