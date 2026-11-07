"""All active releases share one build implementation; no automatic publishers."""
import unittest
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]/'.github/workflows'
def workflow(name):
    return yaml.load((ROOT/name).read_text(),Loader=yaml.BaseLoader)

class LanguagePipelineTests(unittest.TestCase):
    def test_one_language_builder_for_fleet_and_rebuilds(self):
        fleet=workflow('article-features-fleet.yml')
        rebuilt=workflow('publish-wikipedia.yml')
        for job in (fleet['jobs']['build'], rebuilt['jobs']['publish']):
            self.assertEqual(job['uses'],'./.github/workflows/article-features-one.yml')
        single=workflow('article-features-one.yml')
        self.assertEqual(single['on']['workflow_dispatch']['inputs']['publish']['default'],'false')
        self.assertEqual(single['on']['workflow_dispatch']['inputs']['repo_id']['default'],
                         'ffatty/plaintext-wikipedia-full-english')
        self.assertEqual(single['on']['workflow_call']['inputs']['publish']['default'],'false')
        self.assertNotIn('mirror_kaggle',str(single))
        script=str(single['jobs']['bundle']['steps'])
        self.assertIn('-m scripts.wikipedia_bundle',script)
        self.assertNotIn('--max-mb',script)
        self.assertIn('needs.build.result',workflow('rebuild-full.yml')['jobs']['publish']['if'])

    def test_no_automatic_data_mutations_and_no_missing_reusable_workflows(self):
        for file in ROOT.glob('*.yml'):
            data=workflow(file.name)
            if file.name not in ('test.yml','validate-release-workflows.yml'):
                self.assertLessEqual(set(data['on']),{'workflow_dispatch','workflow_call'})
            for job in data['jobs'].values():
                uses=job.get('uses','')
                if uses.startswith('./.github/workflows/'):
                    self.assertTrue((ROOT/Path(uses).name).exists(),uses)

    def test_catalog_is_the_only_language_destination_mapping(self):
        from scripts.wiki_metadata import project_repo, repository_codes
        for repo, code in repository_codes().items():
            self.assertEqual(project_repo(code + 'wiki'), repo)
        self.assertNotIn('case "$code"', (ROOT/'publish-wikipedia.yml').read_text())
