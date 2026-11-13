"""All active releases share one build workflow; publishers are independent."""
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1] / '.github/workflows'


def workflow(name):
    return yaml.load((ROOT / name).read_text(), Loader=yaml.BaseLoader)


class LanguagePipelineTests(unittest.TestCase):
    def test_single_language_workflow_is_dispatchable(self):
        single = workflow('wikipedia.yml')
        triggers = single['on']
        self.assertIn('workflow_dispatch', triggers)
        self.assertIn('workflow_call', triggers)
        inputs = triggers['workflow_dispatch']['inputs']
        self.assertEqual(inputs['publish']['default'], 'true')
        self.assertEqual(inputs['rebuild']['default'], 'false')
        self.assertEqual(inputs['language']['required'], 'true')
        # Feature knobs are exposed with v8 defaults.
        self.assertEqual(inputs['vocab']['default'], '12000')
        self.assertEqual(inputs['dim']['default'], '180')
        self.assertEqual(inputs['window']['default'], '8')
        self.assertEqual(inputs['epoch']['default'], '1')
        script = str(single['jobs']['corpus-stats']['steps'])
        self.assertIn('-m scripts.build_language', script)
        self.assertNotIn('wikipedia_bundle', script)

    def test_build_is_split_into_sequential_stage_jobs(self):
        jobs = workflow('wikipedia.yml')['jobs']
        self.assertNotIn('build', jobs)
        self.assertEqual(jobs['corpus-stats']['needs'], ['resolve', 'stage'])
        self.assertEqual(jobs['fasttext-training']['needs'], 'corpus-stats')
        self.assertEqual(jobs['enrich-parquet']['needs'], ['resolve', 'fasttext-training'])
        for name in ('corpus-stats', 'fasttext-training', 'enrich-parquet'):
            self.assertEqual(jobs[name]['timeout-minutes'], '360', name)
            self.assertIn('--stage ' + name.replace('_', '-'), str(jobs[name]['steps']))
        self.assertIn('--work work', str(jobs['corpus-stats']['steps']))
        self.assertIn('--work work', str(jobs['fasttext-training']['steps']))
        self.assertIn('--release release', str(jobs['enrich-parquet']['steps']))
        # Each stage hands the next one an explicit intermediate artifact.
        self.assertIn('features-${{ needs.resolve.outputs.code }}-corpus',
                      str(jobs['corpus-stats']['steps']))
        self.assertIn('features-${{ needs.corpus-stats.outputs.code }}-corpus',
                      str(jobs['fasttext-training']['steps']))
        self.assertIn('features-${{ needs.corpus-stats.outputs.code }}-model',
                      str(jobs['fasttext-training']['steps']))
        self.assertIn('wikipedia-${{ needs.resolve.outputs.code }}',
                      str(jobs['enrich-parquet']['steps']))

    def test_sources_are_staged_before_build_for_both_modes(self):
        jobs = workflow('wikipedia.yml')['jobs']
        stage, build = jobs['stage'], jobs['corpus-stats']
        self.assertEqual(stage['needs'], ['resolve', 'rebuild'])
        self.assertIn('needs.rebuild.result', stage['if'])
        self.assertIn('--stage-only sources', str(stage['steps']))
        self.assertIn('full-${{ needs.resolve.outputs.project }}-*', str(stage['steps']))
        self.assertIn('staged-sources-${{ needs.resolve.outputs.code }}', str(stage['steps']))
        self.assertEqual(build['needs'], ['resolve', 'stage'])
        self.assertIn("needs.stage.result == 'success'", build['if'])
        self.assertIn('staged-sources-${{ needs.resolve.outputs.code }}', str(build['steps']))
        self.assertIn('--sources sources', str(build['steps']))
        self.assertIn('sources/build-manifest.json', str(jobs['enrich-parquet']['steps']))

    def test_publishers_are_parallel_independent_jobs(self):
        single = workflow('wikipedia.yml')
        for name in ('publish-hf', 'publish-kaggle', 'publish-zenodo'):
            job = single['jobs'][name]
            self.assertEqual(job['needs'], 'enrich-parquet')
            self.assertEqual(
                job['if'],
                "${{ !cancelled() && inputs.publish && needs.enrich-parquet.result == 'success' }}")
            script = str(job['steps'])
            self.assertIn(f'-m scripts.publish_{name.split("-", 1)[1]}', script)
        report = single['jobs']['report']
        self.assertEqual(report['if'], 'always() && inputs.publish')
        self.assertIn('publish-hf', str(report['needs']))

    def test_rebuild_sources_carry_a_verified_manifest_to_publish(self):
        jobs = workflow('wikipedia.yml')['jobs']
        # corpus-stats regenerates build-manifest.json for fresh rebuilds, so it
        # must hand it forward: the enrich stage re-uploads sources/ for HF.
        self.assertIn('source-manifest-${{ needs.resolve.outputs.code }}',
                      str(jobs['corpus-stats']['steps']))
        enrich = str(jobs['enrich-parquet']['steps'])
        self.assertIn('source-manifest-${{ needs.resolve.outputs.code }}', enrich)
        self.assertIn('sources/build-manifest.json', enrich)

    def test_rebuild_path_is_conditional_and_uses_converter_stack(self):
        single = workflow('wikipedia.yml')
        self.assertEqual(single['jobs']['rebuild']['if'],
                         "needs.resolve.outputs.mode == 'rebuild'")
        self.assertEqual(single['jobs']['rebuild']['uses'],
                         './.github/workflows/rebuild-shards.yml')
        shards = workflow('rebuild-shards.yml')
        script = str(shards['jobs']['build']['steps'])
        self.assertIn('convert_adaptive.py', script)
        self.assertIn('multistream_shards.py', script)

    def test_no_automatic_data_mutations_and_no_missing_reusable_workflows(self):
        for file in ROOT.glob('*.yml'):
            data = workflow(file.name)
            if file.name != 'test.yml':
                self.assertLessEqual(set(data['on']), {'workflow_dispatch', 'workflow_call'})
            for job in data['jobs'].values():
                uses = job.get('uses', '')
                if uses.startswith('./.github/workflows/'):
                    self.assertTrue((ROOT / Path(uses).name).exists(), uses)

    def test_other_workflows_stay_build_only(self):
        # publish defaults to true only for the single-language workflow;
        # every other workflow must stay build-only.
        for file in ROOT.glob('*.yml'):
            events = workflow(file.name).get('on', {})
            for trigger in ('workflow_dispatch', 'workflow_call'):
                settings = (events.get(trigger) or {}).get('inputs', {}).get('publish')
                if settings is not None and file.name != 'wikipedia.yml':
                    self.assertEqual(settings.get('default'), 'false', (file.name, trigger))


if __name__ == '__main__':
    unittest.main()
