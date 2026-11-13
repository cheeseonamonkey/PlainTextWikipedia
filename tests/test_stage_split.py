"""The build splits into three explicit stages that must each stand alone.

Each stage reads only its declared inputs and writes only its declared outputs,
so a stage can run in its own process (and its own CI job). These tests prove
the split is behaviour-preserving: the same corpus produces the same verified
bundle whether it is built in one process or one stage at a time.
"""
import gzip
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.article_features import (
    CANDIDATE_NAME,
    CORPUS_STATS_DIR,
    ENRICH_DIR,
    INDEX_NAME,
    MODEL_NAME,
    SPILL_NAME,
    STAGES,
    STATS_NAME,
    TRAINING_DIR,
    TRAINING_META_NAME,
    build,
    stage_corpus_stats,
    stage_enrich_parquet,
    stage_fasttext_training,
)
from scripts.article_parquet import ARTICLE_FIELDS
from scripts.bundle_artifacts import verify_bundle

DOCUMENTS = [
    ('Stars', 'Stars shine brightly across the night sky.'),
    ('Oceans', 'Oceans cover most of the planet surface.'),
    ('Trees', 'Trees grow leaves in dense forests.'),
]


def make_sources(root: Path) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'enwiki_01_of_01.txt.gz'
    with gzip.open(path, 'wt', encoding='utf-8') as stream:
        for title, body in DOCUMENTS:
            stream.write(f'{title}\n{body}\n\n')
    return [path]


class StageSplitTests(unittest.TestCase):
    def test_each_stage_writes_only_its_declared_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = make_sources(root / 'src')
            work = root / 'work'
            stage_corpus_stats(files, work, 'en', vocab_size=500)
            corpus = sorted(p.name for p in (work / CORPUS_STATS_DIR).iterdir())
            self.assertEqual(corpus, sorted([INDEX_NAME, SPILL_NAME, CANDIDATE_NAME, STATS_NAME]))
            self.assertEqual([p.name for p in (work / TRAINING_DIR).iterdir()] if
                             (work / TRAINING_DIR).exists() else [], [])
            stage_fasttext_training(work, dim=16, window=4, epoch=1, threads=1)
            training = sorted(p.name for p in (work / TRAINING_DIR).iterdir())
            self.assertEqual(training, sorted([MODEL_NAME, TRAINING_META_NAME]))
            stats = json.loads((work / CORPUS_STATS_DIR / STATS_NAME).read_text())
            self.assertTrue(stats['vocabulary'])
            self.assertEqual(stats['documents'], len(DOCUMENTS))
            self.assertEqual(stats['language'], 'en')
            meta = json.loads((work / TRAINING_DIR / TRAINING_META_NAME).read_text())
            self.assertEqual((meta['dim'], meta['window'], meta['epoch']), (16, 4, 1))

    def test_stages_in_separate_processes_reproduce_the_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = make_sources(root / 'src')
            work = root / 'work'
            release = root / 'release'
            command = [sys.executable, '-m', 'scripts.article_features']
            common = ['--language', 'en', '--dim', '16', '--window', '4',
                      '--threads', '1', '--vocab', '500']
            stages = [
                command + ['--stage', CORPUS_STATS_DIR, *common, '--work', str(work)]
                + [str(p) for p in files],
                command + ['--stage', TRAINING_DIR, *common, '--work', str(work)],
                command + ['--stage', ENRICH_DIR, '--language', 'en',
                           '--work', str(work), '--release', str(release)],
            ]
            for index, args in enumerate(stages, start=1):
                done = subprocess.run(args, capture_output=True, text=True, check=False)
                self.assertEqual(done.returncode, 0, done.stderr[-2000:])
                self.assertEqual(len(STAGES), 3)
                self.assertEqual(index, STAGES.index(args[4]) + 1)
            manifest = verify_bundle(release)
            self.assertEqual(manifest['documents'], len(DOCUMENTS))
            self.assertEqual(set(manifest['files']), {'articles.parquet', 'corpus-profile.json'})

    def test_split_output_matches_the_single_process_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = make_sources(root / 'src')
            work = root / 'work'
            split = root / 'split'
            whole = root / 'whole'
            stage_corpus_stats(files, work, 'en', vocab_size=500)
            stage_fasttext_training(work, dim=16, window=4, epoch=1, threads=1)
            stage_enrich_parquet(work, split, 'en')
            build(files, whole, 'en', vocab_size=500, dim=16, window=4, epoch=1, threads=1)
            verify_bundle(split)
            verify_bundle(whole)
            for name in ('manifest.json', 'corpus-profile.json'):
                self.assertEqual((split / name).read_bytes(), (whole / name).read_bytes(), name)

    def test_each_stage_refuses_to_run_without_its_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / 'work'
            with self.assertRaises(ValueError):
                stage_fasttext_training(work, dim=16, window=4, epoch=1, threads=1)
            with self.assertRaises(ValueError):
                stage_enrich_parquet(work, Path(tmp) / 'release', 'en')

    def test_enrich_stage_publishes_the_full_article_schema(self):
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = make_sources(root / 'src')
            work = root / 'work'
            release = root / 'release'
            stage_corpus_stats(files, work, 'en', vocab_size=500)
            stage_fasttext_training(work, dim=16, window=4, epoch=1, threads=1)
            stage_enrich_parquet(work, release, 'en')
            table = pq.read_table(release / 'articles.parquet')
            self.assertEqual(table.schema.names, ['text', *ARTICLE_FIELDS])
            rows = table.to_pylist()
            self.assertEqual([r['title'] for r in rows], [d[0] for d in DOCUMENTS])
            for row in rows:
                self.assertEqual(len(row['fasttext']), 16)
                json.loads(row['counts'])
                json.loads(row['tfidf'])


if __name__ == '__main__':
    unittest.main()
