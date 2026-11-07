"""Actual tiny-corpus build; only the network Bekko encoder is substituted."""
import gzip
import json
import lzma
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq
from scripts.article_features import build, tokens
from scripts.wikipedia_bundle import make_bundle, verify_bundle, PAYLOAD, links, complete_sources


class FakeBekko:
    def encode(self, texts):
        vectors = np.zeros((len(texts), 128), dtype=np.float32)
        vectors[:, 0] = 1.
        return vectors, [12] * len(texts)


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.docs = [('Orbit', 'Planets orbit stars. Stars orbit galaxies.'),
                     ('日本', '日本語の文章です。星と惑星。'),
                     ('Last', 'Uncommon tailword survives the entire pipeline.')]
        self.files = []
        for i, doc in enumerate(self.docs):
            file = self.root / f'wiki_{i}.txt.gz'
            with gzip.open(file, 'wt', encoding='utf-8') as out:
                out.write(f'{doc[0]}\n{doc[1]}\n\n')
            self.files.append(file)

    def test_one_parquet_one_text_independent_vectors_full_word_statistics(self):
        dest = self.root / 'out'
        with patch('scripts.bekko_features.BekkoEncoder', FakeBekko):
            report = build(self.files, dest, 'fr', vocab_size=2, threads=1)
        table = pq.read_table(dest / 'articles.parquet')
        self.assertEqual(table.num_rows, 3)
        self.assertEqual(table.num_columns, 32)
        for excluded in ('source_file','source_ordinal','token_count','word_count','byte_count',
                         'embedding_200','contradiction_score','vader_compound','saliency'):
            self.assertNotIn(excluded, table.column_names)
        rows = table.to_pylist()
        self.assertEqual([(r['title'],r['text']) for r in rows], self.docs)
        with lzma.open(dest / 'articles.txt.xz', 'rt', encoding='utf-8') as inp:
            self.assertEqual(inp.read(), ''.join(f'{t}\n{b}\n\n' for t,b in self.docs))
        for row in rows:
            self.assertEqual(row['char_count'], len(row['text']))
            self.assertEqual(len(json.loads(row['bekko'])), 128)
            self.assertEqual(len(json.loads(row['fasttext'])), 160)
            self.assertTrue(row['article_url'].startswith('https://fr.wikipedia.org/'))
        self.assertIn('tailword',json.loads(rows[-1]['word_counts']))
        with gzip.open(dest / 'words.jsonl.gz','rt') as stream:
            words = {r['term']:r for r in map(json.loads,stream)}
        self.assertEqual(words['tailword']['tf'],1)
        self.assertEqual(words['stars']['df'],1)
        self.assertGreater(len(words), report['vocabulary_size'])
        self.assertGreater(report['fasttext']['training_tokens'], 0)
        self.assertLess(report['fasttext']['training_tokens'],sum(len(tokens(b)) for _,b in self.docs))
        self.assertFalse(any('sqlite' in p.name for p in dest.iterdir()))
        import fasttext
        import zstandard as zstd
        model_path = self.root / 'trained.bin'
        with (dest / 'fasttext.bin.zst').open('rb') as source, model_path.open('wb') as target:
            zstd.ZstdDecompressor().copy_stream(source, target)
        with gzip.open(dest / 'vocabulary.json.gz', 'rt') as source:
            vocabulary_terms = {item['term'] for item in json.load(source)['terms']}
        model = fasttext.load_model(str(model_path))
        self.assertEqual(set(model.get_words()) - {'</s>'}, vocabulary_terms)


    def test_bundle_integrity_and_packaging(self):
        import zipfile
        output = self.root / 'bundle'
        with patch('scripts.bekko_features.BekkoEncoder', FakeBekko):
            manifest = make_bundle(self.files,output,'simple',links('owner/repo'),{'fixture':True})
        self.assertEqual(set(manifest['files']),PAYLOAD)
        with zipfile.ZipFile(output/'features.zip') as z:
            self.assertEqual(set(z.namelist()),{'corpus-profile.json','vocabulary.json.gz','words.jsonl.gz','normalized-words.jsonl.gz'})
        with zipfile.ZipFile(output/'models.zip') as z:
            self.assertEqual(set(z.namelist()),{'fasttext.bin.zst','config.json'})
        self.assertEqual(verify_bundle(output),manifest)
        with (output/'articles.txt.xz').open('ab') as f:
            f.write(b'bad')
        with self.assertRaisesRegex(ValueError,'checksum'):
            verify_bundle(output)

    def test_incomplete_or_missing_sources_are_rejected(self):
        p = self.root/'build-manifest.json'
        p.write_text(json.dumps({'complete':False}))
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            complete_sources(self.root)
        p.write_text(json.dumps({'complete':True,'expected_shards':1,'roots':[{'root_shard':1,'complete':True,'files':[self.files[0].name]}]}))
        with self.assertRaisesRegex(ValueError,'differ'):
            complete_sources(self.root)

class PublicationContractTests(unittest.TestCase):
    setUp = BundleTests.setUp
    def test_invalid_bundle_reaches_no_publisher(self):
        from scripts.wikipedia_bundle import publish_all
        with patch('scripts.wikipedia_bundle.publish_hf') as hf, patch('scripts.wikipedia_bundle.publish_kaggle') as kg, patch('scripts.wikipedia_bundle.zenodo_publish') as zn:
            with self.assertRaises(FileNotFoundError):
                publish_all(None,'repo',self.root,123)
            hf.assert_not_called(); kg.assert_not_called(); zn.assert_not_called()

    def test_all_publishers_consume_same_verified_local_bundle(self):
        import os
        from types import SimpleNamespace
        from scripts.wikipedia_bundle import publish_all, sha256
        output=self.root/'release'
        with patch('scripts.bekko_features.BekkoEncoder',FakeBekko):
            make_bundle(self.files,output,'fr',links('ffatty/plaintext-wikipedia-full-french','https://zenodo.org/records/123'),{'fixture':True})
        with patch.dict(os.environ,{'HF_TOKEN':'test','KAGGLE_TOKEN':'test','ZENODO_TOKEN':'test'}), patch('scripts.wikipedia_bundle.publish_hf') as hf, patch('scripts.wikipedia_bundle.publish_kaggle') as kg, patch('scripts.wikipedia_bundle.zenodo_publish') as zn:
            api=SimpleNamespace(repo_info=lambda *a,**k:SimpleNamespace(sha='fixed'))
            publish_all(api,'ffatty/plaintext-wikipedia-full-french',output,123)
            self.assertEqual(hf.call_args.args[2],output)
            self.assertEqual(kg.call_args.args[1],output)
            self.assertEqual(zn.call_args.kwargs['local_folder'],output)
            assets=zn.call_args.args[2]
            self.assertEqual({a.path for a in assets},PAYLOAD|{'manifest.json'})
            self.assertTrue(all(a.sha256==sha256(output/a.path) for a in assets))

    def test_canonical_xz_can_be_rebuilt_without_old_gzip_archives(self):
        source=self.root/'source'
        with patch('scripts.bekko_features.BekkoEncoder',FakeBekko):
            build(self.files,source,'zh',threads=1)
            build([source/'articles.txt.xz'],self.root/'again','zh',threads=1)
        self.assertEqual(pq.read_table(source/'articles.parquet').to_pylist(),pq.read_table(self.root/'again/articles.parquet').to_pylist())
