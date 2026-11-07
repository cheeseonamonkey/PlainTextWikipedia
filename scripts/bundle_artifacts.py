"""The platform-independent release contract, packaging and local verification."""
from __future__ import annotations

import errno
import os
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

from scripts.article_features import ARTICLE_FIELDS, VERSION
from scripts.article_parquet import schema

PAYLOAD = frozenset({'articles.parquet', 'articles.txt.xz', 'features.zip', 'models.zip'})
RELEASE_FILES = PAYLOAD | {'manifest.json'}
ARCHIVE_MEMBERS = {
    'features.zip': {'corpus-profile.json', 'vocabulary.json.gz', 'words.jsonl.gz'},
    'models.zip': {'fasttext.bin.zst', 'config.json'},
}


def file_metadata(path):
    """Both publisher digests in one sequential read, with bounded memory."""
    sha = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    with path.open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            sha.update(chunk)
            md5.update(chunk)
    return dict(bytes=path.stat().st_size, sha256=sha.hexdigest(), md5=md5.hexdigest())


def link_or_copy(source, destination):
    try:
        os.link(source, destination)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        shutil.copyfile(source, destination)


def archive(destination, files):
    """Store already-compressed members with stable metadata and ZIP64 support."""
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as out:
        for file in sorted(files, key=lambda p: p.name):
            info = zipfile.ZipInfo(file.name, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            with file.open('rb') as source, out.open(info, 'w', force_zip64=True) as target:
                shutil.copyfileobj(source, target, 8 * 1024 * 1024)


def read_manifest(folder):
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    if (not isinstance(manifest, dict) or type(manifest.get('version')) is not int or manifest['version'] != VERSION
            or manifest.get('complete') is not True
            or type(manifest.get('documents')) is not int or manifest['documents'] < 1
            or not isinstance(manifest.get('files'), dict) or set(manifest['files']) != PAYLOAD):
        raise ValueError('Incomplete release bundle')
    if (not isinstance(manifest.get('language'), str) or not re.fullmatch('[a-z0-9_-]+', manifest['language'])
            or not isinstance(manifest.get('source'), dict)
            or not isinstance(manifest.get('links'), dict)
            or set(manifest['links']) != {'github', 'huggingface', 'kaggle', 'zenodo'}
            or any(not isinstance(value, str) or not value.startswith('https://')
                   for value in manifest['links'].values())):
        raise ValueError('Invalid release identity/provenance')
    for name, meta in manifest['files'].items():
        if (not isinstance(meta, dict) or set(meta) != {'bytes', 'sha256', 'md5'} or type(meta.get('bytes')) is not int or meta['bytes'] < 1
                or not isinstance(meta.get('sha256'), str) or not re.fullmatch('[0-9a-f]{64}', meta['sha256'])
                or not isinstance(meta.get('md5'), str) or not re.fullmatch('[0-9a-f]{32}', meta['md5'])):
            raise ValueError(f'Invalid release metadata: {name}')
    return manifest


def verify_bundle(folder):
    manifest = read_manifest(folder)
    if {p.name for p in folder.iterdir()} != RELEASE_FILES:
        raise ValueError('Unexpected or missing release files')
    for name, meta in manifest['files'].items():
        file = folder / name
        if not file.is_file() or file.stat().st_size != meta['bytes'] or file_metadata(file) != meta:
            raise ValueError(f'Release checksum mismatch: {name}')
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(folder / 'articles.parquet')
    if parquet.metadata.num_rows != manifest['documents']:
        raise ValueError('Parquet row count differs from release manifest')
    if not parquet.schema_arrow.equals(schema(ARTICLE_FIELDS), check_metadata=True):
        raise ValueError('Unexpected article Parquet schema')
    for name, required in ARCHIVE_MEMBERS.items():
        with zipfile.ZipFile(folder / name) as package:
            members = package.namelist()
            if len(members) != len(required) or set(members) != required:
                raise ValueError(f'Unexpected archive members: {name}')
    return manifest


def source_names(manifest, language=None):
    """Validate complete root coverage before trusting the aggregate flag."""
    def invalid():
        raise ValueError('Incomplete source coverage or inconsistent build manifest')
    if not isinstance(manifest, dict) or manifest.get('complete') is not True:
        invalid()
    count = manifest.get('expected_shards')
    roots = manifest.get('roots')
    if type(count) is not int or count < 1 or not isinstance(roots, list) or len(roots) != count:
        invalid()
    if language and manifest.get('language') != language + 'wiki':
        invalid()
    names, indices = set(), set()
    for root in roots:
        if (not isinstance(root, dict) or root.get('complete') is not True
                or type(root.get('root_shard')) is not int
                or root['root_shard'] in indices
                or root.get('download_failures') or root.get('failed_fragments')):
            invalid()
        indices.add(root['root_shard'])
        files = root.get('files')
        if not isinstance(files, list) or not files:
            invalid()
        for name in files:
            if (not isinstance(name, str) or not re.fullmatch(r'[a-zA-Z0-9_-]+\.txt\.gz', name)
                    or name in names or (language and not name.startswith(language + 'wiki_'))):
                invalid()
            names.add(name)
    if indices != set(range(1, count + 1)):
        invalid()
    return names
