"""The platform-independent release contract, packaging and local verification.

This is the slim v8 contract: the release is ``articles.parquet`` +
``corpus-profile.json`` + ``manifest.json``. No ZIP archives are produced.
Source coverage verification accepts both the modern ``roots`` manifest format
and the legacy ``build_run`` format, and always cross-checks against the actual
staged file listing.
"""
from __future__ import annotations

import errno
import os
import hashlib
import json
import re
import shutil
from pathlib import Path

from scripts.article_parquet import ARTICLE_FIELDS, VERSION, schema

RELEASE_FILES = frozenset({'articles.parquet', 'corpus-profile.json', 'manifest.json'})

# Filename pattern for a source shard: {lang}wiki_NN_of_MM.txt.gz (optionally
# with a recovery a/b suffix or a post-conversion _partNN_of_MM split).
LEAF_RE = re.compile(r"^[a-z0-9_-]+_\d{2}(?:[ab]+)?_of_\d{2}(?:_part\d{2}_of_\d{2})?\.txt\.gz$")


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
        os.link(Path(source).resolve(strict=True), destination)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        shutil.copyfile(source, destination)


def read_manifest(folder):
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    if (not isinstance(manifest, dict) or type(manifest.get('version')) is not int
            or manifest['version'] != VERSION
            or type(manifest.get('documents')) is not int or manifest['documents'] < 1
            or not isinstance(manifest.get('files'), dict)
            or set(manifest['files']) != {'articles.parquet', 'corpus-profile.json'}):
        raise ValueError('Incomplete release bundle')
    if (not isinstance(manifest.get('language'), str) or not re.fullmatch('[a-z0-9_-]+', manifest['language'])):
        raise ValueError('Invalid release identity')
    for name, meta in manifest['files'].items():
        if (not isinstance(meta, dict) or set(meta) != {'bytes', 'sha256'} or type(meta.get('bytes')) is not int
                or meta['bytes'] < 1
                or not isinstance(meta.get('sha256'), str) or not re.fullmatch('[0-9a-f]{64}', meta['sha256'])):
            raise ValueError(f'Invalid release metadata: {name}')
    return manifest


def _schema_equal(actual, expected):
    """Compare two Arrow schemas, tolerating list-field child-name differences
    (pyarrow renames the child to 'element' on write in some versions).
    """
    import pyarrow as pa
    if actual.names != expected.names or len(actual) != len(expected):
        return False
    for fa, fe in zip(actual, expected):
        if fa.name != fe.name:
            return False
        ta, te = fa.type, fe.type
        if isinstance(ta, pa.ListType) and isinstance(te, pa.ListType):
            if ta.value_type != te.value_type:
                return False
        elif ta != te:
            return False
    return True


def verify_bundle(folder):
    manifest = read_manifest(folder)
    if {p.name for p in folder.iterdir()} != RELEASE_FILES:
        raise ValueError('Unexpected or missing release files')
    for name, meta in manifest['files'].items():
        file = folder / name
        actual = file_metadata(file)
        if (not file.is_file() or actual['bytes'] != meta['bytes']
                or actual['sha256'] != meta['sha256']):
            raise ValueError(f'Release checksum mismatch: {name}')
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(folder / 'articles.parquet')
    if parquet.metadata.num_rows != manifest['documents']:
        raise ValueError('Parquet row count differs from release manifest')
    if not _schema_equal(parquet.schema_arrow, schema(ARTICLE_FIELDS)):
        raise ValueError('Unexpected article Parquet schema')
    return manifest


def _manifest_roots(manifest: dict) -> list[dict] | None:
    roots = manifest.get('roots')
    if not isinstance(roots, list) or not roots:
        return None
    return roots


def validate_manifest(manifest: dict, language: str) -> tuple[str, int]:
    """Return (format, expected_shards) after validating a source manifest.

    Modern format: ``roots`` list with per-root completeness. Legacy format:
    global ``complete``/``expected_shards`` without per-root detail. Returns the
    format tag ('roots' or 'legacy') plus the expected shard count.
    """
    if not isinstance(manifest, dict):
        raise ValueError('Malformed source manifest')
    lang_ok = (manifest.get('language') == language + 'wiki')
    roots = _manifest_roots(manifest)
    if 'roots' in manifest and roots is None:
        raise ValueError('Malformed source roots')
    if roots is not None:
        if manifest.get('complete') is not True:
            raise ValueError('Incomplete source coverage')
        if not lang_ok:
            raise ValueError('Inconsistent build manifest language')
        count = manifest.get('expected_shards')
        if type(count) is not int or count < 1 or len(roots) != count:
            raise ValueError('Incomplete source coverage or inconsistent build manifest')
        names, indices = set(), set()
        for root in roots:
            if (not isinstance(root, dict) or root.get('complete') is not True
                    or type(root.get('root_shard')) is not int
                    or root['root_shard'] in indices
                    or root.get('download_failures') or root.get('failed_fragments')):
                raise ValueError('Incomplete source root')
            indices.add(root['root_shard'])
            files = root.get('files')
            if not isinstance(files, list) or not files:
                raise ValueError('Incomplete source root')
            for name in files:
                if (not isinstance(name, str) or not LEAF_RE.fullmatch(name)
                        or name in names or not name.startswith(language + 'wiki_')):
                    raise ValueError('Unexpected source archive name')
                match = re.fullmatch(rf'{re.escape(language)}wiki_(\d{{2}})[ab]*_of_(\d{{2}})(?:_part(\d{{2}})_of_(\d{{2}}))?\.txt\.gz', name)
                if (not match or int(match[1]) != root['root_shard']
                        or int(match[2]) != count
                        or (match[3] is not None and not 1 <= int(match[3]) <= int(match[4]))):
                    raise ValueError('Inconsistent source archive identity')
                names.add(name)
        if indices != set(range(1, count + 1)):
            raise ValueError('Source roots are not contiguous')
        return 'roots', count

    # Legacy format: no per-root detail.
    if not lang_ok or manifest.get('complete') is not True:
        raise ValueError('Incomplete source coverage (legacy manifest)')
    expected = manifest.get('expected_shards')
    published = manifest.get('published_shards')
    if type(expected) is not int or expected < 1:
        raise ValueError('Invalid expected shard count')
    if type(published) is not int or published != expected:
        raise ValueError('Unexpected published shard count')
    return 'legacy', expected


def verify_sources(folder: Path, language: str) -> tuple[str, int]:
    """Validate a staged source directory against its build-manifest.json.

    Cross-checks the actual file list against whatever the manifest declares,
    and returns (format, expected_shards). Hard-fails on missing manifest or
    real coverage gaps; accepts the legacy format with a weaker but honest
    global check.
    """
    manifest_path = folder / 'build-manifest.json'
    if not manifest_path.is_file():
        raise ValueError('Missing source build-manifest.json; rebuild from Wikimedia dumps')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    fmt, expected = validate_manifest(manifest, language)

    staged = sorted(p.name for p in folder.iterdir() if p.name.endswith('.txt.gz'))
    pattern = re.compile(rf'{re.escape(language)}wiki_(\d{{2}})[ab]*_of_(\d{{2}})(?:_part(\d{{2}})_of_(\d{{2}}))?\.txt\.gz')
    indices = set()
    for name in staged:
        match = pattern.fullmatch(name)
        if (not match or not (folder / name).is_file()
                or int(match[2]) != expected or not 1 <= int(match[1]) <= expected
                or (match[3] is not None and not 1 <= int(match[3]) <= int(match[4]))):
            raise ValueError(f'Unexpected source archive: {name}')
        indices.add(int(match[1]))
    if indices != set(range(1, expected + 1)):
        raise ValueError('Missing source roots')
    if not staged:
        raise ValueError('No source archives staged')

    if fmt == 'roots':
        declared = {name for root in manifest['roots'] for name in root['files']}
        if set(declared) != set(staged):
            raise ValueError('Staged sources differ from the complete build manifest')
    else:
        # An old aggregate report cannot prove recovery fragments are complete.
        # Accept only exactly one unsplit archive per root, never a partial split.
        required = {f'{language}wiki_{i:02d}_of_{expected:02d}.txt.gz'
                    for i in range(1, expected + 1)}
        if set(staged) != required:
            raise ValueError('Legacy source fragments cannot establish complete coverage')
        if manifest.get('missing_shards'):
            raise ValueError('Legacy manifest reports missing shards')
        if 'present_shards' in manifest and manifest['present_shards'] != list(range(1, expected + 1)):
            raise ValueError('Legacy manifest shard inventory is inconsistent')
    return fmt, expected
