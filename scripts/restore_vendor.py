#!/usr/bin/env python3
"""Reconstruct and verify large vendored files after checkout."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'vendor/chunks.json').read_text())
    for name, info in manifest.items():
        target = ROOT / name
        if target.is_file():
            with target.open('rb') as existing:
                if hashlib.file_digest(existing, 'sha256').hexdigest() == info['sha256']:
                    continue
        temporary = target.with_name(target.name + '.tmp')
        digest = hashlib.sha256()
        try:
            with temporary.open('wb') as output:
                for index in range(info['parts']):
                    part = target.with_name(target.name + f'.part{index:02d}')
                    with part.open('rb') as source:
                        while data := source.read(8 * 1024 * 1024):
                            output.write(data)
                            digest.update(data)
            if digest.hexdigest() != info['sha256']:
                raise ValueError(f'Vendored dependency checksum mismatch: {name}')
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
