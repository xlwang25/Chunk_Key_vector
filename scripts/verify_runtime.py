"""Verify the bundled official vLLM snapshot and pre-applied KeyChunk files."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '1da94e673c257373280026f75ceb4effac80e892'
VERSION = '0.10.1.1'


def read_version_tuple(path):
    # Read generated metadata without executing it.
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        else:
            continue
        if '__version_tuple__' in names:
            return ast.literal_eval(node.value)
    raise ValueError('generated vLLM version tuple is missing')


def verify(root=ROOT):
    target = root/'vendor/vllm'
    manifest_path = target/'source_manifest.json'
    if not manifest_path.is_file():
        raise ValueError('bundled vLLM source manifest is missing')
    manifest = json.loads(manifest_path.read_text())
    if manifest['commit'] != COMMIT or manifest['version'] != VERSION:
        raise ValueError('bundled vLLM revision differs from the bundled revision')
    expected = {row['path']:row for row in manifest['files']}
    if len(expected) != len(manifest['files']):
        raise ValueError('duplicate source manifest path')
    required = {'setup.py', 'CMakeLists.txt', 'LICENSE', 'vllm/__init__.py'}
    if not required.issubset(expected):
        raise ValueError('incomplete bundled vLLM source manifest')
    for name,row in expected.items():
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('invalid source manifest path')
        path = target/rel
        if not path.is_file() or path.is_symlink():
            raise ValueError('missing or nonregular bundled source: '+name)
        if row['kind'] != 'version_metadata':
            if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('bundled source changed: '+name)
        else:
            # setuptools-scm rewrites this generated module during installation.
            if tuple(read_version_tuple(path)[:4]) != (0,10,1,1):
                raise ValueError('generated vLLM version differs from the bundled revision')
    for patch in (root/'patches/overlay').rglob('*'):
        if not patch.is_file():
            continue
        rel = patch.relative_to(root/'patches/overlay')
        if (target/rel).read_bytes() != patch.read_bytes():
            raise ValueError('KeyChunk overlay differs from bundled runtime: '+str(rel))
    model = root/'patches/models/llama_keychunk.py'
    if (target/'vllm/model_executor/models/llama.py').read_bytes() != model.read_bytes():
        raise ValueError('KeyChunk model hook differs from bundled runtime')
    return dict(passed=True,version=VERSION,commit=COMMIT,
        files=len(expected),bytes=sum((target/name).stat().st_size for name in expected),
        upstream_files=sum(row['kind']=='upstream' for row in expected.values()),
        modified_or_added_files=sum(row['kind']!='upstream' for row in expected.values()))


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(verify(),indent=2))


if __name__=='__main__':main()
