"""Audit distributable names, package imports and bundled source integrity."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from verify_runtime import verify

ROOT=Path(__file__).resolve().parents[1]
GENERATED={'__pycache__','.cache','.pytest_cache','.git','.venv'}
BAD_NAMES=('euro'+'sys','ml'+'sys')
BAD_METHODS=('cache'+'blend','cache'+'slide','prophet'+'kv')
BAD_PATHS=('/new'+'ssd/','/ho'+'me/','/ro'+'ot/')
BINARY={'.so','.o','.pyc','.pt','.pth','.safetensors','.tar','.gz','.zip','.png','.pdf'}


def module_exists(parts):
    path=ROOT.joinpath(*parts)
    return path.with_suffix('.py').is_file() or (path/'__init__.py').is_file()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write',action='store_true');args=parser.parse_args()
    files=[];directories=['.'];errors=[]
    try:runtime=verify(ROOT)
    except (ValueError,KeyError,OSError,SyntaxError) as exc:
        runtime={'passed':False};errors.append('bundled runtime: '+str(exc))
    manifest=ROOT/'vendor/vllm/source_manifest.json'
    expected={row['path']:row for row in json.loads(manifest.read_text())['files']} if manifest.is_file() else {}
    for path in sorted(ROOT.rglob('*')):
        rel=path.relative_to(ROOT)
        if rel==Path('audit.json') or any(part in GENERATED for part in rel.parts):continue
        if rel.parts[0] in {'models','data','results'}:continue
        bundled=rel.parts[:2]==('vendor','vllm')
        if bundled and len(rel.parts)>2 and rel.parts[2] in {'.deps','build','dist','.eggs'}:continue
        if any(word in str(rel).lower() for word in BAD_NAMES+BAD_METHODS):errors.append(str(rel)+': name')
        if path.is_symlink():errors.append(str(rel)+': symlink')
        if path.is_dir():directories.append(str(rel))
        if not path.is_file():continue
        data=path.read_bytes();project=True
        if bundled:
            name=path.relative_to(ROOT/'vendor/vllm').as_posix()
            if name!='source_manifest.json' and name not in expected:errors.append(str(rel)+': unexpected bundled file')
            project=name in expected and expected[name]['kind'] in {'keychunk_overlay','keychunk_model'}
        elif path.suffix in BINARY:errors.append(str(rel)+': generated/binary artifact')
        if project:
            source=data.decode('utf-8')
            if any(word in source.lower() for word in BAD_NAMES+BAD_METHODS) or any(word in source for word in BAD_PATHS):
                errors.append(str(rel)+': embedded legacy identity/path')
            if path.suffix=='.py':
                tree=ast.parse(source,filename=str(rel))
                for node in ast.walk(tree):
                    names=[a.name for a in node.names] if isinstance(node,ast.Import) else []
                    if isinstance(node,ast.ImportFrom) and node.module:
                        if node.level and rel.parts[0]=='keychunk':
                            package=list(rel.parts[:-1]);parts=package[:len(package)-node.level+1]+node.module.split('.')
                            if not module_exists(parts):errors.append(str(rel)+': unresolved relative import '+'.'.join(parts))
                        elif not node.level:names.append(node.module)
                    for name in names:
                        if name.startswith('keychunk.') and not module_exists(name.split('.')):
                            errors.append(str(rel)+': unresolved local import '+name)
        files.append(dict(path=str(rel),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    report=dict(passed=not errors,directories=len(directories),files=len(files),bytes=sum(row['bytes'] for row in files),
        errors=errors,api='keychunk.KeyChunk',metrics=['ttft_s','f1'],default_recompute_chunks=6,
        bundled_runtime=runtime,directory_inventory=directories,inventory=files)
    if args.write:(ROOT/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'inventory','directory_inventory'}},indent=2))
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
