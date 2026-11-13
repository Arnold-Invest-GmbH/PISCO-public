"""Build a native Linux prototype with an included Python runtime.

No publication, installation into system Python, or numerical computation.
All runtime components are copied, then individually hashed. The result stays
an unvalidated candidate until the separate acceptance checks have passed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
R2 = HERE.parent
sys.path.insert(0, str(HERE))
from native_tools import collect
from build_toolchain import ensure, URL as MUSL_URL, SHA256 as MUSL_SHA256


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''): h.update(b)
    return h.hexdigest()


def copy_source(out):
    selected = [R2 / 'code' / n for n in ('solvers.py', 'reproduce_all.py', 'requirements.txt', 'README.md')]
    selected += [p for d in ('code/tests', 'code/paper_support', 'portable', 'verification')
                 for p in (R2 / d).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    selected += [R2 / 'container/requirements.lock']
    allowed = set(json.loads((HERE / 'public-source-files.json').read_text()))
    actual = {p.relative_to(R2).as_posix() for p in selected}
    if actual != allowed:
        raise RuntimeError('Public source allowlist differs: ' + str(sorted(actual ^ allowed)))
    for p in selected:
        dest = out / 'source' / p.relative_to(R2); dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dest)
    licence = next((parent / 'LICENSE' for parent in (R2, *R2.parents)
                    if (parent / 'LICENSE').is_file()), None)
    if licence is None: raise RuntimeError('Repository LICENSE is missing.')
    shutil.copy2(licence, out / 'LICENSE')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--native-root', type=Path, required=True)
    p.add_argument('--prepared-runtime', type=Path)
    p.add_argument('--results-zip', type=Path)
    p.add_argument('--results-release', type=Path)
    args = p.parse_args()
    out = args.output.resolve(); cache = args.cache.resolve(); cache.mkdir(parents=True, exist_ok=True)
    if sys.platform != 'linux' or platform.machine() != 'x86_64':
        p.error('This builder currently validates Linux x86-64 only. Other build recipes must pass their own tests.')
    if out.exists() and any(out.iterdir()): p.error('Output must be empty/new.')
    out.mkdir(parents=True, exist_ok=True)
    runtime_lock = json.loads((HERE / 'runtimes.lock.json').read_text())['x86_64-unknown-linux-gnu']
    if args.prepared_runtime:
        shutil.copytree(args.prepared_runtime, out / 'runtime', symlinks=False)
    else:
        archive = cache / 'python-runtime.tar.gz'
        if not archive.exists() or sha(archive) != runtime_lock['sha256']:
            print('Downloading the locked Python runtime ...', flush=True)
            with urllib.request.urlopen(runtime_lock['url'], timeout=60) as response, archive.open('wb') as stream:
                shutil.copyfileobj(response, stream)
        if sha(archive) != runtime_lock['sha256']: raise RuntimeError('Runtime checksum mismatch.')
        unpack = cache / 'python-unpacked'
        if unpack.exists(): shutil.rmtree(unpack)
        with tarfile.open(archive) as t: t.extractall(unpack, filter='data')
        shutil.copytree(unpack / 'python', out / 'runtime', symlinks=False)
        subprocess.run([str(out / 'runtime/bin/python3'), '-I', '-m', 'pip', 'install',
                        '--disable-pip-version-check', '--only-binary=:all:', '--require-hashes',
                        '-r', str(R2 / 'container/requirements.lock')], check=True)
    python = out / 'runtime/bin/python3'
    subprocess.run([str(python), '-I', '-B', '-m', 'pip', 'install', '--disable-pip-version-check',
                    '--only-binary=:all:', '--require-hashes', '-r', str(HERE / 'gui-linux.lock')], check=True)
    subprocess.run([str(python), '-I', '-m', 'pip', 'check'], check=True)
    (out / 'app').mkdir()
    for name in ('engine.py', 'child.py', 'pisco_app.py'): shutil.copy2(HERE / name, out / 'app' / name)
    compiler = str(ensure(cache))
    subprocess.run([compiler, '-static', '-O2', '-march=x86-64', '-mtune=generic', str(HERE / 'launcher.c'), '-o', str(out / 'PISCO')], check=True)
    collect(args.native_root, out, compiler=compiler)
    shutil.copy2(cache / 'musl-1.2.5/COPYRIGHT', out / 'tools/licenses/musl.txt')
    copy_source(out)
    for name in ('START_HERE.html', 'README.md', 'THIRD_PARTY.md', 'BUILD_MATRIX.md'):
        shutil.copy2(HERE / name, out / name)
    sources = {n: sha(R2 / 'code' / n) for n in ('solvers.py', 'reproduce_all.py')}
    results = None
    if args.results_zip:
        if not args.results_release: p.error('--results-release is needed with --results-zip')
        previous = json.loads(args.results_release.read_text()); results = previous['results_archive']
        if previous['source_sha256'] != sources or sha(args.results_zip) != results['sha256']:
            raise RuntimeError('Publication source/archive mismatch.')
        with zipfile.ZipFile(args.results_zip) as z:
            for name, expected in sources.items():
                if hashlib.sha256(z.read('documentation/source/' + name)).hexdigest() != expected:
                    raise RuntimeError('Archived numerical source mismatch.')
        shutil.copy2(args.results_zip, out / results['filename'])
    manifest = {'package': 'PISCO R2 portable desktop prototype', 'status': 'candidate; not yet validated',
                'platform': 'linux-x86_64', 'python_executable': 'runtime/bin/python3',
                'runtime_distribution': runtime_lock, 'source_sha256': sources, 'results_archive': results,
                'build_host': platform.platform(), 'prepared_runtime_used': bool(args.prepared_runtime),
                'launcher_toolchain': {'source_url': MUSL_URL, 'sha256': MUSL_SHA256, 'cpu_target': 'baseline x86-64'},
                'limits': ['Only the subsequently recorded systems/tests establish support.',
                           'No universal or cross-architecture bitwise equality guarantee.',
                           'Original full publication results were computed in their recorded host environment.']}
    (out / 'release.json').write_text(json.dumps(manifest, indent=2) + '\n')
    subprocess.run([str(python), '-I', '-m', 'pip', 'freeze', '--all'], check=True,
                   stdout=(out / 'python-packages.txt').open('w'))
    for bytecode in out.rglob('__pycache__'):
        if bytecode.is_dir(): shutil.rmtree(bytecode)
    files = sorted(p for p in out.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name != 'results.zip')
    (out / 'checksums.json').write_text(json.dumps({str(f.relative_to(out)): sha(f) for f in files}, indent=2) + '\n')
    print('Built candidate:', out, flush=True)


if __name__ == '__main__': main()
