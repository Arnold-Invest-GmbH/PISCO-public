"""Build a small baseline-x86-64 static launcher toolchain, locally only."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request

URL = 'https://musl.libc.org/releases/musl-1.2.5.tar.gz'
SHA256 = 'a9a118bbe84d8764da0ea0d28b3ab3fae8477fc7e4085d90102b8596fc7c75e4'


def ensure(cache):
    cache = Path(cache).resolve()
    compiler = cache / 'musl-toolchain/bin/musl-gcc'
    if compiler.is_file(): return compiler
    archive = cache / 'musl-1.2.5.tar.gz'
    if not archive.exists():
        with urllib.request.urlopen(URL, timeout=60) as response, archive.open('wb') as stream:
            shutil.copyfileobj(response, stream)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError('Launcher toolchain source checksum mismatch.')
    with tarfile.open(archive) as t: t.extractall(cache, filter='data')
    source = cache / 'musl-1.2.5'
    env = dict(os.environ, CFLAGS='-O2 -march=x86-64 -mtune=generic -std=gnu11 -fno-lto')
    with (cache / 'musl-build.txt').open('w') as log:
        for command in ([str(source / 'configure'), '--prefix=' + str(compiler.parents[1]), '--disable-shared'],
                        ['make', '-j4'], ['make', 'install']):
            subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    return compiler
