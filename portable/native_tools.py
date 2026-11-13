"""Collect native Linux figure tools from a declared build root, without Docker.

Use the oldest supported build system. glibc itself remains an OS dependency.
The release records the source tools and every copied library hash.
"""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

SYSTEM_LIBS = {'libc.so.6', 'libm.so.6', 'libpthread.so.0', 'libdl.so.2',
               'librt.so.1', 'libresolv.so.2', 'libutil.so.1', 'ld-linux-x86-64.so.2'}


def collect(root, destination, compiler='cc'):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    target = destination / 'tools'; (target / 'bin').mkdir(parents=True, exist_ok=True)
    (target / 'lib').mkdir(exist_ok=True)
    site = destination / 'runtime/lib/python3.14/site-packages'
    search = [site / 'PySide6/Qt/lib', site / 'PySide6', site / 'shiboken6']
    search += [root / p for p in ('usr/lib/x86_64-linux-gnu', 'lib/x86_64-linux-gnu', 'usr/lib', 'lib')]
    def library(name):
        for directory in search:
            if (directory / name).is_file(): return directory / name
        raise RuntimeError('Missing native dependency: ' + name)
    poppler = root / 'usr/bin/pdftoppm'
    typst = root / 'usr/local/bin/typst'
    if not typst.is_file(): typst = root / 'usr/bin/typst'
    shutil.copy2(poppler, target / 'bin/pdftoppm.real')
    shutil.copy2(typst, target / 'bin/typst')
    pending = [poppler]
    pending += [site / 'PySide6' / name for name in ('QtCore.abi3.so', 'QtGui.abi3.so', 'QtWidgets.abi3.so')]
    pending += [site / 'PySide6/Qt/plugins/platforms' / name for name in ('libqxcb.so', 'libqoffscreen.so')]
    collected = {}
    while pending:
        binary = pending.pop()
        output = subprocess.check_output(['readelf', '-d', str(binary)], text=True)
        for name in re.findall(r'\(NEEDED\).*?\[(.*?)\]', output):
            if name in SYSTEM_LIBS or name in collected: continue
            source = library(name)
            if not source.is_relative_to(destination / 'runtime'):
                shutil.copy2(source, target / 'lib' / name)
            source_label = ('runtime/' + str(source.relative_to(destination / 'runtime'))
                            if source.is_relative_to(destination / 'runtime')
                            else 'native-inputs/' + str(source.relative_to(root)))
            collected[name] = {'source': source_label, 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
            pending.append(source)
    subprocess.run([compiler, '-static', '-O2', '-march=x86-64', '-mtune=generic',
                    str(Path(__file__).with_name('tool_launcher.c')), '-o', str(target / 'bin/pdftoppm')], check=True)
    fonts = root / 'usr/share/fonts/truetype'
    shutil.copytree(fonts, target / 'fonts', dirs_exist_ok=True)
    (target / 'fonts.conf').write_text('<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "fonts.dtd"><fontconfig><dir prefix="relative">fonts</dir><cachedir prefix="xdg">fontconfig</cachedir></fontconfig>\n')
    cache = root / 'opt/pisco/cache'
    if cache.is_dir(): shutil.copytree(cache, target / 'cache', dirs_exist_ok=True)
    # Include upstream copyright and licence notices for distributed libraries.
    notices = target / 'licenses'; notices.mkdir(exist_ok=True)
    for p in (root / 'usr/share/doc').glob('*/copyright'):
        if p.is_file(): shutil.copy2(p, notices / (p.parent.name + '.txt'))
    (target / 'native-libraries.json').write_text(json.dumps({'libraries': collected,
                  'system_dependencies': sorted(SYSTEM_LIBS),
                  'scope': 'Native library hashes identify this build; a different build root requires its own validation.'}, indent=2) + '\n')
    return collected
