"""Build native Windows x64 or macOS ARM64 bundles on the target OS.

Runtime, wheel, tool and data downloads must match published SHA-256 locks.
This creates a candidate; acceptance and publication are separate operations.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import plistlib
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from build_portable import copy_source


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')


def download(row, path):
    if path.is_file() and sha(path) == row['sha256']:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    for attempt in range(3):
        try:
            with urllib.request.urlopen(row['url'], timeout=120) as source, temporary.open('wb') as target:
                shutil.copyfileobj(source, target)
            if sha(temporary) != row['sha256']:
                raise RuntimeError('Download checksum mismatch: ' + path.name)
            temporary.replace(path)
            return path
        except OSError:
            if attempt == 2:
                raise
    raise RuntimeError('Download failed: ' + path.name)


def unpack(archive, target):
    target.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                destination = (target / member.filename).resolve()
                if not destination.is_relative_to(target.resolve()):
                    raise RuntimeError('Unsafe dependency archive path.')
            source.extractall(target)
    else:
        with tarfile.open(archive) as source:
            source.extractall(target, filter='data')


def windows_environment():
    locator = Path(os.environ['ProgramFiles(x86)']) / 'Microsoft Visual Studio/Installer/vswhere.exe'
    installation = subprocess.check_output([str(locator), '-latest', '-products', '*',
        '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath'], text=True).strip()
    if not installation:
        raise RuntimeError('Visual C++ build tools are required on the Windows build machine.')
    devcmd = Path(installation) / 'Common7/Tools/VsDevCmd.bat'
    # cmd.exe uses its own quoting rules, not the C argv rules applied by
    # subprocess.list2cmdline. Pass the command line intact to CreateProcess.
    command = f'cmd.exe /d /s /c "call "{devcmd}" -arch=x64 -host_arch=x64 >nul && set"'
    output = subprocess.check_output(command, text=True, errors='replace')
    env = dict(os.environ)
    for line in output.splitlines():
        if '=' in line and not line.startswith('='):
            key, value = line.split('=', 1)
            env[key] = value
    return env


def seal(bundle):
    for path in bundle.rglob('__pycache__'):
        if path.is_dir():
            shutil.rmtree(path)
    files = sorted(p for p in bundle.rglob('*') if p.is_file()
                   and p != bundle / 'checksums.json' and p != bundle / 'results.zip')
    write(bundle / 'checksums.json', {p.relative_to(bundle).as_posix(): sha(p) for p in files})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--platform', choices=('windows-x86_64', 'macos-arm64'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    args = p.parse_args()
    windows = args.platform.startswith('windows')
    if windows != (sys.platform == 'win32') or (not windows and sys.platform != 'darwin'):
        p.error('Build on the intended target operating system.')
    expected_machine = ('amd64', 'x86_64') if windows else ('arm64', 'aarch64')
    if platform.machine().lower() not in expected_machine:
        p.error('Build-machine architecture does not match the target.')
    out = args.output.resolve(); cache = args.cache.resolve()
    if out.exists() and any(out.iterdir()):
        p.error('The output directory must be new or empty.')
    out.mkdir(parents=True, exist_ok=True); cache.mkdir(parents=True, exist_ok=True)
    bundle = out if windows else out / 'PISCO.app/Contents/Resources'
    bundle.mkdir(parents=True, exist_ok=True)
    key = 'x86_64-pc-windows-msvc' if windows else 'aarch64-apple-darwin'
    runtime = json.loads((HERE / 'runtimes.lock.json').read_text())[key]
    print('Preparing the locked Python runtime ...', flush=True)
    archive = download(runtime, cache / 'python.tar.gz')
    unpack(archive, cache / 'python-unpacked')
    shutil.copytree(cache / 'python-unpacked/python', bundle / 'runtime', symlinks=False)
    python = bundle / ('runtime/python.exe' if windows else 'runtime/bin/python3')
    command = [str(python), '-I', '-B', '-X', 'utf8']
    subprocess.run([*command, '-m', 'pip', 'install', '--disable-pip-version-check',
        '--no-cache-dir', '--only-binary=:all:', '--require-hashes', '-r',
        str(HERE / ('packages-' + args.platform + '.lock'))], check=True)
    subprocess.run([*command, '-m', 'pip', 'check'], check=True)
    (bundle / 'app').mkdir()
    for name in ('engine.py', 'child.py', 'pisco_app.py', 'rasterize_pdf.py'):
        shutil.copy2(HERE / name, bundle / 'app' / name)
    (bundle / 'tools/bin').mkdir(parents=True)
    native_lock = json.loads((HERE / 'native-inputs.lock.json').read_text())
    native = download(native_lock, cache / native_lock['filename'])
    unpack(native, cache / 'native-inputs')
    for relative, name in [('usr/share/fonts/truetype', 'fonts'), ('opt/pisco/cache', 'cache')]:
        shutil.copytree(cache / 'native-inputs' / relative, bundle / 'tools' / name)
    shutil.copytree(cache / 'native-inputs/usr/share/doc', bundle / 'tools/licenses')
    (bundle / 'tools/fonts.conf').write_text('<?xml version="1.0"?><fontconfig><dir prefix="relative">fonts</dir></fontconfig>\n')
    typst = json.loads((HERE / 'typst-platforms.lock.json').read_text())[args.platform]
    archive = download(typst, cache / ('typst.zip' if windows else 'typst.tar.xz'))
    unpack(archive, cache / 'typst-unpacked')
    executable = 'typst.exe' if windows else 'typst'
    found = list((cache / 'typst-unpacked').rglob(executable))
    if len(found) != 1:
        raise RuntimeError('Unexpected Typst archive layout.')
    shutil.copy2(found[0], bundle / 'tools/bin' / executable)
    launcher = bundle / 'PISCO.exe' if windows else out / 'PISCO.app/Contents/MacOS/PISCO'
    launcher.parent.mkdir(parents=True, exist_ok=True)
    compiler_env = windows_environment() if windows else dict(os.environ)
    for target, rasterizer in [(launcher, False), (bundle / ('tools/bin/pdftoppm.exe' if windows else 'tools/bin/pdftoppm'), True)]:
        if windows:
            options = ['cl.exe', '/nologo', '/O2', '/MT', '/utf-8', str(HERE / 'desktop_launcher.c'),
                       '/Fe:' + str(target), '/Fo:' + str(cache / (target.stem + '.obj'))]
            if rasterizer: options.append('/DPISCO_RASTERIZER')
            options += ['/link', '/SUBSYSTEM:WINDOWS', 'user32.lib']
        else:
            options = ['clang', '-O2', '-arch', 'arm64', '-mmacosx-version-min=14.0',
                       str(HERE / 'desktop_launcher.c'), '-o', str(target)]
            if rasterizer: options.append('-DPISCO_RASTERIZER')
        subprocess.run(options, env=compiler_env, check=True)
    if windows:
        redist = Path(compiler_env['VCToolsRedistDir'])
        for directory in redist.glob('x64/Microsoft.VC*.CRT'):
            for dll in directory.glob('*.dll'):
                shutil.copy2(dll, bundle / 'runtime' / dll.name)
    else:
        (bundle / 'tools/bin/typst').chmod(0o755)
        plist = {'CFBundleExecutable': 'PISCO', 'CFBundleIdentifier': 'de.ai-arnold-invest.pisco',
                 'CFBundleName': 'PISCO', 'CFBundlePackageType': 'APPL',
                 'CFBundleVersion': '2.0', 'CFBundleShortVersionString': '2.0',
                 'LSMinimumSystemVersion': '14.0', 'NSHighResolutionCapable': True}
        with (out / 'PISCO.app/Contents/Info.plist').open('wb') as stream:
            plistlib.dump(plist, stream)
    copy_source(bundle)
    for name in ('START_HERE.html', 'README.md', 'THIRD_PARTY.md', 'BUILD_MATRIX.md'):
        shutil.copy2(HERE / name, bundle / name)
        if not windows: shutil.copy2(HERE / name, out / name)
    publication = json.loads((ROOT / 'release.json').read_text())
    data = publication['results_archive']
    download({'url': 'https://github.com/Arnold-Invest-GmbH/PISCO-public/releases/download/r2-2026-11-13/PISCO-results.zip',
              'sha256': data['sha256']}, bundle / 'results.zip')
    sources = {name: sha(ROOT / 'code' / name) for name in ('solvers.py', 'reproduce_all.py')}
    if sources != publication['source_sha256']:
        raise RuntimeError('The frozen scientific sources have changed.')
    with zipfile.ZipFile(bundle / 'results.zip') as zipped:
        for name, expected in sources.items():
            if hashlib.sha256(zipped.read('documentation/source/' + name)).hexdigest() != expected:
                raise RuntimeError('Archived scientific source differs: ' + name)
    manifest = {'package': 'PISCO paper reproduction', 'platform': args.platform,
        'status': 'candidate; awaiting target-platform acceptance', 'release_id': 'r2-2026-11-13',
        'python_executable': python.relative_to(bundle).as_posix(),
        'launcher_executable': 'PISCO.exe' if windows else '../MacOS/PISCO',
        'source_sha256': sources, 'results_archive': data, 'runtime_distribution': runtime,
        'build_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'build_host': platform.platform(), 'figure_renderer': 'PDFium through pypdfium2; pdftoppm interface for the required figure flags',
        'publisher_signature': False, 'apple_notarization': False if not windows else None,
        'limits': ['No full experiment matrix rerun in this runtime.',
                   'No cross-platform bitwise numerical equality guarantee.',
                   'Saved publication data retain their original execution-environment metadata.']}
    write(bundle / 'release.json', manifest)
    with (bundle / 'python-packages.txt').open('w', encoding='utf-8') as stream:
        subprocess.run([*command, '-m', 'pip', 'freeze', '--all'], stdout=stream, check=True)
    seal(bundle)
    write(out.parent / ('build-' + args.platform + '.json'),
          {'package_root': str(out), 'bundle': str(bundle), 'python': str(python), 'launcher': str(launcher)})
    print('Candidate built:', out, flush=True)


if __name__ == '__main__':
    main()
