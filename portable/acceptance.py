"""Exercise a finished native package, not the developer's Python environment.

May run inside an offline OS test root. All outputs are separate from the
package. GUI testing uses Qt's offscreen display and records a screenshot.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--with-archive', action='store_true')
    args = parser.parse_args(); bundle = args.package.resolve(); output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((bundle / 'release.json').read_text())
    exe = bundle / manifest.get('launcher_executable', 'PISCO.exe' if os.name == 'nt' else 'PISCO')
    records = []
    for label, options in [
        ('installation', ['--action', 'doctor']),
        ('short4', ['--action', 'smoke', '--workers', '4']),
        ('short12', ['--action', 'smoke', '--workers', '12']),
        ('window', ['--gui-test', 'doctor']),
        *([('published', ['--action', 'verify'])] if args.with_archive else []),
    ]:
        command = [str(exe), *options, '--workspace', str(output / (label + ' output'))]
        system = Path(os.environ.get('SystemRoot', 'C:/Windows'))
        clean_path = os.pathsep.join((str(system / 'System32'), str(system))) if os.name == 'nt' else '/no-external-python'
        env = dict(os.environ, QT_QPA_PLATFORM='offscreen', PATH=clean_path)
        started = time.monotonic()
        print('Starting ' + label + ' ...', flush=True)
        with (output / (label + '.txt')).open('w') as log:
            try:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=600)
            except subprocess.TimeoutExpired:
                log.flush()
                print((output / (label + '.txt')).read_text(encoding='utf-8', errors='replace')[-12000:], flush=True)
                raise
        records.append({'name': label, 'command': command, 'exit_code': result.returncode, 'seconds': time.monotonic()-started})
        print(label, result.returncode, flush=True)
        if result.returncode:
            print((output / (label + '.txt')).read_text(encoding='utf-8', errors='replace')[-12000:], flush=True)
            raise RuntimeError(f'{label} failed; see {output / (label + ".txt")}')
    native_gui = False
    if sys.platform in ('win32', 'darwin'):
        # Exercise the actual window-system plugin. A detached Windows process
        # also reproduces a double-click with no inherited console handles.
        native_folder = output / 'native window output'
        command = [str(exe), '--gui-test', 'doctor', '--workspace', str(native_folder)]
        native_env = dict(env, QT_QPA_PLATFORM='windows' if os.name == 'nt' else 'cocoa')
        options = {'creationflags': subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == 'nt' else {}
        subprocess.run(command, env=native_env, timeout=180, check=True, **options)
        window = json.loads((native_folder / 'gui-test.json').read_text())
        native_gui = window['window_created'] and window['exit_code'] == 0
        if not native_gui: raise RuntimeError('Native desktop launch failed.')
        print('Native desktop launch passed.', flush=True)
    comparison_script = bundle / 'source/verification/compare_smoke.py'
    spec = importlib.util.spec_from_file_location('saved_comparison', comparison_script)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    short4 = next((output / 'short4 output').glob('short-test-*'))
    short12 = next((output / 'short12 output').glob('short-test-*'))
    comparison = module.compare(short4, short12)
    (output / 'worker-comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
    if not comparison['all_selected_numerical_fields_exact']:
        raise RuntimeError('The two worker-count runs differ in a checked numerical field.')
    sys.path.insert(0, str(bundle / 'app'))
    from engine import Engine, Stopped
    workspace = output / 'interruption output'; workspace.mkdir()
    cancel = threading.Event(); transcript = []
    def log(line):
        transcript.append(line)
        if line.startswith('[COMPLETED]'): cancel.set()
    engine = Engine(bundle, workspace, workers=4, output=log, cancel=cancel)
    target = workspace / 'short-run'
    command = ['script', bundle / 'source/code/reproduce_all.py', '--smoke', '--workers', '4', '--output', target]
    try: engine.command('interrupted-short-run', command)
    except Stopped: pass
    else: raise RuntimeError('The interruption test did not interrupt a real computation.')
    engine.cancel.clear(); engine.output = transcript.append
    engine.command('continued-short-run', command)
    resumed = json.loads((target / 'smoke_verification.json').read_text())
    if not resumed['passed']: raise RuntimeError('The continued short run did not pass.')
    (output / 'interruption.txt').write_text('\n'.join(transcript) + '\n')
    # Render the five original editable diagrams using only supplied tools.
    figures = output / 'figures'; figures.mkdir()
    support = bundle / 'source/code/paper_support'
    figure_records = []
    env = engine.environment()
    python = bundle / manifest['python_executable']
    direct = output / 'direct script output'
    with (output / 'direct-script.txt').open('w', encoding='utf-8') as log:
        subprocess.run([str(python), '-B', '-X', 'utf8', '-u',
            str(bundle / 'source/code/reproduce_all.py'), '--smoke', '--workers', '4',
            '--fresh', '--output', str(direct)], env=env, cwd=output,
            stdout=log, stderr=subprocess.STDOUT, timeout=600, check=True)
    if not json.loads((direct / 'smoke_verification.json').read_text())['passed']:
        raise RuntimeError('Direct script execution failed its evidence checks.')
    direct_comparison = module.compare(short4, direct)
    if not direct_comparison['all_selected_numerical_fields_exact']:
        raise RuntimeError('Direct script and packaged launch differ numerically.')
    (output / 'direct-script-comparison.json').write_text(json.dumps(direct_comparison, indent=2) + '\n')
    print('Direct reproduce_all.py execution passed.', flush=True)
    for source in sorted((support / 'schemas').glob('*.typ')):
        pdf = figures / (source.stem + '.pdf'); stem = figures / source.stem
        extension = '.exe' if os.name == 'nt' else ''
        for command in ([str(bundle / ('tools/bin/typst' + extension)), 'compile', '--root', str(support), str(source), str(pdf)],
                        [str(bundle / ('tools/bin/pdftoppm' + extension)), '-f', '1', '-singlefile', '-r', '72', '-png', str(pdf), str(stem)]):
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            figure_records.append({'command': command, 'exit_code': result.returncode, 'output': result.stdout + result.stderr})
            if result.returncode: raise RuntimeError('Figure tool failed: ' + result.stderr)
    # Invoke the actual frozen driver method, including its 700 dpi JPEG
    # options and .jpg -> .jpeg rename, rather than approximating its command.
    probe = '''from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
import reproduce_all
from PIL import Image
builder = reproduce_all.PublicationBuilder(Path(sys.argv[2]))
sources = sorted((Path(sys.argv[1]) / 'paper_support/schemas').glob('*.typ'))
assert len(sources) == 5
for source in sources:
    pdf = builder._render_typst_figure(source.name, source.stem + '.pdf')
    with Image.open(pdf.with_suffix('.jpeg')) as picture:
        assert picture.format == 'JPEG'
        assert abs(picture.info['dpi'][0] - 700) < 1
        picture.verify()
print('All five driver JPEG figures passed at 700 dpi.')
'''
    with (output / 'driver-figures.txt').open('w', encoding='utf-8') as log:
        subprocess.run([str(python), '-B', '-X', 'utf8', '-u', '-c', probe,
            str(bundle / 'source/code'), str(output / 'driver figures')], env=env,
            stdout=log, stderr=subprocess.STDOUT, timeout=300, check=True)
    print('Actual driver JPEG figure calls passed.', flush=True)
    report = {'passed': True, 'commands': records, 'worker_comparison_exact': True,
              'native_gui_passed': native_gui,
              'direct_script_smoke_passed': True, 'direct_script_numerical_agreement': True,
              'driver_figure_templates_passed': 5,
              'short_rows': comparison['compared_rows'], 'actual_short_interruption_and_continuation_passed': True,
              'figures': figure_records, 'scope': 'Local author package tests; no complete optimisation or independent editorial confirmation.'}
    (output / 'acceptance.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Package acceptance passed.', flush=True)


if __name__ == '__main__': main()
