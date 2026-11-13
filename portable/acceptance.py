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
        env = dict(os.environ, QT_QPA_PLATFORM='offscreen', PATH='/no-external-python' if os.name != 'nt' else os.environ['PATH'])
        started = time.monotonic()
        with (output / (label + '.txt')).open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        records.append({'name': label, 'command': command, 'exit_code': result.returncode, 'seconds': time.monotonic()-started})
        print(label, result.returncode, flush=True)
        if result.returncode: raise RuntimeError(f'{label} failed; see {output / (label + ".txt")}')
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
    for source in sorted((support / 'schemas').glob('*.typ')):
        pdf = figures / (source.stem + '.pdf'); stem = figures / source.stem
        extension = '.exe' if os.name == 'nt' else ''
        for command in ([str(bundle / ('tools/bin/typst' + extension)), 'compile', '--root', str(support), str(source), str(pdf)],
                        [str(bundle / ('tools/bin/pdftoppm' + extension)), '-f', '1', '-singlefile', '-r', '72', '-png', str(pdf), str(stem)]):
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            figure_records.append({'command': command, 'exit_code': result.returncode, 'output': result.stdout + result.stderr})
            if result.returncode: raise RuntimeError('Figure tool failed: ' + result.stderr)
    report = {'passed': True, 'commands': records, 'worker_comparison_exact': True,
              'short_rows': comparison['compared_rows'], 'actual_short_interruption_and_continuation_passed': True,
              'figures': figure_records, 'scope': 'Local author package tests; no complete optimisation or independent editorial confirmation.'}
    (output / 'acceptance.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Package acceptance passed.', flush=True)


if __name__ == '__main__': main()
