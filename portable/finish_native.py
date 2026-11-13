"""Seal only an acceptance-tested native package and check its ZIP delivery."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_native import seal, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--acceptance', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build = json.loads(args.build.read_text()); bundle = Path(build['bundle'])
    root = Path(build['package_root']); proof = args.acceptance
    acceptance = json.loads((proof / 'acceptance.json').read_text())
    science = json.loads((proof / 'published output/verification/verification.json').read_text())
    comparison = json.loads((proof / 'worker-comparison.json').read_text())
    if not (acceptance['passed'] and acceptance['native_gui_passed'] and science['passed']
            and comparison['all_selected_numerical_fields_exact']
            and acceptance['direct_script_smoke_passed']
            and acceptance['direct_script_numerical_agreement']
            and acceptance['driver_figure_templates_passed'] == 5):
        raise RuntimeError('Native acceptance did not pass.')
    release = json.loads((bundle / 'release.json').read_text())
    # Do not label an altered binary/source tree with evidence from a different build.
    for name, expected in json.loads((bundle / 'checksums.json').read_text()).items():
        if sha(bundle / name) != expected:
            raise RuntimeError('Tested package changed: ' + name)
    summary = {'passed': True, 'platform': release['platform'],
        'host': platform.platform(), 'machine': platform.machine(),
        'build_commit': release['build_commit'], 'source_sha256': release['source_sha256'],
        'ci_run_url': os.environ.get('PISCO_CI_RUN_URL'),
        'saved_data': {k: science[k] for k in ('result_rows', 'artifact_count', 'current_numeric_summaries', 'statistical_contrasts')},
        'saved_data_checks': sum(science['checks_by_category'].values()),
        'short_runs': {'workers': [4, 12], 'rows_each': acceptance['short_rows'],
                       'selected_numerical_fields_exact': True, 'timing_equality_claimed': False},
        'offscreen_gui_passed': True,
        'native_gui_passed': acceptance['native_gui_passed'],
        'stop_and_continue_passed': acceptance['actual_short_interruption_and_continuation_passed'],
        'figure_templates_passed': len(acceptance['figures']) // 2,
        'driver_jpeg_templates_passed': acceptance['driver_figure_templates_passed'],
        'direct_script_smoke_passed': acceptance['direct_script_smoke_passed'],
        'direct_script_numerical_agreement': acceptance['direct_script_numerical_agreement'],
        'network_restriction': os.environ.get('PISCO_NETWORK_TEST'),
        'limits': ['Automated tests on this hosted operating system, not every end-user computer.',
                   'No full experiment matrix recomputed.', 'No cross-platform bitwise equality claim.',
                   'Unsigned distribution; macOS application is not Apple-notarized.']}
    write(bundle / 'validation/summary.json', summary)
    release['status'] = 'automated target-platform acceptance passed'
    release['validation'] = {'report': 'validation/summary.json', 'ci_run_url': summary['ci_run_url']}
    write(bundle / 'release.json', release)
    windows = release['platform'].startswith('windows')
    program = 'PISCO.exe' if windows else 'PISCO.app'
    instructions = f'''# PISCO paper reproduction — {release['platform']}

Extract the complete ZIP, open **{program}**, choose an output folder and select
**Check paper results**. Python, numerical dependencies, figure tools and the
saved current data are included. No Python installation or Docker is needed.

The window also provides short computations, a complete run, progress, stop and
resume. A complete run can take days. Saved-data checking does not start an
optimisation. Resume requires the same environment and worker count.

This distribution passed automated target-platform acceptance, including saved
data, real four/twelve-worker short calculations, selected numerical agreement,
native and offscreen window operation, stop/continue and all five figure templates.
See `validation/summary.json` in the application resources for the exact host,
source commit and test scope. No full matrix was rerun; numerical equality
across platforms is not guaranteed. The scientific source is unchanged.

This is an unsigned research distribution. Operating-system download controls
may require approval before opening it. Do not disable system protection.
For macOS, the application is not Apple-notarized; use the system's normal
Privacy & Security approval for an app you have chosen to trust. Windows can
likewise display a publisher/reputation warning. Verify the published hash.

Windows package: x64. macOS package: Apple Silicon, macOS 14 or later.
The automated test host is recorded in the validation summary. Intel Macs and
Windows ARM are not validated by these packages.

Result files and diagnostic logs remain in the selected output folder. The
saved data retain their original full-run environment metadata. Developer
sources and the exact dependency locks are included under `source/`.

For execution directly through Python, see the included
[script guide](source/code/README.md). It explains `reproduce_all.py`, the
`solvers.py` library, interpreter setup, complete runs, resuming and checking
saved results. No graphical interface is needed for this route.
'''
    (bundle / 'README.md').write_text(instructions, encoding='utf-8')
    (bundle / 'BUILD_MATRIX.md').write_text('# Package validation\n\nThis package passed automated acceptance on '+platform.platform()+'.\nSee validation/summary.json for the exact checks and limits.\n', encoding='utf-8')
    if root != bundle:
        (root / 'README.md').write_text(instructions.replace(
            '](source/code/README.md)', '](PISCO.app/Contents/Resources/source/code/README.md)'), encoding='utf-8')
        shutil.copy2(bundle / 'BUILD_MATRIX.md', root / 'BUILD_MATRIX.md')
    seal(bundle)
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / (root.name + '.zip')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3, strict_timestamps=False) as zipped:
        for path in sorted(root.rglob('*')):
            if path.is_file():
                zipped.write(path, root.name + '/' + path.relative_to(root).as_posix(),
                    compress_type=zipfile.ZIP_STORED if path.name == 'results.zip' else zipfile.ZIP_DEFLATED)
    if archive.stat().st_size >= 2 * 1024**3:
        raise RuntimeError('Package exceeds the release-asset limit.')
    # Test the actual delivery archive in a fresh location containing spaces.
    recipient = args.output.parent / 'recipient with spaces ä'
    recipient.mkdir()
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip():
            raise RuntimeError('ZIP integrity check failed.')
        zipped.extractall(recipient)
        if os.name != 'nt':
            for member in zipped.infolist():
                mode = member.external_attr >> 16
                if mode:
                    (recipient / member.filename).chmod(mode & 0o777)
    restored = recipient / root.name / bundle.relative_to(root)
    launcher = restored / release['launcher_executable']
    with (args.output.parent / 'recipient-check.txt').open('w', encoding='utf-8') as log:
        subprocess.run([str(launcher), '--action', 'verify', '--workspace',
                        str(args.output.parent / 'recipient results')], stdout=log, stderr=subprocess.STDOUT, check=True)
    receipt = {'filename': archive.name, 'bytes': archive.stat().st_size, 'sha256': sha(archive),
        'platform': release['platform'], 'source_sha256': release['source_sha256'],
        'results_sha256': release['results_archive']['sha256'], 'build_commit': release['build_commit'],
        'zip_crc_passed': True, 'extracted_zip_verification_passed': True, 'validation': summary}
    write(args.output / ('release-' + release['platform'] + '.json'), receipt)
    (args.output / (archive.name + '.sha256')).write_text(receipt['sha256'] + '  ' + archive.name + '\n')
    print(json.dumps({'zip': archive.name, 'bytes': receipt['bytes'], 'sha256': receipt['sha256'], 'passed': True}), flush=True)


if __name__ == '__main__':
    main()
