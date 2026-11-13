"""Child entry: isolated imports, cooperative cancellation, unchanged scripts."""
import argparse
import _thread
import importlib
import json
import os
from pathlib import Path
import platform
import runpy
import shutil
import subprocess
import sys
import threading
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stop-file', type=Path, required=True)
    parser.add_argument('mode', choices=('probe', 'script'))
    parser.add_argument('target', type=Path)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    def monitor():
        while True:
            if args.stop_file.exists():
                _thread.interrupt_main(); return
            time.sleep(.1)
    threading.Thread(target=monitor, daemon=True).start()
    if args.mode == 'script':
        sys.path.insert(0, str(args.target.parent))
        sys.argv = [str(args.target), *args.arguments]
        runpy.run_path(str(args.target), run_name='__main__')
        return 0
    bundle = args.target.resolve()
    workspace = Path(args.arguments[0])
    modules = {}
    for name in ('numpy', 'scipy', 'torch', 'matplotlib', 'sklearn', 'PIL', 'threadpoolctl', 'PySide6', 'shiboken6'):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(bundle / 'runtime'):
            raise RuntimeError('External Python dependency detected: ' + str(path))
        modules[name] = {'version': module.__version__, 'path': str(path)}
    if not Path(sys.prefix).resolve().is_relative_to(bundle / 'runtime'):
        raise RuntimeError('Python is not running from the supplied runtime.')
    tools = {}
    for name, flag in (('typst', '--version'), ('pdftoppm', '-v')):
        found = shutil.which(name)
        if not found or not Path(found).resolve().is_relative_to(bundle / 'tools'):
            raise RuntimeError('Missing bundled figure tool: ' + name)
        result = subprocess.run([found, flag], capture_output=True, text=True)
        if result.returncode: raise RuntimeError('Figure tool failed: ' + result.stderr)
        tools[name] = (result.stdout + result.stderr).strip()
    report = {'python': sys.version, 'executable': sys.executable, 'prefix': sys.prefix,
              'platform': platform.platform(), 'machine': platform.machine(),
              'packages': modules, 'figure_tools': tools,
              'free_workspace_bytes': shutil.disk_usage(workspace).free,
              'scope': 'Installation check; no claim of full numerical or independent editorial reproduction.'}
    path = workspace / 'installation-check.json'
    path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('Installation check passed. All Python dependencies and figure tools are supplied in the package.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nCalculation interrupted; completed files are retained.', flush=True)
        raise SystemExit(130)
