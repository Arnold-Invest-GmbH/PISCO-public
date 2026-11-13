"""Portable package operations. Scientific sources are read-only inputs."""
import contextlib
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import zipfile


class Stopped(RuntimeError):
    pass


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temp.replace(path)


def sha(path, cancel=None):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            if cancel is not None and cancel.is_set():
                raise Stopped('Stopped. Existing results are retained.')
            h.update(block)
    return h.hexdigest()


@contextlib.contextmanager
def workspace_lock(folder):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.pisco.lock').open('a+b') as stream:
        stream.seek(0)
        if not stream.read(1):
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError('Another PISCO task is using this output folder.') from error
        try:
            yield
        finally:
            if os.name == 'nt':
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def safe_member(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
        raise RuntimeError('Unsafe archive path: ' + name)
    return path


class Engine:
    def __init__(self, bundle, workspace, workers=4, output=print, cancel=None):
        self.bundle = Path(bundle).resolve()
        self.workspace = Path(workspace).resolve()
        self.workers = int(workers)
        if self.workers < 1:
            raise ValueError('Choose at least one worker.')
        self.output = output
        self.cancel = cancel if cancel is not None else threading.Event()
        self.manifest = read(self.bundle / 'release.json')
        self.python = self.bundle / self.manifest['python_executable']
        self.source = self.bundle / 'source'
        self.last_result = None

    def check_stop(self):
        if self.cancel.is_set():
            raise Stopped('Stopped. Existing results are retained.')

    def integrity(self):
        self.output('Checking the supplied program files ...')
        files = read(self.bundle / 'checksums.json')
        for number, (name, expected) in enumerate(files.items(), 1):
            safe_member(name)
            path = self.bundle / name
            if not path.is_file() or sha(path, self.cancel) != expected:
                raise RuntimeError('Package file is missing or changed: ' + name + '. Obtain the complete matching release again.')
            if number % 500 == 0:
                self.output(f'Program files checked: {number}/{len(files)}')
        for name, expected in self.manifest['source_sha256'].items():
            if sha(self.source / 'code' / name, self.cancel) != expected:
                raise RuntimeError('Scientific source mismatch: ' + name)
        self.output(f'Program integrity passed ({len(files)} files).')

    def environment(self):
        env = dict(os.environ)
        for name in ('PYTHONHOME', 'PYTHONPATH', 'LD_LIBRARY_PATH', 'LD_PRELOAD',
                     'DYLD_LIBRARY_PATH', 'DYLD_INSERT_LIBRARIES', 'TCL_LIBRARY', 'TK_LIBRARY'):
            env.pop(name, None)
        cache = self.workspace / 'cache'
        if (self.bundle / 'tools/cache').is_dir():
            shutil.copytree(self.bundle / 'tools/cache', cache, dirs_exist_ok=True)
        env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
                   MPLCONFIGDIR=str(cache / 'matplotlib'),
                   XDG_CACHE_HOME=str(cache),
                   FONTCONFIG_FILE=str(self.bundle / 'tools/fonts.conf'),
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   PATH=str(self.bundle / 'tools/bin') + os.pathsep + os.defpath)
        return env

    def command(self, name, args):
        self.check_stop()
        logs = self.workspace / 'logs'; logs.mkdir(parents=True, exist_ok=True)
        tag = name + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + str(time.monotonic_ns())
        log = logs / (tag + '.txt')
        stop_file = logs / (tag + '.stop')
        command = [str(self.python), '-I', '-B', '-u', str(self.bundle / 'app/child.py'),
                   '--stop-file', str(stop_file), *map(str, args)]
        self.output('Log: ' + str(log))
        started = time.monotonic()
        flags = {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == 'nt' else {'start_new_session': True}
        lines = queue.Queue()
        stopping = None
        with log.open('w', encoding='utf-8') as transcript:
            transcript.write('Command: ' + json.dumps(command) + '\n'); transcript.flush()
            proc = subprocess.Popen(command, cwd=self.workspace, env=self.environment(),
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding='utf-8', errors='replace', **flags)
            def read_output():
                try:
                    for line in proc.stdout:
                        lines.put(line)
                finally:
                    lines.put(None)
            reader = threading.Thread(target=read_output, daemon=True); reader.start()
            ended = False
            try:
                while not ended or proc.poll() is None:
                    if self.cancel.is_set() and stopping is None:
                        stopping = time.monotonic(); stop_file.touch()
                        self.output('Stopping the calculation; saving completed work ...')
                    if stopping is not None and time.monotonic() - stopping > 30 and proc.poll() is None:
                        if os.name == 'nt':
                            subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'],
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        else:
                            try: os.killpg(proc.pid, signal.SIGKILL)
                            except ProcessLookupError: pass
                    try: line = lines.get(timeout=.1)
                    except queue.Empty: continue
                    if line is None:
                        ended = True; continue
                    transcript.write(line); transcript.flush(); self.output(line.rstrip('\r\n'))
                code = proc.wait()
            finally:
                if proc.poll() is None:
                    proc.kill(); proc.wait()
                reader.join(timeout=2); proc.stdout.close()
                stop_file.unlink(missing_ok=True)
        record = {'command': command, 'exit_code': code, 'cancelled': stopping is not None,
                  'elapsed_seconds': time.monotonic() - started, 'log': str(log)}
        write_json(log.with_suffix('.json'), record)
        if stopping is not None:
            raise Stopped('Stopped. Completed results and the log are retained.')
        if code:
            raise RuntimeError(f'The task failed (exit {code}). See {log}')
        return record

    def doctor(self):
        self.command('installation', ['probe', self.bundle, self.workspace])

    def available_archive(self, info):
        """Find a complete archive or assemble verified release parts locally."""
        original = self.bundle / info['filename']
        if original.is_file() or not info.get('parts'):
            return original
        cache = self.workspace / 'cache'
        cache.mkdir(parents=True, exist_ok=True)
        assembled = cache / ('results-' + info['sha256'] + '.zip')
        if assembled.is_file() and sha(assembled, self.cancel) == info['sha256']:
            return assembled
        sources = []
        missing = []
        for part in info['parts']:
            name = safe_member(part['filename'])
            if len(name.parts) != 1:
                raise RuntimeError('Result part must have a simple filename.')
            path = next((folder / str(name) for folder in (self.bundle, self.bundle.parent)
                         if (folder / str(name)).is_file()), None)
            if path is None: missing.append(str(name))
            else: sources.append((part, path))
        if missing:
            raise RuntimeError('Download the missing result files and place them next to the extracted PISCO folder: '
                               + ', '.join(missing) + '. Then select Check paper results again.')
        total = sum(part['bytes'] for part in info['parts'])
        if total != info['bytes']:
            raise RuntimeError('Result-part sizes do not match the release manifest.')
        if shutil.disk_usage(cache).free < total + 256 * 1024**2:
            raise RuntimeError('Not enough free disk space to assemble the result archive.')
        temporary = cache / ('.assembling-results-' + str(time.monotonic_ns()))
        self.output('Assembling the saved-results archive from the downloaded parts ...')
        digest = hashlib.sha256()
        try:
            with temporary.open('wb') as destination:
                for number, (part, path) in enumerate(sources, 1):
                    self.check_stop()
                    if path.stat().st_size != part['bytes']:
                        raise RuntimeError('Incomplete result file: ' + path.name + '. Download it again.')
                    part_digest = hashlib.sha256()
                    with path.open('rb') as source:
                        for block in iter(lambda: source.read(8 * 1024**2), b''):
                            self.check_stop()
                            destination.write(block); part_digest.update(block); digest.update(block)
                    if part_digest.hexdigest() != part['sha256']:
                        raise RuntimeError('Damaged result file: ' + path.name + '. Download it again.')
                    self.output(f'Result file checked: {number}/{len(sources)}')
            if digest.hexdigest() != info['sha256']:
                raise RuntimeError('Assembled result archive checksum mismatch.')
            temporary.replace(assembled)
        finally:
            temporary.unlink(missing_ok=True)
        return assembled

    def archive(self):
        info = self.manifest.get('results_archive')
        if not info:
            raise RuntimeError('This development build does not contain publication results.')
        archive = self.available_archive(info)
        self.output('Checking the saved-results archive ...')
        if not archive.is_file() or sha(archive, self.cancel) != info['sha256']:
            raise RuntimeError('The supplied results archive is missing or damaged.')
        destination = self.workspace / 'review/code/reproduction_output_corrected'
        marker = destination / '.portable-archive.json'
        if destination.exists():
            if not marker.is_file() or read(marker).get('sha256') != info['sha256']:
                raise RuntimeError('This output folder contains other results. Choose a new output folder; existing files are preserved.')
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zipped:
            entries = zipped.infolist()
            if sum(e.file_size for e in entries) != info['uncompressed_bytes']:
                raise RuntimeError('Archive size differs from the release manifest.')
            names = set()
            for entry in entries:
                safe_member(entry.filename)
                key = entry.filename.casefold()
                if key in names or (entry.external_attr >> 16) & 0o170000 == 0o120000:
                    raise RuntimeError('Duplicate archive path or symbolic link: ' + entry.filename)
                names.add(key)
            if shutil.disk_usage(destination.parent).free < info['uncompressed_bytes'] + 256 * 1024**2:
                raise RuntimeError('Not enough free disk space to unpack the saved results.')
            temporary = Path(tempfile.mkdtemp(prefix='.unpacking-', dir=destination.parent))
            try:
                for number, entry in enumerate(entries, 1):
                    self.check_stop(); zipped.extract(entry, temporary)
                    if number % 250 == 0:
                        self.output(f'Unpacking saved results: {number}/{len(entries)}')
                if not (temporary / 'documentation/artifact_manifest.json').is_file():
                    raise RuntimeError('Unexpected archive layout.')
                write_json(temporary / '.portable-archive.json', {'sha256': info['sha256']})
                temporary.rename(destination)
            finally:
                if temporary.exists(): shutil.rmtree(temporary)
        return destination

    def verify(self):
        archive = self.archive()
        audit = self.workspace / 'verification'
        self.command('current-results', ['script', self.source / 'verification/check_results.py',
                                        '--results', archive, '--source', self.source, '--output', audit])
        report = read(audit / 'verification.json')
        if not report['passed']:
            raise RuntimeError('The current scientific verification reported a failure.')
        self.last_result = audit
        self.output(f"PASS: saved results, {report['current_numeric_summaries']} current numerical summaries "
                    f"and {report['statistical_contrasts']} comparisons checked. No optimisation was run.")

    def compute(self, action):
        if action == 'smoke':
            folder = self.workspace / ('short-test-' + str(time.time_ns()))
            options = ['--smoke', '--fresh']
        else:
            folder = self.workspace / 'full-run'
            if action == 'full' and folder.exists() and any(folder.iterdir()):
                raise RuntimeError('A complete-run folder already exists. Choose Resume or a new output folder.')
            if action == 'resume':
                plan = folder / 'experiment_plan.json'
                if not plan.is_file(): raise RuntimeError('There is no complete run to resume in this output folder.')
                original = read(plan)['numerical_execution']['workers']
                if original != self.workers:
                    raise RuntimeError(f'Resume requires the original worker count: {original}.')
            options = ['--all'] + (['--fresh'] if action == 'full' else [])
        self.last_result = folder
        self.command(action, ['script', self.source / 'code/reproduce_all.py', *options,
                             '--workers', self.workers, '--progress-interval', '30', '--output', folder])
        if action == 'smoke' and not read(folder / 'smoke_verification.json')['passed']:
            raise RuntimeError('The short test did not pass its result/documentation checks.')
        self.output('Completed. Results and documentation: ' + str(folder))

    def execute(self, action):
        if action not in ('doctor', 'verify', 'smoke', 'full', 'resume'):
            raise ValueError('Unknown action: ' + action)
        with workspace_lock(self.workspace):
            self.integrity(); self.doctor()
            if action == 'verify': self.verify()
            elif action != 'doctor': self.compute(action)
