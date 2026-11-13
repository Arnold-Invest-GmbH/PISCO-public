"""Behaviour tests: protect evidence, stop safely and report real failures."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import Engine, Stopped, read, sha, workspace_lock, write_json


class PortableTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='pisco portable test ')
        self.root = Path(self.temp.name)
        self.bundle = self.root / 'bundle'; self.bundle.mkdir()
        (self.bundle / 'app').mkdir()
        shutil.copy2(Path(__file__).with_name('child.py'), self.bundle / 'app/child.py')
        write_json(self.bundle / 'release.json', {'python_executable': 'unused', 'source_sha256': {}})
        self.messages = []
        self.engine = Engine(self.bundle, self.root / 'work', output=self.messages.append)
        self.engine.python = Path(sys.executable)
        self.engine.workspace.mkdir()

    def tearDown(self): self.temp.cleanup()

    def archive(self, names=None):
        names = names or {'documentation/artifact_manifest.json': '{}'}
        archive = self.bundle / 'results.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            for name, value in names.items(): z.writestr(name, value)
        self.engine.manifest['results_archive'] = {'filename': 'results.zip', 'sha256': sha(archive),
                    'uncompressed_bytes': sum(len(v.encode()) for v in names.values())}
        return archive

    def multipart(self):
        archive = self.archive(); data = archive.read_bytes(); archive.unlink()
        parts = []
        for number, chunk in enumerate((data[:len(data)//2], data[len(data)//2:]), 1):
            path = self.bundle.parent / ('PISCO-results.part' + str(number))
            path.write_bytes(chunk)
            parts.append({'filename': path.name, 'bytes': len(chunk), 'sha256': sha(path)})
        self.engine.manifest['results_archive'].update(parts=parts, bytes=len(data))
        return parts

    def test_parts_assemble_and_cached_archive_works_after_downloads_removed(self):
        parts = self.multipart(); destination = self.engine.archive()
        self.assertEqual((destination / 'documentation/artifact_manifest.json').read_text(), '{}')
        for part in parts: (self.bundle.parent / part['filename']).unlink()
        self.assertEqual(self.engine.archive(), destination)

    def test_missing_part_reports_download_name_without_partial_result(self):
        parts = self.multipart(); (self.bundle.parent / parts[1]['filename']).unlink()
        with self.assertRaisesRegex(RuntimeError, 'PISCO-results.part2'): self.engine.archive()
        self.assertFalse((self.engine.workspace / 'review').exists())

    def test_damaged_part_is_rejected_and_partial_archive_removed(self):
        parts = self.multipart(); part = self.bundle.parent / parts[1]['filename']
        part.write_bytes(b'X' * part.stat().st_size)
        with self.assertRaisesRegex(RuntimeError, 'Damaged result file'): self.engine.archive()
        self.assertFalse(list(self.engine.workspace.rglob('.assembling-results-*')))
        self.assertFalse((self.engine.workspace / 'review').exists())

    def test_cancel_during_parts_preserves_downloads_and_removes_partial_archive(self):
        parts = self.multipart()
        def progress(line):
            if line.startswith('Result file checked: 1/'): self.engine.cancel.set()
        self.engine.output = progress
        with self.assertRaises(Stopped): self.engine.archive()
        self.assertTrue(all((self.bundle.parent / part['filename']).is_file() for part in parts))
        self.assertFalse(list(self.engine.workspace.rglob('.assembling-results-*')))
        self.assertFalse((self.engine.workspace / 'review').exists())

    def test_changed_program_rejected(self):
        f = self.bundle / 'program'; f.write_text('original')
        write_json(self.bundle / 'checksums.json', {'program': sha(f)})
        f.write_text('modified')
        with self.assertRaisesRegex(RuntimeError, 'missing or changed'): self.engine.integrity()

    def test_changed_archive_preserves_existing_results(self):
        archive = self.archive(); archive.write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError, 'missing or damaged'): self.engine.archive()
        self.assertFalse((self.engine.workspace / 'review').exists())

    def test_zip_escape_and_windows_drive_rejected(self):
        for name in ('../escape', 'C:/escape', 'x\\escape'):
            self.archive({name: '{}'})
            with self.assertRaisesRegex(RuntimeError, 'Unsafe archive path'): self.engine.archive()
        self.assertFalse((self.root / 'escape').exists())

    def test_bad_layout_cleans_partial_extraction(self):
        self.archive({'wrong.json': '{}'})
        with self.assertRaisesRegex(RuntimeError, 'Unexpected archive layout'): self.engine.archive()
        self.assertFalse(list(self.engine.workspace.rglob('.unpacking-*')))

    def test_reuse_preserves_saved_files(self):
        self.archive(); destination = self.engine.archive()
        (destination / 'retain').write_text('keep')
        self.assertEqual(self.engine.archive(), destination)
        self.assertEqual((destination / 'retain').read_text(), 'keep')

    def test_unknown_existing_folder_protected(self):
        self.archive(); destination = self.engine.workspace / 'review/code/reproduction_output_corrected'
        destination.mkdir(parents=True); (destination / 'retain').touch()
        with self.assertRaisesRegex(RuntimeError, 'other results'): self.engine.archive()
        self.assertTrue((destination / 'retain').exists())

    def test_second_writer_rejected_and_lock_released(self):
        with workspace_lock(self.engine.workspace):
            with self.assertRaisesRegex(RuntimeError, 'Another PISCO task'):
                with workspace_lock(self.engine.workspace): pass
        with workspace_lock(self.engine.workspace): pass

    def test_full_run_never_overwrites_existing(self):
        folder = self.engine.workspace / 'full-run'; folder.mkdir(); (folder / 'retain').touch()
        with patch.object(self.engine, 'command') as command:
            with self.assertRaisesRegex(RuntimeError, 'already exists'): self.engine.compute('full')
            command.assert_not_called()
        self.assertTrue((folder / 'retain').exists())

    def test_resume_requires_existing_plan_and_original_workers(self):
        with self.assertRaisesRegex(RuntimeError, 'no complete run'): self.engine.compute('resume')
        write_json(self.engine.workspace / 'full-run/experiment_plan.json', {'numerical_execution': {'workers': 12}})
        with self.assertRaisesRegex(RuntimeError, 'worker count: 12'): self.engine.compute('resume')

    def test_failure_exit_and_output_are_preserved(self):
        script = self.root / 'fail.py'; script.write_text('print("failure detail"); raise SystemExit(7)')
        with self.assertRaisesRegex(RuntimeError, 'exit 7'): self.engine.command('failure', ['script', script])
        records = list((self.engine.workspace / 'logs').glob('*.json'))
        self.assertEqual(read(records[0])['exit_code'], 7)
        self.assertIn('failure detail', records[0].with_suffix('.txt').read_text())

    def test_cancellation_reaches_child_and_preserves_completed_file(self):
        script = self.root / 'wait.py'
        script.write_text('from pathlib import Path\nimport time\nPath("completed.txt").write_text("keep")\nprint("ready",flush=True)\nwhile True: time.sleep(.05)\n')
        ready = threading.Event(); caught = []
        self.engine.output = lambda line: ready.set() if line == 'ready' else None
        def run():
            try: self.engine.command('stop-test', ['script', script])
            except Stopped: caught.append(True)
        thread = threading.Thread(target=run); thread.start()
        self.assertTrue(ready.wait(10)); self.engine.cancel.set(); thread.join(10)
        self.assertFalse(thread.is_alive()); self.assertTrue(caught)
        self.assertEqual((self.engine.workspace / 'completed.txt').read_text(), 'keep')
        record = read(next((self.engine.workspace / 'logs').glob('*.json')))
        self.assertTrue(record['cancelled']); self.assertEqual(record['exit_code'], 130)


if __name__ == '__main__': unittest.main()
