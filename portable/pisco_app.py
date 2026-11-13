"""Native desktop interface to the unchanged scientific reproduction program."""
import argparse
from pathlib import Path
import os
import queue
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import Engine, Stopped, write_json


def gui(args):
    # Do not load a host desktop's Qt plugins into the supplied Qt build.
    for key in ('QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH', 'QT_QPA_PLATFORMTHEME',
                'QT_STYLE_OVERRIDE', 'QML_IMPORT_PATH', 'QML2_IMPORT_PATH'):
        os.environ.pop(key, None)
    os.environ['FONTCONFIG_FILE'] = str(args.bundle / 'tools/fonts.conf')
    os.environ['FONTCONFIG_PATH'] = str(args.bundle / 'tools')
    if sys.platform == 'linux':
        os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')
        os.environ.setdefault('QT_XCB_GL_INTEGRATION', 'none')
        if os.environ['QT_QPA_PLATFORM'] == 'xcb' and not os.environ.get('DISPLAY'):
            raise RuntimeError('No X11/XWayland desktop is available. Command-line checks can run without a display.')
    import faulthandler
    faulthandler.enable()
    from PySide6 import QtCore, QtGui, QtWidgets


    class Window(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.events = queue.Queue(); self.cancel = threading.Event()
            self.thread = None; self.started = None; self.closing = False; self.exit_code = 0
            self.setWindowTitle('PISCO — paper reproduction'); self.resize(920, 690)
            self.setMinimumSize(760, 580)
            outer = QtWidgets.QVBoxLayout(self); outer.setContentsMargins(24, 22, 24, 22); outer.setSpacing(14)
            title = QtWidgets.QLabel('PISCO · Paper reproduction'); title.setObjectName('title'); outer.addWidget(title)
            outer.addWidget(QtWidgets.QLabel('Inspect the saved results or run calculations on your computer.'))
            location = QtWidgets.QHBoxLayout(); location.addWidget(QtWidgets.QLabel('Output folder'))
            self.folder = QtWidgets.QLineEdit(str(args.workspace)); location.addWidget(self.folder, 1)
            self.browse = QtWidgets.QPushButton('Choose…'); self.browse.clicked.connect(self.choose); location.addWidget(self.browse)
            outer.addLayout(location)
            actions = QtWidgets.QGridLayout(); self.buttons = []
            descriptions = [('Check paper results', 'verify'), ('Run short test', 'smoke'),
                            ('Start complete run', 'full'), ('Resume complete run', 'resume')]
            for column, (label, action) in enumerate(descriptions):
                button = QtWidgets.QPushButton(label); button.setMinimumHeight(44)
                button.clicked.connect(lambda checked=False, a=action: self.start(a))
                actions.addWidget(button, 0, column); self.buttons.append(button)
            options = QtWidgets.QHBoxLayout(); options.addWidget(QtWidgets.QLabel('Workers'))
            self.workers = QtWidgets.QSpinBox(); self.workers.setRange(1, 256); self.workers.setValue(args.workers)
            options.addWidget(self.workers); options.addStretch()
            self.check = QtWidgets.QPushButton('Check installation'); self.check.clicked.connect(lambda: self.start('doctor'))
            self.buttons.append(self.check); options.addWidget(self.check)
            self.stop = QtWidgets.QPushButton('Stop'); self.stop.setEnabled(False); self.stop.clicked.connect(self.request_stop); options.addWidget(self.stop)
            outer.addLayout(actions); outer.addLayout(options)
            self.status = QtWidgets.QLabel('Ready. Checking saved results does not start a new optimisation.')
            self.status.setWordWrap(True); self.status.setMinimumHeight(42); outer.addWidget(self.status)
            self.progress = QtWidgets.QProgressBar(); self.progress.setRange(0, 100); self.progress.setValue(0)
            self.progress.setTextVisible(False); outer.addWidget(self.progress)
            self.log = QtWidgets.QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumBlockCount(4000)
            self.log.setPlaceholderText('Progress and messages will appear here. A complete log is saved in the output folder.')
            self.log.setFont(QtGui.QFont('DejaVu Sans Mono', 10)); outer.addWidget(self.log, 1)
            footer = QtWidgets.QHBoxLayout(); self.elapsed = QtWidgets.QLabel('Results and logs remain in your output folder.'); footer.addWidget(self.elapsed, 1)
            open_button = QtWidgets.QPushButton('Open output folder'); open_button.clicked.connect(self.open_output); footer.addWidget(open_button); outer.addLayout(footer)
            self.timer = QtCore.QTimer(self); self.timer.timeout.connect(self.poll); self.timer.start(100)
            self.setStyleSheet('''QWidget { background: #f5f7fa; color: #1c3040; font-size: 14px; }
              QLabel#title { font-size: 25px; font-weight: 600; }
              QLineEdit, QSpinBox, QPlainTextEdit { background: white; border: 1px solid #c8d2dc; border-radius: 5px; padding: 7px; }
              QPushButton { background: #e8eef4; border: 1px solid #b9c8d5; border-radius: 5px; padding: 8px 10px; }
              QPushButton:hover { background: #d9e7f2; } QPushButton:disabled { color: #7f8c98; background: #edf1f5; }
              QProgressBar { border: 0; background: #e0e7ee; border-radius: 3px; max-height: 7px; }
              QProgressBar::chunk { background: #2878a0; border-radius: 3px; }''')
            if args.gui_test: QtCore.QTimer.singleShot(400, lambda: self.start(args.gui_test))

        def choose(self):
            folder = QtWidgets.QFileDialog.getExistingDirectory(self, 'Choose output folder', self.folder.text())
            if folder: self.folder.setText(folder)

        def open_output(self):
            folder = Path(self.folder.text()).expanduser(); folder.mkdir(parents=True, exist_ok=True)
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(folder.resolve())))

        def start(self, action):
            if self.thread and self.thread.is_alive(): return
            if action == 'full' and QtWidgets.QMessageBox.question(self, 'Start the complete calculation?',
                    'The full calculation took about two days on the author’s computer.\n\nStart it now? Existing results will not be overwritten.',
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No) != QtWidgets.QMessageBox.Yes:
                return
            try:
                workspace = Path(self.folder.text()).expanduser().resolve()
                engine = Engine(args.bundle, workspace, self.workers.value(),
                                output=lambda line: self.events.put(('line', line)), cancel=self.cancel)
            except (OSError, ValueError, KeyError) as error:
                QtWidgets.QMessageBox.critical(self, 'Cannot start', str(error)); return
            self.folder.setText(str(workspace)); self.cancel.clear(); self.started = time.monotonic()
            for widget in (*self.buttons, self.folder, self.browse, self.workers): widget.setEnabled(False)
            self.stop.setEnabled(True); self.progress.setRange(0, 0); self.status.setText('Preparing …')
            def work():
                try:
                    engine.execute(action); result = ('done', 'Completed successfully.', 0)
                except Stopped as error: result = ('stopped', str(error), 130)
                except Exception as error: result = ('error', str(error), 1)
                self.events.put(('finished', result))
            self.thread = threading.Thread(target=work, daemon=False); self.thread.start()
            if args.gui_test:
                workspace.mkdir(parents=True, exist_ok=True)
                QtCore.QTimer.singleShot(750, lambda: self.grab().save(str(workspace / 'gui-window.png')))

        def request_stop(self):
            self.cancel.set(); self.stop.setEnabled(False); self.status.setText('Stopping safely; completed results are retained …')

        def closeEvent(self, event):
            if self.thread and self.thread.is_alive():
                if not self.closing and QtWidgets.QMessageBox.question(self, 'Stop and close?',
                        'A task is running. Stop it and close after it has stopped?',
                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No) == QtWidgets.QMessageBox.Yes:
                    self.closing = True; self.request_stop()
                event.ignore()
            else: event.accept()

        def poll(self):
            try:
                while True:
                    kind, value = self.events.get_nowait()
                    if kind == 'line':
                        self.log.appendPlainText(value)
                        if not self.cancel.is_set(): self.status.setText(value[:220])
                    elif kind == 'finished':
                        state, text, self.exit_code = value
                        self.progress.setRange(0, 100); self.progress.setValue(100 if self.exit_code == 0 else 0); self.stop.setEnabled(False)
                        for widget in (*self.buttons, self.folder, self.browse, self.workers): widget.setEnabled(True)
                        self.status.setText(text)
                        if args.gui_test:
                            write_json(Path(self.folder.text()) / 'gui-test.json', {'action': args.gui_test,
                                       'exit_code': self.exit_code, 'status': text, 'window_created': True, 'toolkit': 'Qt'})
                            QtCore.QTimer.singleShot(300, self.close)
                        elif state == 'error': QtWidgets.QMessageBox.critical(self, 'Task could not finish', text)
                        if self.closing: QtCore.QTimer.singleShot(100, self.close)
            except queue.Empty: pass
            if self.started is not None and self.thread and self.thread.is_alive():
                self.elapsed.setText(f'Elapsed: {int(time.monotonic() - self.started)} s · Complete console output is saved in logs/.')

    application = QtWidgets.QApplication([sys.argv[0]])
    application.setStyle('Fusion')
    for font in (args.bundle / 'tools/fonts').rglob('*.ttf'):
        QtGui.QFontDatabase.addApplicationFont(str(font))
    application.setFont(QtGui.QFont('DejaVu Sans', 10))
    window = Window(); window.show(); application.exec()
    code = window.exit_code
    if window.thread:
        window.thread.join()
    window.timer.stop()
    # Destroy Qt objects while Python is fully alive, before interpreter teardown.
    import shiboken6
    shiboken6.delete(window)
    shiboken6.delete(application)
    return code


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--workspace', type=Path, default=Path.home() / 'PISCO-results')
    p.add_argument('--workers', type=int, default=max(1, min(4, os.cpu_count() or 1)))
    p.add_argument('--action', choices=('doctor', 'verify', 'smoke', 'full', 'resume'))
    p.add_argument('--gui-test', choices=('doctor', 'smoke'), help=argparse.SUPPRESS)
    args = p.parse_args()
    # A Windows double-click starts without console handles. Keep diagnostics
    # usable there too, including faulthandler during Qt startup/shutdown.
    if sys.stdout is None or sys.stderr is None:
        args.workspace.mkdir(parents=True, exist_ok=True)
        startup_log = (args.workspace / 'startup-console.txt').open('a', encoding='utf-8', buffering=1)
        if sys.stdout is None: sys.stdout = startup_log
        if sys.stderr is None: sys.stderr = startup_log
    if args.action:
        cancel = threading.Event()
        import signal
        signal.signal(signal.SIGINT, lambda *_: cancel.set())
        signal.signal(signal.SIGTERM, lambda *_: cancel.set())
        try:
            Engine(args.bundle, args.workspace, args.workers, output=lambda line: print(line, flush=True), cancel=cancel).execute(args.action)
            return 0
        except Stopped as error: print(str(error), file=sys.stderr); return 130
        except Exception as error: print('PISCO: ' + str(error), file=sys.stderr); return 1
    try: return gui(args)
    except Exception as error:
        args.workspace.mkdir(parents=True, exist_ok=True)
        log = args.workspace / 'startup-error.txt'; log.write_text(str(error) + '\n', encoding='utf-8')
        print(f'PISCO could not open its window: {error}\nDetails: {log}', file=sys.stderr)
        return 1


if __name__ == '__main__': raise SystemExit(main())
